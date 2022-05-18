import threading
import logging

from numpy import clip
try:
    import queue
except ImportError:
    import Queue as queue
import time
import os
import math
import torch

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import size, rank
from horovod.torch.mpi_ops import Average
from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import synchronize, poll
from horovod.torch.mpi_ops import size

from grace_dl.torch.optimizer_basic import _DistributedOptimizer, DistributedOptimizer

_hvd_DistributedOptimizer = DistributedOptimizer

class _Scheduled_Optimizer(_DistributedOptimizer):
    def __init__(self, model, hvd_opt, num_steps=10**6):

        self._model = model
        self._opt = hvd_opt

        self._logger = logging.getLogger("Scheduler")
        self._logger.info("Scheduler is enabled.")
        self._logger.debug(" size {}, rank {}".format(size(), rank()))
        self._desc = "rank {}".format(rank())

        self._step = 0
        self._final_step = num_steps
        self._clipping_size = 4096
        self._sending_window = 2

        self._locks = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()
        self._submission_lock = threading.Lock()

        if size() > 1:
            self._register_forward_hooks()
            self._register_hooks()

            # Poll whether the tensor clipping is finished
            """self._submission_queue = queue.LifoQueue()
            self._submission_poller = threading.Thread(target=self._submission_poll, args=())
            self._submission_poller.start()"""

            # Poll whether the tensor allreduce is finished.
            self._completion_queue = queue.Queue()
            self._completion_poller = threading.Thread(target=self._completion_poll, args=())
            self._completion_poller.start()

    """Below are helper methods"""

    def __getattr__(self, item):
        return getattr(self._opt, item)

    def _get_parameter_name(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        return name

    """Overwrite zero_grad function in horovod._DistributedOptimizer"""

    def zero_grad(self):
        """Override the default zero_grad function.
        Clears the gradients of all optimized tensors.
        """
        self._logger.debug("{} calls zero_grad() of step {}".format(self._desc, self._step))
        if size() > 1 and self._step > 0:
            return
        else:
            self._opt.zero_grad()

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as torch accumulates gradients by default.
        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    """Below are actual communication methods"""

    def _synchronize(self):
        """Allreduce missing parameters"""
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = self._instant_allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, value in self._handles.items():
            handle, ctx = value
            if handle is None:
                handle, ctx = self._instant_allreduce_grad_async(p)
                self._handles[p] = (handle, ctx)

    def step(self, closure=None):
        """Override the default step function."""
        self._logger.debug("{} calls step() {}".format(self._desc, self._step))

        # Step 0 is called for parameter initialization after parameter broadcast
        if size() > 1 and self._step > 0:
            self._synchronize()
            # if it is the final training step, wait for the completion of all tensors
            if self._step == self._final_step:
                self._logger.debug("final step {}, waiting for allreduce completion.".format(self._final_step))
                while not self._completion_queue.empty():
                    sleep_time = os.environ.get('SCHEDULER_INTERVAL')
                    if sleep_time is None:
                        sleep_time = 0.001
                    time.sleep(sleep_time)
                self._completion_queue.put((None, None, None))
                self._completion_poller.join()
                self._logger.info("training finished!")
            loss = None
            if closure is not None:
                loss = closure()
            self._step += 1
            return loss
        else:
            # Optimizer.step() will be triggered when user calls byteps.broadcast_optimizer_sate()
            super(self._opt.__class__, self._opt).step()
            self._step += 1

    def _instant_allreduce_grad_async(self, p):
        name = self._get_parameter_name(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)

        if self.op == Average:
            # Split average operation across pre/postscale factors
            # C++ backend will apply additional 1 / size() factor to postscale_factor for op == Average.
            # Above division process will be performed in operations.cc at line 1030 
            prescale_factor = 1.0 / self.gradient_predivide_factor
            postscale_factor = self.gradient_predivide_factor
        else:
            prescale_factor = 1.0
            postscale_factor = 1.0

        self._locks[p].acquire()
        handle = allreduce_async_(tensor_compressed, name=name, op=self.op,
                                  prescale_factor=prescale_factor,
                                  postscale_factor=postscale_factor)
        self._logger.debug("{} calls allreduce_async_ for {}".format(self._desc, self._get_parameter_name(p)))
        # Add to queue to poll completion
        self._completion_queue.put((p, handle, ctx))
        return handle, ctx

    """Call Horovod API to allreduce gradient asynchronously
        Arguments:
            tensor: The tensor to allreduce.
            name: The name of the tensor.
        Returns:
            an allreduce handle and context

        tensor -> tensor_compressed -> clipped_tensors -> submission_queue.put
        poll(submission_queue) -> allreduce_async ->completion_queue.put(handle)
        poll(completion_queue) -> tensor_allreduced -> p.grad.set(tensor_allreduced)
    """
    """def _scheduled_allreduce_grad_async(self, p):
        
        name = self._get_parameter_name(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)

        self._locks[p].acquire()
        tensors = self._tensor_clipping(tensor_compressed)
        self._submission_queue.put((p, tensors, ctx))
        self._logger.info("{} put to submission queue {}".format(self._desc, self._get_parameter_name(p)))
        return None, ctx"""

    """Poll the completion of the tensor's backward or push-pull from a FIFO completion_queue"""
    def _completion_poll(self):
        while True:
            p, handle, ctx = self._completion_queue.get()
            if p is None:
                self._logger.debug("poller exits.")
                break
            # Check whether the push-pull is finished. If so, start updating parameters.
            if handle is not None and poll(handle):
                output = synchronize(handle)
                p.grad.set_(self._compression.decompress(output, ctx))
                self._logger.debug("{} {} finished push-pull".format(self._desc, self._get_parameter_name(p)))
                self._push_pull_delay[p] = self.backward_passes_per_step
                # So only support SGD, Adam and RMSprop optimizers in torch
                if isinstance(self._opt, torch.optim.SGD):
                    self._sgd(p)
                elif isinstance(self._opt, torch.optim.Adam):
                    self._adam(p)
                elif isinstance(self._opt, torch.optim.RMSprop):
                    self._rmsprop(p)
                else:
                    raise ValueError("Invalid optimizer! Only support SGD, Adam and RMSprop.")
                self._zero_one_grad(p)
                # notify update completion and parameter is ready for forward propagation
                if p in self._locks:
                    self._locks[p].release()
            else:
                self._completion_queue.put((p, handle, ctx))

    """def _submission_poll(self):
        while True:
            if self.op == Average:
                # Split average operation across pre/postscale factors
                # C++ backend will apply additional 1 / size() factor to postscale_factor for op == Average.
                # Above division process will be performed in operations.cc at line 1030 
                prescale_factor = 1.0 / self.gradient_predivide_factor
                postscale_factor = self.gradient_predivide_factor
            else:
                prescale_factor = 1.0
                postscale_factor = 1.0
            p, tensors, ctx = self._submission_queue.get()
            self._logger.info("{} submission poll got {}".format(self._desc, self._get_parameter_name(p)))
            name = self._get_parameter_name(p)
            handles = []
            for tensor in tensors:
                handle = allreduce_async_(tensor, name=name, op=self.op,
                                    prescale_factor=prescale_factor,
                                    postscale_factor=postscale_factor)
                handles.append(handle)
            self._completion_queue.put((p, handles, ctx))"""

    """Below are tensor clipping and aggregation"""

    """Called when tensor arrives at _scheduled_allreduce_grad_async()
           to clip tensor into multiple tensors based on SIZE.
           Each clipped tensor has its own handle.
           Clipped tensors will be aggregated in _completion_poll()"""
    """def _tensor_clipping(self, tensor):
        return [tensor]"""

    """def _tensor_aggregation(self, p, clipped_tensors):
        name = self._get_parameter_name(p)
        tensor = p.grad
        shape = p.grad.shape
        return clipped_tensors[0]"""

    """Below are hooks used in forward propagation and backward propagation"""

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._instant_allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)
            with self._locks[p]:
                self._logger.debug("{} {} finished backward.".format(self._desc, self._get_parameter_name(p)))

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    """Add hook before forward propagation of each layer to block forward computation until the allreduce and
        parameter update is finished. The blocking is implemented using a lock."""
    def _register_forward_hooks(self):
        # Recursively find all submodules
        submodules = []
        q = queue.LifoQueue()
        for mod in self._model.children():
            q.put(mod)
        while not q.empty():
            mod = q.get()
            if len(list(mod.children())) == 0:
                submodules.append(mod)
            else:
                for m in mod.children():
                    q.put(m)

        def pre_forward_hook(mod, input):
            for p in mod.parameters():
                if p in self._handles:
                    del self._handles[p]
                if p not in self._locks:
                    continue
                with self._locks[p]:
                    self._logger.debug("{} {} is ready.".format(self._desc, self._get_parameter_name(p)))

            self._logger.debug("{} starts forward {}.".format(self._desc, mod))

        def after_forward_hook(mod, input, result):
            self._logger.debug("{} finished forward {}.".format(self._desc, mod))

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            self._logger.debug("{} registers forward hook on module {}".format(self._desc, mod))
            mod.register_forward_pre_hook(pre_forward_hook)
            mod.register_forward_hook(after_forward_hook)

    """Below are the implementations of optimizers, e.g., SGD, Adam, RMSprop.
    The implementation is derived from Torch's code, except that we update one parameter each time."""

    

def _init_logger():
    logger = logging.getLogger("Scheduler")
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s',
                                  '%H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler('scheduler.log', 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    logger.setLevel(logging.INFO)

def _init_bsc():
    """Replace _register_hook() function in _DistributedOptimizer with empty function."""

    def hijack(obj, func_name):
        orig_func = getattr(obj, func_name)
        # print("hijack function {}".format(orig_func))

        def wrapped_func(*args, **kwargs):
            # print("function {} is hijacked to do nothing.".format(orig_func))
            return
        setattr(obj, func_name, wrapped_func)

    hijack(_DistributedOptimizer, '_register_hooks')

def Scheduled_Optimizer(model,
                         optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1,
                         op=Average,
                         gradient_predivide_factor=1.0,
                         num_groups=0, groups=None,
                         sparse_as_dense=False,
                         num_steps=10**6):
    """Wrap Torch optimizer using Horovod DistributedOptimizer and _Scheduler."""
    hvd_opt = _hvd_DistributedOptimizer(optimizer, named_parameters, compression, backward_passes_per_step)
    return _Scheduled_Optimizer(model, hvd_opt, num_steps)

_init_logger()
_init_bsc()