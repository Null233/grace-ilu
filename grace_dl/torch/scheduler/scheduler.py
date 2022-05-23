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
        self._scheduler = True

        self._logger = logging.getLogger("Scheduler")
        self._logger.info("Scheduler is enabled.")
        self._logger.debug(" size {}, rank {}".format(size(), rank()))
        self._desc = "rank {}".format(rank())

        self._step = 0
        self._final_step = num_steps
        self._clipping_size = 4096
        self._sending_window = 2

        if self.op == Average:
            self.prescale_factor = 1.0 / self.gradient_predivide_factor
            self.postscale_factor = self.gradient_predivide_factor
        else:
            self.prescale_factor = 1.0 
            self.postscale_factor = 1.0

        self._locks = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()
        self._submission_lock = threading.Lock()

        if size() > 1:
            self._register_forward_hooks()
            self._register_hooks()

            # Poll whether the tensor clipping is finished
            if self._scheduler:
                self._submission_queue = queue.LifoQueue()
                self._submission_poller = threading.Thread(target=self._submission_poll, args=())
                self._submission_poller.start()

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
            self._handles[p] = (handle, ctx, True)

        for p, value in self._handles.items():
            handle, ctx, enqueued = value
            if handle is None and not enqueued:
                handle, ctx = self._instant_allreduce_grad_async(p)
                self._logger.debug("None handle occures at {}!".format(self._get_parameter_name(p)))
                self._handles[p] = (handle, ctx, True)

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
        handle = allreduce_async_(tensor_compressed, name=name, op=self.op,
                                  prescale_factor=self.prescale_factor,
                                  postscale_factor=self.postscale_factor)
        self._logger.debug("{} calls instant allreduce_async_ for {}".format(self._desc, self._get_parameter_name(p)))
        # Add to queue to poll completion
        self._completion_queue.put((p, [handle], ctx))
        return handle, ctx

    def _scheduled_allreduce_grad_async(self, p):
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
        name = self._get_parameter_name(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)

        tensors = self._tensor_clipping(tensor_compressed)
        self._submission_queue.put((p, tensors, ctx))
        self._logger.debug("{} put to submission queue {}".format(self._desc, name))
        return None, ctx

    """TODO: Change handle manipulating process to one tensor with multiple handles"""
    def _completion_poll(self):
        """Poll the completion of the tensor's backward or allreduce from a FIFO completion_queue"""
        while True:
            p, handles, ctx = self._completion_queue.get()
            if p is None:
                self._logger.debug("poller exits.")
                break
            # Check whether the allreduce is finished. If so, start updating parameters.
            _aggregate = False
            for handle in handles:
                outputs = []
                if handle is not None and poll(handle):
                    outputs.append(synchronize(handle))
                    _aggregate = True
                else:
                    self._completion_queue.put((p, handles, ctx))
                    _aggregate = False
                    break
            if _aggregate:
                output = self._tensor_aggregation(p, outputs)
                p.grad.set_(self._compression.decompress(output, ctx))
                self._logger.debug("{} {} finished allreduce".format(self._desc, self._get_parameter_name(p)))
                self._allreduce_delay[p] = self.backward_passes_per_step
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

    def _submission_poll(self):
        while True:
            p, tensors, ctx = self._submission_queue.get()
            self._logger.debug("{} submission poll got {}".format(self._desc, self._get_parameter_name(p)))
            name = self._get_parameter_name(p)
            handles = []
            for tensor in tensors:
                handle = allreduce_async_(tensor, name=name, op=self.op,
                                    prescale_factor=self.prescale_factor,
                                    postscale_factor=self.postscale_factor)
                handles.append(handle)
            self._completion_queue.put((p, handles, ctx))
            if self._handles.get(p) is not None:
                (p_handles, p_ctx, enqueued) = self._handles[p]
                p_handles = [] if p_handles is None else p_handles + handles
                self._handles[p] = (p_handles, p_ctx, enqueued)
            else:
                self._handles[p] = (handles, ctx, True)

    """Below are tensor clipping and aggregation"""

    def _tensor_clipping(self, tensor):
        """Called when tensor arrives at _scheduled_allreduce_grad_async()
           to clip tensor into multiple tensors based on SIZE.
           Each clipped tensor has its own handle.
           Clipped tensors will be aggregated in _completion_poll()"""
        return [tensor]

    def _tensor_aggregation(self, p, clipped_tensors):
        name = self._get_parameter_name(p)
        tensor = p.grad
        shape = p.grad.shape
        return clipped_tensors[0]

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
                self._locks[p].acquire()
                if self._scheduler:
                    handle, ctx = self._scheduled_allreduce_grad_async(p)
                else:
                    handle, ctx = self._instant_allreduce_grad_async(p)
            self._handles[p] = (handle, ctx, True)
        return hook

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _register_forward_hooks(self):
        """Add hook before forward propagation of each layer to block forward computation until the allreduce and
        parameter update is finished. The blocking is implemented using a lock."""
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

    def _sgd(self, p):
        """Performs a single optimization step using SGD optimizer on a parameter.
        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for gp in group['params']:
                if self._get_parameter_name(p) != self._get_parameter_name(gp) or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._get_parameter_name(p)))
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                break

    def _adam(self, p):
        """Performs a single optimization step using Adam optimizer on a parameter.
        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            for gp in group['params']:
                if self._get_parameter_name(p) != self._get_parameter_name(gp) or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._get_parameter_name(p)))
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                break

    def _rmsprop(self, p):
        """Performs a single optimization step using RMSprop optimizer on a parameter.
        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            for gp in group['params']:
                if self._get_parameter_name(p) != self._get_parameter_name(gp) or gp.shape != p.shape:
                    continue
                self._logger.debug("{} is updating {}".format(self._desc, self._get_parameter_name(p)))
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)
                break
    

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
    logger.setLevel(logging.DEBUG)

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