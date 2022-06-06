import threading
import logging

try:
    import queue
except ImportError:
    import Queue as queue
import time
import math
import torch

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import size, rank
from horovod.torch.mpi_ops import Average
from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import synchronize, poll
from horovod.torch.mpi_ops import size

import torch.backends.cudnn as cudnn
import torch.nn.functional as F

# from grace_dl.torch.optimizer_basic import _DistributedOptimizer, DistributedOptimizer
from horovod.torch.optimizer import _DistributedOptimizer, DistributedOptimizer

_hvd_DistributedOptimizer = DistributedOptimizer


class _Scheduled_Optimizer(_DistributedOptimizer):
    def __init__(self, model, hvd_opt, num_steps=10**6, **kwargs):

        self._model = model
        self._opt = hvd_opt
        self._scheduler = True

        self._logger = logging.getLogger("Scheduler")
        self._logger.debug(" size {}, rank {}".format(size(), rank()))
        self._desc = "rank {}".format(rank())

        if self._scheduler:
            # number of split tensors of a certain tensor is
            # p.grad.numel() // self._splitting_size + 1.
            # Split tensor has its own handle.
            # The instant_allreduce method will be exposed to compression algorithm.
            self._scheduler_template = {} # KEY: layer_name; VALUE: ([tensors to converge], [tensors to split])
            self._logger.info("Scheduler is enabled.")
            size_ = max([param.numel() for param in self._model.parameters()])
            self._splitting_size = size_ // 2
            self._time_model = []
            self._window_size = 1
            self._p_to_group = {}
            self._group_counts = {}

        self._step = 0
        self._final_step = num_steps

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

        if size() > 1:
            self._tensor_cache = queue.LifoQueue()
            self._forward_start_times = []
            self._step_called_times = []
            self._register_forward_hooks()
            self._register_hooks()

    """Below are helper methods"""

    def __getattr__(self, item):
        return getattr(self._opt, item)

    def _get_parameter_name(self, p):
        # if self._is_tensor_instance:
        #     name = self._parameter_names.get(p.__hash__())
        # else:
        #     name = self._parameter_names.get(p)
        name = self._parameter_names.get(p)
        return name
    
    def _get_param_layer(self, p):
        name = self._parameter_names.get(p)
        return name.split('.')[0]

    def _update_gradient(self, p):
        # So only support SGD, Adam and RMSprop optimizers in torch
        if isinstance(self._opt, torch.optim.SGD):
            self._sgd(p)
        elif isinstance(self._opt, torch.optim.Adam):
            self._adam(p)
        elif isinstance(self._opt, torch.optim.RMSprop):
            self._rmsprop(p)
        else:
            raise ValueError(
                "Invalid optimizer! Only support SGD, Adam and RMSprop.")
        self._zero_one_grad(p)

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as torch accumulates gradients by default.
        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    def _construct_template(self):
        template = {}
        layers = dict(self._model.named_children()).keys()
        ps = list(self._model.parameters())
        for layer in layers:
            tensors_to_converge = []
            tensors_to_split = []
            for p in ps:
                p_layer = self._get_param_layer(p)
                if p_layer == layer:
                    if p.numel() >= self._splitting_size:
                        tensors_to_split.append(p)
                    else:
                        tensors_to_converge.append(p)
            template[layer] = (tensors_to_converge, tensors_to_split)
        return template

    """Actual API for user to call during training process"""

    def zero_grad(self):
        """Override the default zero_grad function.
        Clears the gradients of all optimized tensors.
        """
        self._logger.debug(
            "{} calls zero_grad() of step {}".format(self._desc, self._step))
        if size() > 1 and self._step > 0:
            return
        else:
            self._opt.zero_grad()

    def step(self, closure=None):
        """Override the default step function."""
        # self._logger.debug("{} calls step() {}".format(self._desc, self._step))

        # Step 0 is called for parameter initialization after parameter broadcast
        if size() > 1 and self._step > 0:
            self._step_called_times.append(time.perf_counter())
            self._synchronize()
            # if it is the final training step, wait for the completion of all tensors
            if self._step == self._final_step:
                self._logger.info("training finished!")
            loss = None
            if closure is not None:
                loss = closure()
            self._step += 1
            return loss
        else:
            # Optimizer.step() will be triggered when user calls hvd.broadcast_optimizer_sate()
            super(self._opt.__class__, self._opt).step()
            self._step += 1

    """Below are tensor splitting and aggregation"""

    def _tensor_splitting(self, p, tensor):
        """Called when tensor arrives at _scheduled_allreduce_grad_async()
            to split tensor into multiple tensors based on SIZE.
            Each split tensor has its own handle.
            split tensors will be aggregated in _completion_poll()"""
        tensor = tensor.flatten()
        offset_i = 0
        tensors = []
        num_chunks = 0
        numel = tensor.numel()
        while offset_i < numel:
            next_offset_i = offset_i + self._splitting_size
            begin = offset_i
            end = next_offset_i if next_offset_i <= numel else numel
            tensors.append((tensor[begin:end], num_chunks))
            offset_i = next_offset_i
            num_chunks += 1
        return tensors

    """TODO: Consider alter out-of-place reshape to in-place"""

    def _tensor_aggregation(self, p, split_tensors) -> list:
        # Using LIFO Queue as submission queue will insert split tensors in reverse
        split_tensors.reverse()
        shape = p.grad.shape
        agg_tensor = torch.cat(split_tensors, 0)
        return agg_tensor.view(shape)

    """Below are actual communication methods"""

    def _synchronize(self):
        """Allreduce missing parameters"""
        if self._scheduler:
            self._logger.debug(
                "{} Backward time model: {}".format(self._desc, self._time_model))
            self._time_model = []
        while not self._tensor_cache.empty():
            p, tensor, name, ctx = self._tensor_cache.get()
            handle = allreduce_async_(tensor, name=name, op=self.op,
                                  prescale_factor=self.prescale_factor,
                                  postscale_factor=self.postscale_factor)
            handles, _, __ = self._handles.get(p)
            if handles is not None:
                handles.append(handle)
                self._handles[p] = (handles, _, __)
            else:
                self._handles[p] = ([handle], ctx, True)
        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handles, ctx = self._instant_allreduce_grad_async(p)
            self._handles[p] = (handles, ctx, True)

        for p, value in self._handles.items():
            handles, ctx, enqueued = value
            if handles is None and not enqueued:
                handles, ctx = self._instant_allreduce_grad_async(p)
                self._logger.debug("None handle occures at {}!".format(
                    self._get_parameter_name(p)))
                self._handles[p] = (handles, ctx, True)

    def _instant_allreduce_grad_async(self, p):
        name = self._get_parameter_name(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        handle = allreduce_async_(tensor_compressed, name=name, op=self.op,
                                  prescale_factor=self.prescale_factor,
                                  postscale_factor=self.postscale_factor)
        self._logger.debug("{} calls instant allreduce_async_ for {}".format(
            self._desc, self._get_parameter_name(p)))
        return [handle], ctx

    def _scheduled_allreduce_grad_async(self, p):
        name = self._get_parameter_name(p)
        tensor = p.grad
        tensor_compressed, ctx = self._compression.compress(tensor)
        tensors = self._tensor_splitting(p, tensor_compressed)
        tensors.reverse()
        handles = []
        if len(tensors) > self._window_size:
            
            tensors_to_send = tensors[0 : self._window_size]
            tensors_to_wait = tensors[self._window_size:]
            tensors_to_wait.reverse()
            for t, n in tensors_to_send:
                handle = allreduce_async_(t, name=f'{name}.{str(n)}', op=self.op,
                                  prescale_factor=self.prescale_factor,
                                  postscale_factor=self.postscale_factor)
                handles.append(handle)
            for t, n in tensors_to_wait:
                self._tensor_cache.put((p, t, f'{name}.{str(n)}', ctx))
        else:
            for t, n in tensors:
                handle = allreduce_async_(t, name=f'{name}.{str(n)}', op=self.op,
                                    prescale_factor=self.prescale_factor,
                                    postscale_factor=self.postscale_factor)
                handles.append(handle)
        return handles, ctx

    def _poll_in_hook(self, p):
        handles, ctx, _ = self._handles.get(p)
        # handle_status = {}
        # for p, (handle, ctx, _) in self._handles.items():
        #     handle_status[self._get_parameter_name(p)] = poll(handle)
        # self._logger.debug("{} Handle status {}!".format(
        #             self._desc, handle_status))
        ready_for_aggregation = True
        for handle in handles:
            ready_for_aggregation &= poll(handle)
        if ready_for_aggregation:
            outputs = []
            for handle in handles:
                outputs.append(synchronize(handle))
            aggregated_tensor = self._tensor_aggregation(p, outputs)
            p.grad.set_(self._compression.decompress(aggregated_tensor, ctx))
            self._logger.debug("{} {} finished allreduce".format(
                self._desc, self._get_parameter_name(p)))
            self._allreduce_delay[p] = self.backward_passes_per_step
            self._update_gradient(p)
            if p in self._locks:
                self._locks[p].release()
            return True
        else:
            return False

    """Below are hooks used in forward propagation and backward propagation"""

    def _make_hook(self, p):
        def hook(*ignore):
            if self._scheduler:
                if not len(self._time_model):
                    self._time_model.append((time.perf_counter(), 0))
                else:
                    self._time_model.append(
                        (time.perf_counter(), time.perf_counter() - self._time_model[-1][0]))
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handles, ctx = None, None
            self._allreduce_delay[p] -= 1
            self._locks[p].acquire()
            if self._allreduce_delay[p] == 0:
                if self._scheduler and self._step > 0:
                    handles, ctx = self._scheduled_allreduce_grad_async(p)
                else:
                    handles, ctx = self._instant_allreduce_grad_async(p)
            self._handles[p] = (handles, ctx, True)
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
                    while not self._poll_in_hook(p):
                        continue
                    del self._handles[p]
                if p not in self._locks:
                    continue
                with self._locks[p]:
                    self._logger.debug("{} {} is ready.".format(
                        self._desc, self._get_parameter_name(p)))
            self._logger.debug("{} starts forward {}.".format(self._desc, mod))

        def after_forward_hook(mod, input, result):
            pass
            # self._logger.debug(
            #     "{} finished forward {}.".format(self._desc, mod))

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            self._logger.debug(
                "{} registers forward hook on module {}".format(self._desc, mod))
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
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(d_p, alpha = -group['lr'])
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
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
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
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

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
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
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
                    avg = square_avg.addcmul(-1, grad_avg,
                                             grad_avg).sqrt().add_(group['eps'])
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
                        num_steps=10**6,
                        **kwargs):
    """Wrap Torch optimizer using Horovod DistributedOptimizer and _Scheduler."""
    hvd_opt = _hvd_DistributedOptimizer(
        optimizer, named_parameters, compression, backward_passes_per_step)
    return _Scheduled_Optimizer(model, hvd_opt, num_steps, **kwargs)


_init_logger()
_init_bsc()
