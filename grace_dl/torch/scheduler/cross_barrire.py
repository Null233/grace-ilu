import threading
import logging
try:
    import queue
except ImportError:
    import Queue as queue
import time
import math
import torch
import horovod.torch as hvd

from horovod.common.util import split_list
from horovod.torch.functions import broadcast_object
from horovod.torch.mpi_ops import size, rank
from horovod.torch.mpi_ops import Average
from horovod.torch import Compression

#from grace_dl.torch import 

_DistributedOptimizer = hvd._DistributedOptimizer
_hvd_DistributedOptimizer = hvd.DistributedOptimizer

class _Scheduled_Optimizer(_DistributedOptimizer):
    def __init__(self, model, hvd_opt, num_steps=10**6):

        self._model = model
        self._opt = hvd_opt

        self._logger = logging.getLogger("CrossBarrier")
        self._logger.info("CrossBarrier is enabled.")
        self._logger.debug(" size {}, rank {}".format(size(), rank()))
        self._desc = "rank {}".format(rank())

        self._step = 0
        self._final_step = num_steps

        self._locks = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                self._locks[p] = threading.Lock()

        if size() > 1:
            self._register_forward_hooks()
            self._register_hooks()

            # Poll whether the tensor push-pull is finished.
            self._event_queue = queue.Queue()
            self._poller = threading.Thread(target=self._poll, args=())
            self._poller.start()

    def __getattr__(self, item):
        return getattr(self._opt, item)

    def _get_parameter_name(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        return name

    def _make_hook(self, p):
        def hook(*ignore):
            """if p in self._handles and self._handles[p][0] is not None:
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
                handle, ctx = self._scheduled_allreduce_grad_async(p)
            self._handles[p] = (handle, ctx)"""
            with self._locks[p]:
                self._logger.debug("{} {} finished backward.".format(self._desc, self._get_parameter_name(p)))


    def _register_hook(self):
        if self._groups is not None:
            p_list = []
            # Get list of parameters with grads
            for param_group in self.param_groups:
                for p in param_group['params']:
                    if p.requires_grad:
                        p_list.append(p)

            # To ensure parameter order and group formation is consistent, broadcast p_list order
            # from rank 0 and use for every worker
            p_list_names = [self._parameter_names.get(p) for p in p_list]
            p_list_names = broadcast_object(p_list_names, root_rank=0)
            p_list = sorted(p_list, key=lambda p: p_list_names.index(self._parameter_names.get(p)))

            # Form groups
            if isinstance(self._groups, list):
                p_groups = []
                grouped_id = set()
                p_list_ids = [id(p) for p in p_list]
                for group in self._groups:
                    p_groups.append([p for p in group if id(p) in p_list_ids])
                    for p in p_groups[-1]:
                        grouped_id.add(id(p))
                for p in p_list:
                    if id(p) not in grouped_id:
                        p_groups.append([p])
            else:
                p_groups = split_list(p_list, self._groups)

            p_groups = [tuple(p) for p in p_groups]
            for group in p_groups:
                for p in group:
                    self._p_to_group[p] = group
                self._group_counts[group] = 0

        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _register_forward_hooks(self):
        """Add hook before forward propagation of each layer to block forward computation until the push-pull and
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


def _init_logger():
    logger = logging.getLogger("CrossBarrier")
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s',
                                  '%H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler('cross_barrier.log', 'w')
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
    """Wrap Torch optimizer using BytePS DistributedOptimizer and _CrossBarrier."""
    hvd_opt = _hvd_DistributedOptimizer(optimizer, named_parameters, compression, backward_passes_per_step)
    return _Scheduled_Optimizer(model, hvd_opt, num_steps)

_init_logger()
_init_bsc()