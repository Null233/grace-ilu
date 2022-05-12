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

from horovod.torch.mpi_ops import size, rank
from grace_dl.torch import 

_DistributedOptimizer = hvd._DistributedOptimizer

class Cross_Barrier_Scheduler(_DistributedOptimizer):
    def __init__(self, model, params, named_parameters, compression,
                 backward_passes_per_step=1, op=Average,
                 gradient_predivide_factor=1.0,
                 groups=None, sparse_as_dense=False,
                 num_steps=10**6):

        self._model = model
        self._groups = groups

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

    def register_hook():
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


_init_logger()