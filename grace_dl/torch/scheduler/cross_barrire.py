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

_DistributedOptimizer = hvd.DistributedOptimizer

class DistributedOptimizer_Cross(_DistributedOptimizer):
    def __init__(self, params, named_parameters, compression,
                 backward_passes_per_step=1, op=Average,
                 gradient_predivide_factor=1.0,
                 groups=None,
                 sparse_as_dense=False):
        super(self.__class__, self).__init__(params, 
                 named_parameters, compression,
                 backward_passes_per_step, op,
                 gradient_predivide_factor,
                 groups,sparse_as_dense)
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