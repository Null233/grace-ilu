# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from distutils.version import LooseVersion

import horovod.tensorflow as hvd
import tensorflow as tf
from horovod.tensorflow.gradient_aggregation import LocalGradientAggregationHelper
from horovod.tensorflow.gradient_aggregation_eager import LocalGradientAggregationHelperEager
from horovod.tensorflow.mpi_ops import rank


_PRE_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion('2.4.0')


def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 grace, compression, sparse_as_dense, gradient_predivide_factor,
                                 op, backward_passes_per_step=1,
                                 average_aggregated_gradients=False,
                                 num_groups=0):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        _HAS_AGGREGATE_GRAD = True

        def __init__(self, **kwargs):
            self._name = name or "Distributed%s" % self.__class__.__base__.__name__
            self._aggregated_gradients = False

            self._allreduce_grads = hvd._make_allreduce_grads_fn(
                self._name,
                device_dense,
                device_sparse,
                grace,
                compression,
                sparse_as_dense,
                op,
                gradient_predivide_factor,
                num_groups)

            self._agg_helper = None
            if backward_passes_per_step > 1:
                if hvd._executing_eagerly():
                    self._agg_helper = LocalGradientAggregationHelperEager(
                        backward_passes_per_step=backward_passes_per_step,
                        allreduce_func=self._allreduce_grads,
                        sparse_as_dense=sparse_as_dense,
                        average_aggregated_gradients=average_aggregated_gradients,
                    )
                else:
                    self._agg_helper = LocalGradientAggregationHelper(
                        backward_passes_per_step=backward_passes_per_step,
                        allreduce_func=self._allreduce_grads,
                        sparse_as_dense=sparse_as_dense,
                        average_aggregated_gradients=average_aggregated_gradients,
                        rank=rank(),
                        optimizer_type=LocalGradientAggregationHelper._OPTIMIZER_TYPE_KERAS,
                    )

            super(self.__class__, self).__init__(**kwargs)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            grads = super(self.__class__, self).get_gradients(loss, params)
            gradients = zip(grads, params)
            return self._allreduce(gradients)

        def _aggregate_gradients(self, grads_and_vars):
            grads, vars = list(zip(*grads_and_vars))
            aggregated_grads = self._allreduce(grads)
            if _PRE_TF_2_4_0:
                # Prior to TF 2.4.0, this function was expected to return only a list of
                # grads, not a list of (grad, var) tuples.
                return aggregated_grads
            return list(zip(aggregated_grads, vars))

        def _allreduce(self, grads):
            self._aggregated_gradients = True

            if self._agg_helper:
                return self._agg_helper.compute_gradients(tuple(grads))
            else:
                return self._allreduce_grads(grads)

        def apply_gradients(self, *args, **kwargs):
            if self._agg_helper:
                if isinstance(args[0], zip):
                    # If grad_and_vars are passed in as a zip object
                    # convert to a list. This is necessary for TF2.4+
                    # b/c args[0] is used in both conditional branches
                    # inside _agg_helper.apply_gradients().
                    args = list(args)
                    args[0] = list(args[0])
                    args = tuple(args)

                results = self._agg_helper.apply_gradients(
                    lambda: super(self.__class__, self).apply_gradients(*args, **kwargs),
                    self,
                    *args,
                    **kwargs,
                )
            else:
                results = super(self.__class__, self).apply_gradients(*args, **kwargs)

            if not self._aggregated_gradients:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                                'using TensorFlow 2.0, please specify '
                                '`experimental_run_tf_function=False` in `compile()`.')

            return results

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))

    return cls.from_config(optimizer.get_config())


def _eval(backend, op_or_result):
    if hvd._executing_eagerly():
        return op_or_result
    else:
        return backend.get_session().run(op_or_result)


if hasattr(hvd, 'broadcast_global_variables'):
    def broadcast_global_variables(backend, root_rank):
        return _eval(backend, hvd.broadcast_global_variables(root_rank))


def allreduce(backend, value, name, average, prescale_factor, postscale_factor, op, grace, compression):
    return _eval(backend, hvd.allreduce(tf.constant(value, name=name), average=average,
                                        prescale_factor=prescale_factor,
                                        postscale_factor=postscale_factor,
                                        op=op, grace=grace, compression=compression))


def allgather(backend, value, name):
    return _eval(backend, hvd.allgather(tf.constant(value, name=name)))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, hvd.broadcast(tf.constant(value, name=name), root_rank))


def load_model(keras, wrap_optimizer, optimizer_modules, filepath, custom_optimizers, custom_objects):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ in optimizer_modules
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
