from torch.utils.cpp_extension import load


_backend = load(name='tensor_operations',
                extra_cflags=['-O3', '-std=c++14'],
                sources=['grace_dl/torch/scheduler/backend/tensor_operations.cpp'],
                extra_include_paths=['/usr/local/corex/include']
                )
__all__ = ['_backend']