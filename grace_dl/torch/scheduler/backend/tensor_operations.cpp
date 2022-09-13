#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <algorithm>
#include <cmath>

using namespace std;

namespace torch {

vector<::torch::Tensor> _tensor_clipping_cpp(::torch::Tensor& tensor, const int _splitting_size){
    auto numel = tensor.numel();
    // int64_t chunks = numel / _splitting_size + 1;
    int64_t chunks = ceil((double)numel / (double)_splitting_size);
    return ::torch::chunk(tensor, chunks);
}

::torch::Tensor _tensor_aggregation_cpp(const ::torch::Tensor& p, vector<::torch::Tensor>& tensors){
    // reverse(tensors.begin(), tensors.end());
    auto agg_tensor = ::torch::cat(tensors, 0);
    agg_tensor.reshape_as(p);
    return agg_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_tensor_clipping_cpp", &_tensor_clipping_cpp, "_tensor_clipping_cpp");
    m.def("_tensor_aggregation_cpp", &_tensor_aggregation_cpp, "_tensor_aggregation_cpp");
}

} // namespace torch
