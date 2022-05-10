import torch

from grace_dl.torch import Compressor
from horovod.torch import allreduce_, allgather
from horovod.torch import size

def tensor_clamp(tensor):
    std = (tensor - torch.mean(tensor)) ** 2
    std = torch.sqrt(torch.mean(std))
    c = 2.5 * std.item()
    tensor_ = torch.clamp(tensor, -c, c)
    return tensor_

class TernCompressor(Compressor):
    """Quantize all floating point to ternary from."""

    def __init__(self, tensor_size_threshold, model_layer_threshold, compress_rate, compensate_factor):
        super().__init__()
        self.tensor_size_threshold = tensor_size_threshold
        self.model_layer_threshold = model_layer_threshold
        self.compress_rate = compress_rate
        mul_factor = pow(2, 32//self.compress_rate)
        self.shift_factors = [pow(mul_factor, i) for i in range(self.compress_rate)]
        self._sto_factor = True
        self.compensate_factor = compensate_factor
        self.numels = {}
        self.is_compressed = {}
        self.acc_compensate_cache = {}
        self.index_el = 0
        self.shape{}

    def get_max_scaler(tensor, name):
        scaler = tensor.max().abs().view(1)
        scaler_name = f'{name}.scaler'
        scalers = allgather(scaler, scaler_name)
        unified_scaler = scalers.max()
        return scaler.item()

    def stochastical_binarize_tensor(tensor, scaler, name):
        abs_tensor = torch.abs(tensor)
        sign_tensor = torch.sign(tensor)
        rnd_sample = torch.rand_like(tensor) * scaler
        if self.compensate_factor:
            compensate_tensor = sign_tensor
            compensate_tensor[rnd_sample < abs_tensor] = 0
            self..acc_compensate_cache[name] = compensate_tensor
        sign_tensor[rnd_sample >= abs_tensor] = 0
        return sign_tensor

    def ternary_decoder(encoded_data, shape, name):
        """Decoding the signs to float format """
        scaler_name = f'{name}.scaler'
        scaler = TernCompressor.handles[scaler_name]
        index_original = torch.arange(0, tensor.numel(), device='cuda')
        splits = [torch.div( \
                            encoded_data, \
                            self.shift_factor, \
                            rounding_mode='floor') % \
                  mul_factor for shift_factor in TernCompressor.shift_factors]
        decoded_summed_data = torch.gather(torch.cat(splits, 0), 0, index_original).view(shape)
        decoded_summed_data = decoded_summed_data.sub_(size()).type(torch.float)
        return decoded_summed_data * scaler / size()

    def ternary_encoder(tensor, scaler, name):
        tensor = stochastical_binarize_tensor(tensor, scaler, name)
        sum_all = 0
        e = torch.sign(tensor).type(torch.int) + 1
        redundant_size = self.compress_rate - e.size(dim=0) % self.compress_rate
        e = torch.cat((e, torch.zeros(redundant_size, dtype=torch.int, device='cuda')), 0)
        for split, shift_factor in zip(torch.chunk(e, self.compress_rate), self.shift_factors):
            sum_all += split * shift_factor
        return sum_all

    def is_compressed(name):
        if self.numels[name][0] > self.tensor_size_threshold and \ 
           self.numels[name][1] > len(self.numels)*(1-self.model_layer_threshold)
            return 1
        else:
            return 0

    def compress(tensor, name):
        shape = tensor.shape
        self.shape[name] = shape
        ctx = tensor.dtype
        if not self.numels.get(name):
            self.numels[name] = (tensor.numel(), self.index_el)
            self.index_el += 1
            self.is_compressed[name] = 0
        else:
            self.is_compressed[name] = is_compressed(name)
        if self.is_compressed[name]:
            tensor_compressed = tensor.flatten().requires_grad = False
            if self.compensate_factor and name in self.acc_compensate_cache:
                tensor_compressed.add_(self.acc_compensate_cache[name])
            tensor_compressed = tensor_clamp(tensor_compressed)
            unified_scaler = get_max_scaler(tensor_compressed, name)
            tensor_compressed = ternary_encoder(tensor_compressed, unified_scaler, name)
            return tensor_compressed, ctx, shape
        else:
            return tensor, ctx, shape

    def decompress(tensor, ctx, name):
        if self.is_compressed[name]:
            tensor_decompressed = tensor
            dtype = ctx
            shape = self.shape[name]
            tensor_decompressed = ternary_decoder(tensor, shape, name)
            return tensor_decompressed
        else:
            return tensor