import torch

from grace_dl.torch import Compressor
from horovod.torch.mpi_ops import allgather
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
        super().__init__(average=False)
        self.compress_rate = compress_rate
        mul_factor = pow(2, 32//compress_rate)
        self.shift_factors = [pow(mul_factor, i)
                              for i in range(compress_rate)]

    def get_max_scaler(self, tensor, name):
        scaler = tensor.abs().max().view(1)
        scaler_name = f'{name}.scaler'
        scalers = allgather(scaler, scaler_name)
        unified_scaler = scalers.max().item()
        return unified_scaler

    def stochastical_binarize_tensor(self, tensor, scaler):
        zeros = torch.zeros_like(tensor)
        abs_tensor = torch.abs(tensor)
        sign_tensor = torch.sign(tensor)
        rnd_sample = torch.rand_like(tensor) * scaler
        where_cond = torch.less(rnd_sample, abs_tensor)
        sign_tensor = torch.where(where_cond, sign_tensor, zeros)
        return sign_tensor

    def ternary_decoder(self, encoded_data, scaler, shape):
        """Decoding the signs to float format """
        numel = torch.prod(torch.tensor(shape))
        index_original = torch.arange(0, numel, device=encoded_data.device)
        splits = [torch.div(
            encoded_data,
            shift_factor,
            rounding_mode='floor') %
            shift_factor for shift_factor in self.shift_factors]
        decoded_summed_data = torch.gather(
            torch.cat(splits, 0), 0, index_original).view(shape)
        decoded_summed_data = decoded_summed_data.sub_(
            size()).type(torch.float)
        return decoded_summed_data * scaler / size()

    def ternary_encoder(self, tensor, scaler):
        tensor = self.stochastical_binarize_tensor(tensor, scaler)
        sum_all = 0
        e = torch.sign(tensor).type(torch.int) + 1
        redundant_size = self.compress_rate - \
            e.size(dim=0) % self.compress_rate
        e = torch.cat(
            (e, torch.zeros(redundant_size, dtype=torch.int, device=tensor.device)), 0)
        for split, shift_factor in zip(torch.chunk(e, self.compress_rate), self.shift_factors):
            sum_all += split * shift_factor
        return sum_all

    def compress(self, tensor, name):
        shape = tensor.shape
        ctx = tensor.dtype
        tensor_compressed = tensor_clamp(tensor.flatten())
        unified_scaler = self.get_max_scaler(tensor_compressed, name)
        tensor_compressed = self.ternary_encoder(
            tensor_compressed, unified_scaler)
        return [tensor_compressed], (ctx, shape, unified_scaler)
        # return [tensor], (ctx, shape, unified_scaler)

    def decompress(self, tensors, ctx, name):
        tensor_decompressed, = tensors
        dtype, shape, scaler = ctx
        tensor_decompressed = self.ternary_decoder(
            tensor_decompressed, scaler, shape)
        return tensor_decompressed
