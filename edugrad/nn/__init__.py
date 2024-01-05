import math
from edugrad.tensor import Tensor
from edugrad.helpers import prod, all_int


class Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.weight = Tensor.kaiming_uniform(out_features, in_features, a=math.sqrt(5))
    # TODO: remove this once we can represent Tensor with int shape in typing
    assert isinstance(self.weight.shape[1], int), "does not support symbolic shape"
    bound = 1 / math.sqrt(self.weight.shape[1])
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor):
    return x.linear(self.weight.transpose(), self.bias)


class BatchNorm2d:
    def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

        if affine: self.weight, self.bias = Tensor.ones(sz), Tensor.zeros(sz)
        else: self.weight, self.bias = None, None

        self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)
        self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

    def __call__(self, x:Tensor):
        if Tensor.training:
            # This requires two full memory accesses to x
            # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
            # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
            batch_mean = x.mean(axis=(0,2,3))
            y = (x - batch_mean.reshape(shape=[1, -1, 1, 1]))
            batch_var = (y*y).mean(axis=(0,2,3))
            batch_invstd = batch_var.add(self.eps).pow(-0.5)

            # NOTE: wow, this is done all throughout training in most PyTorch models
            if self.track_running_stats:
                self.running_mean.assign((1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
                self.running_var.assign((1 - self.momentum) * self.running_var + self.momentum * prod(y.shape)/(prod(y.shape) - y.shape[1]) * batch_var.detach() )    # noqa: E501
                self.num_batches_tracked += 1
        else:
            batch_mean = self.running_mean
            # NOTE: this can be precomputed for static inference. we expand it here so it fuses
            batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()

        return x.batchnorm(self.weight, self.bias, batch_mean, batch_invstd)


# TODO: these Conv lines are terrible
def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        self.weight = self.initialize_weight(out_channels, in_channels, groups)
        assert all_int(self.weight.shape), "does not support symbolic shape"
        bound = 1 / math.sqrt(prod(self.weight.shape[1:]))
        self.bias = Tensor.uniform(out_channels, low=-bound, high=bound) if bias else None

    def __call__(self, x:Tensor):
        return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

    def initialize_weight(self, out_channels, in_channels, groups):
        return Tensor.kaiming_uniform(out_channels, in_channels//groups, *self.kernel_size, a=math.sqrt(5))
