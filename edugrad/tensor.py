"""Contains the tensor class that can be used for building neural networks with forward and backward pass.

The module contains the "high-level ops". These are syntax sugar and built on top of the "mid-level ops" containing the
the functions with forward and backward passes in Function.function which is build on top of the "low-level ops"
defining the numpy backend with the most basic operations in data.TensorData.

The high-level ops support many things that you could expect from a tensor library.

"""
# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time
from typing import ClassVar, Sequence, Any
from collections import defaultdict
import math

import numpy as np

from edugrad.helpers import getenv, DEBUG, DType, dtypes, prod, all_int, round_up, shape_int
from edugrad.data import TensorData
from edugrad.ops import LoadOps
from edugrad.function import Function
import edugrad.function as function
from edugrad.autograd import backward, collect_backward_graph

# fmt: off
from edugrad._tensor.tensor_create import _loadop, empty, manual_seed, rand
from edugrad._tensor.tensor_create import randn, randint, normal, uniform, scaled_uniform
from edugrad._tensor.tensor_create import full, zeros, ones, arange, eye, full_like, zeros_like, ones_like
from edugrad._tensor.tensor_combine_segment import cat, stack, repeat, chunk
from edugrad._tensor.tensor_reshape import reshape, expand, permute, flip, shrink, pad, pad2d, transpose, _flatten, squeeze, unsqueeze
from edugrad._tensor.tensor_nn import _pool, avg_pool2d, max_pool2d, conv2d, linear, binary_crossentropy, binary_crossentropy_logits, sparse_categorical_crossentropy
#from edugrad._tensor.tensor_index_slice import __getitem__, __setitem__, slice, gather
from edugrad._tensor.tensor_broadcasted_binary_mlops import _broadcasted, _to_float, add, sub, mul, div, pow, matmul, maximum, minimum, where
from edugrad._tensor.tensor_reduce import _reduce, tsum, tmax, tmin, mean, std, _softmax, softmax, log_softmax, argmax, argmin
# fmt: on
from edugrad.helpers import argfix, fully_flatten




class Tensor:
    __slots__ = "data", "requires_grad", "grad", "_ctx"
    __deletable__ = ("_ctx",)
    training: ClassVar[bool] = False

    class train:
        def __init__(self, val=True):
            self.val = val

        def __enter__(self):
            self.prev, Tensor.training = Tensor.training, self.val

        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
            Tensor.training = self.prev

    no_grad: ClassVar[bool] = False
    default_type: ClassVar[DType] = dtypes.float32

    def __init__(
        self,
        data: None | int | float | list | TensorData | np.ndarray | bytes,
        dtype: DType | None = None,
        requires_grad: bool | None = None,
    ):
        assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"

        # tensors have gradients, buffers do not
        self.grad: Tensor | None = None

        # NOTE: this can be in three states. False and None: no gradient, True: gradient
        # None (the default) will be updated to True if it's put in an optimizer
        self.requires_grad: bool | None = requires_grad

        # internal variables used for autograd graph construction
        self._ctx: Function | None = None

        if isinstance(data, TensorData):
            assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"

        elif isinstance(data, (int, float)):
            data = TensorData.loadop(LoadOps.CONST, tuple(), dtype or Tensor.default_type, data)

        elif isinstance(data, list):
            if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d):
                dtype = dtype or dtypes.bool
            elif d and all_int(d):
                dtype = dtype or dtypes.default_int
            else:
                dtype = dtype or dtypes.default_float
            # NOTE: cast at the end for the dtypes that do not have a numpy dtype
            data = TensorData(np.array(data, dtype.np)).cast(dtype)
            
        elif isinstance(data, bytes):
            data = TensorData(np.frombuffer(data, np.uint8))

        elif isinstance(data, np.ndarray):
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            if data.shape == ():
                data = TensorData.loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), data.item())
            else:
                data = TensorData(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

        if not isinstance(data, TensorData):
            raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
        self.data = data

    # ------------------------------------------------------------------------------------------------------------------
    # basic properties

    def __repr__(self):
        return f"<Tensor {self.data!r} with grad {(self.grad.data if self.grad else None)!r}>"

    # Python has a non moving garbage collector, so this should be okay
    def __hash__(self):
        return id(self)

    @property
    def shape(self) -> tuple[shape_int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> DType:
        return self.data.dtype

    # ------------------------------------------------------------------------------------------------------------------
    # data handlers

    def assign(self, x) -> Tensor:
        # TODO: this is a hack for writing to DISK
        if x.__class__ is not Tensor:
            x = Tensor(x, dtype=self.dtype)
        assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
        assert not x.requires_grad  # tensor requires_grad is okay?
        if DEBUG >= 4:
            print(f"assign {self.data} <- {x.data}")
        if self.dtype == x.dtype and self.data is not None and not getenv("DISALLOW_ASSIGN"):
            x.data.output_buffer = self.data
        self.data = x.data
        return self

    # ------------------------------------------------------------------------------------------------------------------
    # basic tensor manipulations

    def detach(self) -> Tensor:
        return Tensor(self.data, requires_grad=False)

    def numpy(self) -> np.ndarray:
        assert all_int(self.shape), f"no numpy if shape is symbolic, {self.shape=}"
        assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"
        return self.detach().cast(dtypes.from_np(self.dtype.np)).data.data.reshape(self.shape)

    def item(self) -> float | int:
        return self.numpy().item()

    # fmt: off

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_create.py
    # creation low-level op entrypoint

    @staticmethod
    def _loadop(op, sz, dtype:DType | None=None, arg=None, **kwargs): return _loadop(op, sz, dtype, arg, **kwargs)

    @staticmethod
    def empty(*shape, **kwargs): return empty(*shape, **kwargs)

    _seed: int = int(time.time())
    @staticmethod
    def manual_seed(seed=0): return manual_seed(seed)

    @staticmethod
    def rand(*shape, **kwargs): return rand(*shape, **kwargs)

    # creation helper functions

    @staticmethod
    def full(shape:tuple[shape_int, ...], fill_value, **kwargs): return full(shape, fill_value, **kwargs)

    @staticmethod
    def zeros(*shape, **kwargs): return zeros(*shape, **kwargs)

    @staticmethod
    def ones(*shape, **kwargs): return ones(*shape, **kwargs)

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        return arange(start, stop, step, **kwargs)

    @staticmethod
    def eye(dim:int, **kwargs): return eye(dim, **kwargs)

    def full_like(self, fill_value, **kwargs): return full_like(self, fill_value, **kwargs)
    def zeros_like(self, **kwargs): return zeros_like(self, **kwargs)
    def ones_like(self, **kwargs): return ones_like(self, **kwargs)

    # random number generation high level ops

    @staticmethod
    def randn(*shape, dtype:DType | None=None, **kwargs) -> Tensor: return randn(*shape, dtype=dtype, **kwargs)
    @staticmethod
    def randint(*shape, low=0, high=10, **kwargs) -> Tensor: return randint(*shape, low=low, high=high, **kwargs)
    @staticmethod
    def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor:  return normal(*shape, mean=mean, std=std, **kwargs)
    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        return uniform(*shape, low=low, high=high, **kwargs)
    @staticmethod
    def scaled_uniform(*shape, **kwargs) -> Tensor: return scaled_uniform(*shape, **kwargs)

    def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
        assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
        assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
        weight = self.unsqueeze(0) if self.ndim == 1 else self
        cdf = (cw := weight.cumsum(1)) / cw[:, -1].unsqueeze(1)
        unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1)
        indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
        return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_autograd.py
    # toposort and backward pass

    def collect_backward_graph(self): return collect_backward_graph(self)
    def backward(self): return backward(self)

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_reshape.py
    # movement mlops

    def reshape(self, shape, *args) -> Tensor: return reshape(self, shape, *args)
    # def expand(self, shape, *args) -> Tensor:
    #     return expand(self, shape, *args)
    def expand(self, shape, *args) -> Tensor: return function.Expand.apply(self, shape=tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))]))

    def permute(self, order, *args) -> Tensor: return permute(self, order, *args)
    def flip(self, axis, *args) -> Tensor: return flip(self, axis, *args)
    def pad(self, arg:tuple[tuple[int, int] | None, ...], value:float=0.0) -> Tensor: return pad(self, arg, value)
    # (padding_left, padding_right, padding_top, padding_bottom)
    def pad2d(self, padding:list[int] | tuple[int, ...], value:float=0) -> Tensor: return pad2d(self, padding, value)
    def shrink(self, arg:tuple[tuple[shape_int, shape_int] | None, ...]) -> Tensor: return shrink(self, arg)
    def squeeze(self, dim=None) -> Tensor: squeeze(self, dim)
    def unsqueeze(self, dim) -> Tensor: return unsqueeze(self, dim)

    @property
    def T(self) -> Tensor: return self.transpose()
    def transpose(self, ax1=1, ax2=0) -> Tensor: return transpose(self, ax1, ax2)
    def flatten(self, start_dim=0): return _flatten(self, start_dim)

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_index_slice.py
    # movement high level ops

    # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
    # - A slice i:j returns the elements with indices in [i, j)
    #        - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
    #        - Negative values for i and j are taken relative to the end of the sequence
    #        - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
    # - Indexing with None on a given axis will add a new dimension of size one before that axis
    # - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
    # - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
    # - Strides > 1 and < 0 are now allowed!:
    #        - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
    #        - Idea of stride < 0 support:
    #                - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
    #        - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
    #                - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
    #                - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
    #                    is possible.
    #                - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
    # - Fancy indexing and combined indexing is supported
    #        - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
    #        - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
    #                - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
    #        - There's a special case where a permute is needed at the end:
    #                - if first Tensor passed in (expand dims) is not at dim 0
    #                - and following Tensors does not follow consecutively to the end of fancy indexing's dims
    def __getitem__(self, val) -> Tensor: # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
        def normalize_int(e, i, dim_sz):
            if -dim_sz <= e < dim_sz: return e if e != -1 else dim_sz-1
            raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

        orig_slices = list(val) if isinstance(val, tuple) else [val]
        count = defaultdict(list)
        for i,v in enumerate(orig_slices): count[type(v)].append(i)

        if (num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])) > len(self.shape): raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
        if len(ellipsis_found := count[type(Ellipsis)]) > 1: raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

        valid_slices = [v for v in orig_slices if v is not None]
        valid_slices = [v if isinstance(v, slice) else slice(y_ := normalize_int(v, i, dim_sz), y_+1) if isinstance(v, int) else slice(None) for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))]

        start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((), (), ())
        new_slice = tuple(((0, 0) if e < s else (s, e)) if st > 0 else ((0, 0) if e > s else (e+1, s+1)) for s, e, st in zip(start, stop, strides))
        sliced_tensor = self.shrink(new_slice).flip(axis=[i for i, s in enumerate(strides) if s < 0])
        new_shape = sliced_tensor.shape
        if any(abs(s) != 1 for s in strides):
            strides = tuple(abs(s) for s in strides)
            # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
            padded_tensor = sliced_tensor.pad(tuple((0, s-(dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape)))
            # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
            reshaped_tensor = padded_tensor.reshape(flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
            new_shape = reshaped_tensor.shape[::2]
            # Shrink: do [:, 0]
            sliced_tensor = reshaped_tensor.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

        final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
        for i,s in enumerate(orig_slices):
            if s is None: final_shape.append(1)
            else: # s is int or slice or Tensor
                dim_shape = next(it_shape)
                if isinstance(s, int):
                    dim_collapsed += 1
                else:
                    assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
                    final_shape.append(dim_shape)
                    if isinstance(s, Tensor):
                        tensors.append(s)
                        dim.append(i-dim_collapsed)
        ret = sliced_tensor.reshape(tuple(final_shape))

        if tensors: # Fancy/tensor indexing
            # normalize idx
            # TODO: first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
            idx = [t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t for d,t in zip(dim, tensors)]
            max_dim = max(i.ndim for i in idx)
            # compute sum_dim, arange, and idx
            sum_dim = [d if n==0 else d+max_dim-n for n,d in enumerate(dim)]
            arange = [Tensor.arange(ret.shape[d], dtype=dtypes.int32, requires_grad=False, device=self.device).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n,(sd,d) in enumerate(zip(sum_dim, dim))]
            first_idx = [idx[0].reshape(*[1]*dim[0], *[1]*(1 + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - dim[0] - 1))]
            rest_idx = [i.reshape(*[1]*dim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - dim[0] - n)) for n,i in enumerate(idx[1:], 1)]
            idx = first_idx + rest_idx
            ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])
            # iteratively fancy index
            for a,i,sd in zip(arange, idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
            # special permute case
            if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1]+1)):
                ret_dims = list(range(ret.ndim))
                ret = ret.permute(ret_dims[dim[0]:dim[0]+max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0]+max_dim:])
        return ret

    def __setitem__(self,s,v): return self.__getitem__(s).assign(v)

    # NOTE: using slice is discouraged and things should migrate to pad and shrink
    def slice(self, arg:Sequence[Optional[Tuple[int, sint]]], value:float=0) -> Tensor:
        arg_ = tuple([a if a is not None else (0,s) for s,a in zip(self.shape, arg)])
        padding = tuple([(max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg_)])
        return self.pad(padding, value=value).shrink(tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)]))

    def gather(self: Tensor, idx: Tensor, dim: int) -> Tensor:
        assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
        assert all(s >= i for s,i in zip(self.shape, idx.shape)), "all dim of idx.shape must be smaller than self.shape"
        if dim < 0: dim += self.ndim
        idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)
        permarg = list(range(self.ndim))
        permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
        return ((idx == Tensor.arange(self.shape[dim], dtype=dtypes.int32, requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(tuple([*[(0,sh) for sh in idx.shape[1:-1]], (0,self.shape[dim])])).unsqueeze(0)).sum(-1).transpose(ax1=0, ax2=dim)

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_combine_segment.py

    def cat(self, *args, dim=0) -> Tensor: return cat(self, *args, dim=dim)
    @staticmethod
    def stack(tensors, dim=0) -> Tensor: stack(tensors, dim)
    def repeat(self, repeats) -> Tensor: repeat(self, repeats)
    def chunk(self, num:int, dim:int=0) -> list[Tensor]: chunk(self, num, dim)

    # reduce ops

    def _reduce(self, fxn:Type[Function], axis:int | tuple[int, ...] | None=None, keepdim=False) -> Tensor:
        return _reduce(self, fxn, axis, keepdim)

    def sum(self, axis=None, keepdim=False): return tsum(self, axis, keepdim)
    def max(self, axis=None, keepdim=False): return tmax(self, axis, keepdim)
    def min(self, axis=None, keepdim=False): return tmin(self, axis, keepdim)

    def mean(self, axis=None, keepdim=False): return mean(self, axis, keepdim)
    def std(self, axis=None, keepdim=False, correction=1): return std(self, axis, keepdim, correction)

    def _softmax(self, axis): return _softmax(self, axis)
    def softmax(self, axis=-1): return softmax(self, axis)
    def log_softmax(self, axis=-1): return log_softmax(self, axis)

    def argmax(self, axis=None, keepdim=False): return argmax(self, axis, keepdim)
    def argmin(self, axis=None, keepdim=False): return argmin(self, axis, keepdim)

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_nn.py
    # processing ops

    def _pool(self, k_:tuple[shape_int, ...], stride:tuple[int, ...] | int=1, dilation:tuple[int, ...] | int=1) -> Tensor:
      return _pool(self, k_, stride, dilation)

    # NOTE: these work for more than 2D
    def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return avg_pool2d(self, kernel_size, stride, dilation)
    def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return max_pool2d(self, kernel_size, stride, dilation)

    wino = int(getenv("WINO", "0")) # no winograd convolution
    def conv2d(self, weight:Tensor, bias:Tensor | None=None, groups=1, stride=1, dilation=1, padding=0) -> Tensor:
        return conv2d(self, weight, bias, groups, stride, dilation, padding)

    # ------------------------------------------------------------------------------------------------------------------

    def dot(self, w:Tensor) -> Tensor:
        n1, n2 = len(self.shape), len(w.shape)
        assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
        assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"
        x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
        w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
        return (x*w).sum(-1)

    def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor: return self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
    def cumsum(self, axis:int=0) -> Tensor:
        # TODO: someday the optimizer will find this on it's own
        # for now this is a two stage cumsum
        SPLIT = 256
        if self.shape[axis] <= SPLIT*2:
            return self._cumsum(axis)
        ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
        ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1]//SPLIT, SPLIT)._cumsum(-1)
        base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
        base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
        def fix(x:Tensor): return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[..., -self.shape[axis]:].transpose(axis,-1)
        return fix(ret) + fix(base_add)

    # mlops (unary)

    def neg(self): return function.Neg.apply(self)
    def log(self): return function.Log.apply(self)
    def exp(self): return function.Exp.apply(self)
    def relu(self): return function.Relu.apply(self)
    def sigmoid(self): return function.Sigmoid.apply(self)
    def sqrt(self): return function.Sqrt.apply(self)
    def sin(self): return function.Sin.apply(self)
    def cos(self): return ((math.pi/2)-self).sin()

    # math functions (unary) skipped

    # activation functions (unary)
    def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()

    # ------------------------------------------------------------------------------------------------------------------
    # tensor_bradcasted_binary_mlops.py

    # broadcasted binary mlops

    def _broadcasted(self, y:Tensor | float, reverse:bool=False) -> tuple[Tensor, Tensor]:
        return _broadcasted(self, y, reverse)

    def _to_float(self, x:Tensor | float): return _to_float(self, x)

    def add(self, x:Tensor | float, reverse=False) -> Tensor: return add(self, x, reverse)
    def sub(self, x:Tensor | float, reverse=False) -> Tensor: return sub(self, x, reverse)
    def mul(self, x:Tensor | float, reverse=False) -> Tensor: return mul(self, x, reverse)
    def pow(self, x:Tensor | float, reverse=False) -> Tensor: return pow(self, x, reverse)
    def div(self, x:Tensor | float, reverse=False) -> Tensor: return div(self, x, reverse)
    def matmul(self, x:Tensor, reverse=False) -> Tensor: return matmul(self, x, reverse)

    def maximum(self, x:Tensor | float) -> Tensor: return maximum(self, x)
    def minimum(self, x:Tensor | float) -> Tensor: return minimum(self, x)
    def where(self:Tensor, input_:Tensor | float, other:Tensor | float): return where(self, input_, other)

    # op wrappers (wasted lines to make the typechecker happy)

    def __neg__(self) -> Tensor: return self.neg()

    def __add__(self, x) -> Tensor: return self.add(x)
    def __sub__(self, x) -> Tensor: return self.sub(x)
    def __mul__(self, x) -> Tensor: return self.mul(x)
    def __pow__(self, x) -> Tensor: return self.pow(x)
    def __truediv__(self, x) -> Tensor: return self.div(x)
    def __matmul__(self, x) -> Tensor: return self.matmul(x)

    def __radd__(self, x) -> Tensor: return self.add(x, True)
    def __rsub__(self, x) -> Tensor: return self.sub(x, True)
    def __rmul__(self, x) -> Tensor: return self.mul(x, True)
    def __rpow__(self, x) -> Tensor: return self.pow(x, True)
    def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
    def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)

    def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
    def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
    def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
    def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
    def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
    def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))

    def __lt__(self, x) -> Tensor: return function.Less.apply(*self._broadcasted(x, False))
    def __gt__(self, x) -> Tensor: return function.Less.apply(*self._broadcasted(x, True))
    def __ge__(self, x) -> Tensor: return 1.0-(self<x)
    def __le__(self, x) -> Tensor: return 1.0-(self>x)
    def __ne__(self, x) -> Tensor: return (self<x) + (self>x)     # type: ignore
    def __eq__(self, x) -> Tensor: return 1.0-(self != x)             # type: ignore

    # functional nn ops

    def linear(self, weight:Tensor, bias:Tensor | None=None): return linear(self, weight, bias)

    def binary_crossentropy(self, y:Tensor) -> Tensor: return binary_crossentropy(self, y)

    def binary_crossentropy_logits(self, y:Tensor) -> Tensor: return binary_crossentropy_logits(self, y)

    def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
        return sparse_categorical_crossentropy(self, Y, ignore_index)

    # cast ops

    def cast(self, dtype:DType) -> Tensor: return function.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self
    def bitcast(self, dtype:DType) -> Tensor:
        assert self.dtype.itemsize == dtype.itemsize, "can't bitcast mismatched dtype itemsizes"
        return function.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self
    def float(self) -> Tensor: return self.cast(dtypes.float32)
    def half(self) -> Tensor: return self.cast(dtypes.float16)

    # convenience stuff

    @property
    def ndim(self) -> int: return len(self.shape)
    def numel(self) -> shape_int: return prod(self.shape)
    def element_size(self) -> int: return self.dtype.itemsize
    def nbytes(self) -> int: return self.numel() * self.element_size()
    def is_floating_point(self) -> bool: return dtypes.is_float(self.dtype)
