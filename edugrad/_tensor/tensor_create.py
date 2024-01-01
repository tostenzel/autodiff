from __future__ import annotations
import time, math

from edugrad.helpers import argfix, DType, prod, shape_int, dtypes
from edugrad.data import TensorData
from edugrad.ops import LoadOps

# -----------------------------------------------------------------------------------------------------------------------
# creation low-level op entrypoint *****


def _loadop(op, sz, dtype: DType | None = None, arg=None, **kwargs):
    from edugrad.tensor import Tensor

    assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
    return Tensor(
        TensorData.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, arg), dtype=dtype, **kwargs
    )


def empty(*shape, **kwargs):
    from edugrad.tensor import Tensor

    return Tensor._loadop(LoadOps.EMPTY, prod(shape := argfix(*shape)), **kwargs).reshape(shape)


_seed: int = int(time.time())


def manual_seed(seed=0):
    from edugrad.tensor import Tensor

    Tensor._seed = seed


def rand(*shape, **kwargs):
    from edugrad.tensor import Tensor

    Tensor._seed += 1
    return Tensor._loadop(LoadOps.RAND, prod(shape := argfix(*shape)), arg=Tensor._seed, **kwargs).reshape(shape)


# ----------------------------------------------------------------------------------------------------------------------
# creation helper functions


def full(shape: tuple[shape_int, ...], fill_value, **kwargs):
    from edugrad.tensor import Tensor

    return Tensor(fill_value, **kwargs).reshape([1] * len(new_shape := argfix(shape))).expand(new_shape)


def zeros(*shape, **kwargs):
    from edugrad.tensor import Tensor

    return Tensor.full(argfix(*shape), 0, **kwargs)


def ones(*shape, **kwargs):
    from edugrad.tensor import Tensor

    return Tensor.full(argfix(*shape), 1, **kwargs)


def arange(start, stop, step, **kwargs):
    from edugrad.tensor import Tensor

    if stop is None:
        stop, start = start, 0
    return Tensor.full((math.ceil((stop - start) / step),), step, **kwargs).cumsum() + (start - step)


def eye(dim: int, **kwargs):
    from edugrad.tensor import Tensor

    return (
        Tensor.full((dim, 1), 1, **kwargs)
        .pad(((0, 0), (0, dim)))
        .reshape(dim * (dim + 1))
        .shrink(((0, dim * dim),))
        .reshape(dim, dim)
    )


def full_like(self, fill_value, **kwargs):
    from edugrad.tensor import Tensor

    return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), **kwargs)


def zeros_like(self, **kwargs):
    return self.full_like(0, **kwargs)


def ones_like(self, **kwargs):
    return self.full_like(1, **kwargs)


# -----------------------------------------------------------------------------------------------------------------------
# random number generation


def randn(*shape, dtype: DType | None, **kwargs) -> Tensor:
    from edugrad.tensor import Tensor

    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand(2, *shape, **kwargs)
    return (
        src[0]
        .mul(2 * math.pi)
        .cos()
        .mul((1 - src[1]).log().mul(-2).sqrt())
        .cast(Tensor.default_type if dtype is None else dtype)
    )


def randint(*shape, low, high, **kwargs) -> Tensor:
    from edugrad.tensor import Tensor

    return (Tensor.rand(*shape, **kwargs) * (high - low) + low).cast(dtypes.int32)


def normal(*shape, mean, std, **kwargs) -> Tensor:
    from edugrad.tensor import Tensor

    return (std * Tensor.randn(*shape, **kwargs)) + mean


def uniform(*shape, low, high, **kwargs) -> Tensor:
    from edugrad.tensor import Tensor

    dtype = kwargs.pop("dtype", Tensor.default_type)
    return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low


def scaled_uniform(*shape, **kwargs) -> Tensor:
    from edugrad.tensor import Tensor

    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(shape) ** -0.5)
