"""Contains low-level operation entry points and helper functions for tensor creation and manipulation.

It includes functions for creating tensors with specific properties (like being empty, random, or having specific
values) and for random number generation.

"""

from __future__ import annotations
import time
import math

from typing import Any

from edugrad.dtypes import DType, dtypes
from edugrad.helpers import argfix, prod, shape_int
from edugrad.data import TensorData
from edugrad.ops import LoadOps

# -----------------------------------------------------------------------------------------------------------------------
# creation low-level op entrypoint *****


def _loadop(op: LoadOps, sz: int, dtype: DType | None = None, arg: Any = None, **kwargs) -> Tensor:
    """Internal helper function to create a Tensor with a specified operation.

    Args:
    - op: Operation to be performed for tensor creation.
    - sz: Size of the tensor to be created.
    - dtype: Data type of the tensor. Defaults to Tensor's default type if not provided.
    - arg: Additional argument for the operation.
    - kwargs: Additional keyword arguments.

    Returns:
    - Tensor: A new tensor created with the specified operation.

    """
    from edugrad.tensor import Tensor

    assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
    return Tensor(
        TensorData.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, arg), dtype=dtype, **kwargs
    )


def empty(*shape, **kwargs) -> Tensor:
    """Creates an uninitialized tensor with the given shape."""
    from edugrad.tensor import Tensor

    return Tensor._loadop(LoadOps.EMPTY, prod(shape := argfix(*shape)), **kwargs).reshape(shape)


_seed: int = int(time.time())


def manual_seed(seed=0):
    """Sets the manual seed for random number generation."""
    from edugrad.tensor import Tensor

    Tensor._seed = seed


def rand(*shape, **kwargs) -> Tensor:
    """Creates a tensor with elements uniformly distributed between 0 and 1.

    Args:
    - shape: Variable length argument list for the dimensions of the tensor.
    - kwargs: Additional keyword arguments.

    Returns:
    - Tensor: A tensor with random elements uniformly distributed.

    """
    from edugrad.tensor import Tensor

    Tensor._seed += 1
    return Tensor._loadop(LoadOps.RAND, prod(shape := argfix(*shape)), arg=Tensor._seed, **kwargs).reshape(shape)


# ----------------------------------------------------------------------------------------------------------------------
# creation helper functions


def full(shape: tuple[shape_int, ...], fill_value, **kwargs) -> Tensor:
    """Creates a tensor filled entirely with the specified fill value."""
    from edugrad.tensor import Tensor

    return Tensor(fill_value, **kwargs).reshape([1] * len(new_shape := argfix(shape))).expand(new_shape)


def zeros(*shape, **kwargs) -> Tensor:
    """Creates a tensor filled entirely with zeros."""
    from edugrad.tensor import Tensor

    return Tensor.full(argfix(*shape), 0, **kwargs)


def ones(*shape, **kwargs) -> Tensor:
    """Creates a tensor filled entirely with ones."""
    from edugrad.tensor import Tensor

    return Tensor.full(argfix(*shape), 1, **kwargs)


def arange(start: int | float, stop: int | float | None, step: int | float, **kwargs) -> Tensor:
    """Creates a 1D tensor with a sequence of numbers from start to stop with a step size.

    Args:
    - start: The start of the sequence.
    - stop: The end of the sequence.
    - step: The step size between each number in the sequence.
    - kwargs: Additional keyword arguments.

    Returns:
    - Tensor: A 1D tensor containing a sequence of numbers.

    """
    from edugrad.tensor import Tensor

    if stop is None:
        stop, start = start, 0
    return Tensor.full((math.ceil((stop - start) / step),), step, **kwargs).cumsum() + (start - step)


def eye(dim: int, **kwargs) -> Tensor:
    """Creates a 2D identity tensor."""
    from edugrad.tensor import Tensor

    return (
        Tensor.full((dim, 1), 1, **kwargs)
        .pad(((0, 0), (0, dim)))
        .reshape(dim * (dim + 1))
        .shrink(((0, dim * dim),))
        .reshape(dim, dim)
    )


def full_like(tensor, fill_value, **kwargs) -> Tensor:
    """Creates a tensor with the same shape as the given tensor, filled with a specified value."""
    from edugrad.tensor import Tensor

    return Tensor.full(tensor.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", tensor.dtype), **kwargs)


def zeros_like(tensor, **kwargs) -> Tensor:
    """Creates a tensor with the same shape as the given tensor, filled with zeros."""
    return tensor.full_like(0, **kwargs)


def ones_like(tensor, **kwargs) -> Tensor:
    """Creates a tensor with the same shape as the given tensor, filled with ones."""
    return tensor.full_like(1, **kwargs)


# -----------------------------------------------------------------------------------------------------------------------
# random number generation


def randn(*shape, dtype: DType | None, **kwargs) -> Tensor:
    """Creates a tensor with elements sampled from a standard normal distribution."""
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
    """Creates a tensor with elements sampled uniformly from the discrete interval [low, high)."""
    from edugrad.tensor import Tensor

    return (Tensor.rand(*shape, **kwargs) * (high - low) + low).cast(dtypes.int32)


def normal(*shape, mean, std, **kwargs) -> Tensor:
    """Creates a tensor with elements sampled from a normal (Gaussian) distribution."""
    from edugrad.tensor import Tensor

    return (std * Tensor.randn(*shape, **kwargs)) + mean


def uniform(*shape, low, high, **kwargs) -> Tensor:
    """Creates a tensor with elements uniformly distributed over the interval [low, high)."""
    from edugrad.tensor import Tensor

    dtype = kwargs.pop("dtype", Tensor.default_type)
    return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low


def scaled_uniform(*shape, **kwargs) -> Tensor:
    """Creates a scaled tensor with elements uniformly distributed over the interval [-1.0, 1.0)

    It is scaled by the inverse square root of the product of the tensor's shape.

    """
    from edugrad.tensor import Tensor

    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(shape) ** -0.5)
