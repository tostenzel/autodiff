"""Contains tensor operations like concatenation, stacking, repeating, and chunking."""

from __future__ import annotations
import math
from functools import reduce
from itertools import accumulate

from edugrad.helpers import all_int


def cat(tensor: Tensor, *args: Tensor, dim: int) -> Tensor:
    """Concatenates the given tensors along a specified dimension.

    Args:
        tensor (Tensor): The first tensor to concatenate.
        *args (Tensor): Additional tensors to concatenate.
        dim (int): The dimension along which to concatenate.

    Returns:
        Tensor: A new tensor resulting from concatenating the given tensors.

    """
    from edugrad.tensor import Tensor

    # Adjust the dimension if negative.
    dim = (dim + len(tensor.shape)) if dim < 0 else dim

    # Ensure all tensors have compatible shapes for concatenation.
    assert all(
        len(y.shape) == len(tensor.shape) and all(y.shape[i] == s for i, s in enumerate(tensor.shape) if i != dim)
        for y in args
    )

    # Prepare arguments for concatenation.
    catargs = [tensor, *args]

    # Assert that tensors are not zero-dimensional.
    assert all(t.shape for t in catargs), "zero-dimensional tensor cannot be concatenated"

    # Calculate shapes and cumulative shapes for slicing.
    shapes = [s.shape[dim] for s in catargs]
    shape_cumsum = [0, *accumulate(shapes)]
    slc = [[(0, 0) for _ in tensor.shape] for _ in catargs]

    # Adjust slices for each tensor.
    for shp, k, s in zip(shapes, shape_cumsum[:-1], slc):
        s[dim] = (k, shape_cumsum[-1] - k - shp)

    # Concatenate by padding and adding tensors.
    return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg, s in zip(catargs, slc)])


def stack(tensors: list[Tensor], dim: int) -> Tensor:
    """Stacks a list of tensors along a new dimension.

    Args:
        tensors (list[Tensor]): The list of tensors to stack.
        dim (int): The dimension along which to stack.

    Returns:
        Tensor: A new tensor resulting from stacking the given tensors.

    """
    # Unsqueeze the first tensor and prepare the rest.
    first = tensors[0].unsqueeze(dim)
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]

    # Delegate checks for shapes and number of dimensions to cat.
    return first.cat(*unsqueezed_tensors, dim=dim)


def repeat(tensor: Tensor, repeats: list[int]) -> Tensor:
    """Repeats a tensor along specified dimensions.

    Args:
        tensor (Tensor): The tensor to repeat.
        repeats (list[int]): The number of repetitions for each dimension.

    Returns:
        Tensor: A new tensor with repeated values.

    """
    base_shape = (1,) * (len(repeats) - tensor.ndim) + tensor.shape
    new_shape = [x for b in base_shape for x in (1, b)]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]

    # Repeat the tensor by reshaping, expanding, and reshaping again.
    return tensor.reshape(new_shape).expand(expand_shape).reshape(final_shape)


def chunk(tensor: Tensor, num: int, dim: int) -> list[Tensor]:
    """Splits a tensor into a specified number of chunks along a given dimension.

    Args:
        tensor (Tensor): The tensor to split.
        num (int): The number of chunks to create.
        dim (int): The dimension along which to split the tensor.

    Returns:
        list[Tensor]: A list of tensors representing the chunks.

    """
    assert all_int(tensor.shape), f"does not support symbolic shape {tensor.shape}"
    dim, step = (dim + tensor.ndim if dim < 0 else dim), math.ceil(tensor.shape[dim] / num)

    # Generate slice parameters for each chunk.
    slice_params = [[slice(None)] * dim + [slice(k, k + step)] for k in range(0, tensor.shape[dim], step)]

    # Create each chunk by slicing the tensor.
    return [tensor[tuple(sl)] for sl in slice_params]
