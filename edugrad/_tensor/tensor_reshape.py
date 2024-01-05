"""Contains various tensor manipulation operations that can change the shape of a tensor."""

from __future__ import annotations
from typing import List, Tuple, Union

from edugrad.helpers import argfix, prod, shape_int
import edugrad.function as function


def reshape(tensor: Tensor, shape: Union[int, Tuple[int, ...]], *args) -> Tensor:
    """Reshapes a tensor to the specified new shape.

    Args:
        tensor: The tensor to reshape.
        shape: The new shape for the tensor. Can be an int or a tuple of ints.
        args: Additional arguments for the shape.

    """
    new_shape = argfix(shape, *args)
    # Adjust the shape with special handling for -1 which infers the size from other dimensions
    adjusted_shape = tuple(
        -prod(tensor.shape) // prod(new_shape) if s == -1 else (s if s is not None else tensor.shape[i])
        for i, s in enumerate(new_shape)
    )
    return function.Reshape.apply(tensor, shape=adjusted_shape)


def expand(tensor: Tensor, shape: Union[int, Tuple[int, ...]], *args) -> Tensor:
    """Expands the size of the tensor to the specified shape. -1 in the shape means the corresponding dimension is unchanged.

    Args:
        tensor: The tensor to expand.
        shape: The new shape for the tensor.
        args: Additional arguments for the shape.

    """
    new_shape = argfix(shape, *args)
    # Expand the tensor, allowing -1 to keep the original dimension size
    expanded_shape = tuple(x if x != -1 else s for s, x in zip(tensor.shape, new_shape))
    return function.Expand.apply(tensor, shape=expanded_shape)


def permute(tensor: Tensor, order: Union[int, Tuple[int, ...]], *args) -> Tensor:
    """
    Permutes the tensor dimensions according to the specified order.

    Args:
        tensor: The tensor to permute.
        order: The desired order of dimensions.
        args: Additional arguments for the order.

    """
    new_order = argfix(order, *args)
    return function.Permute.apply(tensor, order=new_order)


def flip(tensor: Tensor, axis: Union[int, List[int]], *args) -> Tensor:
    """
    Flips the tensor along the specified axes.

    Args:
        tensor: The tensor to flip.
        axis: The axis or axes to flip.
        args: Additional arguments for the axis.

    """
    # Normalize axis values to be positive
    normalized_axes = [x if x >= 0 else x + len(tensor.shape) for x in argfix(axis, *args)]
    return function.Flip.apply(tensor, axis=normalized_axes)


def shrink(tensor: Tensor, arg: Tuple[Tuple[shape_int, shape_int] | None, ...]) -> Tensor:
    """
    Shrinks the tensor along each dimension according to the specified start and end indices.

    Args:
        tensor: The tensor to shrink.
        arg: Tuple specifying start and end indices for each dimension.

    """
    # Determine the ranges for shrinking each dimension
    shrink_arg = tuple(x if x is not None else (0, s) for x, s in zip(arg, tensor.shape))
    # Apply shrink operation only if necessary
    if any(x is not None and x != (0, s) for x, s in zip(arg, tensor.shape)):
        return function.Shrink.apply(tensor, arg=shrink_arg)
    return tensor


def pad(tensor: Tensor, arg: Tuple[Tuple[int, int] | None, ...], value: float) -> Tensor:
    """
    Pads the tensor along each dimension with the specified padding values.

    Args:
        tensor: The tensor to pad.
        arg: Padding for each dimension.
        value: The padding value.

    """
    from edugrad.tensor import Tensor
    # Determine padding for each dimension, defaulting to (0, 0)
    pad_arg = tuple(x if x is not None else (0, 0) for x in arg)
    # Apply padding operation if necessary
    if all(x is None or x == (0, 0) for x in arg):
        return tensor
    ret = function.Pad.apply(tensor, arg=pad_arg)
    # Add the padding value if it's different from zero
    return ret if value == 0 else ret + function.Pad.apply(Tensor.ones_like(tensor), arg=pad_arg).where(0, value)


def pad2d(tensor: Tensor, padding: Union[List[int], Tuple[int, ...]], value: float) -> Tensor:
    """
    Pads the tensor with 2D padding specified for each side.

    Args:
        tensor: The tensor to pad.
        padding: Padding for each side (left, right, top, bottom).
        value: The padding value.

    """
    # Calculate the slice indices for the 2D padding
    slice_indices = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], tensor.shape[::-1])][::-1]
    return tensor.slice([(0, s) for s in tensor.shape[: -(len(padding) // 2)]] + slice_indices, value=value)


def transpose(tensor: Tensor, ax1: int, ax2: int) -> Tensor:
    """
    Transposes two dimensions of the tensor.

    Args:
        tensor: The tensor to transpose.
        ax1: The first axis to transpose.
        ax2: The second axis to transpose.

    """
    order = list(range(len(tensor.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return tensor.permute(order)


def _flatten(tensor: Tensor, start_dim: int) -> Tensor:
    """
    Flattens the tensor from the specified start dimension.

    Args:
        tensor: The tensor to flatten.
        start_dim: The dimension from which to start flattening.

    """
    return tensor.reshape(shape=tensor.shape[:start_dim] + (-1,))


def squeeze(tensor: Tensor, dim: Optional[int]) -> Tensor:
    """
    Squeezes the tensor by removing dimensions of size 1.

    Args:
        tensor: The tensor to squeeze.
        dim: The specific dimension to squeeze. If None, all dimensions of size 1 are squeezed.

    """
    if dim is None:
        return tensor if 1 not in tensor.shape else tensor.reshape(*[size for size in tensor.shape if size != 1])
    if dim <= 0 and tensor.ndim == 0:
        return tensor  # Match PyTorch behavior for 0-dimensional tensors
    if not -tensor.ndim <= dim < tensor.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-tensor.ndim}, {tensor.ndim-1}], but got {dim})"
        )
    if dim < 0:
        dim += tensor.ndim
    return tensor if tensor.shape[dim] != 1 else tensor.reshape(*[size for idx, size in enumerate(tensor.shape) if idx != dim])


def unsqueeze(tensor: Tensor, dim: int) -> Tensor:
    """
    Adds a dimension of size 1 to the tensor at the specified position.

    Args:
        tensor: The tensor to unsqueeze.
        dim: The position to add the new dimension.

    """
    if dim < 0:
        dim = len(tensor.shape) + dim + 1
    return tensor.reshape(tensor.shape[:dim] + (1,) + tensor.shape[dim:])
