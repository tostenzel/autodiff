"""Contains reduction and transformation functions for tensor operations."""

from __future__ import annotations


from edugrad.dtypes import dtypes
from edugrad.helpers import prod, all_int
from edugrad.function import Function
import edugrad.function as function

# reduce ops


def _reduce(self, fxn: type[Function], axis: int | tuple[int, ...] | None, keepdim) -> Tensor:
    """Applies a reduction operation on the tensor along specified axes.

    This is a generic function used to apply various reduction operations such as sum, max, min, etc.
    It can reduce along a given axis or multiple axes. The keepdim parameter controls whether
    the reduced dimensions are kept in the output tensor with size 1, or removed.

    Args:
        fxn: The reduction function to apply (e.g., Sum, Max).
        axis: The axis or axes to reduce along. If None, reduces along all axes.
        keepdim: Whether to keep the reduced dimensions in the output tensor.

    Returns:
        Tensor: The result of the reduction operation.

    """
    from edugrad.tensor import Tensor

    # Normalize the axis indices and compute the new shape after reduction.
    axis_ = list(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    axis_ = [x if x >= 0 else x + len(self.shape) for x in axis_]
    shape = tuple(s for i, s in enumerate(self.shape) if i not in axis_)

    # Handle edge cases, like reducing an empty tensor.
    if 0 in self.shape and 0 not in shape:
        return Tensor.full(
            tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape,
            {function.Sum: 0, function.Max: -float("inf")}[fxn],
        )

    # Apply the reduction function and reshape the result if necessary.
    ret = fxn.apply(self, new_shape=tuple([1 if i in axis_ else s for i, s in enumerate(self.shape)]))
    return ret if keepdim else ret.reshape(shape=shape)


# ----------------------------------------------------------------------------------------------------------------------
# Functions that use the generic _reduce method for specific reduction operations.


def tsum(tensor: Tensor, axis, keepdim):
    """Computes the sum of elements over the specified axis."""
    return tensor._reduce(function.Sum, axis, keepdim)


def tmax(tensor: Tensor, axis, keepdim):
    """Computes the maximum value of elements over the specified axis."""
    return tensor._reduce(function.Max, axis, keepdim)


def tmin(tensor: Tensor, axis, keepdim):
    """Computes the minimum value of elements over the specified axis."""
    return -tmax((-tensor), axis=axis, keepdim=keepdim)


def mean(tensor: Tensor, axis, keepdim):
    """Computes the mean of elements over the specified axis."""
    assert all_int(tensor.shape), "Does not support symbolic shapes."
    out = tensor.sum(axis=axis, keepdim=keepdim)
    return out.mul(prod(out.shape) / prod(tensor.shape)) if 0 not in tensor.shape else out


def std(tensor: Tensor, axis, keepdim, correction):
    """Computes the standard deviation of elements over the specified axis."""
    assert all_int(tensor.shape), "Does not support symbolic shapes."
    square_sum = ((tensor - tensor.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(prod(tensor.shape) / prod(square_sum.shape) - correction).sqrt()


# ----------------------------------------------------------------------------------------------------------------------
# Functions for softmax and its logarithmic variant, as well as argmax and argmin operations.


def _softmax(tensor: Tensor, axis):
    """Helper function to compute softmax components."""
    m = tensor - tensor.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)


def softmax(tensor: Tensor, axis):
    """Applies the softmax function along the specified axis."""
    _, e, ss = tensor._softmax(axis)
    return e.div(ss)


def log_softmax(tensor: Tensor, axis):
    """Applies the log-softmax function along the specified axis."""
    m, _, ss = tensor._softmax(axis)
    return m - ss.log()


def argmax(tensor: Tensor, axis=None, keepdim=False):
    """Returns the indices of the maximum values along an axis."""
    from edugrad.tensor import Tensor

    if axis is None:
        idx = (tensor == tensor.max(axis)) * Tensor.arange(
            prod(tensor.shape) - 1, -1, -1, dtype=dtypes.int32, requires_grad=False
        ).reshape(tensor.shape)
        return prod(tensor.shape) - idx.max() - 1
    axis = axis + len(tensor.shape) if axis < 0 else axis
    m = tensor == tensor.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(tensor.shape[axis] - 1, -1, -1, dtype=dtypes.int32, requires_grad=False).reshape(
        tensor.shape[axis], *[1] * (tensor.ndim - axis - 1)
    )
    return tensor.shape[axis] - idx.max(axis=axis, keepdim=keepdim) - 1


def argmin(tensor: Tensor, axis=None, keepdim=False):
    """Returns the indices of the minimum values along an axis."""
    return (-tensor).argmax(axis=axis, keepdim=keepdim)
