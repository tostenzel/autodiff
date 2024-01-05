"""Consists broadcasted binary operations for Tensors.

These operations provide element-wise arithmetic operations that support broadcasting for tensors of different shapes.

"""
from __future__ import annotations

import math

from edugrad.dtypes import dtypes
import edugrad.function as function


# broadcasted binary mlops


def _broadcasted(tensor: Tensor, y: Tensor | float, reverse: bool = False) -> tuple[Tensor, Tensor]:
    """Prepares two tensors for broadcasting to a common shape.

    Args:
        tensor (Tensor): The first tensor.
        y (Tensor | float): The second tensor or a scalar value.
        reverse (bool): If True, swaps the tensors before broadcasting.

    Returns:
        tuple[Tensor, Tensor]: A tuple of two tensors broadcasted to a common shape.

    """
    from edugrad.tensor import Tensor

    x: Tensor = tensor
    # If y is not a tensor, convert it to a tensor with the same dtype as the input tensor.
    # If the input tensor is empty, return a tensor full of the scalar value y.
    if not isinstance(y, Tensor):
        if 0 in x.shape:
            return x, x.full_like(y)
        y = Tensor(y, requires_grad=False, dtype=tensor.dtype if tensor.dtype != dtypes.bool else dtypes.float32)

    # Swap tensors if reverse is True.
    if reverse:
        x, y = y, x

    # Directly return tensors if they are already the same shape.
    if (xshape := x.shape) == (yshape := y.shape):
        return (x, y)

    # Adjust shapes to make them broadcastable. This is done by prepending 1's to the shape
    # of the shorter tensor until both shapes have the same length.
    shape_delta = len(xshape) - len(yshape)
    if shape_delta > 0:
        y = y.reshape((1,) * shape_delta + yshape)
    elif shape_delta < 0:
        x = x.reshape((1,) * -shape_delta + xshape)

    # Check if tensors are now the same shape. If yes, return them.
    if (xshape := x.shape) == (yshape := y.shape):
        return (x, y)

    # Determine the final shape after broadcasting. This is the element-wise maximum
    # of the shapes of the two tensors.
    shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])

    # Expand tensors to the final broadcasted shape.
    if xshape != shape_ret:
        x = x.expand(shape_ret)
    if yshape != shape_ret:
        y = y.expand(shape_ret)
    return (x, y)


def _to_float(tensor: Tensor, x: Tensor | float):
    """Converts a tensor to float32 dtype.

    Args:
        tensor (Tensor): The reference tensor to check compatibility.
        x (Tensor | float): The tensor or scalar to be converted.

    Returns:
        The converted tensor or the original scalar.

    """
    from edugrad.tensor import Tensor

    return (
        x.data.base.op.arg
        # tensor is not already a Tensor and suitable for certain operations where float32 dtype is required.
        if isinstance(x, Tensor)
        and x.data.is_unrealized_contiguous_const()
        and not x.requires_grad
        and tensor._broadcasted(x)[0].shape == tensor.shape
        else x
    )


def add(tensor: Tensor, x: Tensor | float, reverse=False) -> Tensor:
    """Adds two tensors or a tensor and a scalar."""
    from edugrad.tensor import Tensor

    x = tensor._to_float(x)
    return function.Add.apply(*tensor._broadcasted(x, reverse)) if x.__class__ is Tensor or x else tensor


def sub(tensor: Tensor, x: Tensor | float, reverse=False) -> Tensor:
    """Subtracts two tensors or a tensor and a scalar."""
    from edugrad.tensor import Tensor

    x = tensor._to_float(x)
    return (
        function.Sub.apply(*tensor._broadcasted(x, reverse))
        if x.__class__ is Tensor or x
        else (-tensor if reverse else tensor)
    )


def mul(tensor: Tensor, x: Tensor | float, reverse=False) -> Tensor:
    """Multiplies two tensors or a tensor and a scalar."""
    from edugrad.tensor import Tensor

    x = tensor._to_float(x)
    if x.__class__ is not Tensor and x == 0.0:
        return function.Zero.apply(tensor)
    if x.__class__ is not Tensor and x == -1.0:
        return -tensor
    return function.Mul.apply(*tensor._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else tensor


def div(tensor: Tensor, x: Tensor | float, reverse=False) -> Tensor:
    """Divides two tensors or a tensor and a scalar."""
    from edugrad.tensor import Tensor

    x = tensor._to_float(x)
    return (
        function.Div.apply(*tensor._broadcasted(x, reverse))
        if x.__class__ is Tensor or reverse or not x or not dtypes.is_float(tensor.dtype)
        else tensor.mul(1 / x)
    )


def pow(tensor: Tensor, x: Tensor | float, reverse=False) -> Tensor:
    """Raises a tensor to the power of another tensor or a scalar."""
    from edugrad.tensor import Tensor

    x = tensor._to_float(x)
    if x.__class__ is not Tensor and not reverse:
        # Simple pow identities
        if x < 0:
            return tensor.reciprocal().pow(-x)
        if x == 3.0:
            return tensor * tensor * tensor
        if x == 2.0:
            return tensor * tensor
        if x == 1.0:
            return tensor
        if x == 0.5:
            return tensor.sqrt()
    if not isinstance(x, Tensor) and reverse and x > 0:
        return tensor.mul(math.log(x)).exp()
    ar = tensor.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else tensor.mul(math.log(abs(x))).exp()
    # Correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
    sign = (
        (x * math.pi).cos()
        if isinstance(x, Tensor)
        else math.cos(x * math.pi)
        if not reverse
        else (tensor * math.pi).cos()
    )
    # We only need to correct the sign if the base is negative
    base_sign = (
        (tensor.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1
    ) / -2
    # We need 0 to be positive so we need to correct base_sign when the base is 0
    base_sign = base_sign - (
        1.5
        * (1 - (tensor.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x)))))
    )
    # Inject nan if the base is negative and the power is not an integer
    to_nan = (
        ((x - x.trunc()) * 1e10).abs().clip(0, 1)
        if isinstance(x, Tensor)
        else int(bool(x - int(x)))
        if not reverse
        else ((tensor - tensor.trunc()) * 1e10).abs().clip(0, 1)
    ) * base_sign
    inject_nan = (((-to_nan) * 2) + 1).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)


def matmul(tensor: Tensor, x: Tensor, reverse=False) -> Tensor:
    """Performs matrix multiplication."""
    return x.dot(tensor) if reverse else tensor.dot(x)


def maximum(tensor: Tensor, x: Tensor | float) -> Tensor:
    """Computes the element-wise maximum of two tensors."""
    return (tensor < x).detach().where(x, (tensor > x).detach().where(tensor, (tensor + x) / 2))


def minimum(tensor: Tensor, x: Tensor | float) -> Tensor:
    """Computes the element-wise minimum of two tensors."""
    return -((-tensor).maximum(-x))


def where(tensor: Tensor, input_: Tensor | float, other: Tensor | float):
    """Selects elements from two tensors based on a condition tensor."""
    x_, y = tensor._broadcasted(input_)
    x, z = x_._broadcasted(other)
    return function.Where.apply(x, *y._broadcasted(z))
