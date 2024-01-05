"""Contains functions for computing the Jacobian and performing gradient checks."""

import numpy as np
from edugrad.tensor import Tensor
from typing import Callable, List, Union


def mask_like(like: np.ndarray, mask_inx: Union[int, List[int]], mask_value: float = 1.0) -> np.ndarray:
    """Creates a mask array that is like the input array but with specified values masked.

    Args:
        like (array): The array to mimic in terms of shape.
        mask_inx (int or array-like): Indices to mask.
        mask_value (float, optional): The value to set at the masked indices. Defaults to 1.0.

    Returns:
        array: Masked array with the same shape as `like`.

    """
    mask = np.zeros_like(like).reshape(-1)
    mask[mask_inx] = mask_value
    return mask.reshape(like.shape)


def jacobian(func: Callable, input: Tensor):
    """Computes the Jacobian matrix for a function at a given input.

    Args:
        func: The function for which to compute the Jacobian.
        input: The input tensor at which to evaluate the Jacobian.

    Returns:
        array: Jacobian matrix evaluated at the given input.

    """
    output = func(input)
    ji = input.numpy().reshape(-1).shape[-1]
    jo = output.numpy().reshape(-1).shape[-1]
    J = np.zeros((jo, ji), dtype=np.float32)

    for o in range(jo):
        input.grad = None
        output = func(input)

        # edugrad doesn't support slicing, workaround to select
        # the needed scalar and backpropagate only through it.
        o_scalar = Tensor(mask_like(output.numpy(), o, 1.0)).mul(output).sum()
        o_scalar.backward()

        for i, grad in enumerate(input.grad.numpy().reshape(-1)):
            J[o, i] = grad
    return J


def numerical_jacobian(func: Callable, input: Tensor, eps: float = 1e-3):
    """Computes an approximation of the Jacobian matrix using finite differences.

    Args:
        func: The function for which to approximate the Jacobian.
        input: The input tensor at which to approximate the Jacobian.
        eps: The epsilon for finite differences. Defaults to 1e-3.

    Returns:
        array: Approximated Jacobian matrix.

    """
    output = func(input)

    ji = input.numpy().reshape(-1).shape[-1]
    jo = output.numpy().reshape(-1).shape[-1]
    NJ = np.zeros((jo, ji), dtype=np.float32)

    for i in range(ji):
        eps_perturb = mask_like(input.numpy(), i, mask_value=eps)

        output_perturb_add = func(Tensor(input.numpy() + eps_perturb)).numpy().reshape(-1)
        output_perturb_sub = func(Tensor(input.numpy() - eps_perturb)).numpy().reshape(-1)

        grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2 * eps)

        NJ[:, i] = grad_approx
    return NJ


def gradcheck(func: Callable, input: Tensor, eps: float = 1e-3, atol: float = 1e-3, rtol: float = 1e-3):
    """Performs a gradient check by comparing the Jacobian to its numerical approximation.

    Args:
        func: The function for which to perform the gradient check.
        input : The input tensor for the function.
        eps: Epsilon for finite differences in numerical Jacobian. Defaults to 1e-3.
        atol: Absolute tolerance for np.allclose. Defaults to 1e-3.
        rtol: Relative tolerance for np.allclose. Defaults to 1e-3.

    Returns:
        bool: True if the computed Jacobian is close to its numerical approximation.

    """
    NJ = numerical_jacobian(func, input, eps)
    J = jacobian(func, input)
    return np.allclose(J, NJ, atol=atol, rtol=rtol)
