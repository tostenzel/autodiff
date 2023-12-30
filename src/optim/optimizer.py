"""Optimization algorithms for gradient-based learning methods.

This module provides a set of optimizer classes for updating the parameters of a neural network model
in response to the gradients computed during backpropagation.

Conceptual explanations: https://www.tobiasstenzel.com/blog/2023/dl-optimization/

"""
from typing import List
from src.helpers import dedup
from src.tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    Optimizers are algorithms or methods used to change the attributes of the neural network,
    such as weights and learning rate, in order to reduce the losses.

    Attributes:
        params (List[Tensor]): List of parameters to be optimized.
        learning_rate (Tensor): Learning rate for the optimizer.
        buffers (List[Tensor]): Tensors without gradient requirement, typically used for internal states of the optimizer.

    """
    def __init__(self, params: List[Tensor], lr: float):
        # Ensure all parameters are set to require gradients if not already specified
        for x in params:
            if x.requires_grad is None: 
                x.requires_grad = True

        # Deduplicate and filter out tensors that require gradients
        self.params = dedup([x for x in params if x.requires_grad])
        assert len(self.params) != 0, "optimizer must have at least one param"

        # Deduplicate and store buffers
        self.buffers = dedup([x for x in params if not x.requires_grad])

        # Set the learning rate as a tensor
        self.learning_rate = Tensor([lr], requires_grad=False)

    def zero_grad(self):
        """Resets the gradients of all optimized parameters to None.
        
        This should be called before each optimization step.
        """
        for param in self.params:
            param.grad = None


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates model parameters by moving them in the opposite direction of their gradients,
    scaled by the learning rate. Optional momentum and weight decay can be applied.

    The basic update rule for a parameter p with gradient g is:
    - p = p - lr * (g + weight_decay * p)

    If momentum is used, the update rule becomes:
    - v = momentum * v + g
    - p = p - lr * v
    If Nesterov momentum is used, it slightly modifies the update rule.

    Attributes:
        momentum (float): Momentum factor.
        weight_decay (float): Weight decay for regularization.
        nesterov (bool): Whether to use Nesterov momentum.
        buffer (List[Tensor]): Buffer storing the momentum values for each parameter.

    Args:
        params (List[Tensor]): List of parameters to optimize.
        lr (float, optional): Learning rate.
        momentum (int, optional): Momentum factor.
        weight_decay (float, optional): Weight decay coefficient.
        nesterov (bool, optional): Whether to use Nesterov momentum.

    """
    def __init__(self, params: List[Tensor], lr: float=0.001, momentum: int=0, weight_decay: float=0.0, nesterov: bool=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Initialize the momentum buffer if momentum is used
        self.buffer = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params] if self.momentum else []

    def step(self):
        """Performs a single optimization step.

        Updates the parameters based on the gradients, learning rate, momentum, and weight decay.
        If Nesterov momentum is used, it modifies the update accordingly.

        """
        for i, t in enumerate(self.params):
            assert t.grad is not None
            grad = t.grad + self.weight_decay * t.detach()
            if self.momentum:
                self.buffer[i].assign(self.momentum * self.buffer[i] + grad)
                grad = (grad + self.momentum * self.buffer[i]) if self.nesterov else self.buffer[i]
            t.assign(t.detach() - grad * self.learning_rate)


def AdamW(params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    """Variant of the Adam optimizer that includes weight decay."""
    return Adam(params, lr, b1, b2, eps, wd)


class Adam(Optimizer):
    """Adam optimizer for gradient-based optimization of stochastic objective functions.

    Adam is an algorithm for first-order gradient-based optimization of stochastic
    objective functions, based on adaptive estimates of lower-order moments.
    
    The update rule for each parameter is:
    - Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
    - Update biased second moment estimate: v = beta2 * v + (1 - beta2) * g^2
    - Compute bias-corrected first moment estimate: m_hat = m / (1 - beta1^t)
    - Compute bias-corrected second moment estimate: v_hat = v / (1 - beta2^t)
    - Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    Where:
    - m and v are estimates of the first moment (mean) and second moment (uncentered variance) of the gradients.
    - g is the gradient.
    - beta1 and beta2 are the exponential decay rates for moment estimates.
    - epsilon is a small scalar used to prevent division by zero in the implementation.
    - t is the timestep.
    - lr is the learning rate.
    - p is the parameter to be updated.

    Attributes:
        beta1 (float): The exponential decay rate for the first moment estimates.
        beta2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay coefficient for regularization.
        moments (List[Tensor]): List of first moment vectors (moving averages of the gradients).
        velocities (List[Tensor]): List of second moment vectors (moving averages of the squared gradients).
        time_step (Tensor): Counter for the number of steps taken.

    Args:
        params (List[Tensor]): List of parameters to optimize.
        lr (float, optional): Learning rate. Default: 0.001.
        b1 (float, optional): Exponential decay rate for the first moment estimates. Default: 0.9.
        b2 (float, optional): Exponential decay rate for the second moment estimates. Default: 0.999.
        eps (float, optional): Term added to the denominator to improve numerical stability. Default: 1e-6.

    """
    def __init__(self, params: List[Tensor], lr: float=0.001, b1: float=0.9, b2: float=0.999, eps: float=1e-6, wd: float=0.0):
        super().__init__(params, lr)
        self.beta1 = b1
        self.beta2 = b2
        self.epsilon = eps
        self.weight_decay = wd

        # Initialize time step, moments, and velocities for the optimizer
        self.time_step = Tensor([0], requires_grad=False)
        self.moments = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]
        self.velocities = [Tensor.zeros(*t.shape, requires_grad=False) for t in self.params]

    def step(self):
        """Performs a single optimization step.

        Updates the parameters based on the Adam algorithm. Applies adaptive learning rates
        for each parameter.
        """
        self.time_step.assign(self.time_step + 1)
        for i, t in enumerate(self.params):
            assert t.grad is not None
            # Compute moments and velocities
            grad = t.grad
            self.moments[i].assign(self.beta1 * self.moments[i] + (1.0 - self.beta1) * grad)
            self.velocities[i].assign(self.beta2 * self.velocities[i] + (1.0 - self.beta2) * (grad * grad))

            # Compute bias-corrected moments
            m_hat = self.moments[i] / (1.0 - self.beta1 ** self.time_step)
            v_hat = self.velocities[i] / (1.0 - self.beta2 ** self.time_step)

            # Compute the update with weight decay
            update = (m_hat / (v_hat.sqrt() + self.epsilon)) + self.weight_decay * t.detach()

            # Update the parameter
            t.assign(t.detach() - self.learning_rate * update)
