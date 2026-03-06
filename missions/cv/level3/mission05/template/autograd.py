"""Gradient checking utilities."""
import numpy as np
from tensor import Tensor


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient using central difference.

    Args:
        f: function that takes a numpy array and returns a scalar
        x: numpy array
        eps: perturbation size

    Returns:
        numpy array of numerical gradients
    """
    # TODO: For each element, compute (f(x+eps) - f(x-eps)) / (2*eps)
    pass


def gradient_check(f, x, eps=1e-5, tol=1e-4):
    """Compare analytical vs numerical gradient.

    Args:
        f: function Tensor -> scalar Tensor
        x: numpy array
        eps: perturbation size
        tol: tolerance

    Returns:
        max relative error
    """
    # TODO:
    # 1. Compute analytical gradient via backward()
    # 2. Compute numerical gradient
    # 3. Return max relative error: max(|a-n| / max(|a|+|n|, 1e-8))
    pass
