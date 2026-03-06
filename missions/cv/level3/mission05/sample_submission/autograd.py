"""Gradient checking utilities."""
import numpy as np
from tensor import Tensor


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient using central difference.

    Args:
        f: function that takes a numpy array and returns a scalar
        x: numpy array at which to compute gradient
        eps: perturbation size

    Returns:
        numpy array of same shape as x with numerical gradients
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        fx_plus = f(x)

        x[idx] = old_val - eps
        fx_minus = f(x)

        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        x[idx] = old_val
        it.iternext()
    return grad


def gradient_check(f, x, eps=1e-5, tol=1e-4):
    """Compare analytical gradient vs numerical gradient.

    Args:
        f: function that takes a Tensor and returns a scalar Tensor
        x: numpy array input
        eps: perturbation for numerical gradient
        tol: tolerance for relative error

    Returns:
        max relative error between analytical and numerical gradients
    """
    # Analytical gradient
    x_tensor = Tensor(x.copy(), requires_grad=True)
    out = f(x_tensor)
    out.backward()
    analytical = x_tensor.grad.copy()

    # Numerical gradient
    def f_np(x_np):
        t = Tensor(x_np.copy(), requires_grad=False)
        return f(t).data.item()

    numerical = numerical_gradient(f_np, x.copy(), eps)

    # Compute max relative error
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    relative_error = np.max(diff / denom)

    return float(relative_error)
