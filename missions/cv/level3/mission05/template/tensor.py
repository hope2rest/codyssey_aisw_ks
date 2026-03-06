"""Tensor class with automatic differentiation support."""
import numpy as np


class Tensor:
    """A simple tensor with autograd support."""

    def __init__(self, data, requires_grad=False, _children=()):
        # TODO: Convert data to np.float64 ndarray
        # Store data, requires_grad, grad=None, _backward=lambda:None, _prev=set(_children)
        pass

    @property
    def shape(self):
        # TODO: return self.data.shape
        pass

    def __add__(self, other):
        # TODO: element-wise addition with backward for gradient computation
        # Handle broadcasting in backward
        pass

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # TODO: element-wise multiplication with backward
        # Handle broadcasting in backward
        pass

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        # TODO: matrix multiplication with backward
        # backward: self.grad += out.grad @ other.data.T
        #           other.grad += self.data.T @ out.grad
        pass

    def __neg__(self):
        # TODO: negation with backward
        pass

    def __sub__(self, other):
        # TODO: use __neg__ and __add__
        pass

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__add__(-self)

    def relu(self):
        # TODO: ReLU activation with backward
        # backward: grad * (self.data > 0)
        pass

    def sigmoid(self):
        # TODO: Sigmoid activation with backward
        # s = 1/(1+exp(-x)), backward: grad * s * (1-s)
        pass

    def sum(self):
        # TODO: sum all elements, backward: ones_like * out.grad
        pass

    def log(self):
        # TODO: element-wise log with backward
        # backward: grad / x (clip x for numerical stability)
        pass

    def backward(self):
        # TODO: Reverse-mode autodiff
        # 1. Build topological sort
        # 2. Set self.grad = ones_like(self.data)
        # 3. Call _backward() in reverse topological order
        pass
