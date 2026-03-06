"""tensor.py - 자동 미분을 지원하는 Tensor 클래스"""
import numpy as np


class Tensor:
    """자동 미분(Autograd)을 지원하는 간단한 Tensor 클래스."""

    def __init__(self, data, requires_grad=False, _children=()):
        # Tensor 유사 객체 처리 (모듈 간 호환을 위한 덕 타이핑)
        if hasattr(data, 'data') and isinstance(getattr(data, 'data'), np.ndarray):
            data = data.data
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other))

        def _backward():
            if self.requires_grad:
                g = out.grad
                # 브로드캐스팅 처리
                if self.data.shape != out.data.shape:
                    # 브로드캐스트된 차원에 대해 합산
                    ndim_diff = len(out.data.shape) - len(self.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g

            if other.requires_grad:
                g = out.grad
                if other.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(other.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += g

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other))

        def _backward():
            if self.requires_grad:
                g = out.grad * other.data
                if self.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(self.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g

            if other.requires_grad:
                g = out.grad * self.data
                if other.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(other.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += g

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=True, _children=(self, other))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=True, _children=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.__add__(-other)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__add__(-self)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=True, _children=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * (self.data > 0).astype(np.float64)

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=True, _children=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * s * (1 - s)

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.array(np.sum(self.data)), requires_grad=True, _children=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), requires_grad=True, _children=(self,))

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad / np.clip(self.data, 1e-12, None)

        out._backward = _backward
        return out

    def backward(self):
        """위상 정렬을 이용한 역전파(Reverse-mode autodiff)."""
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
