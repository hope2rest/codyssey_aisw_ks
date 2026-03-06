"""tensor.py - 자동 미분을 지원하는 Tensor 클래스"""
import numpy as np


class Tensor:
    """자동 미분(Autograd)을 지원하는 간단한 Tensor 클래스."""

    def __init__(self, data, requires_grad=False, _children=()):
        # TODO: data를 np.float64 ndarray로 변환
        # data, requires_grad, grad=None, _backward=lambda:None, _prev=set(_children) 저장
        pass

    @property
    def shape(self):
        # TODO: self.data.shape 반환
        pass

    def __add__(self, other):
        # TODO: 요소별 덧셈 및 backward 구현
        # backward에서 브로드캐스팅 처리
        pass

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # TODO: 요소별 곱셈 및 backward 구현
        # backward에서 브로드캐스팅 처리
        pass

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        # TODO: 행렬 곱셈 및 backward 구현
        # backward: self.grad += out.grad @ other.data.T
        #           other.grad += self.data.T @ out.grad
        pass

    def __neg__(self):
        # TODO: 부정 연산 및 backward 구현
        pass

    def __sub__(self, other):
        # TODO: __neg__와 __add__를 사용하여 구현
        pass

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__add__(-self)

    def relu(self):
        # TODO: ReLU 활성화 함수 및 backward 구현
        # backward: grad * (self.data > 0)
        pass

    def sigmoid(self):
        # TODO: Sigmoid 활성화 함수 및 backward 구현
        # s = 1/(1+exp(-x)), backward: grad * s * (1-s)
        pass

    def sum(self):
        # TODO: 모든 요소의 합, backward: ones_like * out.grad
        pass

    def log(self):
        # TODO: 요소별 로그 및 backward 구현
        # backward: grad / x (수치 안정성을 위해 x 클리핑)
        pass

    def backward(self):
        # TODO: 역전파 (Reverse-mode autodiff)
        # 1. 위상 정렬(topological sort) 구축
        # 2. self.grad = ones_like(self.data) 설정
        # 3. 역순으로 _backward() 호출
        pass
