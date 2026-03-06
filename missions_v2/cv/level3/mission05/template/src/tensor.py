import numpy as np


class Tensor:

    def __init__(self, data, requires_grad=False, _children=()):
        # TODO: data를 np.float64 ndarray로 변환, grad/backward/prev 초기화

    @property
    def shape(self):
        # TODO: self.data.shape 반환

    def __add__(self, other):
        # TODO: 요소별 덧셈 및 backward (브로드캐스팅 처리)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # TODO: 요소별 곱셈 및 backward (브로드캐스팅 처리)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        # TODO: 행렬 곱셈 및 backward

    def __neg__(self):
        # TODO: 부정 연산 및 backward

    def __sub__(self, other):
        # TODO: 뺄셈 (__neg__ + __add__ 활용)

    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other.__add__(-self)

    def relu(self):
        # TODO: ReLU 활성화 및 backward

    def sigmoid(self):
        # TODO: Sigmoid 활성화 및 backward (입력 [-500, 500] 클리핑)

    def sum(self):
        # TODO: 모든 요소의 합 및 backward

    def log(self):
        # TODO: 요소별 로그 및 backward (입력 [1e-12, inf] 클리핑)

    def backward(self):
        # TODO: 위상 정렬 → 역순 _backward() 호출
