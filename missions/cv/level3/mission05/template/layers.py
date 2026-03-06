"""layers.py - Tensor 기반 신경망 레이어"""
import numpy as np
from tensor import Tensor


class Linear:
    """완전 연결(Fully Connected) 레이어."""

    def __init__(self, in_features, out_features, init='he'):
        # TODO: W (in_features, out_features)와 b (1, out_features) 초기화
        # 초기화 옵션: 'zero', 'random' (*0.01), 'he' (*sqrt(2/in_features))
        # W와 b 모두 requires_grad=True인 Tensor로 생성
        pass

    def forward(self, x):
        # TODO: x @ W + b 반환
        pass

    def parameters(self):
        # TODO: [self.W, self.b] 반환
        pass


class Sequential:
    """레이어를 순차적으로 연결하는 컨테이너."""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        # TODO: 각 레이어를 순서대로 통과
        pass

    def parameters(self):
        # TODO: .parameters()를 가진 레이어의 모든 파라미터를 수집
        pass


class ReLU:
    """ReLU 활성화 레이어."""
    def forward(self, x):
        return x.relu()


class Sigmoid:
    """Sigmoid 활성화 레이어."""
    def forward(self, x):
        return x.sigmoid()


def mse_loss(predicted, target):
    """평균 제곱 오차(MSE) 손실 함수. Tensor 스칼라를 반환합니다."""
    # TODO: ((predicted - target)^2).sum() / N
    pass


def binary_cross_entropy(predicted, target):
    """이진 교차 엔트로피(BCE) 손실 함수. Tensor 스칼라를 반환합니다."""
    # TODO: -mean(t*log(p) + (1-t)*log(1-p))
    pass
