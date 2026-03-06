"""trainer.py - 학습 유틸리티"""
import numpy as np
from tensor import Tensor


class SGD:
    """확률적 경사 하강법(SGD) 옵티마이저."""

    def __init__(self, parameters, lr=0.01):
        # TODO: 파라미터와 학습률 저장
        pass

    def step(self):
        # TODO: 각 파라미터에 대해 p.data -= lr * p.grad
        pass

    def zero_grad(self):
        # TODO: 각 파라미터에 대해 p.grad = None 설정
        pass


def train_epoch(model, X, y, loss_fn, optimizer):
    """1 에폭 학습. 손실값을 반환합니다."""
    # TODO: zero_grad, forward, loss, backward, step
    pass


def train(model, X, y, loss_fn, optimizer, epochs=100):
    """전체 학습 루프. 에폭별 손실 리스트를 반환합니다."""
    # TODO: 각 에폭마다 train_epoch 호출
    pass
