"""trainer.py - 학습 유틸리티"""
import numpy as np
from tensor import Tensor


class SGD:
    """확률적 경사 하강법(SGD) 옵티마이저."""

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """그래디언트를 사용하여 파라미터를 업데이트합니다."""
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """모든 파라미터의 그래디언트를 초기화합니다."""
        for p in self.parameters:
            p.grad = None


def train_epoch(model, X, y, loss_fn, optimizer):
    """1 에폭 학습을 수행합니다.

    Args:
        model: forward()와 parameters()를 가진 모델
        X: 입력 numpy 배열
        y: 목표 numpy 배열
        loss_fn: 손실 함수
        optimizer: 옵티마이저

    Returns:
        float 손실값
    """
    optimizer.zero_grad()
    x_tensor = Tensor(X, requires_grad=False)
    y_tensor = Tensor(y, requires_grad=False)

    pred = model.forward(x_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    optimizer.step()

    return float(loss.data)


def train(model, X, y, loss_fn, optimizer, epochs=100):
    """전체 학습 루프.

    Args:
        model: forward()와 parameters()를 가진 모델
        X: 입력 numpy 배열
        y: 목표 numpy 배열
        loss_fn: 손실 함수
        optimizer: 옵티마이저
        epochs: 에폭 수

    Returns:
        에폭별 손실값 리스트
    """
    losses = []
    for _ in range(epochs):
        loss_val = train_epoch(model, X, y, loss_fn, optimizer)
        losses.append(loss_val)
    return losses
