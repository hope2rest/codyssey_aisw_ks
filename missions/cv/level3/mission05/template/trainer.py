import numpy as np
from tensor import Tensor


class SGD:

    def __init__(self, parameters, lr=0.01):
        # TODO: 파라미터와 학습률 저장

    def step(self):
        # TODO: p.data -= lr * p.grad

    def zero_grad(self):
        # TODO: p.grad = None


def train_epoch(model, X, y, loss_fn, optimizer):
    # TODO: zero_grad → forward → loss → backward → step, 손실값 반환


def train(model, X, y, loss_fn, optimizer, epochs=100):
    # TODO: 각 에폭마다 train_epoch 호출, 손실 리스트 반환
