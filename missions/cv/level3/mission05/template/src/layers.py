import numpy as np
from tensor import Tensor


class Linear:

    def __init__(self, in_features, out_features, init='he'):
        # TODO: W (in_features, out_features), b (1, out_features) 초기화

    def forward(self, x):
        # TODO: x @ W + b 반환

    def parameters(self):
        # TODO: [self.W, self.b] 반환


class Sequential:

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        # TODO: 각 레이어를 순서대로 통과

    def parameters(self):
        # TODO: 모든 레이어의 파라미터 수집


class ReLU:
    def forward(self, x):
        return x.relu()


class Sigmoid:
    def forward(self, x):
        return x.sigmoid()


def mse_loss(predicted, target):
    # TODO: ((predicted - target)^2).sum() / N


def binary_cross_entropy(predicted, target):
    # TODO: -mean(t*log(p) + (1-t)*log(1-p))
