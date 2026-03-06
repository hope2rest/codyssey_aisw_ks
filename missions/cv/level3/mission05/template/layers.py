"""Neural network layers built on Tensor."""
import numpy as np
from tensor import Tensor


class Linear:
    """Fully connected linear layer."""

    def __init__(self, in_features, out_features, init='he'):
        # TODO: Initialize W (in_features, out_features) and b (1, out_features)
        # init options: 'zero', 'random' (*0.01), 'he' (*sqrt(2/in_features))
        # Both W and b should be Tensor with requires_grad=True
        pass

    def forward(self, x):
        # TODO: return x @ W + b
        pass

    def parameters(self):
        # TODO: return [self.W, self.b]
        pass


class Sequential:
    """Sequential container for layers."""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        # TODO: pass x through each layer
        pass

    def parameters(self):
        # TODO: collect all parameters from layers that have .parameters()
        pass


class ReLU:
    def forward(self, x):
        return x.relu()


class Sigmoid:
    def forward(self, x):
        return x.sigmoid()


def mse_loss(predicted, target):
    """Mean squared error loss returning Tensor scalar."""
    # TODO: ((predicted - target)^2).sum() / N
    pass


def binary_cross_entropy(predicted, target):
    """Binary cross entropy loss returning Tensor scalar."""
    # TODO: -mean(t*log(p) + (1-t)*log(1-p))
    pass
