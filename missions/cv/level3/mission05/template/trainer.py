"""Training utilities."""
import numpy as np
from tensor import Tensor


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters, lr=0.01):
        # TODO: store parameters and learning rate
        pass

    def step(self):
        # TODO: p.data -= lr * p.grad for each parameter
        pass

    def zero_grad(self):
        # TODO: set p.grad = None for each parameter
        pass


def train_epoch(model, X, y, loss_fn, optimizer):
    """One training epoch. Returns loss value."""
    # TODO: zero_grad, forward, loss, backward, step
    pass


def train(model, X, y, loss_fn, optimizer, epochs=100):
    """Full training loop. Returns list of losses."""
    # TODO: call train_epoch for each epoch
    pass
