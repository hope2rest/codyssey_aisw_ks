"""Training utilities."""
import numpy as np
from tensor import Tensor


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Update parameters using gradients."""
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.parameters:
            p.grad = None


def train_epoch(model, X, y, loss_fn, optimizer):
    """Run one training epoch.

    Args:
        model: model with forward() and parameters()
        X: input numpy array
        y: target numpy array
        loss_fn: loss function
        optimizer: optimizer

    Returns:
        float loss value
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
    """Full training loop.

    Args:
        model: model with forward() and parameters()
        X: input numpy array
        y: target numpy array
        loss_fn: loss function
        optimizer: optimizer
        epochs: number of epochs

    Returns:
        list of loss values per epoch
    """
    losses = []
    for _ in range(epochs):
        loss_val = train_epoch(model, X, y, loss_fn, optimizer)
        losses.append(loss_val)
    return losses
