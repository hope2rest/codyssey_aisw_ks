"""Neural network layers built on Tensor."""
import numpy as np
from tensor import Tensor


class Linear:
    """Fully connected linear layer."""

    def __init__(self, in_features, out_features, init='he'):
        """Initialize linear layer.

        Args:
            in_features: number of input features
            out_features: number of output features
            init: weight initialization ('zero', 'random', 'he')
        """
        self.in_features = in_features
        self.out_features = out_features

        if init == 'zero':
            w = np.zeros((in_features, out_features))
        elif init == 'random':
            np.random.seed(42)
            w = np.random.randn(in_features, out_features) * 0.01
        elif init == 'he':
            np.random.seed(42)
            w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        else:
            raise ValueError(f"Unknown init: {init}")

        self.W = Tensor(w, requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x):
        """Forward pass: x @ W + b."""
        return (x @ self.W) + self.b

    def parameters(self):
        """Return list of parameters."""
        return [self.W, self.b]


class Sequential:
    """Sequential container for layers and activation functions."""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            if callable(layer) and not hasattr(layer, 'forward'):
                x = layer(x)
            else:
                x = layer.forward(x)
        return x

    def parameters(self):
        """Return list of all parameters."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class ReLU:
    """ReLU activation as a layer."""

    def forward(self, x):
        return x.relu()


class Sigmoid:
    """Sigmoid activation as a layer."""

    def forward(self, x):
        return x.sigmoid()


def mse_loss(predicted, target):
    """Mean squared error loss.

    Args:
        predicted: Tensor of predictions
        target: Tensor of targets

    Returns:
        Tensor scalar loss
    """
    diff = predicted - target
    sq = diff * diff
    return sq.sum() * Tensor(np.array(1.0 / predicted.data.size))


def binary_cross_entropy(predicted, target):
    """Binary cross entropy loss.

    Args:
        predicted: Tensor of predictions (after sigmoid)
        target: Tensor of targets (0 or 1)

    Returns:
        Tensor scalar loss
    """
    eps = 1e-12
    pred_clipped = Tensor(np.clip(predicted.data, eps, 1 - eps),
                          requires_grad=predicted.requires_grad,
                          _children=predicted._prev)
    # We need to build proper grad flow through clipping
    # Instead, use log on the predicted directly with safety
    log_p = predicted.log()
    one_minus_p = Tensor(np.ones_like(predicted.data)) - predicted
    log_one_minus_p = one_minus_p.log()

    # -(t * log(p) + (1-t) * log(1-p))
    one_minus_t = Tensor(np.ones_like(target.data)) - target
    loss = (target * log_p + one_minus_t * log_one_minus_p)
    neg_loss = -loss
    mean_loss = neg_loss.sum() * Tensor(np.array(1.0 / predicted.data.shape[0]))

    return mean_loss
