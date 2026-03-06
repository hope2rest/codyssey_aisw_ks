"""Performance diagnostics for bias/variance analysis."""
import numpy as np
from tensor import Tensor


def compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn):
    """Compute train and test loss. Returns (train_loss, test_loss)."""
    # TODO
    pass


def diagnose_bias_variance(train_loss, test_loss, threshold=0.1):
    """Diagnose: 'high_bias', 'high_variance', or 'good_fit'."""
    # TODO:
    # if train_loss > threshold: "high_bias"
    # elif test_loss - train_loss > threshold: "high_variance"
    # else: "good_fit"
    pass


def learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs,
                   train_sizes=None):
    """Compute learning curve at different training set sizes.

    Returns dict with train_sizes, train_losses, val_losses.
    """
    # TODO
    pass
