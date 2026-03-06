"""Performance diagnostics for bias/variance analysis."""
import numpy as np
from tensor import Tensor


def compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn):
    """Compute loss on train and test sets.

    Returns:
        (train_loss, test_loss) as floats
    """
    x_tr = Tensor(X_train, requires_grad=False)
    y_tr = Tensor(y_train, requires_grad=False)
    pred_tr = model.forward(x_tr)
    train_loss = float(loss_fn(pred_tr, y_tr).data)

    x_te = Tensor(X_test, requires_grad=False)
    y_te = Tensor(y_test, requires_grad=False)
    pred_te = model.forward(x_te)
    test_loss = float(loss_fn(pred_te, y_te).data)

    return train_loss, test_loss


def diagnose_bias_variance(train_loss, test_loss, threshold=0.1):
    """Diagnose model performance.

    Args:
        train_loss: training loss
        test_loss: test/validation loss
        threshold: decision threshold

    Returns:
        "high_bias", "high_variance", or "good_fit"
    """
    if train_loss > threshold:
        return "high_bias"
    elif test_loss - train_loss > threshold:
        return "high_variance"
    else:
        return "good_fit"


def learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs,
                   train_sizes=None):
    """Compute learning curve at different training set sizes.

    Args:
        model_fn: callable that returns a fresh model
        X: full training features (numpy array)
        y: full training targets (numpy array)
        loss_fn: loss function
        optimizer_fn: callable that takes model parameters and returns optimizer
        epochs: number of training epochs
        train_sizes: list of fractions (default [0.2, 0.4, 0.6, 0.8, 1.0])

    Returns:
        dict with keys: train_sizes, train_losses, val_losses
    """
    if train_sizes is None:
        train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    n = len(X)
    # Use last 20% as validation
    val_n = max(1, int(n * 0.2))
    X_val = X[-val_n:]
    y_val = y[-val_n:]
    X_pool = X[:-val_n]
    y_pool = y[:-val_n]

    result_train_losses = []
    result_val_losses = []

    for frac in train_sizes:
        k = max(1, int(len(X_pool) * frac))
        X_sub = X_pool[:k]
        y_sub = y_pool[:k]

        model = model_fn()
        optimizer = optimizer_fn(model.parameters())

        from trainer import train as train_loop
        train_loop(model, X_sub, y_sub, loss_fn, optimizer, epochs=epochs)

        tr_loss, vl_loss = compute_train_test_loss(
            model, X_sub, y_sub, X_val, y_val, loss_fn
        )
        result_train_losses.append(tr_loss)
        result_val_losses.append(vl_loss)

    return {
        "train_sizes": train_sizes,
        "train_losses": result_train_losses,
        "val_losses": result_val_losses
    }
