"""diagnostics.py - Bias/Variance 성능 진단"""
import numpy as np
from tensor import Tensor


def compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn):
    """학습/테스트 세트의 손실을 계산합니다.

    Returns:
        (train_loss, test_loss) float 튜플
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
    """모델 성능을 진단합니다.

    Args:
        train_loss: 학습 손실
        test_loss: 테스트/검증 손실
        threshold: 판별 임계값

    Returns:
        "high_bias", "high_variance", 또는 "good_fit"
    """
    if train_loss > threshold:
        return "high_bias"
    elif test_loss - train_loss > threshold:
        return "high_variance"
    else:
        return "good_fit"


def learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs,
                   train_sizes=None):
    """다양한 학습 데이터 크기에서 학습 곡선을 계산합니다.

    Args:
        model_fn: 새 모델을 반환하는 호출 가능 객체
        X: 전체 학습 특성 (numpy 배열)
        y: 전체 학습 목표값 (numpy 배열)
        loss_fn: 손실 함수
        optimizer_fn: 모델 파라미터를 받아 옵티마이저를 반환하는 호출 가능 객체
        epochs: 학습 에폭 수
        train_sizes: 비율 리스트 (기본값 [0.2, 0.4, 0.6, 0.8, 1.0])

    Returns:
        dict: {"train_sizes", "train_losses", "val_losses"}
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
