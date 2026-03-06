"""diagnostics.py - Bias/Variance 성능 진단"""
import numpy as np
from tensor import Tensor


def compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn):
    """학습/테스트 손실을 계산합니다. (train_loss, test_loss) 반환."""
    # TODO: 구현하세요
    pass


def diagnose_bias_variance(train_loss, test_loss, threshold=0.1):
    """모델 성능을 진단합니다: 'high_bias', 'high_variance', 또는 'good_fit' 반환."""
    # TODO:
    # train_loss > threshold 이면: "high_bias"
    # test_loss - train_loss > threshold 이면: "high_variance"
    # 그 외: "good_fit"
    pass


def learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs,
                   train_sizes=None):
    """다양한 학습 데이터 크기에서 학습 곡선을 계산합니다.

    Returns:
        dict: {"train_sizes": list, "train_losses": list, "val_losses": list}
    """
    # TODO: 구현하세요
    pass
