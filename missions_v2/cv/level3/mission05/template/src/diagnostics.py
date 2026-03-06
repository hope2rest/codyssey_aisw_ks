import numpy as np
from tensor import Tensor


def compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn):
    # TODO: 학습/테스트 손실 계산, (train_loss, test_loss) 반환


def diagnose_bias_variance(train_loss, test_loss, threshold=0.1):
    # TODO: "high_bias", "high_variance", 또는 "good_fit" 반환


def learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs, train_sizes=None):
    # TODO: 다양한 학습 데이터 크기에서 학습 곡선 계산
