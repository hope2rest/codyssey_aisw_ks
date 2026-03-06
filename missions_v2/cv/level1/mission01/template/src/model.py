"""model.py - 모델 학습, 평가, 비교"""
import numpy as np


def split_time_series(X, y, test_ratio=0.2):
    """시계열 데이터를 셔플 없이 분할합니다."""
    # TODO: 뒤쪽 test_ratio 비율을 테스트로 분할
    pass


def train_linear(X_train, y_train):
    """정규방정식으로 선형 회귀 가중치를 계산합니다."""
    # TODO: w = (X^T X)^(-1) X^T y, 절편 열 추가
    pass


def train_ridge(X_train, y_train, alpha=1.0):
    """L2 정규화된 회귀 가중치를 계산합니다."""
    # TODO: w = (X^T X + αI)^(-1) X^T y
    pass


def predict(X, weights, bias):
    """예측값을 계산합니다."""
    # TODO: X @ weights + bias
    pass


def evaluate(y_true, y_pred):
    """MAE, RMSE, R2, MAPE를 계산합니다."""
    # TODO: 4개 지표 계산, round 6자리
    pass


def compare_models(results):
    """R2 기준 최적 모델명을 반환합니다."""
    # TODO: R2가 가장 높은 모델명 반환
    pass
