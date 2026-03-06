"""model.py - 모델 학습, 평가, 비교"""
import numpy as np


def split_time_series(X, y, test_ratio=0.2):
    """시계열 데이터를 셔플 없이 분할합니다."""
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_linear(X_train, y_train):
    """정규방정식으로 선형 회귀 가중치를 계산합니다."""
    n = X_train.shape[0]
    X_b = np.column_stack([np.ones(n), X_train])
    try:
        w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train
    return w[1:], w[0]


def train_ridge(X_train, y_train, alpha=1.0):
    """L2 정규화된 회귀 가중치를 계산합니다."""
    n = X_train.shape[0]
    X_b = np.column_stack([np.ones(n), X_train])
    d = X_b.shape[1]
    I = np.eye(d)
    I[0, 0] = 0  # bias에는 정규화 적용 안 함
    try:
        w = np.linalg.inv(X_b.T @ X_b + alpha * I) @ X_b.T @ y_train
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(X_b.T @ X_b + alpha * I) @ X_b.T @ y_train
    return w[1:], w[0]


def predict(X, weights, bias):
    """예측값을 계산합니다."""
    return X @ weights + bias


def evaluate(y_true, y_pred):
    """MAE, RMSE, R2, MAPE를 계산합니다."""
    y_true = np.array(y_true, dtype=np.float64).flatten()
    y_pred = np.array(y_pred, dtype=np.float64).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    nonzero = y_true != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = 0.0

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r_squared": round(r_squared, 6),
        "mape": round(mape, 6),
    }


def compare_models(results):
    """R2 기준 최적 모델명을 반환합니다."""
    best_name = max(results, key=lambda k: results[k]["r_squared"])
    return best_name
