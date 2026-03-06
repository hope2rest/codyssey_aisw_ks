"""feature_engineer.py - 시계열 피처 생성"""
import numpy as np
from datetime import datetime


def add_lag_features(demand, lags):
    """지정된 시차의 수요값을 피처로 추가합니다."""
    n = len(demand)
    result = np.zeros((n, len(lags)), dtype=np.float64)
    for j, lag in enumerate(lags):
        for i in range(n):
            if i >= lag:
                result[i, j] = demand[i - lag]
            else:
                result[i, j] = 0.0
    return result


def add_rolling_features(demand, windows):
    """지정된 윈도우 크기의 이동 평균을 계산합니다."""
    n = len(demand)
    result = np.zeros((n, len(windows)), dtype=np.float64)
    for j, w in enumerate(windows):
        for i in range(n):
            start = max(0, i - w + 1)
            result[i, j] = np.mean(demand[start:i + 1])
    return result


def add_time_features(datetimes):
    """datetime에서 hour, day_of_week, month, is_weekend 피처를 추출합니다."""
    n = len(datetimes)
    result = np.zeros((n, 4), dtype=np.float64)
    for i, dt_str in enumerate(datetimes):
        dt = datetime.strptime(str(dt_str), "%Y-%m-%d %H:%M:%S")
        result[i, 0] = dt.hour
        result[i, 1] = dt.weekday()
        result[i, 2] = dt.month
        result[i, 3] = 1.0 if dt.weekday() >= 5 else 0.0
    return result


def add_holiday_flag(datetimes, holidays):
    """공휴일 플래그 배열을 반환합니다."""
    n = len(datetimes)
    result = np.zeros((n, 1), dtype=np.float64)
    for i, dt_str in enumerate(datetimes):
        date_str = str(dt_str)[:10]
        if date_str in holidays:
            result[i, 0] = 1.0
    return result
