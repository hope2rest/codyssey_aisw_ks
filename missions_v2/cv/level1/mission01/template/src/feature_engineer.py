"""feature_engineer.py - 시계열 피처 생성"""
import numpy as np
from datetime import datetime


def add_lag_features(demand, lags):
    """지정된 시차의 수요값을 피처로 추가합니다."""
    # TODO: 각 lag에 대해 시차된 수요값 배열 생성, 초기 NaN은 0으로
    pass


def add_rolling_features(demand, windows):
    """지정된 윈도우 크기의 이동 평균을 계산합니다."""
    # TODO: 각 window에 대해 이동 평균 계산
    pass


def add_time_features(datetimes):
    """datetime에서 hour, day_of_week, month, is_weekend 피처를 추출합니다."""
    # TODO: datetime 파싱 후 4개 피처 추출
    pass


def add_holiday_flag(datetimes, holidays):
    """공휴일 플래그 배열을 반환합니다."""
    # TODO: 공휴일이면 1, 아니면 0
    pass
