"""preprocessor.py - 결측 처리, 이상치 제거, 단위 변환, 검증"""
import csv
import numpy as np


def handle_missing(data, col_idx):
    """결측값을 전후 유효값의 선형 보간으로 대체합니다."""
    # TODO: NaN을 선형 보간으로 대체, 시작/끝은 가장 가까운 유효값
    pass


def detect_outliers_iqr(data, col_idx):
    """IQR 기반 이상치 행 인덱스를 반환합니다."""
    # TODO: Q1, Q3 계산, IQR = Q3 - Q1, lower/upper bound, 이상치 인덱스 반환
    pass


def convert_fahrenheit(temp_f):
    """화씨를 섭씨로 변환합니다."""
    # TODO: °C = (°F - 32) × 5/9
    pass


def validate_data(monthly_calculated, monthly_stats_path):
    """월별 평균 전력 수요를 비교합니다."""
    # TODO: 월별 계산값과 참조값 비교, 차이율(%) 반환
    pass
