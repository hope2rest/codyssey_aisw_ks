"""
문항 1: 이커머스 데이터 전처리 및 이상치 탐지 파이프라인

NumPy만 사용하여 구현하세요.
"""
import csv
import json
import os

import numpy as np


def load_and_clean(filepath: str):
    """
    CSV 파일을 로드하고 정제합니다.

    - customer_id 열 제외
    - 중복 행 제거 (첫 번째만 유지)
    - 결측값(NaN)은 해당 열의 중앙값으로 대체

    Returns:
        (np.ndarray, list[str], int, int, int):
        (정제된 2D 배열, 열 이름 리스트, 원본 행수, 제거된 중복수, 대체된 결측수)
    """
    # TODO: 구현하세요
    pass


def compute_statistics(data: np.ndarray, columns: list) -> dict:
    """
    각 열의 기술 통계량을 계산합니다.
    mean, std(ddof=0), min, max, median
    모든 값은 소수점 6자리 반올림.

    Returns:
        {열이름: {"mean", "std", "min", "max", "median"}}
    """
    # TODO: 구현하세요
    pass


def detect_outliers_iqr(data: np.ndarray, col_idx: int) -> list:
    """
    IQR 기반 이상치 행 인덱스를 반환합니다.

    Q1 = np.percentile(col, 25), Q3 = np.percentile(col, 75)
    IQR = Q3 - Q1
    이상치: 값 < Q1 - 1.5*IQR 또는 값 > Q3 + 1.5*IQR
    """
    # TODO: 구현하세요
    pass


def detect_outliers_zscore(data: np.ndarray, col_idx: int, threshold: float = 3.0) -> list:
    """
    Z-score 기반 이상치 행 인덱스를 반환합니다.

    Z = (값 - 평균) / 표준편차 (ddof=0)
    이상치: |Z| > threshold
    """
    # TODO: 구현하세요
    pass


def standardize(data: np.ndarray) -> np.ndarray:
    """
    각 열을 Z-score 표준화합니다 (ddof=0).
    표준편차가 0인 열은 0으로 유지합니다.
    """
    # TODO: 구현하세요
    pass


def segment_customers(data: np.ndarray, columns: list) -> dict:
    """
    annual_income과 spending_score 열 기준으로 4개 세그먼트 분류.

    Returns:
        {세그먼트명: {"count": int, "mean_income": float, "mean_spending": float}}
    """
    # TODO: 구현하세요
    pass


def main(data_path: str) -> dict:
    """전체 파이프라인 실행 및 result_q1.json 저장"""
    # TODO: 구현하세요
    pass


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "customers.csv")
    result = main(data_path)
    with open(os.path.join(base_dir, "result_q1.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
