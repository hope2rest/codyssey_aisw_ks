import csv
import json
import os

import numpy as np


def load_and_clean(filepath):
    # TODO: CSV 로드, customer_id 제외, 중복 제거, 결측값 중앙값 대체


def compute_statistics(data, columns):
    # TODO: 각 열의 mean, std(ddof=0), min, max, median 계산


def detect_outliers_iqr(data, col_idx):
    # TODO: IQR 기반 이상치 행 인덱스 반환


def detect_outliers_zscore(data, col_idx, threshold=3.0):
    # TODO: Z-score 기반 이상치 행 인덱스 반환


def standardize(data):
    # TODO: 각 열을 Z-score 표준화 (ddof=0, 표준편차 0인 열은 0 유지)


def segment_customers(data, columns):
    # TODO: annual_income과 spending_score 기준 4개 세그먼트 분류


def main(data_path):
    # TODO: 전체 파이프라인 실행 및 result_q1.json 저장


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "customers.csv")
    result = main(data_path)
    with open(os.path.join(base_dir, "result_q1.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
