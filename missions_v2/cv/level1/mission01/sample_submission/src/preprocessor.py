"""preprocessor.py - 결측 처리, 이상치 제거, 단위 변환, 검증"""
import csv
import numpy as np


def handle_missing(data, col_idx):
    """결측값을 전후 유효값의 선형 보간으로 대체합니다."""
    result = data.copy()
    col = np.array([float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan
                     for x in result[:, col_idx]], dtype=np.float64)

    nan_mask = np.isnan(col)
    if not np.any(nan_mask):
        return result

    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return result

    for i in range(len(col)):
        if np.isnan(col[i]):
            left = valid_idx[valid_idx < i]
            right = valid_idx[valid_idx > i]
            if len(left) > 0 and len(right) > 0:
                li, ri = left[-1], right[0]
                col[i] = col[li] + (col[ri] - col[li]) * (i - li) / (ri - li)
            elif len(left) > 0:
                col[i] = col[left[-1]]
            elif len(right) > 0:
                col[i] = col[right[0]]

    result[:, col_idx] = col
    return result


def detect_outliers_iqr(data, col_idx):
    """IQR 기반 이상치 행 인덱스를 반환합니다."""
    col = np.array(data[:, col_idx], dtype=np.float64)
    valid = col[~np.isnan(col)]
    q1 = np.percentile(valid, 25)
    q3 = np.percentile(valid, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_idx = []
    for i, v in enumerate(col):
        if not np.isnan(v) and (v < lower or v > upper):
            outlier_idx.append(i)
    return outlier_idx


def convert_fahrenheit(temp_f):
    """화씨를 섭씨로 변환합니다."""
    return (np.array(temp_f, dtype=np.float64) - 32) * 5 / 9


def validate_data(monthly_calculated, monthly_stats_path):
    """월별 평균 전력 수요를 비교합니다."""
    ref = {}
    with open(monthly_stats_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref[row["month"]] = float(row["mean_demand"])

    result = {}
    for month, calc_mean in monthly_calculated.items():
        ref_mean = ref.get(month, calc_mean)
        diff_pct = abs(calc_mean - ref_mean) / ref_mean * 100 if ref_mean != 0 else 0
        result[month] = {
            "calculated_mean": round(calc_mean, 6),
            "reference_mean": round(ref_mean, 6),
            "diff_pct": round(diff_pct, 6),
        }
    return result
