"""
문항 1: 이커머스 데이터 전처리 및 이상치 탐지 파이프라인 (정답 코드)
"""
import csv
import json
import os

import numpy as np


def load_and_clean(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = [row for row in reader]

    columns = header[1:]
    total_raw = len(raw_rows)

    data_list = []
    for row in raw_rows:
        vals = []
        for v in row[1:]:
            if v.strip() == "":
                vals.append(np.nan)
            else:
                vals.append(float(v))
        data_list.append(vals)

    data = np.array(data_list, dtype=np.float64)

    _, unique_idx = np.unique(data, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    duplicates = total_raw - len(unique_idx)
    data = data[unique_idx]

    missing_count = int(np.isnan(data).sum())

    for col in range(data.shape[1]):
        col_data = data[:, col]
        mask = np.isnan(col_data)
        if mask.any():
            median_val = np.nanmedian(col_data)
            col_data[mask] = median_val

    return data, columns, total_raw, duplicates, missing_count


def compute_statistics(data: np.ndarray, columns: list) -> dict:
    stats = {}
    for i, col_name in enumerate(columns):
        col = data[:, i]
        stats[col_name] = {
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col, ddof=0)), 6),
            "min": round(float(np.min(col)), 6),
            "max": round(float(np.max(col)), 6),
            "median": round(float(np.median(col)), 6),
        }
    return stats


def detect_outliers_iqr(data: np.ndarray, col_idx: int) -> list:
    col = data[:, col_idx]
    q1 = np.percentile(col, 25)
    q3 = np.percentile(col, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [int(i) for i in range(len(col)) if col[i] < lower or col[i] > upper]


def detect_outliers_zscore(data: np.ndarray, col_idx: int, threshold: float = 3.0) -> list:
    col = data[:, col_idx]
    mean = np.mean(col)
    std = np.std(col, ddof=0)
    if std == 0:
        return []
    z = np.abs((col - mean) / std)
    return [int(i) for i in range(len(z)) if z[i] > threshold]


def standardize(data: np.ndarray) -> np.ndarray:
    result = np.zeros_like(data)
    for col in range(data.shape[1]):
        mean = np.mean(data[:, col])
        std = np.std(data[:, col], ddof=0)
        if std == 0:
            result[:, col] = 0.0
        else:
            result[:, col] = (data[:, col] - mean) / std
    return result


def segment_customers(data: np.ndarray, columns: list) -> dict:
    income_idx = columns.index("annual_income")
    spending_idx = columns.index("spending_score")

    income = data[:, income_idx]
    spending = data[:, spending_idx]

    income_median = np.median(income)
    spending_median = np.median(spending)

    segments = {
        "high_income_high_spend": (income >= income_median) & (spending >= spending_median),
        "high_income_low_spend": (income >= income_median) & (spending < spending_median),
        "low_income_high_spend": (income < income_median) & (spending >= spending_median),
        "low_income_low_spend": (income < income_median) & (spending < spending_median),
    }

    result = {}
    for name, mask in segments.items():
        count = int(mask.sum())
        result[name] = {
            "count": count,
            "mean_income": round(float(np.mean(income[mask])), 6) if count > 0 else 0.0,
            "mean_spending": round(float(np.mean(spending[mask])), 6) if count > 0 else 0.0,
        }
    return result


def main(data_path: str) -> dict:
    data, columns, total_raw, duplicates, missing_filled = load_and_clean(data_path)

    statistics = compute_statistics(data, columns)

    outlier_counts_iqr = {}
    for i, col_name in enumerate(columns):
        outlier_counts_iqr[col_name] = len(detect_outliers_iqr(data, i))

    outlier_counts_zscore = {}
    for i, col_name in enumerate(columns):
        outlier_counts_zscore[col_name] = len(detect_outliers_zscore(data, i))

    standardized = standardize(data)
    std_mean_check = {}
    std_std_check = {}
    for i, col_name in enumerate(columns):
        std_mean_check[col_name] = round(float(np.mean(standardized[:, i])), 6)
        std_std_check[col_name] = round(float(np.std(standardized[:, i], ddof=0)), 6)

    segments = segment_customers(data, columns)

    return {
        "total_rows_raw": total_raw,
        "total_rows_cleaned": int(data.shape[0]),
        "duplicates_removed": duplicates,
        "missing_values_filled": missing_filled,
        "statistics": statistics,
        "outlier_counts_iqr": outlier_counts_iqr,
        "outlier_counts_zscore": outlier_counts_zscore,
        "standardized_mean_check": std_mean_check,
        "standardized_std_check": std_std_check,
        "segments": segments,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "customers.csv")
    result = main(data_path)
    with open(os.path.join(base_dir, "result_q1.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("result_q1.json saved.")
