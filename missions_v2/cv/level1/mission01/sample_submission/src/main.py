"""main.py - 전력 수요 예측 전체 파이프라인"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_power_data, load_weather_data, load_hourly_features, load_holidays
from preprocessor import handle_missing, detect_outliers_iqr, validate_data
from feature_engineer import add_lag_features, add_rolling_features, add_time_features, add_holiday_flag
from model import split_time_series, train_linear, train_ridge, predict, evaluate, compare_models


def main(data_dir=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    mission_dir = os.path.dirname(project_dir)

    if data_dir is None:
        data_dir = os.path.join(mission_dir, "data")

    # 1. 데이터 로드
    power = load_power_data([
        os.path.join(data_dir, "power_hourly_2023.csv"),
        os.path.join(data_dir, "power_hourly_2024.csv"),
    ])

    weather = load_weather_data([
        os.path.join(data_dir, "weather_daily_2023.json"),
        os.path.join(data_dir, "weather_daily_2024.json"),
    ])

    hourly_feat = load_hourly_features(
        os.path.join(data_dir, "temperature_hourly.tsv"),
        os.path.join(data_dir, "humidity_hourly.tsv"),
    )

    holidays = load_holidays(os.path.join(data_dir, "holidays.csv"))

    total_files = 10
    total_rows = len(power)

    # 2. 전처리 - 결측 처리
    demand_col = np.array([float(x) if x is not None and str(x).strip() != "" and not (isinstance(x, float) and np.isnan(x))
                           else np.nan for x in power[:, 1]], dtype=np.float64)
    missing_count = int(np.sum(np.isnan(demand_col)))

    # 선형 보간
    nan_mask = np.isnan(demand_col)
    valid_idx = np.where(~nan_mask)[0]
    for i in range(len(demand_col)):
        if np.isnan(demand_col[i]):
            left = valid_idx[valid_idx < i]
            right = valid_idx[valid_idx > i]
            if len(left) > 0 and len(right) > 0:
                li, ri = left[-1], right[0]
                demand_col[i] = demand_col[li] + (demand_col[ri] - demand_col[li]) * (i - li) / (ri - li)
            elif len(left) > 0:
                demand_col[i] = demand_col[left[-1]]
            elif len(right) > 0:
                demand_col[i] = demand_col[right[0]]

    # 3. 이상치 제거
    q1 = np.percentile(demand_col, 25)
    q3 = np.percentile(demand_col, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (demand_col < lower) | (demand_col > upper)
    outlier_count = int(np.sum(outlier_mask))

    # Replace outliers with interpolated values
    for i in np.where(outlier_mask)[0]:
        neighbors = []
        if i > 0 and not outlier_mask[i - 1]:
            neighbors.append(demand_col[i - 1])
        if i < len(demand_col) - 1 and not outlier_mask[i + 1]:
            neighbors.append(demand_col[i + 1])
        if neighbors:
            demand_col[i] = np.mean(neighbors)

    # 4. 검증
    datetimes = power[:, 0]
    monthly_calc = {}
    for i, dt_str in enumerate(datetimes):
        month_key = str(dt_str)[:7]
        if month_key not in monthly_calc:
            monthly_calc[month_key] = []
        monthly_calc[month_key].append(demand_col[i])
    monthly_calc_means = {k: float(np.mean(v)) for k, v in monthly_calc.items()}
    validation = validate_data(monthly_calc_means, os.path.join(data_dir, "power_monthly_stats.csv"))
    monthly_diff_pct_mean = round(float(np.mean([v["diff_pct"] for v in validation.values()])), 6)

    # 5. 피처 엔지니어링
    lags = [1, 24, 168]
    windows = [24, 168]

    lag_feat = add_lag_features(demand_col, lags)
    roll_feat = add_rolling_features(demand_col, windows)
    time_feat = add_time_features(datetimes)
    holiday_feat = add_holiday_flag(datetimes, holidays)

    # hourly temp/humidity
    hourly_dict = {}
    for row in hourly_feat:
        hourly_dict[str(row[0])] = [float(row[1]), float(row[2])]

    temp_humid = np.zeros((len(datetimes), 2), dtype=np.float64)
    for i, dt_str in enumerate(datetimes):
        hf = hourly_dict.get(str(dt_str), [0.0, 60.0])
        temp_humid[i] = hf

    X = np.column_stack([lag_feat, roll_feat, time_feat, holiday_feat, temp_humid])
    y = demand_col

    # Remove initial rows where lag features are 0
    skip = max(lags)
    X = X[skip:]
    y = y[skip:]

    feature_count = X.shape[1]

    # 6. 모델 학습
    X_train, X_test, y_train, y_test = split_time_series(X, y, test_ratio=0.2)

    # Standardize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1.0
    X_train_s = (X_train - mean) / std
    X_test_s = (X_test - mean) / std

    w_linear, b_linear = train_linear(X_train_s, y_train)
    pred_linear = predict(X_test_s, w_linear, b_linear)
    metrics_linear = evaluate(y_test, pred_linear)

    w_ridge, b_ridge = train_ridge(X_train_s, y_train, alpha=1.0)
    pred_ridge = predict(X_test_s, w_ridge, b_ridge)
    metrics_ridge = evaluate(y_test, pred_ridge)

    results_dict = {"linear": metrics_linear, "ridge": metrics_ridge}
    best = compare_models(results_dict)

    # 7. 결과 저장
    result = {
        "data_summary": {
            "total_files_loaded": total_files,
            "total_rows_merged": total_rows,
            "missing_values_filled": missing_count,
            "outliers_removed": outlier_count,
            "feature_count": feature_count,
        },
        "validation": {
            "monthly_diff_pct_mean": monthly_diff_pct_mean,
        },
        "model_linear": metrics_linear,
        "model_ridge": metrics_ridge,
        "best_model": best,
        "feature_count": feature_count,
        "lag_features": lags,
        "rolling_windows": windows,
    }

    output_path = os.path.join(project_dir, "output", "result_q1.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"result_q1.json saved to {output_path}")
    return result


if __name__ == "__main__":
    main()
