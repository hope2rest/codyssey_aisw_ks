## 문항 1 정답지 — 다중 소스 데이터 통합 기반 전력 수요 예측

### 정답 코드

#### data_loader.py

```python
"""data_loader.py - 다중 소스 데이터 로드 및 통합"""
import csv
import json
import os
import numpy as np
from datetime import datetime


def load_power_data(csv_paths):
    """복수의 CSV 파일을 로드하여 시간순으로 연결합니다."""
    all_rows = []
    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = row["datetime"]
                val = row["demand_kwh"]
                demand = float(val) if val.strip() != "" else np.nan
                all_rows.append([dt, demand])
    all_rows.sort(key=lambda x: x[0])
    return np.array(all_rows, dtype=object)


def load_weather_data(json_paths):
    """JSON 파일들을 로드하여 일별 기상 데이터를 통합합니다."""
    all_entries = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_entries.extend(data)
    all_entries.sort(key=lambda x: x["date"])
    result = []
    for e in all_entries:
        result.append([
            e["date"], e["avg_temp_c"], e["max_temp_c"],
            e["min_temp_c"], e["precipitation_mm"], e["wind_speed_ms"]
        ])
    return np.array(result, dtype=object)


def load_hourly_features(temp_tsv, humid_tsv):
    """TSV 파일에서 시간별 기온(F->C 변환)과 습도를 로드합니다."""
    temps = []
    with open(temp_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dt = row["datetime"]
            temp_f = float(row["temperature_f"])
            temp_c = (temp_f - 32) * 5 / 9
            temps.append([dt, round(temp_c, 2)])

    humids = {}
    with open(humid_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            humids[row["datetime"]] = float(row["humidity_pct"])

    result = []
    for dt, temp_c in temps:
        humid = humids.get(dt, np.nan)
        result.append([dt, temp_c, humid])
    return np.array(result, dtype=object)


def load_holidays(csv_path):
    """공휴일 CSV를 로드합니다."""
    holidays = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holidays.add(row["date"])
    return holidays


def merge_all(power, weather, hourly_features, holidays):
    """시간 기준으로 모든 데이터를 병합합니다."""
    weather_dict = {}
    for row in weather:
        weather_dict[str(row[0])] = [float(row[i]) for i in range(1, 6)]
    hourly_dict = {}
    for row in hourly_features:
        hourly_dict[str(row[0])] = [float(row[1]), float(row[2])]
    merged = []
    for row in power:
        dt_str = str(row[0])
        demand = row[1]
        date_str = dt_str[:10]
        w = weather_dict.get(date_str, [np.nan] * 5)
        hf = hourly_dict.get(dt_str, [np.nan, np.nan])
        is_holiday = 1.0 if date_str in holidays else 0.0
        merged.append([dt_str, float(demand) if not isinstance(demand, float) or not np.isnan(demand) else np.nan]
                      + w + hf + [is_holiday])
    return np.array(merged, dtype=object)
```

#### preprocessor.py

```python
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
```

#### feature_engineer.py

```python
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
```

#### model.py

```python
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
    I[0, 0] = 0
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
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | data_loader.py 필수 함수 3개 정의 (load_power_data, load_weather_data, load_hourly_features) | 5점 | AST 자동 |
| 2 | preprocessor.py 필수 함수 3개 정의 (handle_missing, detect_outliers_iqr, convert_fahrenheit) | 5점 | AST 자동 |
| 3 | feature_engineer.py 필수 함수 4개 정의 (add_lag_features, add_rolling_features, add_time_features, add_holiday_flag) | 5점 | AST 자동 |
| 4 | model.py 필수 함수 5개 정의 (split_time_series, train_linear, train_ridge, evaluate, compare_models) | 5점 | AST 자동 |
| 5 | 전력 데이터 로드 (17,000행 이상) | 5점 | import 자동 |
| 6 | 시간별 기온/습도 로드 및 화씨->섭씨 변환 (범위: -30~50) | 10점 | import 자동 |
| 7 | IQR 이상치 탐지 (이상치 200 감지, 정상값 10 통과) | 5점 | import 자동 |
| 8 | 화씨->섭씨 변환 (32F->0C, 212F->100C, 77F->25C) | 5점 | import 자동 |
| 9 | 시차 피처 생성 (shape, lag-1/lag-2 값, 초기값 0) | 10점 | import 자동 |
| 10 | 이동 평균 피처 생성 (shape, 윈도우=3 평균값) | 5점 | import 자동 |
| 11 | 시간 피처 추출 (hour, day_of_week, month, is_weekend) | 10점 | import 자동 |
| 12 | 선형 회귀 학습 및 평가 (R2 > 0.9) | 10점 | import 자동 |
| 13 | Ridge 회귀 학습 (R2 > 0.8) | 5점 | import 자동 |
| 14 | result_q1.json 필수 키 확인 (data_summary, model_linear, model_ridge, best_model, feature_count) | 5점 | JSON 자동 |
| 15 | 모델 성능 지표 범위 (R2 > 0.3, MAE > 0) | 5점 | JSON 자동 |
| 16 | 데이터 요약 검증 (파일 수=10, 행 수>17000, 결측 보간>0, 피처>5) | 5점 | JSON 자동 |

- Pass 기준: 총 100점 중 100점 (16개 전체 정답)
- AI 트랩: 화씨->섭씨 변환 누락, IQR 이상치 경계 포함/미포함, 시계열 분할 시 셔플, 정규방정식 절편 열 누락, lag 초기값 NaN 대신 0 처리

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `power_hourly_*.csv` | CSV | `datetime,demand_kwh` 형태, 시간별 전력 수요 |
| `weather_daily_*.json` | JSON | 일별 기상 데이터 (avg_temp, max_temp, min_temp, precipitation, wind_speed) |
| `temperature_hourly.tsv` | TSV | 시간별 기온 (화씨, 섭씨 변환 필요) |
| `humidity_hourly.tsv` | TSV | 시간별 상대 습도 (%) |
| `holidays.csv` | CSV | 공휴일 목록 (date, holiday_name) |
| lag/rolling 피처 | `np.ndarray` | 시차/이동평균 피처 배열 |
| 모델 가중치 | `np.ndarray` | 정규방정식으로 계산된 회귀 가중치 |
| 평가 지표 | `dict` | MAE, RMSE, R2, MAPE |

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 다중 형식 데이터 통합 (CSV/JSON/TSV) | test_load_power_data, test_load_hourly_features |
| 단위 변환 (화씨->섭씨) | test_convert_fahrenheit, test_load_hourly_features |
| IQR 이상치 탐지 | test_detect_outliers_iqr |
| 시계열 피처 엔지니어링 | test_lag_features, test_rolling_features, test_time_features |
| 정규방정식 기반 회귀 모델 | test_train_and_evaluate, test_ridge_regression |
| 파이프라인 통합 및 결과 검증 | test_result_structure, test_model_performance, test_data_summary |
