## 문항: 다중 소스 데이터 통합 기반 전력 수요 예측

### 문제

서로 다른 형식과 단위로 수집된 10개의 전력·기상 데이터 파일을 통합·전처리하고, 시계열 특성을 활용한 전력 수요 예측 모델을 구축하세요.

### 제공 데이터

```
data/
├── power_hourly_2023.csv          # 시간별 전력 수요 (8,760행, kWh)
├── power_hourly_2024.csv          # 시간별 전력 수요 (8,784행, kWh)
├── weather_daily_2023.json        # 일별 기상 데이터 (365행)
├── weather_daily_2024.json        # 일별 기상 데이터 (366행)
├── temperature_hourly.tsv         # 시간별 기온 (°F 단위, 변환 필요)
├── humidity_hourly.tsv            # 시간별 습도 (%)
├── holidays.csv                   # 공휴일 목록 (날짜, 휴일명)
├── region_info.json               # 지역별 인구·산업 비율
├── power_monthly_stats.csv        # 월별 전력 통계 (평균, 최대, 최소)
└── missing_log.json               # 결측 구간 기록 (센서 오류 등)
```

- `power_hourly_2023.csv`, `power_hourly_2024.csv`: `datetime,demand_kwh` 형태의 시간별 전력 수요 데이터입니다. 일부 행에 결측값과 이상치가 포함되어 있습니다.
- `weather_daily_2023.json`, `weather_daily_2024.json`: `{"date", "avg_temp_c", "max_temp_c", "min_temp_c", "precipitation_mm", "wind_speed_ms"}` 구조의 일별 기상 데이터입니다.
- `temperature_hourly.tsv`: 시간별 기온 데이터로, 단위가 화씨(°F)이므로 섭씨 변환이 필요합니다.
- `humidity_hourly.tsv`: 시간별 상대 습도(%) 데이터입니다.
- `holidays.csv`: `date,holiday_name` 형태의 공휴일 목록입니다.
- `region_info.json`: 지역별 인구수와 산업 비율을 포함합니다 (메타데이터용).
- `power_monthly_stats.csv`: 월별 전력 수요 통계로, 검증용으로 활용합니다.
- `missing_log.json`: 센서 오류로 인한 결측 구간 기록입니다.

### 프로젝트 구조

| 파일 | 역할 | 핵심 함수 |
|------|------|----------|
| `data_loader.py` | 다중 소스 데이터 로드 및 통합 | `load_power_data()`, `load_weather_data()`, `load_hourly_features()`, `merge_all()` |
| `preprocessor.py` | 결측 처리, 이상치 제거, 단위 변환 | `handle_missing()`, `detect_outliers_iqr()`, `convert_fahrenheit()`, `validate_data()` |
| `feature_engineer.py` | 시계열 피처 생성 | `add_lag_features()`, `add_rolling_features()`, `add_time_features()`, `add_holiday_flag()` |
| `model.py` | 모델 학습, 평가, 비교 | `split_time_series()`, `train_linear()`, `train_ridge()`, `evaluate()`, `compare_models()` |
| `main.py` | 전체 파이프라인 실행 | `main()` |

### 구현 요구사항

#### Part A: data_loader.py - 데이터 로드 및 통합

#### 1. `load_power_data(csv_paths: list) -> np.ndarray`
- 복수의 CSV 파일을 로드하여 시간순으로 연결합니다.
- `datetime` 열을 파싱하고, `demand_kwh` 열을 float으로 변환합니다.
- 반환: `(N, 2)` 배열 (datetime 문자열, 수요값)

#### 2. `load_weather_data(json_paths: list) -> np.ndarray`
- JSON 파일들을 로드하여 일별 기상 데이터를 통합합니다.
- 반환: `(N, 6)` 배열 (date, avg_temp, max_temp, min_temp, precipitation, wind_speed)

#### 3. `load_hourly_features(temp_tsv: str, humid_tsv: str) -> np.ndarray`
- TSV 파일에서 시간별 기온(°F→°C 변환)과 습도를 로드합니다.
- 변환 공식: `°C = (°F - 32) × 5/9`
- 반환: `(N, 3)` 배열 (datetime, temp_celsius, humidity)

#### 4. `merge_all(power, weather, hourly_features, holidays) -> np.ndarray`
- 시간 기준으로 전력 수요, 기상, 기온/습도 데이터를 병합합니다.
- 일별 기상 데이터는 해당 날짜의 모든 시간에 동일 값을 적용합니다.
- 공휴일 여부를 0/1 플래그로 추가합니다.
- 반환: 통합된 2D 배열

#### Part B: preprocessor.py - 전처리

#### 5. `handle_missing(data: np.ndarray, col_idx: int) -> np.ndarray`
- 결측값(`NaN`)을 전후 유효값의 선형 보간으로 대체합니다.
- 선형 보간이 불가한 경우(시작/끝) 가장 가까운 유효값으로 대체합니다.
- 반환: 결측 처리된 배열

#### 6. `detect_outliers_iqr(data: np.ndarray, col_idx: int) -> list[int]`
- IQR 기반 이상치 행 인덱스를 반환합니다.
- `Q1 = np.percentile(col, 25)`, `Q3 = np.percentile(col, 75)`
- 이상치 조건: `값 < Q1 - 1.5 * IQR` 또는 `값 > Q3 + 1.5 * IQR`

#### 7. `convert_fahrenheit(temp_f: np.ndarray) -> np.ndarray`
- 화씨를 섭씨로 변환합니다: `°C = (°F - 32) × 5/9`

#### 8. `validate_data(monthly_calculated: dict, monthly_stats_path: str) -> dict`
- 월별 평균 전력 수요를 계산하여 제공된 월별 통계 파일과 비교합니다.
- 각 월의 차이율(%)을 계산합니다.
- 반환: `{"월": {"calculated_mean": float, "reference_mean": float, "diff_pct": float}}`

#### Part C: feature_engineer.py - 피처 엔지니어링

#### 9. `add_lag_features(demand: np.ndarray, lags: list) -> np.ndarray`
- 지정된 시차(예: [1, 24, 168])의 수요값을 피처로 추가합니다.
- 시차로 인한 초기 NaN은 0으로 채웁니다.
- 반환: `(N, len(lags))` 배열

#### 10. `add_rolling_features(demand: np.ndarray, windows: list) -> np.ndarray`
- 지정된 윈도우 크기(예: [24, 168])의 이동 평균을 계산합니다.
- 윈도우 부족 구간은 가용 데이터의 평균으로 채웁니다.
- 반환: `(N, len(windows))` 배열

#### 11. `add_time_features(datetimes: list) -> np.ndarray`
- datetime에서 `hour`, `day_of_week`, `month`, `is_weekend` 피처를 추출합니다.
- `is_weekend`: 토요일(5), 일요일(6)이면 1, 아니면 0
- 반환: `(N, 4)` 배열

#### 12. `add_holiday_flag(datetimes: list, holidays: list) -> np.ndarray`
- 공휴일이면 1, 아니면 0인 플래그 배열을 반환합니다.
- 반환: `(N, 1)` 배열

#### Part D: model.py - 모델 학습 및 평가

#### 13. `split_time_series(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2) -> tuple`
- 시계열 데이터이므로 셔플 없이 뒤쪽 `test_ratio` 비율을 테스트로 분할합니다.
- 반환: `(X_train, X_test, y_train, y_test)`

#### 14. `train_linear(X_train, y_train) -> tuple`
- 정규방정식으로 선형 회귀 가중치를 계산합니다: `w = (X^T X)^(-1) X^T y`
- X에 절편 열(1)을 추가하여 계산합니다.
- 반환: `(weights, bias)`

#### 15. `train_ridge(X_train, y_train, alpha: float = 1.0) -> tuple`
- L2 정규화된 회귀: `w = (X^T X + αI)^(-1) X^T y`
- 반환: `(weights, bias)`

#### 16. `evaluate(y_true, y_pred) -> dict`
- MAE: `mean(|y_true - y_pred|)`
- RMSE: `sqrt(mean((y_true - y_pred)^2))`
- R²: `1 - SS_res / SS_tot`
- MAPE: `mean(|y_true - y_pred| / |y_true|) * 100` (y_true=0인 경우 제외)
- 반환: `{"mae": float, "rmse": float, "r_squared": float, "mape": float}`

#### 17. `compare_models(results: dict) -> str`
- 모델별 평가 지표를 비교하여 R² 기준 최적 모델명을 반환합니다.

#### Part E: main.py - 파이프라인 실행

#### 18. `main(data_dir: str) -> dict`
- 데이터 로드 → 전처리 → 피처 엔지니어링 → 모델 학습/평가 → 결과 저장
- 선형 회귀와 Ridge 회귀 두 모델을 학습·비교합니다.
- `result_q1.json` 파일로 결과를 저장합니다.

### 출력 형식

`result_q1.json` 파일로 다음 구조를 저장합니다:

```json
{
  "data_summary": {
    "total_files_loaded": 정수,
    "total_rows_merged": 정수,
    "missing_values_filled": 정수,
    "outliers_removed": 정수,
    "feature_count": 정수
  },
  "validation": {
    "monthly_diff_pct_mean": 실수
  },
  "model_linear": {
    "mae": 실수,
    "rmse": 실수,
    "r_squared": 실수,
    "mape": 실수
  },
  "model_ridge": {
    "mae": 실수,
    "rmse": 실수,
    "r_squared": 실수,
    "mape": 실수
  },
  "best_model": "문자열",
  "feature_count": 정수,
  "lag_features": [정수, ...],
  "rolling_windows": [정수, ...]
}
```

- 모든 실수값은 `round(..., 6)`으로 반올림합니다.

### 제약 사항
- NumPy만 사용하여 모든 수치 계산을 수행합니다 (`pandas`, `sklearn`, `scipy` 사용 금지).
- CSV/JSON/TSV 로드에는 표준 라이브러리(`csv`, `json`) 사용 가능합니다.
- 선형 회귀와 Ridge 회귀는 정규방정식으로 직접 구현합니다.
- 시계열 분할 시 셔플하지 않습니다 (시간 순서 유지).
- `datetime` 파싱에는 표준 라이브러리 `datetime` 사용 가능합니다.

### 제출 폴더 구조

다음 폴더 구조를 zip으로 묶어 제출합니다.

```
submission/
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── model.py
│   └── main.py
├── config/
│   └── config.json
├── logs/
│   └── pipeline_log.json
└── output/
    └── result_q1.json
```

- `template/` 디렉토리의 각 파일의 `# TODO` 부분을 채우세요.
