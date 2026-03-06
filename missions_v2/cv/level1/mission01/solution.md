## 문항 1 정답지 — 다중 소스 데이터 통합 기반 전력 수요 예측

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
| 평가 지표 | `dict` | MAE, RMSE, R², MAPE |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | data_loader.py 필수 함수 3개 정의 (load_power_data, load_weather_data, load_hourly_features) | AST 자동 |
| 2 | preprocessor.py 필수 함수 3개 정의 (handle_missing, detect_outliers_iqr, convert_fahrenheit) | AST 자동 |
| 3 | feature_engineer.py 필수 함수 4개 정의 (add_lag_features, add_rolling_features, add_time_features, add_holiday_flag) | AST 자동 |
| 4 | model.py 필수 함수 5개 정의 (split_time_series, train_linear, train_ridge, evaluate, compare_models) | AST 자동 |
| 5 | 전력 데이터 로드 (17,000행 이상) | import 자동 |
| 6 | 시간별 기온/습도 로드 및 화씨→섭씨 변환 (범위: -30~50) | import 자동 |
| 7 | IQR 이상치 탐지 (이상치 200 감지, 정상값 10 통과) | import 자동 |
| 8 | 화씨→섭씨 변환 (32°F→0°C, 212°F→100°C, 77°F→25°C) | import 자동 |
| 9 | 시차 피처 생성 (shape, lag-1/lag-2 값, 초기값 0) | import 자동 |
| 10 | 이동 평균 피처 생성 (shape, 윈도우=3 평균값) | import 자동 |
| 11 | 시간 피처 추출 (hour, day_of_week, month, is_weekend) | import 자동 |
| 12 | 선형 회귀 학습 및 평가 (R² > 0.9) | import 자동 |
| 13 | Ridge 회귀 학습 (R² > 0.8) | import 자동 |
| 14 | result_q1.json 필수 키 확인 (data_summary, model_linear, model_ridge, best_model, feature_count) | JSON 자동 |
| 15 | 모델 성능 지표 범위 (R² > 0.3, MAE > 0) | JSON 자동 |
| 16 | 데이터 요약 검증 (파일 수=10, 행 수>17000, 결측 보간>0, 피처>5) | JSON 자동 |

- Pass 기준: 16개 전체 통과
- AI 트랩: 화씨→섭씨 변환 누락, IQR 이상치 경계 포함/미포함, 시계열 분할 시 셔플, 정규방정식 절편 열 누락, lag 초기값 NaN 대신 0 처리

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 다중 형식 데이터 통합 (CSV/JSON/TSV) | test_load_power_data, test_load_hourly_features |
| 단위 변환 (화씨→섭씨) | test_convert_fahrenheit, test_load_hourly_features |
| IQR 이상치 탐지 | test_detect_outliers_iqr |
| 시계열 피처 엔지니어링 | test_lag_features, test_rolling_features, test_time_features |
| 정규방정식 기반 회귀 모델 | test_train_and_evaluate, test_ridge_regression |
| 파이프라인 통합 및 결과 검증 | test_result_structure, test_model_performance, test_data_summary |
