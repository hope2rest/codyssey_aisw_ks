## 문항: 이커머스 데이터 전처리 및 이상치 탐지 파이프라인

### 문제

이커머스 고객 데이터를 전처리하고, IQR/Z-score 기반 이상치를 탐지하며, 고객 세그먼트를 분류하는 파이프라인을 구현하세요. 
고객 데이터는 `data/customers.csv`에 저장되어 있습니다.

### customers.csv 구조

| 열 이름 | 타입 | 설명 |
|---------|------|------|
| `customer_id` | int | 고객 고유 ID (일부 중복 존재) |
| `age` | float | 나이 (일부 결측) |
| `annual_income` | float | 연간 소득 (만원, 일부 결측) |
| `spending_score` | float | 소비 점수 (1~100, 일부 이상치) |
| `purchase_count` | int | 구매 횟수 |
| `avg_order_value` | float | 평균 주문 금액 (만원, 일부 결측) |
| `days_since_last` | int | 마지막 구매 후 경과일 |
| `total_spent` | float | 총 구매 금액 (만원) |

### 구현 요구사항

#### 1. `load_and_clean(filepath: str) -> tuple[np.ndarray, list[str]]`
- CSV 파일을 `numpy`로 로드합니다 (`customer_id` 열 제외).
- 중복 행(모든 열 값이 동일)을 제거합니다 (첫 번째만 유지).
- 결측값(`NaN`)은 해당 열의 **중앙값(median)**으로 대체합니다.
- 반환: `(정제된 2D 배열, 열 이름 리스트)`

#### 2. `compute_statistics(data: np.ndarray, columns: list[str]) -> dict`
- 각 열에 대해 `mean`, `std`(ddof=0), `min`, `max`, `median`을 계산합니다.
- 반환: `{열이름: {"mean": float, "std": float, "min": float, "max": float, "median": float}}`

#### 3. `detect_outliers_iqr(data: np.ndarray, col_idx: int) -> list[int]`
- 해당 열에서 IQR 기반 이상치의 **행 인덱스** 리스트를 반환합니다.
- `Q1 = np.percentile(data[:, col_idx], 25)`, `Q3 = np.percentile(data[:, col_idx], 75)`
- 이상치 조건: `값 < Q1 - 1.5 * IQR` 또는 `값 > Q3 + 1.5 * IQR`

#### 4. `detect_outliers_zscore(data: np.ndarray, col_idx: int, threshold: float = 3.0) -> list[int]`
- Z-score = `(값 - 평균) / 표준편차` (표준편차는 `ddof=0`)
- `|Z-score| > threshold`인 행 인덱스 리스트를 반환합니다.

#### 5. `standardize(data: np.ndarray) -> np.ndarray`
- 각 열을 Z-score 표준화합니다: `(값 - 평균) / 표준편차` (ddof=0)
- 표준편차가 0인 열은 0으로 유지합니다.
- 반환: 표준화된 2D 배열

#### 6. `segment_customers(data: np.ndarray, columns: list[str]) -> dict`
- `annual_income`과 `spending_score` 열을 사용합니다.
- 각 열의 중앙값을 기준으로 4개 세그먼트로 분류합니다:
  - `high_income_high_spend`: 소득 >= 중앙값 AND 소비 >= 중앙값
  - `high_income_low_spend`: 소득 >= 중앙값 AND 소비 < 중앙값
  - `low_income_high_spend`: 소득 < 중앙값 AND 소비 >= 중앙값
  - `low_income_low_spend`: 소득 < 중앙값 AND 소비 < 중앙값
- 반환: `{세그먼트명: {"count": int, "mean_income": float, "mean_spending": float}}`

#### 7. `main(data_path: str) -> dict`
- 전체 파이프라인을 실행하고 `result_q1.json`을 저장합니다.

### 출력 형식

`result_q1.json` 파일로 다음 구조를 저장합니다:

```json
{
  "total_rows_raw": 정수,
  "total_rows_cleaned": 정수,
  "duplicates_removed": 정수,
  "missing_values_filled": 정수,
  "statistics": {
    "열이름": {"mean": 실수, "std": 실수, "min": 실수, "max": 실수, "median": 실수}
  },
  "outlier_counts_iqr": {"열이름": 정수},
  "outlier_counts_zscore": {"열이름": 정수},
  "standardized_mean_check": {"열이름": 실수},
  "standardized_std_check": {"열이름": 실수},
  "segments": {
    "세그먼트명": {"count": 정수, "mean_income": 실수, "mean_spending": 실수}
  }
}
```

- 모든 부동소수점 결과는 소수점 이하 6자리로 반올림합니다.

### 제약 사항
- NumPy만 사용하여 모든 계산을 수행합니다 (`pandas`, `sklearn`, `scipy` 사용 금지).
- CSV 로드 시 표준 라이브러리 `csv` 모듈 사용 가능합니다.
- 표준편차 계산 시 `ddof=0`을 사용합니다.

### 제출 방식
- `q1_solution.py`와 `result_q1.json` 두 파일을 제출합니다.
- `template/q1_solution.py`의 `# TODO` 부분을 채우세요.
