## 문항 1: SVD 기반 데이터 차원 축소 및 복원 정확도 분석

### 문제

공장의 100개 센서에서 수집된 500건의 측정 데이터를 SVD로 분해하여 차원을 축소하고, 복원 정확도를 분석하는 프로그램을 구현하세요.
- 센서 데이터는 `data/sensor_data.csv`에 저장되어 있습니다 (헤더 없는 500x100 숫자 행렬, 쉼표 구분).
- SVD(Singular Value Decomposition)는 행렬을 U, S, Vt 세 행렬로 분해하는 기법으로, 데이터의 핵심 패턴만 보존하면서 노이즈를 제거하는 차원 축소에 활용됩니다.

### sensor_data.csv 구조

| 구분 | 내용 |
|------|------|
| 형태 | 500x100 숫자 행렬 (헤더 없음) |
| 행 | 1건의 측정 데이터 (총 500건) |
| 열 | 1개 센서 값 (총 100개 센서) |
| 구분자 | 쉼표(,) |

### 구현 요구사항

#### 1. 데이터 로드 및 전처리
- `sensor_data.csv`를 로드하세요 (헤더 없음).
- 각 열(feature)에 대해 평균 0, 표준편차 1로 표준화(Standardization)하세요.
- 표준편차 계산 시 `ddof=0`을 사용하세요.
- 상수 열(표준편차가 0에 가까운 열)은 제거 후 표준화하세요.

#### 2. SVD 분해
- 표준화된 데이터 행렬 X에 대해 `numpy.linalg.svd`를 사용하여 U, S, Vt로 분해하세요.
- 반드시 `full_matrices=False` 옵션을 사용하세요.

#### 3. Explained Variance Ratio 계산
- 각 특이값에 대한 Explained Variance Ratio를 다음 수식으로 계산하세요:
  ```
  explained_variance_ratio[i] = S[i]** 2 / sum(S** 2)
  ```

#### 4. 최적 k 결정
- Cumulative Explained Variance Ratio가 처음으로 95% 이상이 되는 최소 k값을 구하세요.

#### 5. 차원 축소 및 복원
- 결정된 k값으로 데이터를 축소하고 다시 복원하세요:
  ```
  X_reduced = U[:, :k] * S[:k]
  X_reconstructed = X_reduced @ Vt[:k, :]
  ```

#### 6. 복원 오차 계산 및 결과 저장
- 원본 표준화 데이터와 복원 데이터 간의 MSE를 계산하세요:
  ```
  MSE = mean((X - X_reconstructed)** 2)
  ```
- 결과를 `result_q1.json` 파일로 다음 구조에 맞게 저장하세요:
  ```json
  {
    "optimal_k": 정수,
    "cumulative_variance_at_k": 실수 (소수점 6자리),
    "reconstruction_mse": 실수 (소수점 6자리),
    "top_5_singular_values": [실수 5개],
    "explained_variance_ratio_top5": [실수 5개]
  }
  ```

### 제약 사항
- NumPy만 사용하여 모든 계산을 수행할 것 (`pandas`는 데이터 로드에만 허용, `sklearn`/`scipy` 금지)
- `numpy.linalg.svd` 호출 시 `full_matrices=False` 옵션을 반드시 사용할 것
- 모든 부동소수점 결과는 소수점 이하 6자리로 반올림하여 출력할 것

### 제출 방식
- `q1_solution.py`와 `result_q1.json` 두 파일을 제출합니다.
