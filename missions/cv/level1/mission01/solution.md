## 문항 1 정답지 — SVD 기반 데이터 차원 축소 및 복원 정확도 분석

### 데이터 타입

| 키 | 타입 | 설명 |
|----|------|------|
| optimal_k | int | 누적 분산 95% 이상이 되는 최소 k |
| cumulative_variance_at_k | float | 최적 k에서의 누적 분산 비율 (소수점 6자리) |
| reconstruction_mse | float | 원본-복원 데이터 간 MSE (소수점 6자리) |
| top_5_singular_values | list[float] | 상위 5개 특이값 (소수점 6자리) |
| explained_variance_ratio_top5 | list[float] | 상위 5개 설명 분산 비율 (소수점 6자리) |

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | optimal_k 일치 | 20점 | 정답과 정수 값 일치 여부 |
| 2 | cumulative_variance_at_k 정확도 | 20점 | 정답 대비 소수점 6자리 오차 범위 검증 |
| 3 | reconstruction_mse 정확도 | 20점 | 정답 대비 소수점 6자리 오차 범위 검증 |
| 4 | top_5_singular_values 정확도 | 20점 | 상위 5개 특이값 각각의 오차 범위 검증 |
| 5 | explained_variance_ratio_top5 정확도 | 20점 | 상위 5개 분산 비율 각각의 오차 범위 검증 |

- **Pass 기준**: 5개 체크 항목 모두 충족 (100점 만점)
- **AI 트랩**:
  - `ddof=0` (모집단 표준편차) 사용 필수 — `ddof=1` 사용 시 결과 불일치
  - 상수 열(std 근사 0) 제거 후 표준화 필요 — 제거하지 않으면 division by zero 또는 결과 왜곡
  - `full_matrices=False`로 SVD 수행 — 기본값(`True`) 사용 시 행렬 크기 불일치

### 학습 목표 매핑

| 학습 목표 | 관련 구현 요구사항 |
|----------|------------------|
| SVD 분해 원리 이해 | 요구사항 2: numpy.linalg.svd로 U, S, Vt 분해 |
| 분산 기반 차원 선택 | 요구사항 3, 4: Explained Variance Ratio 계산 및 최적 k 결정 |
| 데이터 전처리 역량 | 요구사항 1: 상수 열 제거, ddof=0 표준화 |
| 차원 축소 및 복원 | 요구사항 5: k개 주성분으로 축소 후 복원 |
| 복원 품질 평가 | 요구사항 6: MSE 기반 복원 오차 측정 |
