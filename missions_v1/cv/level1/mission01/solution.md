## 문항 1 정답지 — 이커머스 데이터 전처리 및 이상치 탐지 파이프라인

### 데이터 타입

| 키 | 타입 | 설명 |
|----|------|------|
| total_rows_raw | int | 원본 CSV 행 수 (205) |
| total_rows_cleaned | int | 정제 후 행 수 (200) |
| duplicates_removed | int | 제거된 중복 행 수 (5) |
| missing_values_filled | int | 대체된 결측값 수 (15) |
| statistics | dict | 열별 기술 통계량 |
| outlier_counts_iqr | dict | IQR 이상치 개수 |
| outlier_counts_zscore | dict | Z-score 이상치 개수 |
| standardized_mean_check | dict | 표준화 후 평균 (≈0) |
| standardized_std_check | dict | 표준화 후 표준편차 (≈1) |
| segments | dict | 4개 고객 세그먼트 |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | 필수 함수 7개 정의 | AST 분석으로 함수 존재 확인 |
| 2 | sklearn/scipy/pandas 미사용 | 소스 코드 문자열 검색 |
| 3 | 데이터 로드 후 형태 | (200, 7) ndarray 반환 |
| 4 | NaN 제거 확인 | 정제 후 NaN 없음 |
| 5 | 통계량 정확도 | mean/std/min/max/median 값 검증 |
| 6 | IQR 이상치 탐지 | 알려진 이상치 탐지 여부 |
| 7 | Z-score 이상치 탐지 | threshold 기반 탐지 검증 |
| 8 | 표준화 정확도 | 평균≈0, 표준편차≈1 |
| 9 | 세그먼트 합계 | 4개 세그먼트 합 = 전체 행 수 |
| 10 | result_q1.json 구조 | 필수 키 10개 존재 확인 |
| 11 | result_q1.json 값 | 정량적 값 일치 검증 |

- **AI 트랩**:
  - `ddof=0` (모집단 표준편차) 사용 필수
  - 중복 행 제거 시 첫 번째만 유지
  - 결측값 대체 시 중앙값 사용 (평균이 아닌)
  - IQR 이상치: 1.5배 규칙 정확히 적용
