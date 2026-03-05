## 문항 2 정답지 — MAC 연산 심화 분석

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `patterns` | `dict[str, list[list[int\|float]]]` | 키: `img_01`~`img_04`, 값: 3×3 정수 또는 실수 배열 |
| `filters` | `dict[str, list[list[int]]]` | 키: `cross`, `block`, `line`, 값: 3×3 정수 배열 |
| `labels` | `dict[str, str]` | 키: 대소문자 불규칙, 값: 라벨 문자열 |
| MAC 결과 | `int` 또는 `float` | 두 배열의 요소별 곱의 합 |
| 벤치마크 시간 | `float` | 초 단위 평균 실행 시간 |
| 진단 결과 | `dict` | `category`(data_schema/numerical/logic/none)와 `reason` |

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | 필수 함수 12개 정의 | 5점 | AST 자동 |
| 2 | json, time 외 외부 라이브러리 미사용 | 5점 | AST 자동 |
| 3 | MAC 연산 (정수) | 5점 | import 자동 |
| 4 | MAC 연산 (부동소수점) | 5점 | import 자동 |
| 5 | 최적 필터 매칭 | 5점 | import 자동 |
| 6 | 라벨 키 소문자 정규화 | 5점 | import 자동 |
| 7 | epsilon 기반 비교 (True/False) | 5점 | import 자동 |
| 8 | MAC 연산 시간 측정 | 10점 | import 자동 |
| 9 | 시간 복잡도 분석 구조 | 10점 | import 자동 |
| 10 | 실패 원인 3가지 분류 | 10점 | import 자동 |
| 11 | MAC 설명 (한국어, 50자+) | 5점 | import 자동 |
| 12 | 정규화 필요성 (한국어, 30자+) | 5점 | import 자동 |
| 13 | epsilon 비교 필요성 (한국어, 30자+) | 5점 | import 자동 |
| 14 | 전체 파이프라인 결과 | 10점 | import 자동 |

- Pass 기준: 총 90점 중 90점 (14개 전체 정답)
- AI 트랩: labels 키 대소문자 불규칙, img_04 부동소수점 패턴

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| MAC 연산 이해 및 AI 중요성 | test_mac_basic, test_mac_floats, test_mac_explanation |
| 유사도 계산 원리 | test_find_best_match, test_main_result |
| 라벨 표준화 이유 | test_normalize_labels, test_normalization_reason |
| 부동소수점 오차와 epsilon 비교 | test_is_close, test_epsilon_reason |
| 시간 복잡도 O(N²) 분석 | test_measure_mac_time, test_analyze_complexity |
| 실패 케이스 진단 | test_diagnose_failure, test_main_result |
