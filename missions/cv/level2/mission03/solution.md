## 문항 1 정답지 — 이미지 기반 객체 카운팅

### 정답 파일 구성

| 파일 | 주요 내용 |
|------|----------|
| `conv2d.py` | NumPy stride_tricks 기반 conv2d, Sobel 엣지 검출, 그레이스케일 변환, 좌우/상하 반전, 밝기 조절, Min-Max 정규화 |
| `counter.py` | PIL 이미지 로드, 이진화, scipy connected component, min_area 필터, 앙상블 카운팅, 바운딩 박스 추출 |
| `metrics.py` | MAE/Accuracy 계산, worst case 탐색, 실패 원인 분석, 기본 vs 증강 비교, 일자별 검출 로그, 주간 보고서 |
| `main.py` | labels.json 로드, 유효 이미지 필터, 기본/증강 카운팅, 방법 비교, 주간 일지, result_q1.json 저장 |
| `result_q1.json` | 전체 실행 결과 |

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | conv2d.py 필수 함수 7개 정의 (conv2d, to_grayscale, compute_edge_magnitude, flip_horizontal, flip_vertical, adjust_brightness, normalize_image) | 5점 | AST 자동 |
| 2 | conv2d.py에서 filter2D 미사용 | 5점 | 소스코드 검색 |
| 3 | counter.py 필수 함수 4개 + THRESHOLD/MIN_AREA 정의 | 5점 | AST 자동 |
| 4 | metrics.py 필수 함수 7개 정의 | 5점 | AST 자동 |
| 5 | conv2d identity 커널 테스트 | 5점 | import 자동 |
| 6 | conv2d Sobel 커널 테스트 | 5점 | import 자동 |
| 7 | 그레이스케일 변환 검증 | 5점 | import 자동 |
| 8 | 좌우/상하 반전 검증 | 5점 | import 자동 |
| 9 | 밝기 조절 및 정규화 검증 | 5점 | import 자동 |
| 10 | 중앙값 앙상블 카운팅 검증 | 5점 | import 자동 |
| 11 | 증강 앙상블 카운팅 정수 반환 | 5점 | import 자동 |
| 12 | 바운딩 박스 출력 형식 검증 | 5점 | import 자동 |
| 13 | compute_metrics MAE/Accuracy 계산 | 5점 | import 자동 |
| 14 | find_worst_case 검증 | 5점 | import 자동 |
| 15 | compare_methods 기본 vs 증강 비교 | 5점 | import 자동 |
| 16 | 검출 로그 및 주간 보고서 검증 | 5점 | import 자동 |
| 17 | 유효 이미지만 분석 (test_01 제외) | 5점 | JSON 자동 |
| 18 | 증강 카운팅 결과 포함 | 5점 | JSON 자동 |
| 19 | 방법 비교 결과 포함 | 5점 | JSON 자동 |
| 20 | 주간 보고서 결과 포함 | 5점 | JSON 자동 |
| 21 | failure_reasons (3개+, 한국어, 20자+) | 5점 | JSON 자동 |
| 22 | why_learning_based (30~200자, 한국어) | 5점 | JSON 자동 |

- Pass 기준: 총 110점 중 110점 (22개 전체 정답)
- AI 트랩: filter2D 사용 금지, test_01 팬텀 라벨, MAE=0.0 과적합 의심

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 2D 컨볼루션 이해 및 NumPy 구현 | test_conv2d_identity, test_conv2d_sobel |
| 엣지 검출 파이프라인 설계 | test_grayscale, test_conv2d_functions |
| 이미지 증강 기법 구현 | test_flip_operations, test_brightness_and_normalize |
| 이미지 처리 파이프라인 통합 | test_counter_functions, test_ensemble_count |
| 증강 기반 정확도 개선 | test_count_boxes_augmented, test_augmented_predictions |
| 바운딩 박스 추출 | test_bounding_boxes_format |
| 정량적 성능 평가 | test_compute_metrics, test_find_worst_case |
| 방법론 비교 분석 | test_compare_methods, test_method_comparison |
| 객체 검출 일지 관리 | test_detection_log_and_weekly_report, test_weekly_report |
| 규칙 기반 한계 분석 | test_failure_reasons, test_why_learning_based |
| 데이터 유효성 검증 | test_valid_images_only |
