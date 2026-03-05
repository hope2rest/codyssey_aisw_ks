## 문항 3 정답지 — 이미지 기반 객체 카운팅 (멀티 모듈)

### 정답 파일 구성

| 파일 | 주요 내용 |
|------|----------|
| `conv2d.py` | NumPy stride_tricks 기반 conv2d, Sobel 엣지 검출, 그레이스케일 변환 |
| `counter.py` | PIL 이미지 로드, 이진화, scipy connected component, min_area 필터 |
| `metrics.py` | MAE/Accuracy 계산, worst case 탐색, 실패 원인 분석 텍스트 |
| `main.py` | labels.json 로드, 유효 이미지 필터, 파이프라인 실행, result_q3.json 저장 |
| `result_q3.json` | 전체 실행 결과 |

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|------------|
| 1 | 필수 파일 4개 존재 (conv2d, counter, metrics, main) | 10점 | 파일 존재 확인 |
| 2 | conv2d.py 필수 함수 3개 정의 | 10점 | AST 자동 |
| 3 | conv2d.py에서 filter2D 미사용 | 10점 | 소스코드 검색 |
| 4 | conv2d identity 커널 테스트 | 10점 | import 자동 |
| 5 | conv2d Sobel 커널 테스트 | 10점 | import 자동 |
| 6 | 그레이스케일 변환 검증 | 5점 | import 자동 |
| 7 | counter.py 구조 검증 (count_boxes, THRESHOLD, MIN_AREA) | 5점 | AST 자동 |
| 8 | compute_metrics MAE/Accuracy 계산 | 10점 | import 자동 |
| 9 | find_worst_case 검증 | 5점 | import 자동 |
| 10 | 유효 이미지만 분석 (test_01 제외) | 10점 | JSON 자동 |
| 11 | failure_reasons (3개+, 한국어, 20자+) | 10점 | JSON 자동 |
| 12 | why_learning_based (30~200자, 한국어) | 5점 | JSON 자동 |

- Pass 기준: 총 100점 중 100점 (12개 전체 정답)
- AI 트랩: filter2D 사용 금지, test_01 팬텀 라벨, MAE=0.0 과적합 의심

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 2D 컨볼루션 이해 및 NumPy 구현 | test_conv2d_identity, test_conv2d_sobel |
| 엣지 검출 파이프라인 설계 | test_grayscale, test_conv2d_functions |
| 이미지 처리 파이프라인 통합 | test_counter_functions |
| 정량적 성능 평가 | test_compute_metrics, test_find_worst_case |
| 모듈 분리 및 프로젝트 구조 | test_required_files_exist |
| 규칙 기반 한계 분석 | test_failure_reasons, test_why_learning_based |
| 데이터 유효성 검증 | test_valid_images_only |
