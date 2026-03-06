## 문항 2 정답지 — 2D 컨볼루션 기반 특징 추출

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `images` | `dict[str, list[list[int]]]` | 키: `img_01`~`img_03`, 값: 5×5 정수 배열 |
| `kernels` | `dict[str, list[list[int]]]` | 키: `edge_h`, `edge_v`, `sharpen`, 값: 3×3 정수 배열 |
| conv2d 결과 | `list[list[int]]` | (H-kH+1)×(W-kW+1) 크기 배열 |
| relu 결과 | `list[list[int]]` | 음수가 0으로 치환된 배열 |
| 통계 | `dict` | `min`, `max`, `mean` |
| 특징맵 | `dict[str, list[list[int]]]` | 커널별 relu(conv2d) 결과 |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | 필수 함수 9개 정의 | AST 자동 |
| 2 | json 외 외부 라이브러리 미사용 | AST 자동 |
| 3 | 제로 패딩 적용 | import 자동 |
| 4 | 기본 컨볼루션 연산 | import 자동 |
| 5 | 컨볼루션 출력 크기 | import 자동 |
| 6 | ReLU 활성화 | import 자동 |
| 7 | Flatten 변환 | import 자동 |
| 8 | 통계 계산 (min/max/mean) | import 자동 |
| 9 | 다중 커널 특징 추출 | import 자동 |
| 10 | 최강 특징 커널 탐색 | import 자동 |
| 11 | 전체 파이프라인 특징맵 합계 | import 자동 |
| 12 | 전체 파이프라인 최강 특징 | import 자동 |

- Pass 기준: 12개 전체 통과
- AI 트랩: conv2d 자동 패딩 적용, relu in-place 수정, compute_stats 정수 나눗셈, 출력 크기 계산 오류

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 제로 패딩 이해 | test_pad_matrix |
| 컨볼루션 연산 (MAC 활용) | test_conv2d_basic, test_conv2d_output_size |
| 활성화 함수 (ReLU) | test_relu |
| 데이터 변환 (flatten) | test_flatten |
| 통계 분석 | test_compute_stats |
| 특징 추출 파이프라인 | test_extract_features, test_find_strongest_feature |
| 전체 파이프라인 통합 | test_main_feature_sums, test_main_strongest |
