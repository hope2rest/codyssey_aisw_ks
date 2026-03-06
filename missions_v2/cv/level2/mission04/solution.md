## 문항 4 정답지 — 이미지 객체 검출 및 데이터 증강을 통한 정확도 향상

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `labels.json` | JSON | 이미지별 정답 박스 개수 {"easy_01": 3, ...} |
| `images/*.png` | PNG | 640×480 RGB 이미지 (easy/medium/hard 각 5장) |
| 그레이스케일 | `np.ndarray` | (H, W) float 배열 |
| 엣지 크기 | `np.ndarray` | sqrt(Gx² + Gy²) |
| 바운딩 박스 | `list[dict]` | x_min, y_min, x_max, y_max, area |
| 메트릭 | `dict` | MAE, Accuracy |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | conv2d.py 필수 함수 7개 정의 (conv2d, to_grayscale, compute_edge_magnitude, flip_horizontal, flip_vertical, adjust_brightness, normalize_image) | AST 자동 |
| 2 | counter.py 필수 함수 4개 정의 (count_boxes, ensemble_count, count_boxes_augmented, extract_bounding_boxes) | AST 자동 |
| 3 | metrics.py 필수 함수 3개 정의 (compute_metrics, find_worst_case, compare_methods) | AST 자동 |
| 4 | conv2d 연산 (valid 모드, NumPy 직접 구현) | import 자동 |
| 5 | 그레이스케일 변환 (0.299R + 0.587G + 0.114B) | import 자동 |
| 6 | Sobel 엣지 검출 (수평/수직 커널) | import 자동 |
| 7 | 이미지 증강 (좌우/상하 반전, 밝기 조절) | import 자동 |
| 8 | 이진화 + Connected Component 카운팅 | import 자동 |
| 9 | THRESHOLD, MIN_AREA 변수 명시적 정의 | AST 자동 |
| 10 | 앙상블 카운팅 (중앙값) | import 자동 |
| 11 | 바운딩 박스 추출 | import 자동 |
| 12 | MAE/Accuracy 계산 | import 자동 |
| 13 | 최악 케이스 탐색 | import 자동 |
| 14 | 기본 vs 증강 방법 비교 | import 자동 |
| 15 | result_q4.json 필수 키 확인 | JSON 자동 |
| 16 | 예측 결과 15개 이미지 포함 | JSON 자동 |
| 17 | easy 카테고리 MAE < 2.0 | JSON 자동 |
| 18 | 증강이 기본보다 MAE 개선 | JSON 자동 |

- Pass 기준: 18개 전체 통과
- AI 트랩: conv2d에서 cv2.filter2D 사용 (금지), 그레이스케일 변환 계수 오류, Sobel 커널 방향 혼동, 이진화 임계값 부적절, Connected Component에서 최소 면적 필터 누락, 밝기 조절 후 클리핑 누락

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 2D 컨볼루션 직접 구현 | test_conv2d_valid, test_conv2d_identity |
| 그레이스케일 변환 | test_to_grayscale |
| Sobel 엣지 검출 | test_edge_magnitude |
| 이미지 증강 기법 | test_flip_horizontal, test_flip_vertical, test_adjust_brightness |
| 객체 카운팅 (Connected Component) | test_count_boxes |
| 앙상블 기법 (중앙값 투표) | test_ensemble_count, test_count_boxes_augmented |
| 바운딩 박스 추출 | test_extract_bounding_boxes |
| 정량적 성능 평가 | test_compute_metrics, test_find_worst_case, test_compare_methods |
| 파이프라인 통합 | test_result_structure, test_predictions_count, test_easy_mae, test_augmented_improvement |
