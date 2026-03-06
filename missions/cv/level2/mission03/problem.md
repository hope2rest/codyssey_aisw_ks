## 문항: 이미지 기반 객체 카운팅

### 문제

제공되는 이미지에서 박스(Box)의 개수를 카운팅하는 규칙 기반 파이프라인을 4개의 모듈 파일로 나누어 구현하세요.
이미지(easy/medium/hard 각 5장, 총 15장)는 `data/images/`에, 정답 박스 개수는 `data/labels.json`에 저장되어 있습니다.

### 프로젝트 구조

| 파일 | 역할 | 핵심 함수 |
|------|------|----------|
| `conv2d.py` | 2D 컨볼루션, 엣지 검출, 이미지 증강 | `conv2d()`, `to_grayscale()`, `compute_edge_magnitude()`, `flip_horizontal()`, `flip_vertical()`, `adjust_brightness()`, `normalize_image()` |
| `counter.py` | 박스 카운팅 및 증강 앙상블 | `count_boxes()`, `ensemble_count()`, `count_boxes_augmented()`, `extract_bounding_boxes()` |
| `metrics.py` | 정량적 성능 지표, 방법 비교 | `compute_metrics()`, `find_worst_case()`, `compare_methods()` |
| `main.py` | 전체 파이프라인 실행 | `main()` |

### 입력 데이터

| 파일/폴더 | 타입 | 설명 |
|-----------|------|------|
| `data/images/` | PNG (640x480 RGB) | `easy_01.png` ~ `easy_05.png`, `medium_01.png` ~ `medium_05.png`, `hard_01.png` ~ `hard_05.png` |
| `data/labels.json` | `dict[str, int]` | `{"easy_01": 3, "easy_02": 5, ...}` 형태 |

이미지 카테고리:

| 카테고리 | 장수 | 특징 |
|----------|------|------|
| `easy` | 5장 | 밝은 배경, 박스 간 간격 충분, 균일한 간격 |
| `medium` | 5장 | 박스 일부 겹침, 약간의 그림자 존재 |
| `hard` | 5장 | 적재(Stacked) 형태 포함, 불규칙한 다각형, 크기 편차 큰 |

### 구현 요구사항

#### Part A: conv2d.py - 컨볼루션 기반 엣지 검출

#### 1. `conv2d(image, kernel)`
- NumPy만으로 2D 컨볼루션을 구현합니다 (valid 모드).

#### 2. `to_grayscale(rgb)`
- `gray = 0.299*R + 0.587*G + 0.114*B` 공식으로 변환합니다.

#### 3. `compute_edge_magnitude(gray)`
- Sobel 커널(3x3)로 수평/수직 엣지를 검출합니다.
- `edge_magnitude = sqrt(Gx^2 + Gy^2)`

#### Part A-2: conv2d.py - 이미지 증강

#### 4. `flip_horizontal(image)`
- 이미지를 좌우 반전합니다 (2D/3D 배열 지원).

#### 5. `flip_vertical(image)`
- 이미지를 상하 반전합니다 (2D/3D 배열 지원).

#### 6. `adjust_brightness(image, factor)`
- `factor`를 곱한 뒤 0~255 범위로 클리핑합니다.

#### 7. `normalize_image(image)`
- Min-Max 정규화로 0~255 범위에 매핑합니다.

#### Part B: counter.py - 박스 카운팅 파이프라인

#### 8. `count_boxes(image_path)`
- 엣지 이미지를 이진화(thresholding)합니다.
- Connected Component 분석으로 박스 개수를 추정합니다 (직접 구현(BFS/DFS) 또는 `scipy.ndimage.label` 사용 가능).
- 최소 면적 필터(`min_area`)로 노이즈를 제거합니다.
- `THRESHOLD`, `MIN_AREA` 변수를 명시적으로 정의합니다.

#### Part B-2: counter.py - 증강 앙상블 카운팅

#### 9. `ensemble_count(counts)`
- 여러 카운팅 결과 리스트를 받아 중앙값(median)을 정수로 반환합니다.

#### 10. `count_boxes_augmented(image_path)`
- 원본 이미지에 증강(좌우/상하 반전, 밝기 조절 등)을 적용하여 여러 버전을 생성합니다.
- 각 버전의 카운팅 결과를 `ensemble_count()`로 앙상블합니다.

#### 11. `extract_bounding_boxes(image_path)`
- 검출된 각 박스의 바운딩 박스 좌표를 추출합니다.
- 반환: `[{"x_min": 정수, "y_min": 정수, "x_max": 정수, "y_max": 정수, "area": 정수}, ...]`

#### Part C: metrics.py - 정량적 성능 분석

#### 12. `compute_metrics(predictions, labels, category)`
- MAE (Mean Absolute Error): 예측 개수와 실제 개수 차이의 평균
- Accuracy: 정확히 맞춘 이미지 수 / 전체 이미지 수

#### 13. `find_worst_case(predictions, labels, category)`
- 해당 카테고리에서 오차가 가장 큰 이미지 이름을 반환합니다.

#### 14. `compare_methods(predictions_base, predictions_aug, labels)`
- 기본 카운팅과 증강 앙상블 카운팅의 카테고리별 MAE/Accuracy를 비교합니다.
- 반환: 카테고리별 `base_mae`, `augmented_mae`, `mae_improvement`, `base_accuracy`, `augmented_accuracy`

#### Part D: main.py - 전체 파이프라인

#### 15. `main()`
- `labels.json` 로드 → 유효 이미지 필터
- 기본 카운팅 (`count_boxes`) 및 증강 앙상블 카운팅 (`count_boxes_augmented`)
- 바운딩 박스 추출 (`extract_bounding_boxes`)
- 메트릭 계산 (기본 + 증강) 및 방법 비교 (`compare_methods`)
- `result_q3.json` 파일로 결과를 저장

### 출력 형식

`result_q3.json` 파일로 다음 구조를 저장합니다:

```json
{
  "predictions": {"easy_01": 정수, ...},
  "predictions_augmented": {"easy_01": 정수, ...},
  "sample_bounding_boxes": [
    {"x_min": 정수, "y_min": 정수, "x_max": 정수, "y_max": 정수, "area": 정수}
  ],
  "metrics": {
    "easy":   {"mae": 실수, "accuracy": 실수},
    "medium": {"mae": 실수, "accuracy": 실수},
    "hard":   {"mae": 실수, "accuracy": 실수}
  },
  "metrics_augmented": {
    "easy":   {"mae": 실수, "accuracy": 실수},
    "medium": {"mae": 실수, "accuracy": 실수},
    "hard":   {"mae": 실수, "accuracy": 실수}
  },
  "method_comparison": {
    "easy":   {"base_mae": 실수, "augmented_mae": 실수, "mae_improvement": 실수, ...},
    "medium": {"base_mae": 실수, "augmented_mae": 실수, "mae_improvement": 실수, ...},
    "hard":   {"base_mae": 실수, "augmented_mae": 실수, "mae_improvement": 실수, ...}
  },
  "worst_case_image": "hard_XX"
}
```

### 제약 사항

- `conv2d` 함수는 반드시 NumPy만으로 직접 구현 (`cv2.filter2D` 등 사용 금지)
- 이미지 로드에는 `PIL` 또는 `cv2`를 사용할 수 있습니다.
- `THRESHOLD`와 `MIN_AREA` 값은 코드 내에서 명시적으로 변수로 정의
- `counter.py`는 `conv2d.py`를, `main.py`는 `counter.py`와 `metrics.py`를 import하여 사용

### 제출 방식

- `conv2d.py`, `counter.py`, `metrics.py`, `main.py`, `result_q3.json` 총 5개 파일을 zip으로 묶어 제출합니다.
- `template/` 디렉토리의 각 파일의 `# TODO` 부분을 채우세요.
