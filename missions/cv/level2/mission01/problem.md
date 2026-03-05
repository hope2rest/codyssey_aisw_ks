## 문제 3: 이미지 기반 객체 카운팅

### [ 문제 ]

제공되는 이미지에서 **박스(Box)의 개수**를 카운팅하는 규칙 기반 파이프라인을 **4개의 모듈 파일**로 나누어 구현하시오.

#### 프로젝트 구조

| 파일 | 역할 | 핵심 함수 |
|------|------|----------|
| `conv2d.py` | 2D 컨볼루션 및 엣지 검출 | `conv2d()`, `to_grayscale()`, `compute_edge_magnitude()` |
| `counter.py` | 박스 카운팅 파이프라인 | `count_boxes()` |
| `metrics.py` | 성능 지표 및 한계 분석 | `compute_metrics()`, `find_worst_case()`, `get_failure_reasons()`, `get_why_learning_based()` |
| `main.py` | 전체 파이프라인 실행 | `main()` |

제공되는 이미지 셋은 3개 카테고리로 구분됩니다:

| 카테고리 | 장수 | 특징 |
|----------|------|------|
| `easy` | 5장 | 밝은 배경, 박스 간 간격 충분, 균일한 간격 |
| `medium` | 5장 | 박스 일부 겹침, 약간의 그림자 존재 |
| `hard` | 5장 | 적재(Stacked) 형태 포함, 불규칙한 다각형, 크기 편차 큰 |

각 이미지의 정답 박스 개수는 `data/labels.json`에 제공됩니다.

---

### [ 요구사항 ]

#### conv2d.py — Part A: 컨볼루션 기반 엣지 검출

1. **NumPy만으로 2D 컨볼루션 함수 `conv2d(image, kernel)`를 구현하시오.**
   - 입력: 2D 배열(이미지)과 2D 배열(커널)
   - 출력: **valid 모드**의 컨볼루션 결과

2. **`to_grayscale(rgb)` 함수를 구현하시오.**
   - 공식: `gray = 0.299*R + 0.587*G + 0.114*B`

3. **`compute_edge_magnitude(gray)` 함수를 구현하시오.**
   - Sobel 커널(3x3)을 정의하여 수평/수직 엣지를 각각 검출하시오.
   - `edge_magnitude = sqrt(Gx^2 + Gy^2)`

#### counter.py — Part B: 박스 카운팅 파이프라인

4. **`count_boxes(image_path)` 함수를 구현하시오.**
   - 엣지 이미지를 **이진화(thresholding)** 하시오.
   - **Connected Component 분석**으로 박스 개수를 추정하시오. (직접 구현(BFS/DFS) 또는 `scipy.ndimage.label` 사용 가능)
   - **최소 면적 필터**: `min_area` 기준으로 노이즈를 제거하시오.
   - `THRESHOLD`, `MIN_AREA` 변수를 **명시적으로 정의**하시오.

#### metrics.py — Part C: 정량적 성능 분석 및 한계 보고

5. **`compute_metrics(predictions, labels, category)` 함수를 구현하시오.**
   - MAE (Mean Absolute Error): 예측 개수와 실제 개수 차이의 평균
   - Accuracy: 정확히 맞춘 이미지 수 / 전체 이미지 수

6. **`find_worst_case(predictions, labels, category)` 함수를 구현하시오.**
   - 해당 카테고리에서 오차가 가장 큰 이미지 이름을 반환하시오.

7. **`get_failure_reasons()` 함수를 구현하시오.**
   - hard 카테고리에서 규칙 기반 방식이 실패하는 기술적 원인을 **3가지 이상** 서술하시오.
   - 각 항목: 한국어, 20자 이상

8. **`get_why_learning_based()` 함수를 구현하시오.**
   - 학습 기반 접근법(CNN 등)이 필요한 이유를 **200자 이내** 한국어로 서술하시오.

#### main.py — 전체 파이프라인

9. **`main()` 함수를 구현하시오.**
   - `labels.json` 로드 → 유효 이미지 필터 → 카운팅 → 메트릭 계산 → JSON 출력
   - `result_q3.json` 파일로 결과를 저장하시오.

---

### [ 제약 사항 ]
- `conv2d` 함수는 반드시 **NumPy만으로 직접 구현** (`cv2.filter2D` 등 사용 금지)
- 이미지 로드에는 `PIL` 또는 `cv2` 사용 가능
- `threshold` 값과 `min_area` 값은 코드 내에서 **명시적으로 변수**로 정의할 것
- **모듈 간 import**: `counter.py`는 `conv2d.py`를, `main.py`는 `counter.py`와 `metrics.py`를 import하여 사용

---

### [ 입력 형식 ]

| 파일/폴더 | 설명 |
|-----------|------|
| `data/images/` | `easy_01.png` ~ `easy_05.png`, `medium_01.png` ~ `medium_05.png`, `hard_01.png` ~ `hard_05.png` (640x480 RGB) |
| `data/labels.json` | `{"easy_01": 3, "easy_02": 5, ...}` 형태 |

> **주의**: `labels.json`에는 `test_01` 키가 포함되어 있으나, 해당 이미지 파일은 존재하지 않습니다. 실제 이미지 파일이 있는 항목만 처리해야 합니다.

---

### [ 출력 형식 ]

`result_q3.json` 파일로 다음 구조를 저장하시오:

```json
{
  "predictions": {"easy_01": 정수, "easy_02": 정수, ...},
  "metrics": {
    "easy":   {"mae": 실수, "accuracy": 실수},
    "medium": {"mae": 실수, "accuracy": 실수},
    "hard":   {"mae": 실수, "accuracy": 실수}
  },
  "worst_case_image": "hard_XX",
  "failure_reasons": ["이유1 (20자 이상)", "이유2", "이유3"],
  "why_learning_based": "200자 이내 서술"
}
```

---

### [ 제출 방식 ]

아래 **5개 파일**을 제출하시오.

| 파일 | 설명 |
|------|------|
| `conv2d.py` | 2D 컨볼루션 및 엣지 검출 구현 |
| `counter.py` | 박스 카운팅 파이프라인 구현 |
| `metrics.py` | 성능 지표 계산 및 한계 분석 |
| `main.py` | 전체 파이프라인 실행 |
| `result_q3.json` | 실행 결과 JSON |
