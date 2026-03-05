## 문제 1: 이미지 기반 객체 카운팅

### 문제

제공되는 이미지에서 **박스(Box)의 개수**를 카운팅하는 규칙 기반 파이프라인을 **4개의 모듈 파일**로 나누어 구현하세요.
데이터 증강을 통한 앙상블 카운팅으로 정확도를 개선하고, 객체 검출 결과를 일주일 단위로 관리하는 일지 시스템을 구축하세요.

#### 프로젝트 구조

| 파일 | 역할 | 핵심 함수 |
|------|------|----------|
| `conv2d.py` | 2D 컨볼루션, 엣지 검출, 이미지 증강 | `conv2d()`, `to_grayscale()`, `compute_edge_magnitude()`, `flip_horizontal()`, `flip_vertical()`, `adjust_brightness()`, `normalize_image()` |
| `counter.py` | 박스 카운팅 및 증강 앙상블 | `count_boxes()`, `ensemble_count()`, `count_boxes_augmented()`, `extract_bounding_boxes()` |
| `metrics.py` | 성능 지표, 방법 비교, 검출 일지 | `compute_metrics()`, `find_worst_case()`, `get_failure_reasons()`, `get_why_learning_based()`, `compare_methods()`, `create_detection_log()`, `generate_weekly_report()` |
| `main.py` | 전체 파이프라인 실행 | `main()` |

제공되는 이미지 셋은 3개 카테고리로 구분됩니다.

| 카테고리 | 장수 | 특징 |
|----------|------|------|
| `easy` | 5장 | 밝은 배경, 박스 간 간격 충분, 균일한 간격 |
| `medium` | 5장 | 박스 일부 겹침, 약간의 그림자 존재 |
| `hard` | 5장 | 적재(Stacked) 형태 포함, 불규칙한 다각형, 크기 편차 큰 |

각 이미지의 정답 박스 개수는 `data/labels.json`에 제공됩니다.

---

### 요구사항

#### conv2d.py — Part A: 컨볼루션 기반 엣지 검출

1. **NumPy만으로 2D 컨볼루션 함수 `conv2d(image, kernel)`를 구현하세요.**
   - 입력: 2D 배열(이미지)과 2D 배열(커널)
   - 출력: **valid 모드**의 컨볼루션 결과

2. **`to_grayscale(rgb)` 함수를 구현하세요.**
   - 공식: `gray = 0.299*R + 0.587*G + 0.114*B`

3. **`compute_edge_magnitude(gray)` 함수를 구현하세요.**
   - Sobel 커널(3x3)을 정의하여 수평/수직 엣지를 각각 검출하세요.
   - `edge_magnitude = sqrt(Gx^2 + Gy^2)`

#### conv2d.py — Part A-2: 이미지 증강

4. **`flip_horizontal(image)` 함수를 구현하세요.**
   - 이미지를 좌우 반전하세요. (2D 또는 3D 배열 모두 지원)

5. **`flip_vertical(image)` 함수를 구현하세요.**
   - 이미지를 상하 반전하세요. (2D 또는 3D 배열 모두 지원)

6. **`adjust_brightness(image, factor)` 함수를 구현하세요.**
   - `factor`를 곱한 뒤 0~255 범위로 클리핑하세요.

7. **`normalize_image(image)` 함수를 구현하세요.**
   - Min-Max 정규화로 0~255 범위에 매핑하세요.

#### counter.py — Part B: 박스 카운팅 파이프라인

8. **`count_boxes(image_path)` 함수를 구현하세요.**
   - 엣지 이미지를 **이진화(thresholding)** 하세요.
   - **Connected Component 분석**으로 박스 개수를 추정하세요. (직접 구현(BFS/DFS) 또는 `scipy.ndimage.label` 사용 가능)
   - **최소 면적 필터**: `min_area` 기준으로 노이즈를 제거하세요.
   - `THRESHOLD`, `MIN_AREA` 변수를 **명시적으로 정의**하세요.

#### counter.py — Part B-2: 증강 앙상블 카운팅

9. **`ensemble_count(counts)` 함수를 구현하세요.**
   - 여러 카운팅 결과 리스트를 받아 **중앙값(median)**을 정수로 반환하세요.

10. **`count_boxes_augmented(image_path)` 함수를 구현하세요.**
    - 원본 이미지에 증강(좌우 반전, 상하 반전, 밝기 조절 등)을 적용하여 여러 버전을 생성하세요.
    - 각 버전의 카운팅 결과를 `ensemble_count()`로 앙상블하세요.

11. **`extract_bounding_boxes(image_path)` 함수를 구현하세요.**
    - 검출된 각 박스의 바운딩 박스 좌표를 추출하세요.
    - 반환: `[{"x_min": 정수, "y_min": 정수, "x_max": 정수, "y_max": 정수, "area": 정수}, ...]`

#### metrics.py — Part C: 정량적 성능 분석 및 한계 보고

12. **`compute_metrics(predictions, labels, category)` 함수를 구현하세요.**
    - MAE (Mean Absolute Error): 예측 개수와 실제 개수 차이의 평균
    - Accuracy: 정확히 맞춘 이미지 수 / 전체 이미지 수

13. **`find_worst_case(predictions, labels, category)` 함수를 구현하세요.**
    - 해당 카테고리에서 오차가 가장 큰 이미지 이름을 반환하세요.

14. **`get_failure_reasons()` 함수를 구현하세요.**
    - hard 카테고리에서 규칙 기반 방식이 실패하는 기술적 원인을 **3가지 이상** 서술하세요.
    - 각 항목은 한국어로 20자 이상 작성하세요.

15. **`get_why_learning_based()` 함수를 구현하세요.**
    - 학습 기반 접근법(CNN 등)이 필요한 이유를 **200자 이내** 한국어로 서술하세요.

#### metrics.py — Part D: 방법 비교 및 검출 일지 관리

16. **`compare_methods(predictions_base, predictions_aug, labels)` 함수를 구현하세요.**
    - 기본 카운팅과 증강 앙상블 카운팅의 카테고리별 MAE/Accuracy를 비교하세요.
    - 반환: 카테고리별 `base_mae`, `augmented_mae`, `mae_improvement`, `base_accuracy`, `augmented_accuracy`

17. **`create_detection_log(predictions, labels, date_str)` 함수를 구현하세요.**
    - 일자별 검출 결과를 로그로 기록하세요.
    - 반환: `date`, `total_images`, `results`(이미지별 예측/실제/오차/정답여부), `daily_accuracy`

18. **`generate_weekly_report(daily_logs)` 함수를 구현하세요.**
    - 7일분 일일 로그를 집계하여 주간 보고서를 생성하세요.
    - 반환: `week_start`, `week_end`, `total_images_processed`, `average_daily_accuracy`, `best_day`, `worst_day`

#### main.py — 전체 파이프라인

19. **`main()` 함수를 구현하세요.**
    - `labels.json` 로드 → 유효 이미지 필터
    - 기본 카운팅 (`count_boxes`) 및 증강 앙상블 카운팅 (`count_boxes_augmented`)
    - 바운딩 박스 추출 (`extract_bounding_boxes`)
    - 메트릭 계산 (기본 + 증강) 및 방법 비교 (`compare_methods`)
    - 7일분 검출 일지 생성 및 주간 보고서 (`create_detection_log`, `generate_weekly_report`)
    - `result_q1.json` 파일로 결과를 저장하세요.

---

### 제약 사항
- `conv2d` 함수는 반드시 **NumPy만으로 직접 구현**하세요. (`cv2.filter2D` 등 사용 금지)
- 이미지 로드에는 `PIL` 또는 `cv2`를 사용할 수 있습니다.
- `threshold` 값과 `min_area` 값은 코드 내에서 **명시적으로 변수**로 정의하세요.
- `counter.py`는 `conv2d.py`를, `main.py`는 `counter.py`와 `metrics.py`를 import하여 사용합니다.

---

### 입력 형식

| 파일/폴더 | 타입 | 설명 |
|-----------|------|------|
| `data/images/` | PNG (640x480 RGB) | `easy_01.png` ~ `easy_05.png`, `medium_01.png` ~ `medium_05.png`, `hard_01.png` ~ `hard_05.png` |
| `data/labels.json` | `dict[str, int]` | `{"easy_01": 3, "easy_02": 5, ...}` 형태 |

---

### 출력 형식

`result_q1.json` 파일로 다음 구조를 저장하세요.

```json
{
  "predictions": {"easy_01": 정수, ...},
  "predictions_augmented": {"easy_01": 정수, ...},
  "sample_bounding_boxes": [
    {"x_min": 정수, "y_min": 정수, "x_max": 정수, "y_max": 정수, "area": 정수},
    ...
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
  "worst_case_image": "hard_XX",
  "failure_reasons": ["이유1 (20자 이상)", "이유2", "이유3"],
  "why_learning_based": "200자 이내 서술",
  "weekly_report": {
    "week_start": "YYYY-MM-DD",
    "week_end": "YYYY-MM-DD",
    "total_images_processed": 정수,
    "average_daily_accuracy": 실수,
    "best_day": "YYYY-MM-DD",
    "worst_day": "YYYY-MM-DD"
  }
}
```

---

### 제출물 구조

아래 **5개 파일**을 하나의 **zip 파일**로 묶어 제출하세요.

```
submission.zip
└── submission/
├── conv2d.py           # Part A: 2D 컨볼루션, 엣지 검출, 이미지 증강
│   ├── conv2d()                  — NumPy 기반 2D 컨볼루션 (valid 모드)
│   ├── to_grayscale()            — RGB → 그레이스케일 변환
│   ├── compute_edge_magnitude()  — Sobel 엣지 크기 계산
│   ├── flip_horizontal()         — 이미지 좌우 반전
│   ├── flip_vertical()           — 이미지 상하 반전
│   ├── adjust_brightness()       — 밝기 조절 (0~255 클리핑)
│   └── normalize_image()         — Min-Max 정규화
│
├── counter.py          # Part B: 박스 카운팅 및 증강 앙상블
│   ├── THRESHOLD, MIN_AREA       — 하이퍼파라미터 변수
│   ├── count_boxes()             — 이미지 → 박스 개수 반환
│   ├── ensemble_count()          — 중앙값 기반 앙상블
│   ├── count_boxes_augmented()   — 증강 + 앙상블 카운팅
│   └── extract_bounding_boxes()  — 바운딩 박스 좌표 추출
│
├── metrics.py          # Part C/D: 성능 분석, 비교, 검출 일지
│   ├── compute_metrics()         — 카테고리별 MAE, Accuracy 계산
│   ├── find_worst_case()         — 오차가 가장 큰 이미지 탐색
│   ├── get_failure_reasons()     — 규칙 기반 실패 원인 (한국어)
│   ├── get_why_learning_based()  — 학습 기반 필요성 서술 (한국어)
│   ├── compare_methods()         — 기본 vs 증강 성능 비교
│   ├── create_detection_log()    — 일자별 검출 로그 생성
│   └── generate_weekly_report()  — 주간 요약 보고서 생성
│
├── main.py             # 전체 파이프라인 실행
│   └── main()                    — 카운팅 → 증강 → 비교 → 일지 → JSON 저장
│
└── result_q1.json      # 실행 결과
```
