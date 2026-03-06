## 문항 4 정답지 — 이미지 객체 검출 및 데이터 증강을 통한 정확도 향상

### 정답 코드

#### conv2d.py

```python
"""conv2d.py — NumPy 기반 2D 컨볼루션, 엣지 검출, 이미지 증강 모듈"""
import numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
GAUSS3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0


def conv2d(image, kernel):
    """2D 컨볼루션 (valid 모드, NumPy stride_tricks 활용)."""
    kH, kW = kernel.shape
    iH, iW = image.shape
    oH = iH - kH + 1
    oW = iW - kW + 1
    k_flip = kernel[::-1, ::-1]
    shape = (oH, oW, kH, kW)
    strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    output = np.einsum('ijkl,kl->ij', windows, k_flip)
    return output


def to_grayscale(rgb):
    """gray = 0.299*R + 0.587*G + 0.114*B"""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def pad_to(arr, target_h, target_w):
    """valid 모드로 줄어든 배열을 원본 크기로 복원."""
    ph = target_h - arr.shape[0]
    pw = target_w - arr.shape[1]
    return np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), mode='edge')


def compute_edge_magnitude(gray):
    """가우시안 블러 -> Sobel Gx/Gy -> edge_magnitude = sqrt(Gx^2 + Gy^2)"""
    h, w = gray.shape
    blurred = conv2d(gray, GAUSS3)
    blurred = pad_to(blurred, h, w)
    Gx = conv2d(blurred, SOBEL_X)
    Gy = conv2d(blurred, SOBEL_Y)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    magnitude = pad_to(magnitude, h, w)
    return magnitude


def flip_horizontal(image):
    """이미지 좌우 반전."""
    return image[:, ::-1].copy()


def flip_vertical(image):
    """이미지 상하 반전."""
    return image[::-1, :].copy()


def adjust_brightness(image, factor):
    """밝기 조절 (factor 곱한 뒤 0~255 클리핑)."""
    return np.clip(image * factor, 0, 255).astype(image.dtype)


def normalize_image(image):
    """Min-Max 정규화 (0~255 범위)."""
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.float64)
    return (image - min_val) / (max_val - min_val) * 255.0
```

#### counter.py

```python
"""counter.py — 박스 카운팅 파이프라인 및 증강 앙상블 모듈"""
import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_closing
from conv2d import (to_grayscale, compute_edge_magnitude,
                    flip_horizontal, flip_vertical, adjust_brightness)

THRESHOLD = 30
MIN_AREA = 100


def _count_from_rgb(rgb, threshold=THRESHOLD, min_area=MIN_AREA):
    """RGB 배열에서 박스 개수 카운팅."""
    gray = to_grayscale(rgb)
    edge_mag = compute_edge_magnitude(gray)
    binary = (edge_mag > threshold).astype(np.uint8)
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct, iterations=3)
    labeled_array, num_features = scipy_label(closed)
    valid_count = 0
    for comp_id in range(1, num_features + 1):
        area = int(np.sum(labeled_array == comp_id))
        if area >= min_area:
            valid_count += 1
    return valid_count


def count_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    """이미지에서 박스 개수를 카운팅하는 파이프라인."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)
    return _count_from_rgb(rgb, threshold, min_area)


def ensemble_count(counts):
    """여러 카운팅 결과의 중앙값 반환 (정수)."""
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    if n % 2 == 1:
        return sorted_counts[n // 2]
    return round((sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2)


def count_boxes_augmented(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    """원본 + 증강 이미지에 대해 앙상블 카운팅."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)
    augmented = [rgb, flip_horizontal(rgb), flip_vertical(rgb),
                 adjust_brightness(rgb, 0.8), adjust_brightness(rgb, 1.2)]
    counts = [_count_from_rgb(aug, threshold, min_area) for aug in augmented]
    return ensemble_count(counts)


def extract_bounding_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    """검출된 박스의 바운딩 박스 좌표 추출."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)
    gray = to_grayscale(rgb)
    edge_mag = compute_edge_magnitude(gray)
    binary = (edge_mag > threshold).astype(np.uint8)
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct, iterations=3)
    labeled_array, num_features = scipy_label(closed)
    bboxes = []
    for comp_id in range(1, num_features + 1):
        component = (labeled_array == comp_id)
        area = int(np.sum(component))
        if area >= min_area:
            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            y_min, y_max = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            x_min, x_max = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
            bboxes.append({"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "area": area})
    return bboxes
```

#### metrics.py

```python
"""metrics.py — 정량적 성능 지표 및 방법 비교 모듈"""
import numpy as np


def compute_metrics(predictions, labels, category):
    """특정 카테고리의 MAE와 Accuracy를 계산."""
    keys = sorted([k for k in labels if k.startswith(category + "_") and k in predictions])
    if not keys:
        return {"mae": 0.0, "accuracy": 0.0}
    errors = [abs(predictions[k] - labels[k]) for k in keys]
    mae = float(np.mean(errors))
    accuracy = float(sum(1 for e in errors if e == 0) / len(errors))
    return {"mae": round(mae, 4), "accuracy": round(accuracy, 4)}


def find_worst_case(predictions, labels, category):
    """카테고리에서 오차가 가장 큰 이미지 이름 반환."""
    keys = [k for k in labels if k.startswith(category + "_") and k in predictions]
    if not keys:
        return ""
    return max(keys, key=lambda k: abs(predictions[k] - labels[k]))


def compare_methods(predictions_base, predictions_aug, labels):
    """기본 vs 증강 방식의 카테고리별 성능 비교."""
    categories = ["easy", "medium", "hard"]
    comparison = {}
    for cat in categories:
        base = compute_metrics(predictions_base, labels, cat)
        aug = compute_metrics(predictions_aug, labels, cat)
        comparison[cat] = {
            "base_mae": base["mae"],
            "augmented_mae": aug["mae"],
            "mae_improvement": round(base["mae"] - aug["mae"], 4),
            "base_accuracy": base["accuracy"],
            "augmented_accuracy": aug["accuracy"],
        }
    return comparison
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | conv2d.py 필수 함수 7개 정의 | 5점 | AST 자동 |
| 2 | counter.py 필수 함수 4개 정의 | 5점 | AST 자동 |
| 3 | metrics.py 필수 함수 3개 정의 | 3점 | AST 자동 |
| 4 | conv2d 연산 (valid 모드, NumPy 직접 구현) | 10점 | import 자동 |
| 5 | 그레이스케일 변환 (0.299R + 0.587G + 0.114B) | 5점 | import 자동 |
| 6 | Sobel 엣지 검출 (수평/수직 커널) | 10점 | import 자동 |
| 7 | 이미지 증강 (좌우/상하 반전, 밝기 조절) | 5점 | import 자동 |
| 8 | 이진화 + Connected Component 카운팅 | 10점 | import 자동 |
| 9 | THRESHOLD, MIN_AREA 변수 명시적 정의 | 2점 | AST 자동 |
| 10 | 앙상블 카운팅 (중앙값) | 5점 | import 자동 |
| 11 | 바운딩 박스 추출 | 5점 | import 자동 |
| 12 | MAE/Accuracy 계산 | 5점 | import 자동 |
| 13 | 최악 케이스 탐색 | 5점 | import 자동 |
| 14 | 기본 vs 증강 방법 비교 | 5점 | import 자동 |
| 15 | result_q4.json 유효 이미지만 포함 | 5점 | JSON 자동 |
| 16 | 증강 예측 정수·양수 검증 | 5점 | JSON 자동 |
| 17 | 기본 vs 증강 비교 결과 필수 키 포함 | 5점 | JSON 자동 |
| 18 | filter2D 사용 금지 검증 | 5점 | AST 자동 |

- Pass 기준: 총 100점 중 100점 (18개 전체 정답)
- AI 트랩: conv2d에서 cv2.filter2D 사용 (금지), 그레이스케일 변환 계수 오류, Sobel 커널 방향 혼동, 이진화 임계값 부적절, Connected Component에서 최소 면적 필터 누락, 밝기 조절 후 클리핑 누락

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `labels.json` | JSON | 이미지별 정답 박스 개수 {"easy_01": 3, ...} |
| `images/*.png` | PNG | 640x480 RGB 이미지 (easy/medium/hard 각 5장) |
| 그레이스케일 | `np.ndarray` | (H, W) float 배열 |
| 엣지 크기 | `np.ndarray` | sqrt(Gx^2 + Gy^2) |
| 바운딩 박스 | `list[dict]` | x_min, y_min, x_max, y_max, area |
| 메트릭 | `dict` | MAE, Accuracy |

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 2D 컨볼루션 직접 구현 | test_conv2d_identity, test_conv2d_sobel |
| 그레이스케일 변환 | test_grayscale |
| 이미지 증강 기법 | test_flip_operations, test_brightness_and_normalize |
| 객체 카운팅 (Connected Component) | test_count_boxes_augmented |
| 앙상블 기법 (중앙값 투표) | test_ensemble_count |
| 바운딩 박스 추출 | test_bounding_boxes_format |
| 정량적 성능 평가 | test_compute_metrics, test_find_worst_case, test_compare_methods |
| 파이프라인 통합 | test_valid_images_only, test_augmented_predictions, test_method_comparison |
| 코드 구조 검증 | test_conv2d_functions, test_counter_functions, test_metrics_functions, test_no_filter2d |
