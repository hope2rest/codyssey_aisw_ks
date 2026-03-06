"""counter.py — 박스 카운팅 파이프라인 및 증강 앙상블 모듈"""

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_closing

from conv2d import (to_grayscale, compute_edge_magnitude,
                    flip_horizontal, flip_vertical, adjust_brightness)


THRESHOLD = 30
MIN_AREA = 100


# ─── Part B: 기본 박스 카운팅 ───


def _count_from_rgb(rgb, threshold=THRESHOLD, min_area=MIN_AREA):
    """RGB 배열에서 박스 개수 카운팅 (내부 헬퍼)."""
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


# ─── Part D: 증강 앙상블 카운팅 ───


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

    augmented = [
        rgb,
        flip_horizontal(rgb),
        flip_vertical(rgb),
        adjust_brightness(rgb, 0.8),
        adjust_brightness(rgb, 1.2),
    ]

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
            bboxes.append({
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "area": area,
            })

    return bboxes
