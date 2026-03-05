"""counter.py — 박스 카운팅 파이프라인 모듈"""

import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_closing

from conv2d import to_grayscale, compute_edge_magnitude


THRESHOLD = 30
MIN_AREA = 100


def count_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    """이미지에서 박스 개수를 카운팅하는 파이프라인."""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)

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
