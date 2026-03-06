"""conv2d_analyzer.py — 2D 컨볼루션 기반 특징 추출 모듈"""

import json


def load_data(filepath):
    """JSON 파일을 읽어 딕셔너리로 반환한다."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def pad_matrix(matrix, pad_size):
    """2D 행렬에 제로 패딩을 적용한다."""
    rows = len(matrix)
    cols = len(matrix[0])
    new_rows = rows + 2 * pad_size
    new_cols = cols + 2 * pad_size
    result = [[0] * new_cols for _ in range(new_rows)]
    for i in range(rows):
        for j in range(cols):
            result[i + pad_size][j + pad_size] = matrix[i][j]
    return result


def conv2d(image, kernel):
    """2D 컨볼루션을 수행한다 (패딩 없이, stride=1)."""
    img_h = len(image)
    img_w = len(image[0])
    ker_h = len(kernel)
    ker_w = len(kernel[0])
    out_h = img_h - ker_h + 1
    out_w = img_w - ker_w + 1
    output = []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            total = 0
            for ki in range(ker_h):
                for kj in range(ker_w):
                    total += image[i + ki][j + kj] * kernel[ki][kj]
            row.append(total)
        output.append(row)
    return output


def relu(matrix):
    """ReLU 활성화 함수를 적용한다 (음수를 0으로 변환)."""
    result = []
    for row in matrix:
        new_row = []
        for val in row:
            if val > 0:
                new_row.append(val)
            else:
                new_row.append(0)
        result.append(new_row)
    return result


def flatten(matrix):
    """2D 행렬을 1D 리스트로 변환한다 (행 우선)."""
    result = []
    for row in matrix:
        for val in row:
            result.append(val)
    return result


def compute_stats(matrix):
    """2D 행렬의 통계(min, max, mean)를 계산한다."""
    flat = flatten(matrix)
    total = 0
    min_val = flat[0]
    max_val = flat[0]
    for val in flat:
        total += val
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    return {
        "min": min_val,
        "max": max_val,
        "mean": total / len(flat),
    }


def extract_features(image, kernels):
    """각 커널로 컨볼루션 후 ReLU를 적용한 특징맵 딕셔너리를 반환한다."""
    features = {}
    for name, kernel in kernels.items():
        conv_result = conv2d(image, kernel)
        features[name] = relu(conv_result)
    return features


def find_strongest_feature(image, kernels):
    """ReLU 적용 후 합이 가장 큰 커널 이름을 반환한다."""
    features = extract_features(image, kernels)
    best_name = None
    best_sum = -1
    for name, fmap in features.items():
        s = 0
        for row in fmap:
            for val in row:
                s += val
        if s > best_sum:
            best_sum = s
            best_name = name
    return best_name


def main(data_path):
    """전체 특징 추출 파이프라인을 실행한다."""
    data = load_data(data_path)
    images = data["images"]
    kernels = data["kernels"]

    feature_maps = {}
    feature_sums = {}
    strongest_features = {}

    for img_name, image in images.items():
        features = extract_features(image, kernels)
        feature_maps[img_name] = features

        sums = {}
        for kern_name, fmap in features.items():
            s = 0
            for row in fmap:
                for val in row:
                    s += val
            sums[kern_name] = s
        feature_sums[img_name] = sums

        strongest_features[img_name] = find_strongest_feature(image, kernels)

    return {
        "feature_maps": feature_maps,
        "feature_sums": feature_sums,
        "strongest_features": strongest_features,
    }
