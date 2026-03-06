## 문항 2 정답지 — 2D 컨볼루션 기반 특징 추출

### 정답 코드

```python
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
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | functions_exist: 필수 함수 9개 정의 | 5점 | AST 자동 |
| 2 | no_external_lib: json 외 외부 라이브러리 미사용 | 5점 | AST 자동 |
| 3 | pad_matrix: 제로 패딩 적용 | 10점 | import 자동 |
| 4 | conv2d_basic: 기본 컨볼루션 연산 | 10점 | import 자동 |
| 5 | conv2d_output_size: 컨볼루션 출력 크기 | 10점 | import 자동 |
| 6 | relu: ReLU 활성화 | 10점 | import 자동 |
| 7 | flatten: 2D -> 1D 변환 | 5점 | import 자동 |
| 8 | compute_stats: min/max/mean 계산 | 10점 | import 자동 |
| 9 | extract_features: 다중 커널 특징 추출 | 10점 | import 자동 |
| 10 | find_strongest_feature: 최강 특징 커널 탐색 | 10점 | import 자동 |
| 11 | main_feature_sums: 전체 파이프라인 특징맵 합계 | 10점 | import 자동 |
| 12 | main_strongest: 전체 파이프라인 최강 특징 | 5점 | import 자동 |

- Pass 기준: 총 100점 중 100점 (12개 전체 정답)
- AI 트랩: conv2d 자동 패딩 적용, relu in-place 수정, compute_stats 정수 나눗셈, 출력 크기 계산 오류

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `images` | `dict[str, list[list[int]]]` | 키: `img_01`~`img_03`, 값: 5x5 정수 배열 |
| `kernels` | `dict[str, list[list[int]]]` | 키: `edge_h`, `edge_v`, `sharpen`, 값: 3x3 정수 배열 |
| conv2d 결과 | `list[list[int]]` | (H-kH+1)x(W-kW+1) 크기 배열 |
| relu 결과 | `list[list[int]]` | 음수가 0으로 치환된 배열 |
| 통계 | `dict` | `min`, `max`, `mean` |
| 특징맵 | `dict[str, list[list[int]]]` | 커널별 relu(conv2d) 결과 |

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
