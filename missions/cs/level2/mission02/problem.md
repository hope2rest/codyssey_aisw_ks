## 문항 2: 2D 컨볼루션 기반 특징 추출

### 문제

5×5 이미지와 3×3 커널을 읽어, 2D 컨볼루션으로 특징맵을 추출하고 가장 강한 반응을 보이는 커널을 찾는 프로그램을 구현하세요.
- 5×5 이미지와 3×3 커널은 `data/data.json`에 저장되어 있습니다.
- 2D 컨볼루션은 이미지 위에서 커널(작은 행렬)을 슬라이딩하며 MAC 연산을 반복 적용하는 방식으로, 엣지 검출·블러·샤프닝 등 이미지의 특징을 추출하는 데 활용됩니다.

### data.json 구조

| 키 | 타입 | 내용 |
|----|------|------|
| `images` | `dict[str, list[list[int]]]` | 분석 대상 이미지 3개 (`img_01`~`img_03`), 각 5×5 정수 배열 |
| `kernels` | `dict[str, list[list[int]]]` | 특징 추출 커널 3개 (`edge_h`, `edge_v`, `sharpen`), 각 3×3 정수 배열 |

### 구현 요구사항

#### 1. `load_data(filepath: str) → dict`

- JSON 파일을 읽어 딕셔너리로 반환합니다.

#### 2. `pad_matrix(matrix: list[list], pad_size: int) → list[list]`

- 2D 행렬의 상하좌우에 `pad_size`만큼 0을 추가합니다.

**예시:**
```python
pad_matrix([[1, 2], [3, 4]], 1)
# [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
```

#### 3. `conv2d(image: list[list], kernel: list[list]) → list[list]`

- 패딩 없이 stride=1로 2D 컨볼루션을 수행합니다.
- 출력 크기: `(H - kH + 1) × (W - kW + 1)`
- 각 위치에서 이미지 패치와 커널의 같은 위치 값을 곱한 뒤 전부 더합니다.

**예시:**
```python
image = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
kernel = [[1,0,0], [0,1,0], [0,0,1]]
conv2d(image, kernel)  # [[18, 21], [30, 33]]
```

#### 4. `relu(matrix: list[list]) → list[list]`

- 행렬의 모든 음수 값을 0으로 변환한 새 행렬을 반환합니다.

#### 5. `flatten(matrix: list[list]) → list`

- 2D 행렬을 행 우선(row-major) 순서로 1D 리스트로 변환합니다.

#### 6. `compute_stats(matrix: list[list]) → dict`

- 2D 행렬의 통계를 계산합니다.
- 반환 형식: `{"min": 최솟값, "max": 최댓값, "mean": 평균}`

#### 7. `extract_features(image: list[list], kernels: dict) → dict`

- 각 커널로 컨볼루션 후 ReLU를 적용한 특징맵 딕셔너리를 반환합니다.
- 반환 형식: `{"커널명": relu(conv2d(image, kernel)), ...}`

#### 8. `find_strongest_feature(image: list[list], kernels: dict) → str`

- ReLU 적용 후 특징맵의 전체 합이 가장 큰 커널 이름을 반환합니다.

#### 9. `main(data_path: str) → dict`

- 데이터 로드 → 특징 추출 → 합계 계산 → 최강 특징 판별을 순서대로 실행합니다.
- 반환 형식은 다음과 같습니다.

```python
{
    "feature_maps": {
        "img_01": {
            "edge_h": [[정수, ...], ...],
            "edge_v": [[정수, ...], ...],
            "sharpen": [[정수, ...], ...]
        },
        ...
    },
    "feature_sums": {
        "img_01": {"edge_h": 정수, "edge_v": 정수, "sharpen": 정수},
        ...
    },
    "strongest_features": {
        "img_01": "커널명",
        ...
    }
}
```

### 제약 사항

- **외부 라이브러리 사용 금지** - `json` 외 외부 라이브러리는 사용할 수 없습니다.
- Python 기본 문법(반복문, 조건문, 리스트, 딕셔너리)만으로 구현하세요.

### 제출 방식

- `conv2d_analyzer.py` 파일 1개를 제출합니다.
- `template/conv2d_analyzer.py`의 `# TODO` 부분을 채우세요.
