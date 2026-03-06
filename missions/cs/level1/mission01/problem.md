## 문항 1: MAC 연산 기반 패턴 매칭

### 문제

`data/data.json`에 저장된 3×3 크기의 패턴과 필터를 읽어, 각 패턴과 가장 유사한 필터를 MAC 연산으로 찾고 성능을 분석하는 프로그램을 구현하세요.

### data.json 구조

| 키 | 타입 | 내용 |
|----|------|------|
| `patterns` | `dict[str, list[list[int\|float]]]` | 매칭 대상 이미지 4개 (`img_01`~`img_04`), 각 3×3 정수 또는 실수 배열 |
| `filters` | `dict[str, list[list[int]]]` | 비교 기준 필터 3개 (`cross`, `block`, `line`), 각 3×3 정수 배열 |
| `labels` | `dict[str, str]` | 패턴별 정답 라벨 |

### 구현 요구사항

#### 1. `load_data(filepath: str) → dict`

- JSON 파일을 읽어 딕셔너리로 반환합니다.

#### 2. `mac(a: list[list], b: list[list]) → int | float`

- 두 개의 2D 리스트에 대해 MAC 연산을 수행합니다.
- 같은 위치의 값을 곱한 뒤 전부 더합니다.

**예시:**
```python
a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
b = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
mac(a, b)  # 5
```

#### 3. `normalize_labels(labels: dict) → dict`

- 딕셔너리의 키를 모두 소문자로 변환한 새 딕셔너리를 반환합니다.
- 값은 변경하지 않습니다.

#### 4. `is_close(a: float, b: float, epsilon: float = 1e-6) → bool`

- 두 수의 차이가 epsilon 미만이면 `True`를 반환합니다.

#### 5. `find_best_match(pattern: list[list], filters: dict) → str`

- 패턴과 각 필터의 MAC 점수를 계산하여, 가장 높은 점수를 받은 필터 이름을 반환합니다.

#### 6. `measure_mac_time(n: int, repeat: int = 5) → float`

- NxN 크기의 패턴 두 개를 생성하여 MAC 연산의 평균 실행 시간(초)을 측정합니다.
- `time.time()`을 사용하여 `repeat`회 반복 후 평균을 반환합니다.

#### 7. `analyze_complexity(sizes: list, times: list) → dict`

- 크기별 측정 시간 데이터를 받아 시간 복잡도를 분석합니다.
- 반환 형식은 다음과 같습니다.

```python
{
    "size_pairs": [[1, 2], [2, 4], ...],
    "time_ratios": [실수, ...],
    "estimated_order": "O(N^2)"
}
```

#### 8. `diagnose_failure(scores: dict, best_match: str, expected_label: str, filter_names: list) → dict`

- 매칭 결과의 실패 원인을 다음 3가지로 분류합니다.
  - `"data_schema"` - 키 누락, 데이터 불일치 등 데이터/스키마 문제
  - `"numerical"` - 점수 차이가 epsilon 미만인 부동소수점 비교 문제
  - `"logic"` - 점수 차이가 명확하지만 잘못된 필터가 선택된 로직 오류
- 예측이 정답과 일치하면 `"none"`을 반환합니다.
- 반환 형식: `{"category": 문자열, "reason": 문자열}`

#### 9. `main(data_path: str) → dict`

- 데이터 로드 → 점수 계산 → 최적 매칭 → 성능 벤치마크 → 실패 진단을 순서대로 실행합니다.
- 벤치마크 크기: `sizes=[1, 2, 4, 8, 16, 32]`
- 반환 형식은 다음과 같습니다.

```python
{
    "scores": {
        "img_01": {"cross": 정수, "block": 정수, "line": 정수},
        ...
    },
    "best_matches": {
        "img_01": "필터명",
        ...
    },
    "labels": {
        "img_01": "라벨명",
        ...
    },
    "benchmarks": {
        "sizes": [1, 2, 4, 8, 16, 32],
        "times": [실수, ...],
        "complexity_analysis": {
            "size_pairs": [[1, 2], [2, 4], ...],
            "time_ratios": [실수, ...],
            "estimated_order": "O(N^2)"
        }
    },
    "diagnosis": {
        "img_01": {"category": "분류", "reason": "설명"},
        ...
    }
}
```

### 제약 사항

- **외부 라이브러리 사용 금지** - `json`과 `time` 외 외부 라이브러리는 사용할 수 없습니다.
- Python 기본 문법(반복문, 조건문, 딕셔너리)만으로 구현하세요.

### 제출 방식

- `mac_scorer.py` 파일 1개를 제출합니다.
- `template/mac_scorer.py`의 `# TODO` 부분을 채우세요.
