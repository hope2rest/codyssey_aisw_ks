## 문항 2: MAC 연산 심화 분석

### 문제

문항 1에서 구현한 MAC 연산을 기반으로, 성능 벤치마크와 시간 복잡도 분석, 실패 원인 진단, 핵심 개념 설명 기능을 추가한 심화 분석 프로그램을 작성하세요.

#### data.json 구조

| 키 | 타입 | 내용 |
|----|------|------|
| `patterns` | `dict[str, list[list[int\|float]]]` | 매칭 대상 이미지 4개 (`img_01`~`img_04`), 각 3×3 정수 또는 실수 배열 |
| `filters` | `dict[str, list[list[int]]]` | 비교 기준 필터 3개 (`cross`, `block`, `line`), 각 3×3 정수 배열 |
| `labels` | `dict[str, str]` | 패턴별 정답 라벨 |

---

### 요구사항

#### Part A: MAC 연산 기본

1. **`load_data(filepath: str) → dict`를 구현하세요.**
   - JSON 파일을 읽어 딕셔너리로 반환합니다.

2. **`mac(a: list[list], b: list[list]) → int | float`를 구현하세요.**
   - 두 개의 2D 리스트에 대해 MAC 연산을 수행합니다.
   - 같은 위치의 값을 곱한 뒤 전부 더합니다.

3. **`normalize_labels(labels: dict) → dict`를 구현하세요.**
   - 딕셔너리의 키를 모두 소문자로 변환한 새 딕셔너리를 반환합니다.
   - 값은 변경하지 않습니다.

4. **`is_close(a: float, b: float, epsilon: float = 1e-6) → bool`를 구현하세요.**
   - 두 수의 차이가 epsilon 미만이면 `True`를 반환합니다.

5. **`find_best_match(pattern: list[list], filters: dict) → str`를 구현하세요.**
   - 패턴과 각 필터의 MAC 점수를 계산하여, 가장 높은 점수를 받은 필터 이름을 반환합니다.

#### Part B: 성능 벤치마크 및 시간 복잡도 분석

6. **`measure_mac_time(n: int, repeat: int = 5) → float`를 구현하세요.**
   - NxN 크기의 패턴 두 개를 생성하여 MAC 연산의 평균 실행 시간(초)을 측정합니다.
   - `time.time()`을 사용하여 `repeat`회 반복 후 평균을 반환합니다.

7. **`analyze_complexity(sizes: list, times: list) → dict`를 구현하세요.**
   - 크기별 측정 시간 데이터를 받아 시간 복잡도를 분석합니다.
   - 반환: `size_pairs`(인접 크기 쌍), `time_ratios`(시간 비율), `estimated_order`(`"O(N^2)"`)

#### Part C: 실패 원인 진단

8. **`diagnose_failure(scores: dict, best_match: str, expected_label: str, filter_names: list) → dict`를 구현하세요.**
   - 매칭 결과의 실패 원인을 다음 3가지로 분류합니다.
     - `"data_schema"`: 키 누락, 데이터 불일치 등 데이터/스키마 문제
     - `"numerical"`: 점수 차이가 epsilon 미만인 부동소수점 비교 문제
     - `"logic"`: 점수 차이가 명확하지만 잘못된 필터가 선택된 로직 오류
   - 예측이 정답과 일치하면 `"none"`을 반환합니다.
   - 반환: `{"category": 문자열, "reason": 문자열}`

#### Part D: 개념 설명

9. **`get_mac_explanation() → str`를 구현하세요.**
   - MAC 연산의 정의와 AI에서의 중요성을 한국어로 50자 이상 서술합니다.

10. **`get_normalization_reason() → str`를 구현하세요.**
    - 라벨 키를 표준화(정규화)하는 이유를 한국어로 30자 이상 서술합니다.

11. **`get_epsilon_reason() → str`를 구현하세요.**
    - 허용오차(epsilon) 기반 비교가 필요한 이유를 한국어로 30자 이상 서술합니다.

#### 전체 파이프라인

12. **`main(data_path: str) → dict`를 구현하세요.**
    - 데이터 로드 → 점수 계산 → 최적 매칭
    - 성능 벤치마크 (sizes=[1, 2, 4, 8, 16, 32])
    - 각 패턴별 실패 진단
    - 개념 설명 포함
    - 전체 결과 딕셔너리를 반환합니다.

---

### 제약 사항
- `json`과 `time` 외 외부 라이브러리는 사용할 수 없습니다.
- Python 기본 문법(반복문, 조건문, 딕셔너리)만으로 구현하세요.

---

### 출력 형식

`main(data_path)` 함수는 다음 구조의 딕셔너리를 반환합니다.

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
    },
    "explanations": {
        "mac_explanation": "50자 이상 한국어 서술",
        "normalization_reason": "30자 이상 한국어 서술",
        "epsilon_reason": "30자 이상 한국어 서술"
    }
}
```

---

### 제출물 구조

`mac_analyzer.py` 파일을 제출하세요.

```
mac_analyzer.py
├── load_data()
├── mac()
├── normalize_labels()
├── is_close()
├── find_best_match()
├── measure_mac_time()
├── analyze_complexity()
├── diagnose_failure()
├── get_mac_explanation()
├── get_normalization_reason()
├── get_epsilon_reason()
└── main()
```
