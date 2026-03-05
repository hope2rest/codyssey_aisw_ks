## 문항 1: MAC 연산 기반 패턴 매칭

### 문제

`data/data.json`에 저장된 3×3 크기의 패턴과 필터를 읽어, 각 패턴과 가장 유사한 필터를 MAC 연산으로 찾는 프로그램을 작성하세요.

#### data.json 구조

| 키 | 타입 | 내용 |
|----|------|------|
| `patterns` | `dict[str, list[list[int]]]` | 매칭 대상 이미지 3개 (`img_01`, `img_02`, `img_03`), 각 3×3 정수 배열 |
| `filters` | `dict[str, list[list[int]]]` | 비교 기준 필터 3개 (`cross`, `block`, `line`), 각 3×3 정수 배열 |
| `labels` | `dict[str, str]` | 패턴별 정답 라벨 |

---

### 요구사항

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

6. **`main(data_path: str) → dict`를 구현하세요.**
   - 위 함수들을 조합하여 전체 파이프라인을 실행합니다.

---

### 제약 사항
- `json` 외 외부 라이브러리는 사용할 수 없습니다.
- Python 기본 문법(반복문, 조건문, 딕셔너리)만으로 구현하세요.

---

### 출력 형식

`main(data_path)` 함수는 다음 구조의 딕셔너리를 반환합니다.

```python
{
    "scores": {
        "img_01": {"cross": 정수, "block": 정수, "line": 정수},
        "img_02": {"cross": 정수, "block": 정수, "line": 정수},
        "img_03": {"cross": 정수, "block": 정수, "line": 정수}
    },
    "best_matches": {
        "img_01": "필터명",
        "img_02": "필터명",
        "img_03": "필터명"
    }
}
```

---

### 제출물 구조

`mac_scorer.py` 파일을 제출하세요.

```
mac_scorer.py
├── load_data()
├── mac()
├── normalize_labels()
├── is_close()
├── find_best_match()
└── main()
```
