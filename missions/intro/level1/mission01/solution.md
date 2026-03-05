## 문항 1 정답지 — MAC 연산 기반 패턴 매칭

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `patterns` | `dict[str, list[list[int]]]` | 키: `img_01`~`img_03`, 값: 3×3 정수 배열 |
| `filters` | `dict[str, list[list[int]]]` | 키: `cross`, `block`, `line`, 값: 3×3 정수 배열 |
| `labels` | `dict[str, str]` | 키: 대소문자 불규칙 (`IMG_01`, `Img_02`, `img_03`), 값: 라벨 문자열 |
| MAC 결과 | `int` 또는 `float` | 두 배열의 요소별 곱의 합 |
| `scores` | `dict[str, dict[str, int]]` | 패턴별 필터 MAC 점수 |
| `best_matches` | `dict[str, str]` | 패턴별 최적 필터 이름 |

### 정답 코드

```python
import json


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def mac(a, b):
    total = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            total += a[i][j] * b[i][j]
    return total


def normalize_labels(labels):
    result = {}
    for key, value in labels.items():
        result[key.lower()] = value
    return result


def is_close(a, b, epsilon=1e-6):
    return abs(a - b) < epsilon


def find_best_match(pattern, filters):
    best_name = None
    best_score = -1
    for name, filt in filters.items():
        score = mac(pattern, filt)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def main(data_path):
    data = load_data(data_path)
    patterns = data["patterns"]
    filters = data["filters"]
    labels = normalize_labels(data["labels"])

    scores = {}
    best_matches = {}

    for pat_name, pat_data in patterns.items():
        scores[pat_name] = {}
        for filt_name, filt_data in filters.items():
            scores[pat_name][filt_name] = mac(pat_data, filt_data)
        best_matches[pat_name] = find_best_match(pat_data, filters)

    return {
        "scores": scores,
        "best_matches": best_matches,
        "labels": labels,
    }
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | 필수 함수 6개 정의 | 10점 | AST 자동 |
| 2 | json 외 외부 라이브러리 미사용 | 10점 | AST 자동 |
| 3 | MAC 연산 (동일 패턴) | 10점 | import 자동 |
| 4 | MAC 연산 (다른 패턴) | 10점 | import 자동 |
| 5 | MAC 연산 (부동소수점) | 10점 | import 자동 |
| 6 | 라벨 키 소문자 정규화 | 10점 | import 자동 |
| 7 | is_close True (0.1+0.2 ≈ 0.3) | 10점 | import 자동 |
| 8 | is_close False (1.0 ≠ 2.0) | 10점 | import 자동 |
| 9 | 전체 파이프라인 결과 | 10점 | import 자동 |

- Pass 기준: 총 100점 중 100점 (9개 전체 정답)
- AI 트랩: labels 키 대소문자 불규칙 (IMG_01, Img_02, img_03)

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| MAC 연산 이해 | test_mac_basic, test_mac_different |
| 패턴-필터 유사도 계산 | test_main_result (scores, best_matches) |
| data.json 키/라벨 정규화 | test_normalize_labels |
| 부동소수점 오차와 epsilon 비교 | test_is_close_true, test_is_close_false, test_mac_floats |
