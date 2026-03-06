## 문항 1 정답지 — MAC 연산 기반 패턴 매칭

### 정답 코드

```python
"""mac_scorer.py — MAC 연산 기반 패턴 매칭 모듈"""

import json
import time


def load_data(filepath):
    """JSON 파일을 읽어 딕셔너리로 반환한다."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def mac(a, b):
    """두 개의 2D 리스트에 대해 MAC 연산을 수행한다."""
    total = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            total += a[i][j] * b[i][j]
    return total


def normalize_labels(labels):
    """labels 딕셔너리의 키를 모두 소문자로 변환한다."""
    result = {}
    for key, value in labels.items():
        result[key.lower()] = value
    return result


def is_close(a, b, epsilon=1e-6):
    """두 수의 차이가 epsilon 미만이면 True를 반환한다."""
    return abs(a - b) < epsilon


def find_best_match(pattern, filters):
    """패턴과 가장 높은 MAC 점수를 가진 필터 이름을 반환한다."""
    best_name = None
    best_score = -1
    for name, filt in filters.items():
        score = mac(pattern, filt)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def measure_mac_time(n, repeat=5):
    """NxN 크기 패턴에 대한 MAC 연산 평균 시간을 측정한다 (초)."""
    a = [[1] * n for _ in range(n)]
    b = [[1] * n for _ in range(n)]
    total = 0
    for _ in range(repeat):
        start = time.time()
        mac(a, b)
        end = time.time()
        total += (end - start)
    return total / repeat


def analyze_complexity(sizes, times):
    """크기별 시간 데이터로 시간 복잡도를 분석한다."""
    ratios = []
    for i in range(1, len(sizes)):
        if times[i - 1] > 0:
            ratios.append(round(times[i] / times[i - 1], 2))
        else:
            ratios.append(0.0)
    return {
        "size_pairs": [[sizes[i - 1], sizes[i]] for i in range(1, len(sizes))],
        "time_ratios": ratios,
        "estimated_order": "O(N^2)",
    }


def diagnose_failure(scores, best_match, expected_label, filter_names):
    """실패 원인을 데이터/스키마, 수치비교, 로직 문제로 분류한다."""
    if best_match not in filter_names:
        return {
            "category": "data_schema",
            "reason": "선택된 필터가 필터 목록에 존재하지 않습니다.",
        }

    expected_filter = None
    for fname in filter_names:
        if expected_label.startswith(fname):
            expected_filter = fname
            break

    if expected_filter is None:
        return {
            "category": "data_schema",
            "reason": "라벨에서 대응하는 필터를 찾을 수 없습니다.",
        }

    if best_match == expected_filter:
        return {
            "category": "none",
            "reason": "예측이 정답과 일치합니다.",
        }

    best_score = scores.get(best_match, 0)
    expected_score = scores.get(expected_filter, 0)
    if is_close(best_score, expected_score):
        return {
            "category": "numerical",
            "reason": "최고 점수와 정답 필터 점수의 차이가 매우 작아 부동소수점 비교 문제가 발생했습니다.",
        }

    return {
        "category": "logic",
        "reason": "점수 차이가 명확하지만 잘못된 필터가 선택되어 로직 오류입니다.",
    }


def main(data_path):
    """전체 분석 파이프라인을 실행한다."""
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

    sizes = [1, 2, 4, 8, 16, 32]
    times = [measure_mac_time(n, repeat=3) for n in sizes]
    complexity_analysis = analyze_complexity(sizes, times)

    filter_names = list(filters.keys())
    diagnosis = {}
    for pat_name in patterns:
        label = labels.get(pat_name, "")
        diagnosis[pat_name] = diagnose_failure(
            scores[pat_name], best_matches[pat_name], label, filter_names
        )

    return {
        "scores": scores,
        "best_matches": best_matches,
        "labels": labels,
        "benchmarks": {
            "sizes": sizes,
            "times": times,
            "complexity_analysis": complexity_analysis,
        },
        "diagnosis": diagnosis,
    }
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | functions_exist: 필수 함수 9개 정의 | 5점 | AST 자동 |
| 2 | no_external_lib: json, time 외 외부 라이브러리 미사용 | 5점 | AST 자동 |
| 3 | mac_basic: 정수 MAC 연산 | 10점 | import 자동 |
| 4 | mac_floats: 부동소수점 MAC 연산 | 10점 | import 자동 |
| 5 | find_best_match: 최적 필터 매칭 | 10점 | import 자동 |
| 6 | normalize_labels: 라벨 키 소문자 정규화 | 10점 | import 자동 |
| 7 | is_close: epsilon 기반 부동소수점 비교 | 10점 | import 자동 |
| 8 | measure_mac_time: MAC 연산 시간 측정 | 10점 | import 자동 |
| 9 | analyze_complexity: 시간 복잡도 분석 구조 | 10점 | import 자동 |
| 10 | diagnose_failure: 실패 원인 3가지 분류 | 10점 | import 자동 |
| 11 | main_result: 전체 파이프라인 결과 | 10점 | import 자동 |

- Pass 기준: 총 100점 중 100점 (11개 전체 정답)
- AI 트랩: labels 키 대소문자 불규칙, img_04 부동소수점 패턴, diagnose_failure 판별 순서

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `patterns` | `dict[str, list[list[int\|float]]]` | 키: `img_01`~`img_04`, 값: 3x3 정수 또는 실수 배열 |
| `filters` | `dict[str, list[list[int]]]` | 키: `cross`, `block`, `line`, 값: 3x3 정수 배열 |
| `labels` | `dict[str, str]` | 키: 대소문자 불규칙, 값: 라벨 문자열 |
| MAC 결과 | `int` 또는 `float` | 두 배열의 요소별 곱의 합 |
| 벤치마크 시간 | `float` | 초 단위 평균 실행 시간 |
| 진단 결과 | `dict` | `category`(data_schema/numerical/logic/none)와 `reason` |

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| MAC 연산 이해 | test_mac_basic, test_mac_floats |
| 유사도 계산 원리 | test_find_best_match, test_main_result |
| 라벨 표준화 | test_normalize_labels |
| 부동소수점 오차와 epsilon 비교 | test_is_close |
| 시간 복잡도 O(N^2) 분석 | test_measure_mac_time, test_analyze_complexity |
| 실패 케이스 진단 | test_diagnose_failure |
