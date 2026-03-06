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
