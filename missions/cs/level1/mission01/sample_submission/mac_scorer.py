# mac_scorer.py — MAC 연산 기반 패턴 매칭 (정답)

import json


def load_data(filepath):
    """JSON 파일을 읽어 딕셔너리로 반환한다."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def mac(a, b):
    """두 개의 2D 리스트에 대해 MAC(Multiply-Accumulate) 연산을 수행한다."""
    total = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            total += a[i][j] * b[i][j]
    return total


def normalize_labels(labels):
    """labels 딕셔너리의 키를 모두 소문자로 변환한 새 딕셔너리를 반환한다."""
    result = {}
    for key, value in labels.items():
        result[key.lower()] = value
    return result


def is_close(a, b, epsilon=1e-6):
    """두 수의 차이가 epsilon 미만이면 True, 아니면 False를 반환한다."""
    return abs(a - b) < epsilon


def find_best_match(pattern, filters):
    """pattern과 가장 높은 MAC 점수를 가진 필터 이름을 반환한다."""
    best_name = None
    best_score = -1
    for name, filt in filters.items():
        score = mac(pattern, filt)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def main(data_path):
    """데이터를 로드하고 전체 매칭 파이프라인을 실행한다."""
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
