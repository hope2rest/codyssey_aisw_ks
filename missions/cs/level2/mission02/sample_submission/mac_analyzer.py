"""mac_analyzer.py — MAC 연산 심화 분석 모듈"""

import json
import time


# ─── Part A: MAC 연산 기본 ───


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


# ─── Part B: 성능 벤치마크 및 시간 복잡도 분석 ───


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
            time_ratio = times[i] / times[i - 1]
            ratios.append(round(time_ratio, 2))
        else:
            ratios.append(0.0)
    return {
        "size_pairs": [[sizes[i - 1], sizes[i]] for i in range(1, len(sizes))],
        "time_ratios": ratios,
        "estimated_order": "O(N^2)",
    }


# ─── Part C: 실패 진단 ───


def diagnose_failure(scores, best_match, expected_label, filter_names):
    """실패 원인을 데이터/스키마, 로직, 수치비교 문제로 분류한다."""
    # 데이터/스키마 문제: 키 누락
    if best_match not in filter_names:
        return {
            "category": "data_schema",
            "reason": "선택된 필터가 필터 목록에 존재하지 않습니다.",
        }

    # expected_label에서 필터명 추출 (예: "cross_pattern" → "cross")
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

    # 예측이 정답과 일치하는 경우
    if best_match == expected_filter:
        return {
            "category": "none",
            "reason": "실패가 아닙니다. 예측이 정답과 일치합니다.",
        }

    # 수치 비교 문제: 점수 차이가 매우 작은 경우
    best_score = scores.get(best_match, 0)
    expected_score = scores.get(expected_filter, 0)
    if is_close(best_score, expected_score):
        return {
            "category": "numerical",
            "reason": "최고 점수와 정답 필터 점수의 차이가 매우 작아 부동소수점 비교 문제가 발생했습니다.",
        }

    # 로직 문제
    return {
        "category": "logic",
        "reason": "점수 차이가 명확하지만 잘못된 필터가 선택되어 로직 오류입니다.",
    }


# ─── Part D: 개념 설명 ───


def get_mac_explanation():
    """MAC 연산의 정의와 AI에서의 중요성을 설명한다."""
    return (
        "MAC(Multiply-Accumulate) 연산은 두 배열의 같은 위치 값을 곱한 뒤 "
        "모두 더하는 연산으로, 신경망의 컨볼루션과 완전연결 계층에서 핵심적으로 "
        "사용됩니다. AI 가속기(GPU, NPU)는 MAC 연산을 병렬로 처리하도록 "
        "설계되어 있어 딥러닝 추론과 학습의 성능을 결정하는 기본 단위입니다."
    )


def get_normalization_reason():
    """라벨 키를 표준화(정규화)하는 이유를 설명한다."""
    return (
        "데이터 소스마다 키 표기법이 다를 수 있으므로(대소문자 혼용 등), "
        "라벨 키를 소문자로 통일하면 비교 시 불일치를 방지하고 "
        "프로그램의 안정성과 일관성을 확보할 수 있습니다."
    )


def get_epsilon_reason():
    """허용오차(epsilon) 기반 비교가 필요한 이유를 설명한다."""
    return (
        "부동소수점은 이진 표현의 한계로 0.1+0.2≠0.3 같은 미세 오차가 발생합니다. "
        "== 비교 대신 허용오차(epsilon) 기반 비교를 사용해야 "
        "오차로 인한 잘못된 판정을 방지할 수 있습니다."
    )


# ─── 전체 파이프라인 ───


def main(data_path):
    """전체 분석 파이프라인을 실행한다."""
    data = load_data(data_path)
    patterns = data["patterns"]
    filters = data["filters"]
    labels = normalize_labels(data["labels"])

    # 점수 계산 및 최적 매칭
    scores = {}
    best_matches = {}
    for pat_name, pat_data in patterns.items():
        scores[pat_name] = {}
        for filt_name, filt_data in filters.items():
            scores[pat_name][filt_name] = mac(pat_data, filt_data)
        best_matches[pat_name] = find_best_match(pat_data, filters)

    # 성능 벤치마크
    sizes = [1, 2, 4, 8, 16, 32]
    times = [measure_mac_time(n, repeat=3) for n in sizes]
    complexity_analysis = analyze_complexity(sizes, times)

    # 실패 진단 (각 패턴에 대해)
    filter_names = list(filters.keys())
    diagnosis = {}
    for pat_name in patterns:
        label = labels.get(pat_name, "")
        diag = diagnose_failure(
            scores[pat_name], best_matches[pat_name], label, filter_names
        )
        diagnosis[pat_name] = diag

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
        "explanations": {
            "mac_explanation": get_mac_explanation(),
            "normalization_reason": get_normalization_reason(),
            "epsilon_reason": get_epsilon_reason(),
        },
    }
