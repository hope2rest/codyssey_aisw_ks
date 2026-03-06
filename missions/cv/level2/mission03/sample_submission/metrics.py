"""metrics.py — 성능 지표, 방법 비교, 검출 일지 관리 모듈"""

import numpy as np


# ─── Part C: 정량적 성능 분석 ───


def compute_metrics(predictions, labels, category):
    """특정 카테고리의 MAE와 Accuracy를 계산."""
    keys = sorted([
        k for k in labels
        if k.startswith(category + "_") and k in predictions
    ])
    if not keys:
        return {"mae": 0.0, "accuracy": 0.0}

    errors = [abs(predictions[k] - labels[k]) for k in keys]
    mae = float(np.mean(errors))
    accuracy = float(sum(1 for e in errors if e == 0) / len(errors))
    return {"mae": round(mae, 4), "accuracy": round(accuracy, 4)}


def find_worst_case(predictions, labels, category):
    """카테고리에서 오차가 가장 큰 이미지 이름 반환."""
    keys = [
        k for k in labels
        if k.startswith(category + "_") and k in predictions
    ]
    if not keys:
        return ""
    return max(keys, key=lambda k: abs(predictions[k] - labels[k]))


def get_failure_reasons():
    """규칙 기반 방식의 기술적 실패 원인 3가지 이상."""
    return [
        "박스들이 밀집하거나 서로 겹쳐 있을 경우 Sobel 엣지가 연결되어 여러 박스가 하나의 연결 컴포넌트로 병합되므로, 규칙 기반 카운팅은 실제 개수를 심각하게 과소 추정한다.",
        "적재(Stacked) 형태나 불규칙한 다각형 형태에서는 단일 고정 임계값과 2D 엣지만으로 박스 경계를 올바르게 분리할 수 없으며, 깊이 정보 없이는 앞뒤 박스를 구분하기 불가능하다.",
        "크기 편차가 매우 큰 환경에서는 하나의 고정 min_area 값으로 소형 박스(노이즈와 유사)와 대형 박스를 동시에 처리할 수 없어 소형 박스가 노이즈로 오인되어 필터링된다.",
        "조명 불균일, 그림자, 박스 표면 질감에 의해 박스 내부에도 강한 엣지가 생성되어 단일 박스가 여러 컴포넌트로 분리되거나, 배경 텍스처가 박스로 오인식되는 위양성이 발생한다."
    ]


def get_why_learning_based():
    """학습 기반 접근법이 필요한 이유 (200자 이내 한국어)."""
    return (
        "규칙 기반 방법은 고정 임계값과 단순 형태 분석에 의존하므로 조명 변화, "
        "박스 겹침, 크기 편차, 적재 구조 등 복잡한 실세계 조건에 일반화할 수 없다. "
        "CNN 등 학습 기반 모델은 대규모 데이터로부터 특징을 자동 학습하여 "
        "다양한 환경에서도 강인한 객체 탐지가 가능하다."
    )


# ─── Part D: 방법 비교 분석 ───


def compare_methods(predictions_base, predictions_aug, labels):
    """기본 vs 증강 방식의 카테고리별 성능 비교."""
    categories = ["easy", "medium", "hard"]
    comparison = {}
    for cat in categories:
        base = compute_metrics(predictions_base, labels, cat)
        aug = compute_metrics(predictions_aug, labels, cat)
        comparison[cat] = {
            "base_mae": base["mae"],
            "augmented_mae": aug["mae"],
            "mae_improvement": round(base["mae"] - aug["mae"], 4),
            "base_accuracy": base["accuracy"],
            "augmented_accuracy": aug["accuracy"],
        }
    return comparison


# ─── Part E: 객체 검출 일지 관리 ───


def create_detection_log(predictions, labels, date_str):
    """일자별 검출 로그 생성."""
    log = {
        "date": date_str,
        "total_images": len(predictions),
        "results": [],
    }
    for name in sorted(predictions.keys()):
        pred = predictions[name]
        actual = labels.get(name, 0)
        log["results"].append({
            "image": name,
            "predicted": pred,
            "actual": actual,
            "error": abs(pred - actual),
            "correct": pred == actual,
        })
    correct_count = sum(1 for r in log["results"] if r["correct"])
    log["daily_accuracy"] = round(
        correct_count / len(log["results"]), 4
    ) if log["results"] else 0.0
    return log


def generate_weekly_report(daily_logs):
    """주간 단위 요약 보고서 생성."""
    if not daily_logs:
        return {}

    dates = [log["date"] for log in daily_logs]
    total_images = sum(log["total_images"] for log in daily_logs)
    all_accuracies = [log["daily_accuracy"] for log in daily_logs]
    avg_accuracy = round(sum(all_accuracies) / len(all_accuracies), 4)

    best_day = dates[all_accuracies.index(max(all_accuracies))]
    worst_day = dates[all_accuracies.index(min(all_accuracies))]

    return {
        "week_start": min(dates),
        "week_end": max(dates),
        "total_images_processed": total_images,
        "average_daily_accuracy": avg_accuracy,
        "best_day": best_day,
        "worst_day": worst_day,
        "daily_accuracies": dict(zip(dates, all_accuracies)),
    }
