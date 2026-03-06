"""metrics.py — 정량적 성능 지표 및 방법 비교 모듈"""

import numpy as np


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
