"""main.py — 전체 파이프라인 실행 및 결과 JSON 생성"""

import json
import os

from counter import count_boxes
from metrics import compute_metrics, find_worst_case, get_failure_reasons, get_why_learning_based


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DATA_DIR, "data", "images")
LABELS_FILE = os.path.join(DATA_DIR, "data", "labels.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "result_q3.json")


def main():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    valid_names = sorted([
        k for k in labels
        if not k.startswith("test")
        and os.path.exists(os.path.join(IMAGES_DIR, k + ".png"))
    ])

    predictions = {}
    for name in valid_names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        pred = count_boxes(img_path)
        predictions[name] = pred

    categories = ["easy", "medium", "hard"]
    metrics = {}
    for cat in categories:
        metrics[cat] = compute_metrics(predictions, labels, cat)

    worst_case = find_worst_case(predictions, labels, "hard")

    result = {
        "predictions": predictions,
        "metrics": metrics,
        "worst_case_image": worst_case if worst_case else "",
        "failure_reasons": get_failure_reasons(),
        "why_learning_based": get_why_learning_based(),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    main()
