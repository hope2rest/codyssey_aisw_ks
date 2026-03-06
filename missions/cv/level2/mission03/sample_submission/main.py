"""main.py — 전체 파이프라인 실행 및 결과 JSON 생성"""

import json
import os
from datetime import datetime, timedelta

from counter import count_boxes, count_boxes_augmented, extract_bounding_boxes
from metrics import (compute_metrics, find_worst_case,
                     get_failure_reasons, get_why_learning_based,
                     compare_methods, create_detection_log,
                     generate_weekly_report)


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DATA_DIR, "data", "images")
LABELS_FILE = os.path.join(DATA_DIR, "data", "labels.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "result_q1.json")


def main():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    valid_names = sorted([
        k for k in labels
        if not k.startswith("test")
        and os.path.exists(os.path.join(IMAGES_DIR, k + ".png"))
    ])

    # ── Part B: 기본 카운팅 ──
    predictions = {}
    for name in valid_names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        predictions[name] = count_boxes(img_path)

    # ── Part D: 증강 앙상블 카운팅 ──
    predictions_augmented = {}
    for name in valid_names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        predictions_augmented[name] = count_boxes_augmented(img_path)

    # ── 바운딩 박스 추출 (첫 번째 이미지 샘플) ──
    first_image = os.path.join(IMAGES_DIR, valid_names[0] + ".png")
    sample_bounding_boxes = extract_bounding_boxes(first_image)

    # ── Part C: 메트릭 계산 ──
    categories = ["easy", "medium", "hard"]
    metrics = {}
    metrics_augmented = {}
    for cat in categories:
        metrics[cat] = compute_metrics(predictions, labels, cat)
        metrics_augmented[cat] = compute_metrics(predictions_augmented, labels, cat)

    method_comparison = compare_methods(predictions, predictions_augmented, labels)

    worst_case = find_worst_case(predictions, labels, "hard")

    # ── Part E: 주간 검출 일지 ──
    base_date = datetime(2024, 1, 1)
    daily_logs = []
    for day in range(7):
        date_str = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        if day < 3:
            log = create_detection_log(predictions, labels, date_str)
        else:
            log = create_detection_log(predictions_augmented, labels, date_str)
        daily_logs.append(log)

    weekly_report = generate_weekly_report(daily_logs)

    # ── 결과 저장 ──
    result = {
        "predictions": predictions,
        "predictions_augmented": predictions_augmented,
        "sample_bounding_boxes": sample_bounding_boxes,
        "metrics": metrics,
        "metrics_augmented": metrics_augmented,
        "method_comparison": method_comparison,
        "worst_case_image": worst_case if worst_case else "",
        "failure_reasons": get_failure_reasons(),
        "why_learning_based": get_why_learning_based(),
        "weekly_report": weekly_report,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    main()
