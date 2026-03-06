"""main.py — 전체 파이프라인 실행 및 결과 JSON 생성"""

import json
import os

from counter import count_boxes, count_boxes_augmented, extract_bounding_boxes
from metrics import compute_metrics, find_worst_case, compare_methods


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MISSION_DIR = os.path.dirname(PROJECT_DIR)
IMAGES_DIR = os.path.join(MISSION_DIR, "data", "images")
LABELS_FILE = os.path.join(MISSION_DIR, "data", "labels.json")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "output", "result_q3.json")


def main():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    valid_names = sorted([
        k for k in labels
        if not k.startswith("test")
        and os.path.exists(os.path.join(IMAGES_DIR, k + ".png"))
    ])

    # 기본 카운팅
    predictions = {}
    for name in valid_names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        predictions[name] = count_boxes(img_path)

    # 증강 앙상블 카운팅
    predictions_augmented = {}
    for name in valid_names:
        img_path = os.path.join(IMAGES_DIR, name + ".png")
        predictions_augmented[name] = count_boxes_augmented(img_path)

    # 바운딩 박스 추출 (첫 번째 이미지 샘플)
    first_image = os.path.join(IMAGES_DIR, valid_names[0] + ".png")
    sample_bounding_boxes = extract_bounding_boxes(first_image)

    # 메트릭 계산
    categories = ["easy", "medium", "hard"]
    metrics = {}
    metrics_augmented = {}
    for cat in categories:
        metrics[cat] = compute_metrics(predictions, labels, cat)
        metrics_augmented[cat] = compute_metrics(predictions_augmented, labels, cat)

    method_comparison = compare_methods(predictions, predictions_augmented, labels)
    worst_case = find_worst_case(predictions, labels, "hard")

    # 결과 저장
    result = {
        "predictions": predictions,
        "predictions_augmented": predictions_augmented,
        "sample_bounding_boxes": sample_bounding_boxes,
        "metrics": metrics,
        "metrics_augmented": metrics_augmented,
        "method_comparison": method_comparison,
        "worst_case_image": worst_case if worst_case else "",
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    main()
