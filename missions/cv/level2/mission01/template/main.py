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
    # TODO: labels.json 로드

    # TODO: 유효한 이미지만 필터 (실제 파일이 존재하는 것만)

    # TODO: 각 이미지에 대해 count_boxes 실행 → predictions 딕셔너리

    # TODO: 카테고리별 metrics 계산 (easy, medium, hard)

    # TODO: worst_case_image 찾기 (hard 카테고리)

    # TODO: failure_reasons, why_learning_based 가져오기

    result = {
        "predictions": {},
        "metrics": {},
        "worst_case_image": "",
        "failure_reasons": [],
        "why_learning_based": "",
    }

    # TODO: result를 JSON 파일로 저장

    return result


if __name__ == "__main__":
    main()
