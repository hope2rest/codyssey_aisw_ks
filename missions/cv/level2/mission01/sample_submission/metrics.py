"""metrics.py — 성능 지표 계산 및 한계 분석 모듈"""

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
