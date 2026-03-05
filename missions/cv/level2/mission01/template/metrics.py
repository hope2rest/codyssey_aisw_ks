"""metrics.py — 성능 지표 계산 및 한계 분석 모듈"""

import numpy as np


def compute_metrics(predictions, labels, category):
    """
    특정 카테고리의 MAE와 Accuracy를 계산.

    Parameters
    ----------
    predictions : dict — {"이미지명": 예측개수, ...}
    labels      : dict — {"이미지명": 정답개수, ...}
    category    : str  — "easy", "medium", "hard"

    Returns
    -------
    dict — {"mae": float, "accuracy": float}
    """
    # TODO: 해당 카테고리의 이미지만 필터링
    # TODO: MAE = 절대오차 평균
    # TODO: Accuracy = 정확히 맞춘 비율
    pass


def find_worst_case(predictions, labels, category):
    """
    특정 카테고리에서 오차가 가장 큰 이미지 이름을 반환.

    Parameters
    ----------
    predictions : dict
    labels      : dict
    category    : str

    Returns
    -------
    str — 가장 큰 오차의 이미지 이름
    """
    # TODO: 오차가 가장 큰 이미지 이름 반환
    pass


def get_failure_reasons():
    """
    규칙 기반 방식이 hard 카테고리에서 실패하는 기술적 원인을 3가지 이상 반환.
    각 항목: 한국어, 20자 이상.

    Returns
    -------
    list[str] — 실패 원인 목록
    """
    # TODO: 기술적 원인 3가지 이상 서술
    return []


def get_why_learning_based():
    """
    학습 기반 접근법(CNN 등)이 필요한 이유를 200자 이내 한국어로 서술.

    Returns
    -------
    str — 30~200자 한국어 서술
    """
    # TODO: 200자 이내 서술
    return ""
