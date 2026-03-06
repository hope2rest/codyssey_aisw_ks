"""autograd.py - 그래디언트 검증 유틸리티"""
import numpy as np
from tensor import Tensor


def numerical_gradient(f, x, eps=1e-5):
    """Central Difference로 수치적 그래디언트를 계산합니다.

    Args:
        f: numpy 배열을 받아 스칼라를 반환하는 함수
        x: numpy 배열
        eps: 섭동(perturbation) 크기

    Returns:
        수치적 그래디언트 numpy 배열
    """
    # TODO: 각 요소에 대해 (f(x+eps) - f(x-eps)) / (2*eps) 계산
    pass


def gradient_check(f, x, eps=1e-5, tol=1e-4):
    """분석적 그래디언트와 수치적 그래디언트를 비교합니다.

    Args:
        f: Tensor를 받아 스칼라 Tensor를 반환하는 함수
        x: numpy 배열 입력
        eps: 수치적 그래디언트의 섭동 크기
        tol: 허용 오차

    Returns:
        최대 상대 오차
    """
    # TODO:
    # 1. backward()로 분석적 그래디언트 계산
    # 2. 수치적 그래디언트 계산
    # 3. 최대 상대 오차 반환: max(|a-n| / max(|a|+|n|, 1e-8))
    pass
