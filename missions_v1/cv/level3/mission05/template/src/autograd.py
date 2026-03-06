import numpy as np
from tensor import Tensor


def numerical_gradient(f, x, eps=1e-5):
    # TODO: Central Difference로 수치적 그래디언트 계산


def gradient_check(f, x, eps=1e-5, tol=1e-4):
    # TODO: 분석적 vs 수치적 그래디언트 비교, 최대 상대 오차 반환
