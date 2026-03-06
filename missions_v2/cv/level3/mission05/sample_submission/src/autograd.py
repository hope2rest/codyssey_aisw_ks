"""autograd.py - 그래디언트 검증 유틸리티"""
import numpy as np
from tensor import Tensor


def numerical_gradient(f, x, eps=1e-5):
    """Central Difference로 수치적 그래디언트를 계산합니다.

    Args:
        f: numpy 배열을 받아 스칼라를 반환하는 함수
        x: 그래디언트를 계산할 numpy 배열
        eps: 섭동(perturbation) 크기

    Returns:
        x와 같은 형태의 수치적 그래디언트 numpy 배열
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        fx_plus = f(x)

        x[idx] = old_val - eps
        fx_minus = f(x)

        grad[idx] = (fx_plus - fx_minus) / (2 * eps)
        x[idx] = old_val
        it.iternext()
    return grad


def gradient_check(f, x, eps=1e-5, tol=1e-4):
    """분석적 그래디언트와 수치적 그래디언트를 비교합니다.

    Args:
        f: Tensor를 받아 스칼라 Tensor를 반환하는 함수
        x: numpy 배열 입력
        eps: 수치적 그래디언트의 섭동 크기
        tol: 상대 오차 허용 범위

    Returns:
        분석적/수치적 그래디언트 간 최대 상대 오차
    """
    # 분석적 그래디언트
    x_tensor = Tensor(x.copy(), requires_grad=True)
    out = f(x_tensor)
    out.backward()
    analytical = x_tensor.grad.copy()

    # 수치적 그래디언트
    def f_np(x_np):
        t = Tensor(x_np.copy(), requires_grad=False)
        return f(t).data.item()

    numerical = numerical_gradient(f_np, x.copy(), eps)

    # 최대 상대 오차 계산
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    relative_error = np.max(diff / denom)

    return float(relative_error)
