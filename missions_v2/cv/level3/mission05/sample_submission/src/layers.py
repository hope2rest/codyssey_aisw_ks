"""layers.py - Tensor 기반 신경망 레이어"""
import numpy as np
from tensor import Tensor


class Linear:
    """완전 연결(Fully Connected) 레이어."""

    def __init__(self, in_features, out_features, init='he'):
        """Linear 레이어를 초기화합니다.

        Args:
            in_features: 입력 특성 수
            out_features: 출력 특성 수
            init: 가중치 초기화 방식 ('zero', 'random', 'he')
        """
        self.in_features = in_features
        self.out_features = out_features

        if init == 'zero':
            w = np.zeros((in_features, out_features))
        elif init == 'random':
            np.random.seed(42)
            w = np.random.randn(in_features, out_features) * 0.01
        elif init == 'he':
            np.random.seed(42)
            w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        else:
            raise ValueError(f"Unknown init: {init}")

        self.W = Tensor(w, requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x):
        """순전파: x @ W + b."""
        return (x @ self.W) + self.b

    def parameters(self):
        """파라미터 리스트를 반환합니다."""
        return [self.W, self.b]


class Sequential:
    """레이어와 활성화 함수를 순차적으로 연결하는 컨테이너."""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        """모든 레이어를 순서대로 통과하는 순전파."""
        for layer in self.layers:
            if callable(layer) and not hasattr(layer, 'forward'):
                x = layer(x)
            else:
                x = layer.forward(x)
        return x

    def parameters(self):
        """모든 파라미터 리스트를 반환합니다."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class ReLU:
    """ReLU 활성화 레이어."""

    def forward(self, x):
        return x.relu()


class Sigmoid:
    """Sigmoid 활성화 레이어."""

    def forward(self, x):
        return x.sigmoid()


def mse_loss(predicted, target):
    """평균 제곱 오차(MSE) 손실 함수.

    Args:
        predicted: 예측값 Tensor
        target: 목표값 Tensor

    Returns:
        Tensor 스칼라 손실값
    """
    diff = predicted - target
    sq = diff * diff
    return sq.sum() * Tensor(np.array(1.0 / predicted.data.size))


def binary_cross_entropy(predicted, target):
    """이진 교차 엔트로피(BCE) 손실 함수.

    Args:
        predicted: 예측값 Tensor (sigmoid 이후)
        target: 목표값 Tensor (0 또는 1)

    Returns:
        Tensor 스칼라 손실값
    """
    eps = 1e-12
    pred_clipped = Tensor(np.clip(predicted.data, eps, 1 - eps),
                          requires_grad=predicted.requires_grad,
                          _children=predicted._prev)
    # 클리핑을 통한 그래디언트 흐름 구축
    # 예측값에 직접 log를 안전하게 적용
    log_p = predicted.log()
    one_minus_p = Tensor(np.ones_like(predicted.data)) - predicted
    log_one_minus_p = one_minus_p.log()

    # -(t * log(p) + (1-t) * log(1-p))
    one_minus_t = Tensor(np.ones_like(target.data)) - target
    loss = (target * log_p + one_minus_t * log_one_minus_p)
    neg_loss = -loss
    mean_loss = neg_loss.sum() * Tensor(np.array(1.0 / predicted.data.shape[0]))

    return mean_loss
