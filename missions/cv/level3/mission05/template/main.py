"""main.py - 미니 딥러닝 프레임워크 전체 파이프라인"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor import Tensor
from autograd import gradient_check
from layers import Linear, Sequential, ReLU, Sigmoid, mse_loss, binary_cross_entropy
from trainer import SGD, train
from diagnostics import compute_train_test_loss, diagnose_bias_variance, learning_curve


def main():
    """전체 파이프라인을 실행하고 result_q5.json을 저장합니다."""
    np.random.seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    # TODO: 1. XOR 데이터 로드 및 모델 학습
    # 모델: Linear(2,4) -> ReLU -> Linear(4,1) -> Sigmoid
    # 옵티마이저: SGD(lr=0.1), 1000 에폭, binary_cross_entropy

    # TODO: 2. 간단한 함수에 대한 Gradient check 수행

    # TODO: 3. 회귀 데이터 로드 및 회귀 모델 학습

    # TODO: 4. 초기화 전략 비교 (zero, random, he)

    # TODO: 5. Bias/Variance 진단

    # TODO: 6. 학습 곡선(Learning curve) 계산

    # TODO: 7. result_q5.json 저장
    pass


if __name__ == "__main__":
    main()
