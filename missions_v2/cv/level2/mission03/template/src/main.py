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
    # TODO: 전체 파이프라인 실행 및 result_q3.json 저장
    # 1. XOR 데이터 로드 및 모델 학습
    # 2. Gradient check 수행
    # 3. 회귀 데이터 로드 및 모델 학습
    # 4. 초기화 전략 비교 (zero, random, he)
    # 5. Bias/Variance 진단
    # 6. 학습 곡선 계산
    # 7. result_q3.json 저장


if __name__ == "__main__":
    main()
