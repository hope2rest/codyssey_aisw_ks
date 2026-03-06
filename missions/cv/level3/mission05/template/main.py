"""Main pipeline for mini deep learning framework exam."""
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
    np.random.seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    # TODO: 1. Load XOR data and train model
    # Model: Linear(2,4) -> ReLU -> Linear(4,1) -> Sigmoid
    # Optimizer: SGD(lr=0.1), 1000 epochs, binary_cross_entropy

    # TODO: 2. Gradient check on a simple function

    # TODO: 3. Load regression data, train regression model

    # TODO: 4. Init comparison (zero, random, he)

    # TODO: 5. Diagnostics (bias/variance)

    # TODO: 6. Learning curve

    # TODO: 7. Save result_q5.json
    pass


if __name__ == "__main__":
    main()
