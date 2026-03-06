"""Generate XOR and regression datasets for exam 5."""
import numpy as np
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float64)
np.savez(os.path.join(data_dir, "xor_data.npz"), X=X_xor, y=y_xor)

# Regression dataset
np.random.seed(42)
X_reg = np.random.randn(50, 2)
y_reg = (2 * X_reg[:, 0] + 3 * X_reg[:, 1] + np.random.randn(50) * 0.1).reshape(-1, 1)
np.savez(os.path.join(data_dir, "regression_data.npz"), X=X_reg, y=y_reg)

print("Data generated successfully.")
