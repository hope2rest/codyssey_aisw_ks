"""main.py - 미니 딥러닝 프레임워크 전체 파이프라인"""
import sys
import os
import json
import numpy as np

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tensor import Tensor
from autograd import gradient_check
from layers import Linear, Sequential, ReLU, Sigmoid, mse_loss, binary_cross_entropy
from trainer import SGD, train
from diagnostics import compute_train_test_loss, diagnose_bias_variance, learning_curve


def main():
    np.random.seed(42)

    # 데이터 디렉토리 결정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    mission_dir = os.path.dirname(project_dir)
    data_dir = os.path.join(mission_dir, "data")

    # ========== 1. XOR 문제 ==========
    xor_data = np.load(os.path.join(data_dir, "xor_data.npz"))
    X_xor = xor_data["X"].astype(np.float64)
    y_xor = xor_data["y"].astype(np.float64)

    # 모델 구축
    np.random.seed(42)
    l1 = Linear(2, 4, init='he')
    l2 = Linear(4, 1, init='he')
    model_xor = Sequential(l1, ReLU(), l2, Sigmoid())

    optimizer_xor = SGD(model_xor.parameters(), lr=0.1)
    xor_losses = train(model_xor, X_xor, y_xor, binary_cross_entropy, optimizer_xor, epochs=1000)
    xor_final_loss = xor_losses[-1]

    # 예측
    x_t = Tensor(X_xor)
    xor_preds = model_xor.forward(x_t)
    xor_predictions = xor_preds.data.tolist()

    # 정확도
    preds_rounded = np.round(xor_preds.data)
    xor_accuracy = float(np.mean(preds_rounded == y_xor))

    # ========== 2. 그래디언트 검증 ==========
    np.random.seed(42)

    def simple_fn(x):
        """f(x) = sum(x^2 + 2*x)"""
        return (x * x + Tensor(np.full_like(x.data, 2.0)) * x).sum()

    x_check = np.random.randn(3, 2)
    max_error = gradient_check(simple_fn, x_check)
    gradient_check_passed = max_error < 1e-4

    # ========== 3. 회귀 문제 ==========
    reg_data = np.load(os.path.join(data_dir, "regression_data.npz"))
    X_reg = reg_data["X"].astype(np.float64)
    y_reg = reg_data["y"].astype(np.float64)

    np.random.seed(42)
    reg_model = Sequential(
        Linear(2, 8, init='he'),
        ReLU(),
        Linear(8, 1, init='he')
    )
    optimizer_reg = SGD(reg_model.parameters(), lr=0.01)
    reg_losses = train(reg_model, X_reg, y_reg, mse_loss, optimizer_reg, epochs=500)
    regression_final_loss = reg_losses[-1]

    # R-squared (결정 계수)
    x_reg_t = Tensor(X_reg)
    reg_preds = reg_model.forward(x_reg_t).data
    ss_res = np.sum((y_reg - reg_preds) ** 2)
    ss_tot = np.sum((y_reg - np.mean(y_reg)) ** 2)
    r_squared = float(1 - ss_res / ss_tot)

    # ========== 4. 초기화 전략 비교 ==========
    init_results = {}
    for init_type in ['zero', 'random', 'he']:
        np.random.seed(42)
        m = Sequential(
            Linear(2, 4, init=init_type),
            ReLU(),
            Linear(4, 1, init=init_type),
            Sigmoid()
        )
        opt = SGD(m.parameters(), lr=0.1)
        losses = train(m, X_xor, y_xor, binary_cross_entropy, opt, epochs=1000)
        init_results[f"{init_type}_final_loss"] = losses[-1]

    # ========== 5. 성능 진단 ==========
    # 회귀 데이터 분할
    n = len(X_reg)
    split = int(n * 0.8)
    X_train_diag = X_reg[:split]
    y_train_diag = y_reg[:split]
    X_test_diag = X_reg[split:]
    y_test_diag = y_reg[split:]

    np.random.seed(42)
    diag_model = Sequential(
        Linear(2, 8, init='he'),
        ReLU(),
        Linear(8, 1, init='he')
    )
    opt_diag = SGD(diag_model.parameters(), lr=0.01)
    train(diag_model, X_train_diag, y_train_diag, mse_loss, opt_diag, epochs=500)

    diag_train_loss, diag_test_loss = compute_train_test_loss(
        diag_model, X_train_diag, y_train_diag, X_test_diag, y_test_diag, mse_loss
    )
    diagnosis = diagnose_bias_variance(diag_train_loss, diag_test_loss)

    # ========== 6. 학습 곡선 ==========
    def model_factory():
        np.random.seed(42)
        return Sequential(
            Linear(2, 8, init='he'),
            ReLU(),
            Linear(8, 1, init='he')
        )

    def optimizer_factory(params):
        return SGD(params, lr=0.01)

    lc = learning_curve(model_factory, X_reg, y_reg, mse_loss, optimizer_factory,
                        epochs=200, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0])

    # ========== 결과 생성 ==========
    def r6(val):
        """소수점 6자리로 반올림."""
        if isinstance(val, list):
            return [r6(v) for v in val]
        return round(float(val), 6)

    result = {
        "xor_final_loss": r6(xor_final_loss),
        "xor_predictions": [r6(p) for p in xor_predictions],
        "xor_accuracy": r6(xor_accuracy),
        "gradient_check_max_error": r6(max_error),
        "gradient_check_passed": gradient_check_passed,
        "regression_final_loss": r6(regression_final_loss),
        "regression_r_squared": r6(r_squared),
        "init_comparison": {k: r6(v) for k, v in init_results.items()},
        "diagnostics": {
            "train_loss": r6(diag_train_loss),
            "test_loss": r6(diag_test_loss),
            "diagnosis": diagnosis
        },
        "learning_curve": {
            "train_sizes": r6(lc["train_sizes"]),
            "train_losses": r6(lc["train_losses"]),
            "val_losses": r6(lc["val_losses"])
        }
    }

    output_path = os.path.join(project_dir, "output", "result_q5.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Result saved to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
