## 문항 3 정답지 — 미니 딥러닝 프레임워크 설계를 통한 성능 검증

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `xor_data.npz` | NPZ | X(4×2), y(4×1) XOR 데이터 |
| `regression_data.npz` | NPZ | X_train(40×2), y_train(40×1), X_test(10×2), y_test(10×1) |
| Tensor.data | `np.ndarray(float64)` | 텐서 데이터 |
| Tensor.grad | `np.ndarray(float64)` | 그래디언트 |
| 손실값 | `float` | MSE 또는 BCE 손실 |
| 진단 결과 | `str` | "high_bias", "high_variance", "good_fit" |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | tensor.py: Tensor 클래스, backward 메서드 | AST 자동 |
| 2 | autograd.py: numerical_gradient, gradient_check 함수 | AST 자동 |
| 3 | layers.py: Linear, Sequential 클래스, mse_loss, binary_cross_entropy 함수 | AST 자동 |
| 4 | trainer.py: SGD 클래스, train_epoch, train 함수 | AST 자동 |
| 5 | diagnostics.py: compute_train_test_loss, diagnose_bias_variance, learning_curve 함수 | AST 자동 |
| 6 | Tensor 덧셈/곱셈/행렬곱 + backward | import 자동 |
| 7 | Tensor relu/sigmoid + backward | import 자동 |
| 8 | Tensor 위상 정렬 backward | import 자동 |
| 9 | 수치적 그래디언트 검증 (central difference) | import 자동 |
| 10 | Linear 레이어 (zero/random/he 초기화) | import 자동 |
| 11 | MSE/BCE 손실 함수 | import 자동 |
| 12 | SGD 옵티마이저 + 학습 루프 | import 자동 |
| 13 | Bias/Variance 진단 | import 자동 |
| 14 | result_q3.json 전체 구조 검증 | JSON 자동 |

- Pass 기준: 14개 전체 통과
- AI 트랩: backward에서 그래디언트 누적 `+=` 대신 `=` 사용, 브로드캐스팅 backward shape 복원 누락, sigmoid 수치 안정성 클리핑 누락, log 입력 클리핑 누락, 위상 정렬 순서 오류

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 자동 미분 (Autograd) 원리 | test_tensor_add_backward, test_tensor_mul_backward, test_tensor_matmul_backward |
| 활성화 함수 구현 | test_relu_forward_backward, test_sigmoid_forward_backward |
| 수치적 그래디언트 검증 | test_gradient_check |
| 신경망 레이어 설계 | test_linear_layer, test_sequential_forward |
| 손실 함수 구현 | test_mse_loss, test_bce_loss |
| 학습 루프 (SGD) | test_train_xor |
| 성능 진단 (Bias/Variance) | test_diagnose_bias_variance |
| 파이프라인 통합 | test_result_structure |
