## 문항: 딥러닝 프레임워크 설계를 통한 성능 검증

### 문제

NumPy만으로 Tensor 기반 자동 미분(Autograd) 엔진을 구현하고, 신경망 레이어, 학습 루프, Bias/Variance 진단까지 포함하는 미니 딥러닝 프레임워크를 구축하세요.
데이터는 `data/xor_data.npz` (XOR 4샘플)과 `data/regression_data.npz` (50샘플, y=2*x1+3*x2+noise)에 저장되어 있습니다.

### 데이터 구조

| 파일 | 키 | 설명 |
|------|-----|------|
| `data/xor_data.npz` | `X` (4x2), `y` (4x1) | XOR 논리 연산 데이터 |
| `data/regression_data.npz` | `X_train` (40x2), `y_train` (40x1), `X_test` (10x2), `y_test` (10x1) | 선형 회귀 데이터 |

### 프로젝트 구조

| 파일 | 역할 | 핵심 클래스/함수 |
|------|------|------------------|
| `tensor.py` | Tensor 클래스, 자동 미분 | `Tensor` (`__add__`, `__mul__`, `__matmul__`, `relu`, `sigmoid`, `log`, `sum`, `backward`) |
| `autograd.py` | 수치적 그래디언트 검증 | `numerical_gradient()`, `gradient_check()` |
| `layers.py` | 신경망 레이어, 손실 함수 | `Linear`, `Sequential`, `ReLU`, `Sigmoid`, `mse_loss()`, `binary_cross_entropy()` |
| `trainer.py` | SGD 옵티마이저, 학습 루프 | `SGD`, `train_epoch()`, `train()` |
| `diagnostics.py` | 성능 진단 | `compute_train_test_loss()`, `diagnose_bias_variance()`, `learning_curve()` |
| `main.py` | 전체 파이프라인 실행 | `main()` |

### 구현 요구사항

#### Part A: tensor.py - Tensor 클래스와 자동 미분

1. `Tensor.__init__(data, requires_grad, _children)` - `data`를 `np.float64` ndarray로 변환합니다. `grad=None`, `_backward=lambda:None`, `_prev=set(_children)`을 저장합니다.
2. `__add__(other)` - 요소별 덧셈과 backward를 구현합니다. 브로드캐스팅 시 backward에서 합산하여 원래 shape으로 복원합니다.
3. `__mul__(other)` - 요소별 곱셈과 backward를 구현합니다. 브로드캐스팅 처리 포함.
4. `__matmul__(other)` - 행렬 곱셈과 backward를 구현합니다. `self.grad += out.grad @ other.data.T`, `other.grad += self.data.T @ out.grad`.
5. `__neg__()`, `__sub__(other)` - 부정과 뺄셈을 구현합니다.
6. `relu()` - ReLU 활성화 함수. backward: `grad * (self.data > 0)`.
7. `sigmoid()` - Sigmoid 활성화 함수. 수치 안정성을 위해 입력을 `[-500, 500]` 범위로 클리핑합니다. `s = 1/(1+exp(-x))`, backward: `grad * s * (1-s)`.
8. `sum()` - 모든 요소의 합. backward: `ones_like * out.grad`.
9. `log()` - 요소별 로그. 수치 안정성을 위해 입력을 `[1e-12, inf]`로 클리핑합니다. backward: `grad / x`.
10. `backward()` - 역전파. 위상 정렬(topological sort)로 순서를 구한 후 역순으로 `_backward()` 호출. 그래디언트 누적은 `+=` 사용.

#### Part B: autograd.py - 그래디언트 검증

11. `numerical_gradient(f, x, eps=1e-5)` - Central Difference로 수치적 그래디언트를 계산합니다: `(f(x+eps) - f(x-eps)) / (2*eps)`.
12. `gradient_check(f, x, eps=1e-5, tol=1e-4)` - 분석적 그래디언트(backward)와 수치적 그래디언트를 비교합니다. 최대 상대 오차를 반환합니다: `max(|a-n| / max(|a|+|n|, 1e-8))`.

#### Part C: layers.py - 신경망 레이어

13. `Linear(in_features, out_features, init)` - 완전 연결 레이어. 초기화 옵션:
    - `'zero'`: 모두 0 (대칭 깨지지 않음)
    - `'random'`: `N(0, 0.01)` (기울기 소실 가능)
    - `'he'`: `N(0, sqrt(2/fan_in))` (ReLU 네트워크에 권장)
    - `W`는 `(in_features, out_features)`, `b`는 `(1, out_features)` 형태. 둘 다 `requires_grad=True`.
14. `Sequential(*layers)` - 레이어를 순차적으로 연결합니다. `forward(x)`와 `parameters()` 구현.
15. `mse_loss(predicted, target)` - MSE 손실: `((predicted - target)^2).sum() / N`.
16. `binary_cross_entropy(predicted, target)` - BCE 손실: `-mean(t*log(p) + (1-t)*log(1-p))`.

#### Part D: trainer.py - 학습 루프

17. `SGD(parameters, lr)` - `step()`: `p.data -= lr * p.grad`, `zero_grad()`: `p.grad = None`.
18. `train_epoch(model, X, y, loss_fn, optimizer)` - 1 에폭 학습: zero_grad → forward → loss → backward → step. 손실값 반환.
19. `train(model, X, y, loss_fn, optimizer, epochs)` - 전체 학습 루프. 에폭별 손실 리스트 반환.

#### Part E: diagnostics.py - 성능 진단

20. `compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn)` - 학습/테스트 손실을 계산하여 `(train_loss, test_loss)` 반환.
21. `diagnose_bias_variance(train_loss, test_loss, threshold=0.1)` - 진단 결과 문자열 반환:
    - `train_loss > threshold` → `"high_bias"`
    - `test_loss - train_loss > threshold` → `"high_variance"`
    - 그 외 → `"good_fit"`
22. `learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs, train_sizes)` - 다양한 학습 데이터 크기에서 학습/검증 손실을 계산합니다. `{"train_sizes": list, "train_losses": list, "val_losses": list}` 반환.

#### Part F: main.py - 파이프라인 실행

23. `main()` - 전체 파이프라인:
    1. XOR 데이터 로드 → 모델(`Linear(2,4)->ReLU->Linear(4,1)->Sigmoid`) 학습 (SGD lr=0.1, 1000 에폭, BCE 손실)
    2. Gradient check 수행
    3. 회귀 데이터 로드 → 회귀 모델 학습 → R-squared 계산
    4. 초기화 전략 비교 (zero/random/he)
    5. Bias/Variance 진단
    6. Learning curve 계산
    7. `result_q5.json` 저장

### 출력 형식

`result_q5.json` 파일로 다음 구조를 저장합니다:

```json
{
  "xor_final_loss": 실수,
  "xor_predictions": [[실수], [실수], [실수], [실수]],
  "xor_accuracy": 실수,
  "gradient_check_max_error": 실수,
  "gradient_check_passed": 불리언,
  "regression_final_loss": 실수,
  "regression_r_squared": 실수,
  "init_comparison": {
    "zero_final_loss": 실수,
    "random_final_loss": 실수,
    "he_final_loss": 실수
  },
  "diagnostics": {
    "train_loss": 실수,
    "test_loss": 실수,
    "diagnosis": "문자열"
  },
  "learning_curve": {
    "train_sizes": [실수, ...],
    "train_losses": [실수, ...],
    "val_losses": [실수, ...]
  }
}
```

- 모든 실수값은 `round(..., 6)`으로 반올림합니다.

### 제약 사항

- NumPy만으로 모든 수치 계산을 수행합니다 (`torch`, `tensorflow`, `sklearn` 사용 금지).
- `Tensor.backward()`에서 그래디언트 누적은 반드시 `+=`를 사용합니다 (`=` 사용 시 공유 노드 오류 발생).
- `sigmoid` 입력은 `[-500, 500]`, `log` 입력은 `[1e-12, inf]`로 클리핑합니다.
- `np.random.seed(42)`를 `main()` 시작 시 설정합니다.

### 제출 폴더 구조

다음 폴더 구조를 zip으로 묶어 제출합니다.

```
submission/
├── src/
│   ├── tensor.py
│   ├── autograd.py
│   ├── layers.py
│   ├── trainer.py
│   ├── diagnostics.py
│   └── main.py
├── config/
│   └── config.json
├── models/
│   └── model_info.json
├── logs/
│   └── training_log.json
└── output/
    └── result_q5.json
```

- `template/` 디렉토리의 각 파일의 `# TODO` 부분을 채우세요.
