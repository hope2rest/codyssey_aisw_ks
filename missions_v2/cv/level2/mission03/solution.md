## 문항 3 정답지 — 미니 딥러닝 프레임워크 설계를 통한 성능 검증

### 정답 코드

#### tensor.py

```python
"""tensor.py - 자동 미분을 지원하는 Tensor 클래스"""
import numpy as np


class Tensor:
    """자동 미분(Autograd)을 지원하는 간단한 Tensor 클래스."""

    def __init__(self, data, requires_grad=False, _children=()):
        if hasattr(data, 'data') and isinstance(getattr(data, 'data'), np.ndarray):
            data = data.data
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other))
        def _backward():
            if self.requires_grad:
                g = out.grad
                if self.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(self.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g
            if other.requires_grad:
                g = out.grad
                if other.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(other.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += g
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other))
        def _backward():
            if self.requires_grad:
                g = out.grad * other.data
                if self.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(self.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(self.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += g
            if other.requires_grad:
                g = out.grad * self.data
                if other.data.shape != out.data.shape:
                    ndim_diff = len(out.data.shape) - len(other.data.shape)
                    axes = list(range(ndim_diff))
                    for i, (s, o) in enumerate(zip(other.data.shape, out.data.shape[ndim_diff:])):
                        if s == 1 and o != 1:
                            axes.append(i + ndim_diff)
                    g = np.sum(g, axis=tuple(axes)).reshape(other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += g
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=True, _children=(self, other))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=True, _children=(self,))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self.__add__(-other)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=True, _children=(self,))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * (self.data > 0).astype(np.float64)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=True, _children=(self,))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * s * (1 - s)
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.array(np.sum(self.data)), requires_grad=True, _children=(self,))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-12, None)), requires_grad=True, _children=(self,))
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad / np.clip(self.data, 1e-12, None)
        out._backward = _backward
        return out

    def backward(self):
        """위상 정렬을 이용한 역전파."""
        topo = []
        visited = set()
        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
```

#### autograd.py

```python
"""autograd.py - 그래디언트 검증 유틸리티"""
import numpy as np
from tensor import Tensor


def numerical_gradient(f, x, eps=1e-5):
    """Central Difference로 수치적 그래디언트를 계산합니다."""
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
    """분석적 그래디언트와 수치적 그래디언트를 비교합니다."""
    x_tensor = Tensor(x.copy(), requires_grad=True)
    out = f(x_tensor)
    out.backward()
    analytical = x_tensor.grad.copy()
    def f_np(x_np):
        t = Tensor(x_np.copy(), requires_grad=False)
        return f(t).data.item()
    numerical = numerical_gradient(f_np, x.copy(), eps)
    diff = np.abs(analytical - numerical)
    denom = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-8)
    relative_error = np.max(diff / denom)
    return float(relative_error)
```

#### layers.py

```python
"""layers.py - Tensor 기반 신경망 레이어"""
import numpy as np
from tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features, init='he'):
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
        return (x @ self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            if callable(layer) and not hasattr(layer, 'forward'):
                x = layer(x)
            else:
                x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


class ReLU:
    def forward(self, x):
        return x.relu()


class Sigmoid:
    def forward(self, x):
        return x.sigmoid()


def mse_loss(predicted, target):
    diff = predicted - target
    sq = diff * diff
    return sq.sum() * Tensor(np.array(1.0 / predicted.data.size))


def binary_cross_entropy(predicted, target):
    log_p = predicted.log()
    one_minus_p = Tensor(np.ones_like(predicted.data)) - predicted
    log_one_minus_p = one_minus_p.log()
    one_minus_t = Tensor(np.ones_like(target.data)) - target
    loss = (target * log_p + one_minus_t * log_one_minus_p)
    neg_loss = -loss
    mean_loss = neg_loss.sum() * Tensor(np.array(1.0 / predicted.data.shape[0]))
    return mean_loss
```

#### trainer.py

```python
"""trainer.py - 학습 유틸리티"""
import numpy as np
from tensor import Tensor


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


def train_epoch(model, X, y, loss_fn, optimizer):
    optimizer.zero_grad()
    x_tensor = Tensor(X, requires_grad=False)
    y_tensor = Tensor(y, requires_grad=False)
    pred = model.forward(x_tensor)
    loss = loss_fn(pred, y_tensor)
    loss.backward()
    optimizer.step()
    return float(loss.data)


def train(model, X, y, loss_fn, optimizer, epochs=100):
    losses = []
    for _ in range(epochs):
        loss_val = train_epoch(model, X, y, loss_fn, optimizer)
        losses.append(loss_val)
    return losses
```

#### diagnostics.py

```python
"""diagnostics.py - Bias/Variance 성능 진단"""
import numpy as np
from tensor import Tensor


def compute_train_test_loss(model, X_train, y_train, X_test, y_test, loss_fn):
    x_tr = Tensor(X_train, requires_grad=False)
    y_tr = Tensor(y_train, requires_grad=False)
    pred_tr = model.forward(x_tr)
    train_loss = float(loss_fn(pred_tr, y_tr).data)
    x_te = Tensor(X_test, requires_grad=False)
    y_te = Tensor(y_test, requires_grad=False)
    pred_te = model.forward(x_te)
    test_loss = float(loss_fn(pred_te, y_te).data)
    return train_loss, test_loss


def diagnose_bias_variance(train_loss, test_loss, threshold=0.1):
    if train_loss > threshold:
        return "high_bias"
    elif test_loss - train_loss > threshold:
        return "high_variance"
    else:
        return "good_fit"


def learning_curve(model_fn, X, y, loss_fn, optimizer_fn, epochs, train_sizes=None):
    if train_sizes is None:
        train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    n = len(X)
    val_n = max(1, int(n * 0.2))
    X_val = X[-val_n:]
    y_val = y[-val_n:]
    X_pool = X[:-val_n]
    y_pool = y[:-val_n]
    result_train_losses = []
    result_val_losses = []
    for frac in train_sizes:
        k = max(1, int(len(X_pool) * frac))
        X_sub = X_pool[:k]
        y_sub = y_pool[:k]
        model = model_fn()
        optimizer = optimizer_fn(model.parameters())
        from trainer import train as train_loop
        train_loop(model, X_sub, y_sub, loss_fn, optimizer, epochs=epochs)
        tr_loss, vl_loss = compute_train_test_loss(model, X_sub, y_sub, X_val, y_val, loss_fn)
        result_train_losses.append(tr_loss)
        result_val_losses.append(vl_loss)
    return {
        "train_sizes": train_sizes,
        "train_losses": result_train_losses,
        "val_losses": result_val_losses
    }
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | tensor.py: Tensor 클래스, backward 메서드 | 5점 | AST 자동 |
| 2 | autograd.py: numerical_gradient, gradient_check 함수 | 5점 | AST 자동 |
| 3 | layers.py: Linear, Sequential 클래스, mse_loss, binary_cross_entropy 함수 | 5점 | AST 자동 |
| 4 | trainer.py: SGD 클래스, train_epoch, train 함수 | 5점 | AST 자동 |
| 5 | diagnostics.py: compute_train_test_loss, diagnose_bias_variance, learning_curve 함수 | 5점 | AST 자동 |
| 6 | Tensor 덧셈/곱셈/행렬곱 + backward | 10점 | import 자동 |
| 7 | Tensor relu/sigmoid + backward | 10점 | import 자동 |
| 8 | Tensor 위상 정렬 backward | 5점 | import 자동 |
| 9 | 수치적 그래디언트 검증 (central difference) | 10점 | import 자동 |
| 10 | Linear 레이어 (zero/random/he 초기화) | 5점 | import 자동 |
| 11 | MSE/BCE 손실 함수 | 10점 | import 자동 |
| 12 | SGD 옵티마이저 + 학습 루프 | 10점 | import 자동 |
| 13 | Bias/Variance 진단 | 5점 | import 자동 |
| 14 | result_q3.json 전체 구조 검증 | 10점 | JSON 자동 |

- Pass 기준: 총 100점 중 100점 (14개 전체 정답)
- AI 트랩: backward에서 그래디언트 누적 `+=` 대신 `=` 사용, 브로드캐스팅 backward shape 복원 누락, sigmoid 수치 안정성 클리핑 누락, log 입력 클리핑 누락, 위상 정렬 순서 오류

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `xor_data.npz` | NPZ | X(4x2), y(4x1) XOR 데이터 |
| `regression_data.npz` | NPZ | X_train(40x2), y_train(40x1), X_test(10x2), y_test(10x1) |
| Tensor.data | `np.ndarray(float64)` | 텐서 데이터 |
| Tensor.grad | `np.ndarray(float64)` | 그래디언트 |
| 손실값 | `float` | MSE 또는 BCE 손실 |
| 진단 결과 | `str` | "high_bias", "high_variance", "good_fit" |

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
