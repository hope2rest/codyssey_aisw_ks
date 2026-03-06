# Q5 Solution: Mini Deep Learning Framework

## Module Structure

| File | Role |
|------|------|
| tensor.py | Tensor class with reverse-mode autodiff |
| autograd.py | Numerical gradient checking (central difference) |
| layers.py | Linear, Sequential, ReLU, Sigmoid, MSE, BCE |
| trainer.py | SGD optimizer, train loop |
| diagnostics.py | bias/variance diagnosis, learning curve |
| main.py | Full pipeline, outputs result_q5.json |

## Key Implementation Details

### tensor.py - Backward Pass
- Topological sort via DFS, then reverse iteration
- Gradient accumulation uses `+=` (not `=`) to handle shared nodes
- Broadcasting handled in add/mul backward by summing over broadcasted axes
- `sigmoid`: clips input to [-500, 500] for numerical stability
- `log`: clips input to [1e-12, inf]

### layers.py - Initialization
- `'zero'`: all zeros (will fail to break symmetry)
- `'random'`: N(0, 0.01) (may vanish)
- `'he'`: N(0, sqrt(2/fan_in)) (recommended for ReLU networks)

### diagnostics.py - Bias/Variance
- `train_loss > threshold` => high_bias (underfitting)
- `test_loss - train_loss > threshold` => high_variance (overfitting)
- Otherwise => good_fit

## Expected Results

- XOR accuracy: 1.0 (all 4 patterns correct)
- Gradient check: max relative error ~0 (passed)
- Regression R-squared: >0.99
- Init comparison: he >> random >> zero
- Diagnosis on regression: good_fit

## Grading Criteria (all quantitative)

| Check | Points | Metric |
|-------|--------|--------|
| Structure (AST) | 15 | 3 tests pass |
| Tensor ops | 25 | 4 tests pass (values + gradients) |
| Layers | 15 | 2 tests pass (shape + chaining) |
| Trainer | 10 | 1 test pass (decreasing loss) |
| Diagnostics | 15 | 2 tests pass (strings + grad check) |
| Result JSON | 20 | 2 tests pass (structure + xor_accuracy >= 0.75) |
