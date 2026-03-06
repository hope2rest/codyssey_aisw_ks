"""Validator for Exam 5: Mini Deep Learning Framework + Performance Diagnostics.

Test groups:
- TestStructure (3): AST-based code structure checks
- TestTensor (4): Tensor operations and backward
- TestLayers (2): Linear and Sequential forward
- TestTrainer (1): Training produces decreasing losses
- TestDiagnostics (2): diagnose_bias_variance and gradient_check
- TestResult (2): result_q5.json structure and values
"""
import ast
import json
import os
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_submission_to_path(submission_dir):
    src_dir = os.path.join(submission_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _load_module(submission_dir, module_name):
    """Import a module from submission_dir/src."""
    _add_submission_to_path(submission_dir)
    if module_name in sys.modules:
        del sys.modules[module_name]
    mod = __import__(module_name)
    return mod


# ===========================================================================
# TestStructure: AST checks (3 tests)
# ===========================================================================

class TestStructure:
    """AST-based structure checks on submitted source files."""

    def _parse(self, submission_dir, filename):
        path = os.path.join(submission_dir, "src", filename)
        assert os.path.isfile(path), f"src/{filename} 파일 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return ast.parse(f.read(), filename)

    def _class_names(self, tree):
        return [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    def _func_names_in_class(self, tree, class_name):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return []

    def _top_func_names(self, tree):
        return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def test_tensor_structure(self, submission_dir):
        """tensor.py: Tensor class with __add__, __mul__, __matmul__, relu, sigmoid, backward"""
        tree = self._parse(submission_dir, "tensor.py")
        classes = self._class_names(tree)
        assert "Tensor" in classes, "Tensor class not found"
        methods = self._func_names_in_class(tree, "Tensor")
        for m in ["__add__", "__mul__", "__matmul__", "relu", "sigmoid", "backward"]:
            assert m in methods, f"Tensor.{m} not found"

    def test_layers_structure(self, submission_dir):
        """layers.py: Linear and Sequential classes"""
        tree = self._parse(submission_dir, "layers.py")
        classes = self._class_names(tree)
        assert "Linear" in classes, "Linear class not found"
        assert "Sequential" in classes, "Sequential class not found"
        lin_methods = self._func_names_in_class(tree, "Linear")
        assert "forward" in lin_methods, "Linear.forward not found"
        assert "parameters" in lin_methods, "Linear.parameters not found"

    def test_diagnostics_structure(self, submission_dir):
        """diagnostics.py: diagnose_bias_variance function"""
        tree = self._parse(submission_dir, "diagnostics.py")
        funcs = self._top_func_names(tree)
        assert "diagnose_bias_variance" in funcs, "diagnose_bias_variance function not found"


# ===========================================================================
# TestTensor: Tensor operations (4 tests)
# ===========================================================================

class TestTensor:
    """Test Tensor operations and backward pass."""

    def test_add_mul_matmul(self, submission_dir):
        """add/mul/matmul produce correct values"""
        tensor_mod = _load_module(submission_dir, "tensor")
        Tensor = tensor_mod.Tensor

        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))

        c = a + b
        np.testing.assert_allclose(c.data, [[6, 8], [10, 12]])

        d = a * b
        np.testing.assert_allclose(d.data, [[5, 12], [21, 32]])

        e = a @ b
        np.testing.assert_allclose(e.data, [[19, 22], [43, 50]])

    def test_backward_gradients(self, submission_dir):
        """backward computes correct gradients for simple expression"""
        tensor_mod = _load_module(submission_dir, "tensor")
        Tensor = tensor_mod.Tensor

        # f(x) = sum(x * x) => df/dx = 2x
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        y = (x * x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, [2.0, 4.0, 6.0], atol=1e-6)

    def test_relu_sigmoid(self, submission_dir):
        """relu and sigmoid correct behavior"""
        tensor_mod = _load_module(submission_dir, "tensor")
        Tensor = tensor_mod.Tensor

        x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))
        r = x.relu()
        np.testing.assert_allclose(r.data, [0.0, 0.0, 1.0, 2.0])

        s = x.sigmoid()
        expected = 1.0 / (1.0 + np.exp(-np.array([-1.0, 0.0, 1.0, 2.0])))
        np.testing.assert_allclose(s.data, expected, atol=1e-6)

    def test_sum_backward(self, submission_dir):
        """sum backward produces ones gradient"""
        tensor_mod = _load_module(submission_dir, "tensor")
        Tensor = tensor_mod.Tensor

        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        s = x.sum()
        s.backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 2)))


# ===========================================================================
# TestLayers (2 tests)
# ===========================================================================

class TestLayers:
    """Test Linear and Sequential layers."""

    def test_linear_forward_shape(self, submission_dir):
        """Linear forward produces correct output shape"""
        _add_submission_to_path(submission_dir)
        layers_mod = _load_module(submission_dir, "layers")
        tensor_mod = _load_module(submission_dir, "tensor")

        np.random.seed(0)
        layer = layers_mod.Linear(3, 5, init='he')
        x = tensor_mod.Tensor(np.random.randn(4, 3))
        out = layer.forward(x)
        assert out.shape == (4, 5), f"Expected (4,5), got {out.shape}"

    def test_sequential_forward(self, submission_dir):
        """Sequential forward chains layers"""
        _add_submission_to_path(submission_dir)
        layers_mod = _load_module(submission_dir, "layers")
        tensor_mod = _load_module(submission_dir, "tensor")

        np.random.seed(0)
        model = layers_mod.Sequential(
            layers_mod.Linear(2, 4, init='he'),
            layers_mod.ReLU(),
            layers_mod.Linear(4, 1, init='he')
        )
        x = tensor_mod.Tensor(np.random.randn(3, 2))
        out = model.forward(x)
        assert out.shape == (3, 1), f"Expected (3,1), got {out.shape}"
        assert len(model.parameters()) == 4, "Expected 4 parameters (2 layers x 2)"


# ===========================================================================
# TestTrainer (1 test)
# ===========================================================================

class TestTrainer:
    """Test training utilities."""

    def test_train_decreasing_loss(self, submission_dir):
        """train() returns decreasing losses on simple problem"""
        _add_submission_to_path(submission_dir)
        tensor_mod = _load_module(submission_dir, "tensor")
        layers_mod = _load_module(submission_dir, "layers")
        trainer_mod = _load_module(submission_dir, "trainer")

        np.random.seed(42)
        model = layers_mod.Sequential(
            layers_mod.Linear(2, 4, init='he'),
            layers_mod.ReLU(),
            layers_mod.Linear(4, 1, init='he')
        )
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        optimizer = trainer_mod.SGD(model.parameters(), lr=0.1)
        losses = trainer_mod.train(model, X, y, layers_mod.binary_cross_entropy,
                                   optimizer, epochs=200)
        assert len(losses) == 200
        assert np.mean(losses[:10]) > np.mean(losses[-10:]), \
            "Loss did not decrease during training"


# ===========================================================================
# TestDiagnostics (2 tests)
# ===========================================================================

class TestDiagnostics:
    """Test diagnostics functions."""

    def test_diagnose_bias_variance(self, submission_dir):
        """diagnose_bias_variance returns correct strings"""
        _add_submission_to_path(submission_dir)
        diag_mod = _load_module(submission_dir, "diagnostics")

        assert diag_mod.diagnose_bias_variance(0.5, 0.6) == "high_bias"
        assert diag_mod.diagnose_bias_variance(0.05, 0.3) == "high_variance"
        assert diag_mod.diagnose_bias_variance(0.05, 0.08) == "good_fit"

    def test_gradient_check(self, submission_dir):
        """gradient_check works on simple function"""
        _add_submission_to_path(submission_dir)
        autograd_mod = _load_module(submission_dir, "autograd")
        tensor_mod = _load_module(submission_dir, "tensor")

        def f(x):
            return (x * x).sum()

        x = np.array([1.0, 2.0, 3.0])
        err = autograd_mod.gradient_check(f, x)
        assert err < 1e-4, f"Gradient check error too large: {err}"


# ===========================================================================
# TestResult (2 tests)
# ===========================================================================

class TestResult:
    """Validate result_q5.json structure and values."""

    def _load_result(self, submission_dir):
        path = os.path.join(submission_dir, "output", "result_q5.json")
        assert os.path.isfile(path), "output/result_q5.json 파일 없음"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_result_structure(self, submission_dir):
        """result_q5.json has required keys and types"""
        result = self._load_result(submission_dir)
        required_keys = [
            "xor_final_loss", "xor_predictions", "xor_accuracy",
            "gradient_check_max_error", "gradient_check_passed",
            "regression_final_loss", "regression_r_squared",
            "init_comparison", "diagnostics", "learning_curve"
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

        assert isinstance(result["xor_predictions"], list)
        assert len(result["xor_predictions"]) == 4
        assert isinstance(result["gradient_check_passed"], bool)
        assert isinstance(result["init_comparison"], dict)
        assert isinstance(result["diagnostics"], dict)
        assert "diagnosis" in result["diagnostics"]
        assert isinstance(result["learning_curve"], dict)

    def test_result_values(self, submission_dir):
        """result_q5.json values meet requirements"""
        result = self._load_result(submission_dir)

        assert result["xor_accuracy"] >= 0.75, \
            f"XOR accuracy {result['xor_accuracy']} < 0.75"
        assert result["gradient_check_passed"] is True, \
            "Gradient check did not pass"
        assert result["xor_final_loss"] < 0.5, \
            f"XOR loss {result['xor_final_loss']} too high"
        assert result["diagnostics"]["diagnosis"] in \
            ["high_bias", "high_variance", "good_fit"], \
            f"Invalid diagnosis: {result['diagnostics']['diagnosis']}"
