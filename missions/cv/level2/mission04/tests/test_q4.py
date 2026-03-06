"""
Q4 금융 리스크 예측 + 모델 해석 — pytest 검증 (18개 테스트, 정량적 검증만)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: preprocessor.py, model.py, interpreter.py, main.py, result_q4.json (5파일)
"""
import ast
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_SUBMISSION_DIR = None
_MISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_MISSION_DIR, "data")
_CSV_PATH = os.path.join(_DATA_DIR, "loan_data.csv")


@pytest.fixture(autouse=True, scope="module")
def _configure(submission_dir):
    global _SUBMISSION_DIR
    _SUBMISSION_DIR = submission_dir


def _import_module(module_name):
    if _SUBMISSION_DIR not in sys.path:
        sys.path.insert(0, _SUBMISSION_DIR)
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _parse_ast(filename):
    path = os.path.join(_SUBMISSION_DIR, filename)
    assert os.path.isfile(path), f"{filename} 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path), source


# ========================================================================
# TestStructure — 코드 구조 검증 (4개)
# ========================================================================

class TestStructure:
    def test_preprocessor_functions(self):
        """preprocessor.py에 필수 함수 4개가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("preprocessor.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"load_data", "handle_missing", "encode_categoricals", "scale_features"}
        missing = required - func_names
        assert not missing, f"preprocessor.py 누락 함수: {missing}"

    def test_model_functions(self):
        """model.py에 필수 함수 4개가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("model.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"split_data", "apply_pca", "train_model", "evaluate_model"}
        missing = required - func_names
        assert not missing, f"model.py 누락 함수: {missing}"

    def test_interpreter_functions(self):
        """interpreter.py에 필수 함수 3개가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("interpreter.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"get_feature_importance", "get_pca_variance", "cluster_features"}
        missing = required - func_names
        assert not missing, f"interpreter.py 누락 함수: {missing}"

    def test_main_function(self):
        """main.py에 main() 함수가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("main.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "main" in func_names, "main.py에 main() 함수 없음"


# ========================================================================
# TestPreprocessor — 전처리 기능 검증 (4개)
# ========================================================================

class TestPreprocessor:
    def test_load_data(self):
        """load_data()가 올바른 (X, y) 튜플을 반환하는지 확인"""
        mod = _import_module("preprocessor")
        X, y = mod.load_data(_CSV_PATH)
        assert isinstance(X, pd.DataFrame), "X가 DataFrame이 아닙니다"
        assert X.shape == (200, 9), f"X shape 오류: {X.shape}, 기대: (200, 9)"
        assert len(y) == 200, f"y 길이 오류: {len(y)}"
        assert "loan_id" not in X.columns, "loan_id 컬럼이 제거되지 않았습니다"
        assert "risk_label" not in X.columns, "risk_label이 X에 포함되어 있습니다"

    def test_handle_missing(self):
        """handle_missing()이 결측값을 처리하는지 확인"""
        mod = _import_module("preprocessor")
        X, _ = mod.load_data(_CSV_PATH)
        assert X.isnull().sum().sum() > 0, "원본 데이터에 결측값이 없습니다"
        X_clean = mod.handle_missing(X)
        assert X_clean.isnull().sum().sum() == 0, "결측값이 남아 있습니다"
        assert X_clean.shape == X.shape, f"shape 변경됨: {X_clean.shape}"

    def test_encode_categoricals(self):
        """encode_categoricals()가 DataFrame을 반환하는지 확인"""
        mod = _import_module("preprocessor")
        X, _ = mod.load_data(_CSV_PATH)
        X_clean = mod.handle_missing(X)
        X_enc = mod.encode_categoricals(X_clean)
        assert isinstance(X_enc, pd.DataFrame), "반환값이 DataFrame이 아닙니다"
        assert X_enc.shape == X_clean.shape, f"shape 변경됨: {X_enc.shape}"

    def test_scale_features(self):
        """scale_features()가 StandardScaler로 스케일링하는지 확인"""
        mod = _import_module("preprocessor")
        X, _ = mod.load_data(_CSV_PATH)
        X_clean = mod.handle_missing(X)
        X_enc = mod.encode_categoricals(X_clean)
        scaled, scaler = mod.scale_features(X_enc)
        assert scaled.shape == (200, 9), f"scaled shape 오류: {scaled.shape}"
        # StandardScaler 결과: 평균 ~0
        col_means = np.abs(np.mean(scaled, axis=0))
        assert np.max(col_means) < 1e-10, f"스케일링 후 평균이 0이 아닙니다: {col_means}"


# ========================================================================
# TestModel — 모델 학습/평가 검증 (4개)
# ========================================================================

class TestModel:
    @staticmethod
    def _get_scaled_data():
        prep = _import_module("preprocessor")
        X, y = prep.load_data(_CSV_PATH)
        X = prep.handle_missing(X)
        X = prep.encode_categoricals(X)
        X_scaled, _ = prep.scale_features(X)
        return X_scaled, y

    def test_split_data(self):
        """split_data()가 70/30 stratified split을 수행하는지 확인"""
        X_scaled, y = self._get_scaled_data()
        mod = _import_module("model")
        X_train, X_test, y_train, y_test = mod.split_data(X_scaled, y)
        assert X_train.shape[0] == 140, f"X_train rows: {X_train.shape[0]}, 기대: 140"
        assert X_test.shape[0] == 60, f"X_test rows: {X_test.shape[0]}, 기대: 60"
        # stratify 확인: 테스트셋의 양성 비율이 전체와 유사해야 함
        full_ratio = y.sum() / len(y)
        test_ratio = y_test.sum() / len(y_test)
        assert abs(full_ratio - test_ratio) < 0.05, f"stratify 미적용: 전체={full_ratio:.3f}, 테스트={test_ratio:.3f}"

    def test_apply_pca(self):
        """apply_pca()가 PCA를 올바르게 적용하는지 확인"""
        X_scaled, y = self._get_scaled_data()
        mod = _import_module("model")
        X_train, X_test, _, _ = mod.split_data(X_scaled, y)
        X_train_pca, X_test_pca, pca = mod.apply_pca(X_train, X_test)
        assert X_train_pca.shape[1] == X_test_pca.shape[1], "PCA 차원 불일치"
        assert X_train_pca.shape[1] < 9, f"PCA가 차원을 줄이지 않음: {X_train_pca.shape[1]}"
        total_var = sum(pca.explained_variance_ratio_)
        assert total_var >= 0.95, f"총 분산 설명 비율 부족: {total_var:.4f}"

    def test_train_logistic(self):
        """LogisticRegression 학습 및 평가 검증"""
        X_scaled, y = self._get_scaled_data()
        mod = _import_module("model")
        X_train, X_test, y_train, y_test = mod.split_data(X_scaled, y)
        X_train_pca, X_test_pca, _ = mod.apply_pca(X_train, X_test)
        model = mod.train_model(X_train_pca, y_train, model_type="logistic")
        metrics = mod.evaluate_model(model, X_test_pca, y_test)
        assert "accuracy" in metrics, "accuracy 키 없음"
        assert "f1_macro" in metrics, "f1_macro 키 없음"
        assert metrics["accuracy"] >= 0.7, f"accuracy 너무 낮음: {metrics['accuracy']}"

    def test_train_ridge(self):
        """RidgeClassifier 학습 및 평가 검증"""
        X_scaled, y = self._get_scaled_data()
        mod = _import_module("model")
        X_train, X_test, y_train, y_test = mod.split_data(X_scaled, y)
        X_train_pca, X_test_pca, _ = mod.apply_pca(X_train, X_test)
        model = mod.train_model(X_train_pca, y_train, model_type="ridge")
        metrics = mod.evaluate_model(model, X_test_pca, y_test)
        assert "accuracy" in metrics, "accuracy 키 없음"
        assert metrics["accuracy"] >= 0.7, f"accuracy 너무 낮음: {metrics['accuracy']}"


# ========================================================================
# TestInterpreter — 모델 해석 검증 (3개)
# ========================================================================

class TestInterpreter:
    @staticmethod
    def _get_model_and_data():
        prep = _import_module("preprocessor")
        mod = _import_module("model")
        X, y = prep.load_data(_CSV_PATH)
        X = prep.handle_missing(X)
        X = prep.encode_categoricals(X)
        feature_names = list(X.columns)
        X_scaled, _ = prep.scale_features(X)
        X_train, X_test, y_train, y_test = mod.split_data(X_scaled, y)
        model = mod.train_model(X_train, y_train, model_type="logistic")
        X_train_pca, X_test_pca, pca = mod.apply_pca(X_train, X_test)
        return model, pca, X_scaled, feature_names

    def test_feature_importance(self):
        """get_feature_importance()가 올바른 형식을 반환하는지 확인"""
        model, _, X_scaled, feature_names = self._get_model_and_data()
        interp = _import_module("interpreter")
        importance = interp.get_feature_importance(model, feature_names)
        assert isinstance(importance, list), "반환값이 list가 아닙니다"
        assert len(importance) == 9, f"feature 수 오류: {len(importance)}, 기대: 9"
        for item in importance:
            assert "feature" in item, "feature 키 없음"
            assert "importance" in item, "importance 키 없음"
            assert item["importance"] >= 0, "importance가 음수"
        # 내림차순 확인
        vals = [item["importance"] for item in importance]
        assert vals == sorted(vals, reverse=True), "절댓값 내림차순 정렬이 아닙니다"

    def test_pca_variance(self):
        """get_pca_variance()가 올바른 형식을 반환하는지 확인"""
        _, pca, _, _ = self._get_model_and_data()
        interp = _import_module("interpreter")
        variance = interp.get_pca_variance(pca)
        assert isinstance(variance, list), "반환값이 list가 아닙니다"
        assert len(variance) > 0, "빈 리스트입니다"
        total = sum(item["variance_ratio"] for item in variance)
        assert total >= 0.95, f"총 분산 설명 비율 부족: {total:.4f}"
        for item in variance:
            assert "component" in item, "component 키 없음"
            assert "variance_ratio" in item, "variance_ratio 키 없음"

    def test_clustering(self):
        """cluster_features()가 올바른 형식을 반환하는지 확인"""
        _, _, X_scaled, _ = self._get_model_and_data()
        interp = _import_module("interpreter")
        result = interp.cluster_features(X_scaled, n_clusters=3)
        assert "labels" in result, "labels 키 없음"
        assert "cluster_counts" in result, "cluster_counts 키 없음"
        assert "inertia" in result, "inertia 키 없음"
        assert len(result["labels"]) == 200, f"labels 길이: {len(result['labels'])}, 기대: 200"
        assert len(result["cluster_counts"]) == 3, f"클러스터 수: {len(result['cluster_counts'])}, 기대: 3"
        total_count = sum(result["cluster_counts"].values())
        assert total_count == 200, f"클러스터 총 개수: {total_count}, 기대: 200"
        assert result["inertia"] > 0, "inertia가 0 이하"


# ========================================================================
# TestResult — result_q4.json 결과 검증 (3개)
# ========================================================================

class TestResult:
    @staticmethod
    def _load_result():
        path = os.path.join(_SUBMISSION_DIR, "result_q4.json")
        assert os.path.isfile(path), f"result_q4.json 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_result_structure(self):
        """result_q4.json의 최상위 키 구조 확인"""
        result = self._load_result()
        required_keys = {"preprocessing", "model_logistic", "model_ridge",
                         "pca", "feature_importance", "clustering"}
        missing = required_keys - set(result.keys())
        assert not missing, f"누락 키: {missing}"

    def test_preprocessing_values(self):
        """전처리 결과 값 검증"""
        result = self._load_result()
        prep = result["preprocessing"]
        assert prep["original_shape"] == [200, 9], f"original_shape 오류: {prep['original_shape']}"
        assert prep["missing_values_before"] == 10, f"missing_values_before 오류: {prep['missing_values_before']}"
        assert prep["missing_values_after"] == 0, f"missing_values_after 오류: {prep['missing_values_after']}"

    def test_model_metrics_values(self):
        """모델 성능 지표 범위 검증"""
        result = self._load_result()
        for model_key in ("model_logistic", "model_ridge"):
            metrics = result[model_key]
            for metric_key in ("accuracy", "precision", "recall", "f1_macro"):
                assert metric_key in metrics, f"{model_key}.{metric_key} 없음"
                val = metrics[metric_key]
                assert 0.0 <= val <= 1.0, f"{model_key}.{metric_key} 범위 오류: {val}"
            assert metrics["accuracy"] >= 0.7, f"{model_key} accuracy 너무 낮음: {metrics['accuracy']}"
