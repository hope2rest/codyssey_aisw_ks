"""
Q1 전력 수요 예측 — pytest 검증 (16개 테스트)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: data_loader.py, preprocessor.py, feature_engineer.py, model.py, main.py, result_q1.json
"""
import ast
import importlib
import json
import os
import sys

import numpy as np
import pytest

_SUBMISSION_DIR = None
_MISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_MISSION_DIR, "data")


@pytest.fixture(autouse=True, scope="module")
def _configure(submission_dir):
    global _SUBMISSION_DIR
    _SUBMISSION_DIR = submission_dir


def _import_module(module_name):
    src_dir = os.path.join(_SUBMISSION_DIR, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _parse_ast(filename):
    path = os.path.join(_SUBMISSION_DIR, "src", filename)
    assert os.path.isfile(path), f"src/{filename} 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path), source


# ========================================================================
# TestStructure — 코드 구조 검증 (4개)
# ========================================================================

class TestStructure:
    def test_data_loader_functions(self):
        """data_loader.py에 필수 함수가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("data_loader.py")
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        required = {"load_power_data", "load_weather_data", "load_hourly_features"}
        missing = required - func_names
        assert not missing, f"data_loader.py 누락 함수: {missing}"

    def test_preprocessor_functions(self):
        """preprocessor.py에 필수 함수가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("preprocessor.py")
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        required = {"handle_missing", "detect_outliers_iqr", "convert_fahrenheit"}
        missing = required - func_names
        assert not missing, f"preprocessor.py 누락 함수: {missing}"

    def test_feature_engineer_functions(self):
        """feature_engineer.py에 필수 함수가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("feature_engineer.py")
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        required = {"add_lag_features", "add_rolling_features", "add_time_features", "add_holiday_flag"}
        missing = required - func_names
        assert not missing, f"feature_engineer.py 누락 함수: {missing}"

    def test_model_functions(self):
        """model.py에 필수 함수가 정의되어 있는지 확인"""
        tree, _ = _parse_ast("model.py")
        func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        required = {"split_time_series", "train_linear", "train_ridge", "evaluate", "compare_models"}
        missing = required - func_names
        assert not missing, f"model.py 누락 함수: {missing}"


# ========================================================================
# TestDataLoader — 데이터 로드 검증 (2개)
# ========================================================================

class TestDataLoader:
    def test_load_power_data(self):
        """전력 데이터 로드 검증"""
        mod = _import_module("data_loader")
        power = mod.load_power_data([
            os.path.join(_DATA_DIR, "power_hourly_2023.csv"),
            os.path.join(_DATA_DIR, "power_hourly_2024.csv"),
        ])
        assert power is not None, "load_power_data가 None 반환"
        assert len(power) > 17000, f"행 수 부족: {len(power)}"

    def test_load_hourly_features(self):
        """시간별 기온/습도 로드 및 화씨→섭씨 변환 검증"""
        mod = _import_module("data_loader")
        feat = mod.load_hourly_features(
            os.path.join(_DATA_DIR, "temperature_hourly.tsv"),
            os.path.join(_DATA_DIR, "humidity_hourly.tsv"),
        )
        assert feat is not None, "load_hourly_features가 None 반환"
        assert len(feat) > 17000, f"행 수 부족: {len(feat)}"
        # 섭씨 범위 확인 (한국 기후: -20 ~ 45)
        temps = np.array([float(row[1]) for row in feat[:100]], dtype=np.float64)
        assert np.all(temps > -30) and np.all(temps < 50), "온도 변환 오류"


# ========================================================================
# TestPreprocessor — 전처리 검증 (2개)
# ========================================================================

class TestPreprocessor:
    def test_detect_outliers_iqr(self):
        """IQR 이상치 탐지 검증"""
        mod = _import_module("preprocessor")
        data = np.array([[10], [20], [30], [40], [50], [200]], dtype=np.float64)
        outliers = mod.detect_outliers_iqr(data, 0)
        assert isinstance(outliers, list), "반환값이 list가 아닙니다"
        assert 5 in outliers, "이상치(200)를 감지하지 못함"
        assert 0 not in outliers, "정상값(10)을 이상치로 판정"

    def test_convert_fahrenheit(self):
        """화씨→섭씨 변환 검증"""
        mod = _import_module("preprocessor")
        f_temps = np.array([32.0, 212.0, 77.0])
        c_temps = mod.convert_fahrenheit(f_temps)
        np.testing.assert_allclose(c_temps, [0.0, 100.0, 25.0], atol=0.01)


# ========================================================================
# TestFeatureEngineer — 피처 생성 검증 (3개)
# ========================================================================

class TestFeatureEngineer:
    def test_lag_features(self):
        """시차 피처 생성 검증"""
        mod = _import_module("feature_engineer")
        demand = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        lags = [1, 2]
        result = mod.add_lag_features(demand, lags)
        assert result.shape == (5, 2), f"shape 오류: {result.shape}"
        assert result[2, 0] == 20.0, f"lag-1 오류: {result[2, 0]}"
        assert result[2, 1] == 10.0, f"lag-2 오류: {result[2, 1]}"
        assert result[0, 0] == 0.0, "초기값이 0이 아님"

    def test_rolling_features(self):
        """이동 평균 피처 생성 검증"""
        mod = _import_module("feature_engineer")
        demand = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        result = mod.add_rolling_features(demand, [3])
        assert result.shape == (5, 1), f"shape 오류: {result.shape}"
        assert abs(result[2, 0] - 20.0) < 0.01, f"이동평균 오류: {result[2, 0]}"

    def test_time_features(self):
        """시간 피처 추출 검증"""
        mod = _import_module("feature_engineer")
        datetimes = ["2023-01-01 08:00:00", "2023-01-07 15:00:00"]
        result = mod.add_time_features(datetimes)
        assert result.shape == (2, 4), f"shape 오류: {result.shape}"
        assert result[0, 0] == 8, "hour 오류"
        assert result[0, 1] == 6, "day_of_week 오류 (일요일=6)"
        assert result[0, 2] == 1, "month 오류"
        assert result[0, 3] == 1.0, "is_weekend 오류 (일요일)"


# ========================================================================
# TestModel — 모델 학습/평가 검증 (2개)
# ========================================================================

class TestModel:
    def test_train_and_evaluate(self):
        """선형 회귀 학습 및 평가 검증"""
        mod = _import_module("model")
        np.random.seed(42)
        X = np.random.randn(100, 3)
        w_true = np.array([2.0, -1.0, 0.5])
        y = X @ w_true + 5.0 + np.random.randn(100) * 0.1

        X_train, X_test, y_train, y_test = mod.split_time_series(X, y, test_ratio=0.2)
        assert len(X_train) == 80, f"train 크기 오류: {len(X_train)}"
        assert len(X_test) == 20, f"test 크기 오류: {len(X_test)}"

        weights, bias = mod.train_linear(X_train, y_train)
        pred = mod.predict(X_test, weights, bias)
        metrics = mod.evaluate(y_test, pred)

        assert "mae" in metrics, "mae 키 없음"
        assert "rmse" in metrics, "rmse 키 없음"
        assert "r_squared" in metrics, "r_squared 키 없음"
        assert metrics["r_squared"] > 0.9, f"R² 너무 낮음: {metrics['r_squared']}"

    def test_ridge_regression(self):
        """Ridge 회귀 학습 검증"""
        mod = _import_module("model")
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(50) * 0.5

        weights, bias = mod.train_ridge(X, y, alpha=1.0)
        assert len(weights) == 3, f"weights 길이 오류: {len(weights)}"
        pred = mod.predict(X, weights, bias)
        metrics = mod.evaluate(y, pred)
        assert metrics["r_squared"] > 0.8, f"Ridge R² 너무 낮음: {metrics['r_squared']}"


# ========================================================================
# TestResult — result_q1.json 결과 검증 (3개)
# ========================================================================

class TestResult:
    @staticmethod
    def _load_result():
        path = os.path.join(_SUBMISSION_DIR, "output", "result_q1.json")
        assert os.path.isfile(path), f"output/result_q1.json 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_result_structure(self):
        """result_q1.json 필수 키 확인"""
        result = self._load_result()
        required_keys = ["data_summary", "model_linear", "model_ridge", "best_model", "feature_count"]
        for key in required_keys:
            assert key in result, f"result_q1.json에 '{key}' 없음"
        summary = result["data_summary"]
        for key in ["total_files_loaded", "total_rows_merged", "missing_values_filled", "outliers_removed"]:
            assert key in summary, f"data_summary에 '{key}' 없음"

    def test_model_performance(self):
        """모델 성능 지표 범위 검증"""
        result = self._load_result()
        for model_key in ("model_linear", "model_ridge"):
            metrics = result[model_key]
            assert "mae" in metrics, f"{model_key}.mae 없음"
            assert "rmse" in metrics, f"{model_key}.rmse 없음"
            assert "r_squared" in metrics, f"{model_key}.r_squared 없음"
            assert metrics["r_squared"] > 0.3, f"{model_key} R² 너무 낮음: {metrics['r_squared']}"
            assert metrics["mae"] > 0, f"{model_key} MAE가 0 이하"

    def test_data_summary(self):
        """데이터 요약 정량적 값 검증"""
        result = self._load_result()
        summary = result["data_summary"]
        assert summary["total_files_loaded"] == 10, f"파일 수 오류: {summary['total_files_loaded']}"
        assert summary["total_rows_merged"] > 17000, f"행 수 부족: {summary['total_rows_merged']}"
        assert summary["missing_values_filled"] > 0, "결측 보간 수가 0"
        assert summary["feature_count"] > 5, f"피처 수 부족: {summary['feature_count']}"
