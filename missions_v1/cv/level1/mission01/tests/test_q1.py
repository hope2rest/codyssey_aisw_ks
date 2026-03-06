"""
Q1 이커머스 데이터 전처리 — pytest 검증 (12개 테스트)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: q1_solution.py, result_q1.json (2파일)
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


def _import_module(module_name="q1_solution"):
    if _SUBMISSION_DIR not in sys.path:
        sys.path.insert(0, _SUBMISSION_DIR)
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _parse_ast(filename="q1_solution.py"):
    path = os.path.join(_SUBMISSION_DIR, filename)
    assert os.path.isfile(path), f"{filename} 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path), source


# ========================================================================
# TestStructure — 코드 구조 검증 (2개)
# ========================================================================

class TestStructure:
    def test_required_functions(self):
        """q1_solution.py에 필수 함수 7개가 정의되어 있는지 확인"""
        tree, _ = _parse_ast()
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {
            "load_and_clean", "compute_statistics",
            "detect_outliers_iqr", "detect_outliers_zscore",
            "standardize", "segment_customers", "main"
        }
        missing = required - func_names
        assert not missing, f"누락 함수: {missing}"

    def test_no_sklearn(self):
        """sklearn/scipy/pandas 사용 금지 확인"""
        _, source = _parse_ast()
        for lib in ["sklearn", "scipy", "pandas"]:
            assert lib not in source, f"{lib} 사용 감지 — NumPy만 허용"


# ========================================================================
# TestLoadAndClean — 데이터 로드/정제 검증 (2개)
# ========================================================================

class TestLoadAndClean:
    def test_load_returns_correct_shape(self):
        """데이터 로드 후 올바른 형태 반환"""
        mod = _import_module()
        data_path = os.path.join(_DATA_DIR, "customers.csv")
        result = mod.load_and_clean(data_path)
        assert result is not None, "load_and_clean이 None 반환"
        data = result[0]
        columns = result[1]
        assert isinstance(data, np.ndarray), "반환값이 ndarray가 아닙니다"
        assert data.ndim == 2, f"2D 배열이 아닙니다: {data.ndim}D"
        assert data.shape[1] == 7, f"열 수 오류: {data.shape[1]} (기대: 7)"
        assert len(columns) == 7, f"열 이름 수 오류: {len(columns)}"

    def test_no_nan_after_cleaning(self):
        """정제 후 NaN이 없는지 확인"""
        mod = _import_module()
        data_path = os.path.join(_DATA_DIR, "customers.csv")
        result = mod.load_and_clean(data_path)
        data = result[0]
        assert not np.isnan(data).any(), "정제 후에도 NaN 존재"


# ========================================================================
# TestStatistics — 통계량 검증 (2개)
# ========================================================================

class TestStatistics:
    def test_statistics_keys(self):
        """통계량 결과에 필수 키가 포함되어 있는지 확인"""
        mod = _import_module()
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        cols = ["a", "b"]
        stats = mod.compute_statistics(data, cols)
        assert isinstance(stats, dict), "반환값이 dict가 아닙니다"
        for col in cols:
            assert col in stats, f"{col} 키 없음"
            for key in ("mean", "std", "min", "max", "median"):
                assert key in stats[col], f"{col}.{key} 키 없음"

    def test_statistics_values(self):
        """통계량 계산 정확도 검증"""
        mod = _import_module()
        data = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)
        cols = ["x", "y"]
        stats = mod.compute_statistics(data, cols)
        assert abs(stats["x"]["mean"] - 30.0) < 1e-4, f"mean 오류: {stats['x']['mean']}"
        assert abs(stats["x"]["std"] - np.std([10, 30, 50], ddof=0)) < 1e-4
        assert abs(stats["x"]["min"] - 10.0) < 1e-4
        assert abs(stats["x"]["max"] - 50.0) < 1e-4
        assert abs(stats["x"]["median"] - 30.0) < 1e-4


# ========================================================================
# TestOutliers — 이상치 탐지 검증 (2개)
# ========================================================================

class TestOutliers:
    def test_iqr_outliers(self):
        """IQR 이상치 탐지 검증"""
        mod = _import_module()
        # 정상: 10~90, 이상치: 200 (인덱스 5)
        data = np.array([[10], [20], [30], [40], [50], [200]], dtype=np.float64)
        outliers = mod.detect_outliers_iqr(data, 0)
        assert isinstance(outliers, list), "반환값이 list가 아닙니다"
        assert 5 in outliers, "이상치(200)를 감지하지 못함"
        assert 0 not in outliers, "정상값(10)을 이상치로 판정"

    def test_zscore_outliers(self):
        """Z-score 이상치 탐지 검증"""
        mod = _import_module()
        data = np.array([[10], [12], [11], [13], [10], [100]], dtype=np.float64)
        outliers = mod.detect_outliers_zscore(data, 0, threshold=2.0)
        assert isinstance(outliers, list), "반환값이 list가 아닙니다"
        assert 5 in outliers, "이상치(100)를 감지하지 못함"


# ========================================================================
# TestStandardize — 표준화 검증 (1개)
# ========================================================================

class TestStandardize:
    def test_standardize(self):
        """표준화 후 평균≈0, 표준편차≈1 검증"""
        mod = _import_module()
        data = np.random.randn(100, 5) * 10 + 50
        standardized = mod.standardize(data)
        assert isinstance(standardized, np.ndarray), "반환값이 ndarray가 아닙니다"
        for col in range(5):
            mean = np.mean(standardized[:, col])
            std = np.std(standardized[:, col], ddof=0)
            assert abs(mean) < 1e-10, f"열 {col} 평균: {mean}"
            assert abs(std - 1.0) < 1e-10, f"열 {col} 표준편차: {std}"


# ========================================================================
# TestSegments — 세그먼트 검증 (1개)
# ========================================================================

class TestSegments:
    def test_segment_counts(self):
        """세그먼트 분류 결과 검증"""
        mod = _import_module()
        data = np.array([
            [40, 6000, 80, 10, 30, 10, 300],
            [30, 3000, 20, 5, 20, 30, 100],
            [50, 7000, 30, 15, 40, 5, 600],
            [25, 2000, 90, 8, 25, 20, 200],
        ], dtype=np.float64)
        cols = ["age", "annual_income", "spending_score",
                "purchase_count", "avg_order_value", "days_since_last", "total_spent"]
        segments = mod.segment_customers(data, cols)
        assert isinstance(segments, dict), "반환값이 dict가 아닙니다"
        for seg in ("high_income_high_spend", "high_income_low_spend",
                     "low_income_high_spend", "low_income_low_spend"):
            assert seg in segments, f"{seg} 세그먼트 없음"
            assert "count" in segments[seg], f"{seg}.count 없음"
        total = sum(s["count"] for s in segments.values())
        assert total == 4, f"세그먼트 합계: {total} (기대: 4)"


# ========================================================================
# TestResult — result_q1.json 결과 검증 (2개)
# ========================================================================

class TestResult:
    @staticmethod
    def _load_result():
        path = os.path.join(_SUBMISSION_DIR, "result_q1.json")
        assert os.path.isfile(path), f"result_q1.json 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_result_structure(self):
        """result_q1.json 필수 키 확인"""
        result = self._load_result()
        required_keys = [
            "total_rows_raw", "total_rows_cleaned", "duplicates_removed",
            "missing_values_filled", "statistics", "outlier_counts_iqr",
            "outlier_counts_zscore", "standardized_mean_check",
            "standardized_std_check", "segments"
        ]
        for key in required_keys:
            assert key in result, f"result_q1.json에 '{key}' 없음"

    def test_result_values(self):
        """result_q1.json 정량적 값 검증"""
        result = self._load_result()
        assert result["total_rows_raw"] == 205, \
            f"total_rows_raw: {result['total_rows_raw']} (기대: 205)"
        assert result["total_rows_cleaned"] == 200, \
            f"total_rows_cleaned: {result['total_rows_cleaned']} (기대: 200)"
        assert result["duplicates_removed"] == 5, \
            f"duplicates_removed: {result['duplicates_removed']} (기대: 5)"
        assert result["missing_values_filled"] == 15, \
            f"missing_values_filled: {result['missing_values_filled']} (기대: 15)"

        # 표준화 검증: 모든 열의 평균 ≈ 0
        for col, val in result["standardized_mean_check"].items():
            assert abs(val) < 1e-4, f"{col} 표준화 평균: {val}"

        # 표준화 검증: 모든 열의 표준편차 ≈ 1
        for col, val in result["standardized_std_check"].items():
            assert abs(val - 1.0) < 1e-4, f"{col} 표준화 표준편차: {val}"

        # 세그먼트 합계 = total_rows_cleaned
        total_seg = sum(s["count"] for s in result["segments"].values())
        assert total_seg == result["total_rows_cleaned"], \
            f"세그먼트 합계: {total_seg} != {result['total_rows_cleaned']}"
