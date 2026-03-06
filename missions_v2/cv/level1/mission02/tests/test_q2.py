"""
Q2 TF-IDF + 감성 분석 — pytest 검증 (13개 테스트)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: q2_solution.py, result_q2.json (2파일)
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


def _import_module(module_name="q2_solution"):
    if _SUBMISSION_DIR not in sys.path:
        sys.path.insert(0, _SUBMISSION_DIR)
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _parse_ast(filename="q2_solution.py"):
    path = os.path.join(_SUBMISSION_DIR, filename)
    assert os.path.isfile(path), f"{filename} 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path), source


# ========================================================================
# TestStructure — 코드 구조 검증
# ========================================================================

class TestStructure:
    def test_required_functions(self):
        """필수 함수 정의 확인"""
        tree, _ = _parse_ast()
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {
            "preprocess", "cosine_similarity", "search",
            "rule_based_predict", "compute_sentiment_metrics", "main"
        }
        missing = required - func_names
        assert not missing, f"누락 함수: {missing}"

    def test_no_sklearn(self):
        """sklearn/scipy 사용 금지 확인"""
        _, source = _parse_ast()
        for lib in ["sklearn", "scipy"]:
            assert lib not in source, f"{lib} 사용 감지 — NumPy만 허용"


# ========================================================================
# TestPreprocess — 전처리 검증
# ========================================================================

class TestPreprocess:
    def test_basic_preprocess(self):
        """기본 전처리 기능 검증"""
        mod = _import_module()
        stopwords = {"이", "그", "한"}
        tokens = mod.preprocess("딥러닝 모델은 이 데이터를 학습한다!", stopwords)
        assert isinstance(tokens, list), "반환값이 list가 아닙니다"
        assert "이" not in tokens, "불용어 미제거"
        assert all(len(t) > 1 for t in tokens), "길이 1 이하 토큰 존재"

    def test_nfc_normalization(self):
        """NFC 정규화 검증"""
        mod = _import_module()
        text1 = "\u1100\u1161\u11A8"  # 각 (decomposed)
        text2 = "\uAC01"  # 각 (composed)
        t1 = mod.preprocess(text1 + " 테스트", set())
        t2 = mod.preprocess(text2 + " 테스트", set())
        assert t1 == t2, f"NFC 정규화 미적용: {t1} != {t2}"


# ========================================================================
# TestTFIDF — TF-IDF 검증
# ========================================================================

class TestTFIDF:
    def test_cosine_similarity_orthogonal(self):
        """직교 벡터의 코사인 유사도 = 0"""
        mod = _import_module()
        a = np.array([1, 0, 0], dtype=np.float64)
        b = np.array([0, 1, 0], dtype=np.float64)
        assert abs(mod.cosine_similarity(a, b)) < 1e-9

    def test_cosine_similarity_identical(self):
        """동일 벡터의 코사인 유사도 = 1"""
        mod = _import_module()
        a = np.array([1, 2, 3], dtype=np.float64)
        assert abs(mod.cosine_similarity(a, a) - 1.0) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        """영벡터 처리"""
        mod = _import_module()
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        assert mod.cosine_similarity(a, b) == 0.0


# ========================================================================
# TestSentiment — 감성 분석 검증
# ========================================================================

class TestSentiment:
    @staticmethod
    def _load_sentiment_dict():
        with open(os.path.join(_DATA_DIR, "sentiment_dict.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_positive_prediction(self):
        """긍정 리뷰 예측"""
        mod = _import_module()
        sd = self._load_sentiment_dict()
        result = mod.rule_based_predict("정말 좋은 제품입니다 추천합니다", sd)
        assert result == 1, f"긍정 리뷰에서 0 반환"

    def test_negative_prediction(self):
        """부정 리뷰 예측"""
        mod = _import_module()
        sd = self._load_sentiment_dict()
        result = mod.rule_based_predict("별로입니다 실망입니다", sd)
        assert result == 0, f"부정 리뷰에서 1 반환"

    def test_negation_handling(self):
        """부정어 처리 검증"""
        mod = _import_module()
        sd = self._load_sentiment_dict()
        # "안 좋다" → 좋다(1.0) * -1 = -1.0 → 부정
        result = mod.rule_based_predict("안 좋다", sd)
        assert result == 0, "부정어 처리 실패: '안 좋다'가 긍정으로 판정"

    def test_metrics_computation(self):
        """감성 메트릭 계산 검증"""
        mod = _import_module()
        preds = [1, 1, 0, 0, 1, 0]
        labels = [1, 0, 0, 1, 1, 0]
        metrics = mod.compute_sentiment_metrics(preds, labels)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert abs(metrics["accuracy"] - 4 / 6) < 1e-4, f"Accuracy 오류: {metrics['accuracy']}"


# ========================================================================
# TestResult — result_q2.json 결과 검증
# ========================================================================

class TestResult:
    @staticmethod
    def _load_result():
        path = os.path.join(_SUBMISSION_DIR, "result_q2.json")
        assert os.path.isfile(path), f"result_q2.json 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_result_structure(self):
        """result_q2.json 필수 키 확인"""
        result = self._load_result()
        required = [
            "vocab_size", "tfidf_matrix_shape", "search_results",
            "sentiment_accuracy", "sentiment_precision",
            "sentiment_recall", "sentiment_f1",
            "total_reviews", "positive_count", "negative_count"
        ]
        for key in required:
            assert key in result, f"'{key}' 키 없음"

    def test_result_values(self):
        """result_q2.json 정량적 검증"""
        result = self._load_result()
        assert result["vocab_size"] == 290, \
            f"vocab_size: {result['vocab_size']} (기대: 290)"
        assert result["tfidf_matrix_shape"] == [20, 290], \
            f"tfidf_matrix_shape: {result['tfidf_matrix_shape']}"
        assert result["total_reviews"] == 30, \
            f"total_reviews: {result['total_reviews']} (기대: 30)"
        assert len(result["search_results"]) == 5, \
            f"search_results 개수: {len(result['search_results'])} (기대: 5)"
        assert result["sentiment_accuracy"] >= 0.7, \
            f"sentiment_accuracy 너무 낮음: {result['sentiment_accuracy']}"
