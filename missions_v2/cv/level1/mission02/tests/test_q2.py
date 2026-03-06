"""
Q2 도서 검색 및 추천 서비스 — pytest 검증 (30개 테스트)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: core/ 폴더 구조 (search_engine.py, recommender.py, sentiment.py, main.py)
        + dashboard/, charts/, output/result_q2.json, output/charts/
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
    """core/ 하위 모듈을 import한다."""
    core_dir = os.path.join(_SUBMISSION_DIR, "core")
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    # Clean up any cached imports
    for key in list(sys.modules.keys()):
        if key == module_name or key.startswith(module_name + "."):
            del sys.modules[key]
    return importlib.import_module(module_name)


def _parse_ast(filename, subdir="core"):
    path = os.path.join(_SUBMISSION_DIR, subdir, filename)
    assert os.path.isfile(path), f"{filename} 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path), source


# ========================================================================
# TestStructure — 코드 구조 검증
# ========================================================================

class TestStructure:
    def test_required_files(self):
        """필수 파일 존재 확인"""
        required_files = [
            os.path.join("core", "search_engine.py"),
            os.path.join("core", "recommender.py"),
            os.path.join("core", "sentiment.py"),
            os.path.join("core", "main.py"),
        ]
        for f in required_files:
            path = os.path.join(_SUBMISSION_DIR, f)
            assert os.path.isfile(path), f"필수 파일 없음: {f}"

    def test_search_engine_functions(self):
        """search_engine.py 필수 함수 정의 확인"""
        tree, _ = _parse_ast("search_engine.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {
            "preprocess", "compute_tf", "compute_idf",
            "build_tfidf_matrix", "cosine_similarity", "search"
        }
        missing = required - func_names
        assert not missing, f"search_engine.py 누락 함수: {missing}"

    def test_recommender_functions(self):
        """recommender.py 필수 함수 정의 확인"""
        tree, _ = _parse_ast("recommender.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"compute_book_similarity", "recommend_books", "recommend_by_category"}
        missing = required - func_names
        assert not missing, f"recommender.py 누락 함수: {missing}"

    def test_sentiment_functions(self):
        """sentiment.py 필수 함수 정의 확인"""
        tree, _ = _parse_ast("sentiment.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"rule_based_predict", "compute_metrics"}
        missing = required - func_names
        assert not missing, f"sentiment.py 누락 함수: {missing}"

    def test_no_sklearn(self):
        """sklearn/scipy 사용 금지 확인"""
        for filename in ["search_engine.py", "recommender.py", "sentiment.py", "main.py"]:
            _, source = _parse_ast(filename)
            for lib in ["sklearn", "scipy"]:
                assert lib not in source, f"{filename}에서 {lib} 사용 감지 — NumPy만 허용"


# ========================================================================
# TestPreprocess — 전처리 검증
# ========================================================================

class TestPreprocess:
    def test_basic_preprocess(self):
        """기본 전처리 기능 검증"""
        mod = _import_module("search_engine")
        stopwords = {"이", "그", "한"}
        tokens = mod.preprocess("딥러닝 모델은 이 데이터를 학습한다!", stopwords)
        assert isinstance(tokens, list), "반환값이 list가 아닙니다"
        assert "이" not in tokens, "불용어 미제거"
        assert all(len(t) > 1 for t in tokens), "길이 1 이하 토큰 존재"

    def test_nfc_normalization(self):
        """NFC 정규화 검증"""
        mod = _import_module("search_engine")
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
        mod = _import_module("search_engine")
        a = np.array([1, 0, 0], dtype=np.float64)
        b = np.array([0, 1, 0], dtype=np.float64)
        assert abs(mod.cosine_similarity(a, b)) < 1e-9

    def test_cosine_similarity_identical(self):
        """동일 벡터의 코사인 유사도 = 1"""
        mod = _import_module("search_engine")
        a = np.array([1, 2, 3], dtype=np.float64)
        assert abs(mod.cosine_similarity(a, a) - 1.0) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        """영벡터 처리"""
        mod = _import_module("search_engine")
        a = np.array([1, 2, 3], dtype=np.float64)
        b = np.zeros(3, dtype=np.float64)
        assert mod.cosine_similarity(a, b) == 0.0


# ========================================================================
# TestRecommender — 추천 검증
# ========================================================================

class TestRecommender:
    def test_book_similarity_matrix_shape(self):
        """유사도 행렬 크기 검증"""
        mod_se = _import_module("search_engine")
        # recommender에서 cosine_similarity를 가져오므로 search_engine을 먼저 import
        mod_rec = _import_module("recommender")
        # 간단한 3x4 TF-IDF 행렬로 테스트
        tfidf = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float64)
        sim = mod_rec.compute_book_similarity(tfidf)
        assert sim.shape == (3, 3), f"유사도 행렬 크기 오류: {sim.shape}"
        # 대각선은 1.0
        for i in range(3):
            assert abs(sim[i, i] - 1.0) < 1e-9, f"대각선 값 오류: sim[{i},{i}]={sim[i,i]}"

    def test_recommend_books_excludes_self(self):
        """자기 자신 제외 검증"""
        mod_se = _import_module("search_engine")
        mod_rec = _import_module("recommender")
        tfidf = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        sim = mod_rec.compute_book_similarity(tfidf)
        metadata = [
            {"title": "A", "author": "x", "category": "cat1"},
            {"title": "B", "author": "y", "category": "cat2"},
            {"title": "C", "author": "z", "category": "cat1"},
        ]
        recs = mod_rec.recommend_books(0, sim, metadata, top_k=2)
        indices = [r["index"] for r in recs]
        assert 0 not in indices, "추천에 자기 자신이 포함됨"


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
        mod = _import_module("sentiment")
        sd = self._load_sentiment_dict()
        result = mod.rule_based_predict("정말 좋은 제품입니다 추천합니다", sd)
        assert result == 1, f"긍정 리뷰에서 0 반환"

    def test_negative_prediction(self):
        """부정 리뷰 예측"""
        mod = _import_module("sentiment")
        sd = self._load_sentiment_dict()
        result = mod.rule_based_predict("별로입니다 실망입니다", sd)
        assert result == 0, f"부정 리뷰에서 1 반환"

    def test_negation_handling(self):
        """부정어 처리 검증"""
        mod = _import_module("sentiment")
        sd = self._load_sentiment_dict()
        result = mod.rule_based_predict("안 좋다", sd)
        assert result == 0, "부정어 처리 실패: '안 좋다'가 긍정으로 판정"

    def test_metrics_computation(self):
        """감성 메트릭 계산 검증"""
        mod = _import_module("sentiment")
        preds = [1, 1, 0, 0, 1, 0]
        labels = [1, 0, 0, 1, 1, 0]
        metrics = mod.compute_metrics(preds, labels)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert abs(metrics["accuracy"] - 4 / 6) < 1e-4, f"Accuracy 오류: {metrics['accuracy']}"


# ========================================================================
# TestResult — result_q2.json 결과 검증
# ========================================================================

class TestResult:
    @staticmethod
    def _load_result():
        path = os.path.join(_SUBMISSION_DIR, "output", "result_q2.json")
        assert os.path.isfile(path), f"result_q2.json 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_result_structure(self):
        """result_q2.json 필수 키 확인"""
        result = self._load_result()
        required = [
            "vocab_size", "tfidf_matrix_shape", "search_results",
            "recommendation",
            "sentiment_accuracy", "sentiment_precision",
            "sentiment_recall", "sentiment_f1",
            "total_reviews", "positive_count", "negative_count"
        ]
        for key in required:
            assert key in result, f"'{key}' 키 없음"

    def test_result_values(self):
        """result_q2.json 정량적 검증"""
        result = self._load_result()
        assert result["tfidf_matrix_shape"][0] == 30, \
            f"문서 수: {result['tfidf_matrix_shape'][0]} (기대: 30)"
        assert result["total_reviews"] == 40, \
            f"total_reviews: {result['total_reviews']} (기대: 40)"
        assert len(result["search_results"]) == 5, \
            f"search_results 개수: {len(result['search_results'])} (기대: 5)"
        assert result["sentiment_accuracy"] >= 0.7, \
            f"sentiment_accuracy 너무 낮음: {result['sentiment_accuracy']}"

    def test_result_recommendation(self):
        """result_q2.json 추천 결과 구조 확인"""
        result = self._load_result()
        rec = result["recommendation"]
        assert "target_book" in rec, "target_book 키 없음"
        assert "top5_similar" in rec, "top5_similar 키 없음"
        assert "same_category_top3" in rec, "same_category_top3 키 없음"
        assert len(rec["top5_similar"]) == 5, \
            f"top5_similar 개수: {len(rec['top5_similar'])} (기대: 5)"
        assert len(rec["same_category_top3"]) == 3, \
            f"same_category_top3 개수: {len(rec['same_category_top3'])} (기대: 3)"
        # 각 추천 항목에 필수 키 확인
        for item in rec["top5_similar"]:
            assert "index" in item and "title" in item and "similarity" in item
        for item in rec["same_category_top3"]:
            assert "index" in item and "title" in item and "similarity" in item

    def test_search_results_have_titles(self):
        """검색 결과에 title 포함 확인"""
        result = self._load_result()
        for sr in result["search_results"]:
            assert "query" in sr and "top3" in sr
            for item in sr["top3"]:
                assert "doc_index" in item and "similarity" in item and "title" in item


# ========================================================================
# TestDashboardStructure — 대시보드 폴더 구조 검증
# ========================================================================

class TestDashboardStructure:
    def test_dashboard_folder_exists(self):
        """dashboard/ 폴더 존재 확인"""
        dashboard_dir = os.path.join(_SUBMISSION_DIR, "dashboard")
        assert os.path.isdir(dashboard_dir), f"dashboard/ 폴더 없음: {dashboard_dir}"

    def test_charts_folder_exists(self):
        """charts/ 폴더 존재 확인"""
        charts_dir = os.path.join(_SUBMISSION_DIR, "charts")
        assert os.path.isdir(charts_dir), f"charts/ 폴더 없음: {charts_dir}"

    def test_dashboard_app_exists(self):
        """dashboard/app.py 존재 확인"""
        app_path = os.path.join(_SUBMISSION_DIR, "dashboard", "app.py")
        assert os.path.isfile(app_path), f"dashboard/app.py 없음: {app_path}"

    def test_dashboard_pages_exist(self):
        """dashboard/pages/ 필수 파일 존재 확인"""
        pages_dir = os.path.join(_SUBMISSION_DIR, "dashboard", "pages")
        assert os.path.isdir(pages_dir), f"dashboard/pages/ 폴더 없음"
        for page in ["search.py", "recommend.py", "sentiment.py"]:
            path = os.path.join(pages_dir, page)
            assert os.path.isfile(path), f"dashboard/pages/{page} 없음"

    def test_dashboard_components_exist(self):
        """dashboard/components/ 필수 파일 존재 확인"""
        comp_dir = os.path.join(_SUBMISSION_DIR, "dashboard", "components")
        assert os.path.isdir(comp_dir), f"dashboard/components/ 폴더 없음"
        for comp in ["search_bar.py", "book_card.py", "chart_builder.py"]:
            path = os.path.join(comp_dir, comp)
            assert os.path.isfile(path), f"dashboard/components/{comp} 없음"

    def test_charts_modules_exist(self):
        """charts/ 필수 모듈 존재 확인"""
        charts_dir = os.path.join(_SUBMISSION_DIR, "charts")
        for module in ["search_charts.py", "recommend_charts.py", "sentiment_charts.py"]:
            path = os.path.join(charts_dir, module)
            assert os.path.isfile(path), f"charts/{module} 없음"


# ========================================================================
# TestCharts — 차트 생성 기능 검증
# ========================================================================

class TestCharts:
    def _import_chart_module(self, module_name):
        charts_dir = os.path.join(_SUBMISSION_DIR, "charts")
        if charts_dir not in sys.path:
            sys.path.insert(0, charts_dir)
        if module_name in sys.modules:
            del sys.modules[module_name]
        return importlib.import_module(module_name)

    def test_search_charts_functions(self):
        """search_charts.py 필수 함수 확인"""
        tree, _ = _parse_ast("search_charts.py", subdir="charts")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "save_search_results_chart" in func_names, "save_search_results_chart 함수 없음"

    def test_recommend_charts_functions(self):
        """recommend_charts.py 필수 함수 확인"""
        tree, _ = _parse_ast("recommend_charts.py", subdir="charts")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "save_similarity_heatmap" in func_names, "save_similarity_heatmap 함수 없음"
        assert "save_recommendation_chart" in func_names, "save_recommendation_chart 함수 없음"

    def test_sentiment_charts_functions(self):
        """sentiment_charts.py 필수 함수 확인"""
        tree, _ = _parse_ast("sentiment_charts.py", subdir="charts")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        assert "save_sentiment_distribution" in func_names, "save_sentiment_distribution 함수 없음"
        assert "save_sentiment_metrics_chart" in func_names, "save_sentiment_metrics_chart 함수 없음"

    def test_chart_generation(self, tmp_path):
        """차트 생성 기능 통합 테스트"""
        search_mod = self._import_chart_module("search_charts")
        sentiment_mod = self._import_chart_module("sentiment_charts")

        # 검색 차트 생성 테스트
        search_results = [
            {"query": "테스트", "top3": [
                {"title": "도서A", "similarity": 0.8, "doc_index": 0},
                {"title": "도서B", "similarity": 0.5, "doc_index": 1},
                {"title": "도서C", "similarity": 0.3, "doc_index": 2},
            ]}
        ]
        out = str(tmp_path / "search.png")
        search_mod.save_search_results_chart(search_results, out)
        assert os.path.isfile(out), "검색 결과 차트 생성 실패"
        assert os.path.getsize(out) > 0, "검색 결과 차트 파일이 비어 있음"

        # 감성 분포 차트 생성 테스트
        out2 = str(tmp_path / "sentiment.png")
        sentiment_mod.save_sentiment_distribution(25, 15, out2)
        assert os.path.isfile(out2), "감성 분포 차트 생성 실패"
        assert os.path.getsize(out2) > 0, "감성 분포 차트 파일이 비어 있음"
