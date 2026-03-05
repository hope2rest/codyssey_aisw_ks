"""MAC 연산 심화 분석 — pytest 검증 (14개 테스트, 전체 통과 시 합격)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: mac_analyzer.py (1파일)
"""
import ast
import importlib.util
import os
import re
import sys

import pytest

# ─── 모듈 레벨 변수 ───

_SUBMISSION_DIR = None


@pytest.fixture(autouse=True, scope="module")
def _configure(submission_dir):
    """submission_dir fixture로 모듈 경로 설정"""
    global _SUBMISSION_DIR
    _SUBMISSION_DIR = submission_dir


# ─── 공통 헬퍼 ───


def _load_module():
    """제출물 mac_analyzer.py를 동적 import"""
    path = os.path.join(_SUBMISSION_DIR, "mac_analyzer.py")
    assert os.path.isfile(path), f"mac_analyzer.py 파일 없음: {path}"
    spec = importlib.util.spec_from_file_location("mac_analyzer", path)
    mod = importlib.util.module_from_spec(spec)
    if "mac_analyzer" in sys.modules:
        del sys.modules["mac_analyzer"]
    spec.loader.exec_module(mod)
    return mod


def _parse_ast():
    """제출물 mac_analyzer.py를 AST로 파싱"""
    path = os.path.join(_SUBMISSION_DIR, "mac_analyzer.py")
    assert os.path.isfile(path), f"mac_analyzer.py 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path)


# ========================================================================
# TestStructure — 코드 구조 검증 (2개)
# ========================================================================


class TestStructure:
    """코드 구조 검증"""

    def test_functions_exist(self):
        """필수 함수 12개 정의 확인 (5점)"""
        tree = _parse_ast()
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {
            "load_data", "mac", "normalize_labels", "is_close",
            "find_best_match", "measure_mac_time", "analyze_complexity",
            "diagnose_failure", "get_mac_explanation",
            "get_normalization_reason", "get_epsilon_reason", "main",
        }
        missing = required - func_names
        assert not missing, f"누락된 함수: {missing}"

    def test_no_external_lib(self):
        """json, time 외 외부 라이브러리 미사용 확인 (5점)"""
        tree = _parse_ast()
        allowed = {"json", "time"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name in allowed, (
                        f"허용되지 않은 import: {alias.name}"
                    )
            if isinstance(node, ast.ImportFrom):
                assert node.module in allowed, (
                    f"허용되지 않은 import: from {node.module}"
                )


# ========================================================================
# TestMAC — MAC 연산 및 매칭 검증 (3개)
# ========================================================================


class TestMAC:
    """MAC 연산 기능 검증"""

    def test_mac_basic(self):
        """정수 MAC 연산 (5점)"""
        mod = _load_module()
        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        b = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        assert mod.mac(a, b) == 5, f"기대: 5, 결과: {mod.mac(a, b)}"

    def test_mac_floats(self):
        """부동소수점 MAC 연산 (5점)"""
        mod = _load_module()
        a = [[0.5, 0.5], [0.5, 0.5]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        result = mod.mac(a, b)
        assert abs(result - 1.0) < 1e-9, f"기대: 1.0, 결과: {result}"

    def test_find_best_match(self):
        """최적 필터 매칭 (5점)"""
        mod = _load_module()
        pattern = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
        filters = {
            "cross": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            "block": [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        }
        assert mod.find_best_match(pattern, filters) == "block", (
            "block 필터가 최적 매칭이어야 합니다"
        )


# ========================================================================
# TestNormalize — 라벨 정규화 및 epsilon 비교 (2개)
# ========================================================================


class TestNormalize:
    """라벨 정규화 및 부동소수점 비교 검증"""

    def test_normalize_labels(self):
        """대소문자 섞인 키를 소문자로 통일 (5점)"""
        mod = _load_module()
        labels = {"IMG_01": "cross", "Img_02": "block"}
        result = mod.normalize_labels(labels)
        assert result == {"img_01": "cross", "img_02": "block"}, (
            f"정규화 결과 오류: {result}"
        )

    def test_is_close(self):
        """epsilon 기반 부동소수점 비교 (5점)"""
        mod = _load_module()
        assert mod.is_close(0.1 + 0.2, 0.3) is True, (
            "0.1+0.2와 0.3은 is_close로 True여야 합니다"
        )
        assert mod.is_close(1.0, 2.0) is False, (
            "1.0과 2.0은 is_close로 False여야 합니다"
        )


# ========================================================================
# TestBenchmark — 성능 벤치마크 및 복잡도 분석 (2개)
# ========================================================================


class TestBenchmark:
    """성능 측정 및 시간 복잡도 분석 검증"""

    def test_measure_mac_time(self):
        """MAC 연산 시간 측정 (10점)"""
        mod = _load_module()
        t = mod.measure_mac_time(4, repeat=3)
        assert isinstance(t, float), f"반환 타입 오류: {type(t)}"
        assert t >= 0, f"음수 시간: {t}"

    def test_analyze_complexity(self):
        """시간 복잡도 분석 구조 (10점)"""
        mod = _load_module()
        sizes = [1, 2, 4]
        times = [0.001, 0.004, 0.016]
        result = mod.analyze_complexity(sizes, times)
        assert "size_pairs" in result, "size_pairs 키 없음"
        assert "time_ratios" in result, "time_ratios 키 없음"
        assert "estimated_order" in result, "estimated_order 키 없음"
        assert result["estimated_order"] == "O(N^2)", (
            f"기대: O(N^2), 결과: {result['estimated_order']}"
        )
        assert len(result["time_ratios"]) == len(sizes) - 1, (
            "time_ratios 길이 오류"
        )


# ========================================================================
# TestDiagnosis — 실패 진단 (1개)
# ========================================================================


class TestDiagnosis:
    """실패 원인 분류 검증"""

    def test_diagnose_failure(self):
        """실패 원인 3가지 분류 (10점)"""
        mod = _load_module()

        # 데이터/스키마 문제: 존재하지 않는 필터
        result = mod.diagnose_failure(
            {"cross": 5}, "unknown", "cross_pattern", ["cross", "block"]
        )
        assert result["category"] == "data_schema", (
            f"키 누락 → data_schema 기대, 결과: {result['category']}"
        )

        # 수치 비교 문제: 점수 차이가 epsilon 미만
        result = mod.diagnose_failure(
            {"cross": 5.0000001, "block": 5.0000000},
            "cross", "block_pattern", ["cross", "block"]
        )
        assert result["category"] == "numerical", (
            f"미세 차이 → numerical 기대, 결과: {result['category']}"
        )

        # 로직 문제: 점수 차이가 명확
        result = mod.diagnose_failure(
            {"cross": 10, "block": 2},
            "cross", "block_pattern", ["cross", "block"]
        )
        assert result["category"] == "logic", (
            f"명확한 차이 → logic 기대, 결과: {result['category']}"
        )


# ========================================================================
# TestExplanations — 개념 설명 검증 (3개)
# ========================================================================


class TestExplanations:
    """개념 설명 검증"""

    def test_mac_explanation(self):
        """MAC 설명 (한국어, 50자 이상) (5점)"""
        mod = _load_module()
        text = mod.get_mac_explanation()
        assert len(text) >= 50, f"길이: {len(text)}자 (50자 이상 필요)"
        assert re.search(r"[가-힣]", text), "한국어 미포함"

    def test_normalization_reason(self):
        """정규화 필요성 설명 (한국어, 30자 이상) (5점)"""
        mod = _load_module()
        text = mod.get_normalization_reason()
        assert len(text) >= 30, f"길이: {len(text)}자 (30자 이상 필요)"
        assert re.search(r"[가-힣]", text), "한국어 미포함"

    def test_epsilon_reason(self):
        """epsilon 비교 필요성 설명 (한국어, 30자 이상) (5점)"""
        mod = _load_module()
        text = mod.get_epsilon_reason()
        assert len(text) >= 30, f"길이: {len(text)}자 (30자 이상 필요)"
        assert re.search(r"[가-힣]", text), "한국어 미포함"


# ========================================================================
# TestMain — 전체 파이프라인 검증 (1개)
# ========================================================================


class TestMain:
    """main() 전체 파이프라인 검증"""

    def test_main_result(self, data_path):
        """전체 파이프라인 실행 결과 (10점)"""
        mod = _load_module()
        result = mod.main(data_path)

        # scores 검증
        assert result["scores"]["img_01"]["cross"] == 5
        assert result["scores"]["img_02"]["block"] == 4
        assert result["scores"]["img_03"]["line"] == 3
        assert abs(result["scores"]["img_04"]["block"] - 2.0) < 1e-9

        # best_matches 검증
        assert result["best_matches"]["img_01"] == "cross"
        assert result["best_matches"]["img_02"] == "block"
        assert result["best_matches"]["img_03"] == "line"
        assert result["best_matches"]["img_04"] == "block"

        # labels 정규화 검증
        assert result["labels"]["img_01"] == "cross_pattern"
        assert result["labels"]["img_04"] == "block_pattern"

        # benchmarks 구조 검증
        bm = result["benchmarks"]
        assert len(bm["sizes"]) > 0, "sizes 비어 있음"
        assert len(bm["times"]) == len(bm["sizes"]), "times 길이 불일치"
        assert "complexity_analysis" in bm, "complexity_analysis 없음"
        assert bm["complexity_analysis"]["estimated_order"] == "O(N^2)"

        # diagnosis 구조 검증
        assert "diagnosis" in result, "diagnosis 없음"
        for pat_name in ("img_01", "img_02", "img_03", "img_04"):
            assert pat_name in result["diagnosis"], f"{pat_name} 진단 없음"
            assert "category" in result["diagnosis"][pat_name]
            assert "reason" in result["diagnosis"][pat_name]

        # explanations 구조 검증
        exp = result["explanations"]
        assert len(exp["mac_explanation"]) >= 50
        assert len(exp["normalization_reason"]) >= 30
        assert len(exp["epsilon_reason"]) >= 30
