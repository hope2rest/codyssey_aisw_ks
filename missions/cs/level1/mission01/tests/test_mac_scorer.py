"""MAC 연산 기반 패턴 매칭 — pytest 검증 (10개 테스트)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: mac_scorer.py (1파일)
"""
import ast
import importlib.util
import os
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
    """제출물 mac_scorer.py를 동적 import"""
    path = os.path.join(_SUBMISSION_DIR, "mac_scorer.py")
    assert os.path.isfile(path), f"mac_scorer.py 파일 없음: {path}"
    spec = importlib.util.spec_from_file_location("mac_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    if "mac_scorer" in sys.modules:
        del sys.modules["mac_scorer"]
    spec.loader.exec_module(mod)
    return mod


def _parse_ast():
    """제출물 mac_scorer.py를 AST로 파싱"""
    path = os.path.join(_SUBMISSION_DIR, "mac_scorer.py")
    assert os.path.isfile(path), f"mac_scorer.py 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path)


# ========================================================================
# TestStructure (AST 분석형) — 2개
# ========================================================================


class TestStructure:
    """코드 구조 검증"""

    def test_functions_exist(self):
        """필수 함수 6개가 정의되어 있는지 확인 (10점)"""
        tree = _parse_ast()
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"load_data", "mac", "normalize_labels",
                     "is_close", "find_best_match", "main"}
        missing = required - func_names
        assert not missing, f"누락된 함수: {missing}"

    def test_no_external_lib(self):
        """json 외 외부 라이브러리를 import하지 않는지 확인 (10점)"""
        tree = _parse_ast()
        allowed = {"json"}
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
# TestMAC (MAC 연산 검증) — 3개
# ========================================================================


class TestMAC:
    """MAC 연산 기능 검증"""

    def test_mac_basic(self):
        """정수 2D 배열 MAC 연산 (10점)"""
        mod = _load_module()
        a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        b = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        # 동일 패턴: 1+0+1+0+1+0+1+0+1 = 5
        assert mod.mac(a, b) == 5, f"기대: 5, 결과: {mod.mac(a, b)}"

    def test_mac_different(self):
        """서로 다른 패턴의 MAC 연산 (10점)"""
        mod = _load_module()
        a = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
        b = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        # 1*1+1*0+0*1+1*0+1*1+0*0+0*1+0*0+0*1 = 2
        assert mod.mac(a, b) == 2, f"기대: 2, 결과: {mod.mac(a, b)}"

    def test_mac_floats(self):
        """부동소수점 배열 MAC 연산 (10점)"""
        mod = _load_module()
        a = [[0.5, 0.5], [0.5, 0.5]]
        b = [[1.0, 0.0], [0.0, 1.0]]
        # 0.5*1.0 + 0.5*0.0 + 0.5*0.0 + 0.5*1.0 = 1.0
        result = mod.mac(a, b)
        assert abs(result - 1.0) < 1e-9, f"기대: 1.0, 결과: {result}"


# ========================================================================
# TestNormalize (라벨 정규화) — 1개
# ========================================================================


class TestNormalize:
    """라벨 키 정규화 검증"""

    def test_normalize_labels(self):
        """대소문자 섞인 키를 소문자로 통일 (10점)"""
        mod = _load_module()
        labels = {"IMG_01": "cross", "Img_02": "block", "img_03": "line"}
        result = mod.normalize_labels(labels)
        assert result == {"img_01": "cross", "img_02": "block", "img_03": "line"}, (
            f"정규화 결과 오류: {result}"
        )


# ========================================================================
# TestFloatingPoint (부동소수점 비교) — 2개
# ========================================================================


class TestFloatingPoint:
    """epsilon 기반 부동소수점 비교 검증"""

    def test_is_close_true(self):
        """0.1+0.2 ≈ 0.3 비교 (10점)"""
        mod = _load_module()
        # 0.1 + 0.2 = 0.30000000000000004 (부동소수점 오차)
        assert mod.is_close(0.1 + 0.2, 0.3) is True, (
            "0.1+0.2와 0.3은 is_close로 True여야 합니다"
        )

    def test_is_close_false(self):
        """차이가 큰 두 수 비교 (10점)"""
        mod = _load_module()
        assert mod.is_close(1.0, 2.0) is False, (
            "1.0과 2.0은 is_close로 False여야 합니다"
        )


# ========================================================================
# TestMain (전체 파이프라인) — 1개
# ========================================================================


class TestMain:
    """main() 전체 파이프라인 검증"""

    def test_main_result(self, data_path):
        """전체 파이프라인 실행 결과 (10점)"""
        mod = _load_module()
        result = mod.main(data_path)

        # scores 검증
        assert result["scores"]["img_01"]["cross"] == 5
        assert result["scores"]["img_01"]["block"] == 2
        assert result["scores"]["img_01"]["line"] == 1
        assert result["scores"]["img_02"]["block"] == 4
        assert result["scores"]["img_03"]["line"] == 3

        # best_matches 검증
        assert result["best_matches"]["img_01"] == "cross", (
            f"img_01 최적 매칭: 기대 cross, 결과 {result['best_matches']['img_01']}"
        )
        assert result["best_matches"]["img_02"] == "block", (
            f"img_02 최적 매칭: 기대 block, 결과 {result['best_matches']['img_02']}"
        )
        assert result["best_matches"]["img_03"] == "line", (
            f"img_03 최적 매칭: 기대 line, 결과 {result['best_matches']['img_03']}"
        )

        # labels 정규화 검증
        assert result["labels"]["img_01"] == "cross_pattern"
        assert result["labels"]["img_02"] == "block_pattern"
        assert result["labels"]["img_03"] == "line_pattern"
