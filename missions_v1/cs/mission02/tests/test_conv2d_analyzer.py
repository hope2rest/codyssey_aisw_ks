"""2D 컨볼루션 기반 특징 추출 — pytest 검증 (12개 테스트, 전체 통과 시 합격)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: conv2d_analyzer.py (1파일)
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
    """제출물 conv2d_analyzer.py를 동적 import"""
    path = os.path.join(_SUBMISSION_DIR, "conv2d_analyzer.py")
    assert os.path.isfile(path), f"conv2d_analyzer.py 파일 없음: {path}"
    spec = importlib.util.spec_from_file_location("conv2d_analyzer", path)
    mod = importlib.util.module_from_spec(spec)
    if "conv2d_analyzer" in sys.modules:
        del sys.modules["conv2d_analyzer"]
    spec.loader.exec_module(mod)
    return mod


def _parse_ast():
    """제출물 conv2d_analyzer.py를 AST로 파싱"""
    path = os.path.join(_SUBMISSION_DIR, "conv2d_analyzer.py")
    assert os.path.isfile(path), f"conv2d_analyzer.py 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path)


# ========================================================================
# TestStructure — 코드 구조 검증 (2개)
# ========================================================================


class TestStructure:
    """코드 구조 검증"""

    def test_functions_exist(self):
        """필수 함수 9개 정의 확인"""
        tree = _parse_ast()
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {
            "load_data", "pad_matrix", "conv2d", "relu",
            "flatten", "compute_stats", "extract_features",
            "find_strongest_feature", "main",
        }
        missing = required - func_names
        assert not missing, f"누락된 함수: {missing}"

    def test_no_external_lib(self):
        """json 외 외부 라이브러리 미사용 확인"""
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
# TestConv2d — 컨볼루션 및 패딩 검증 (3개)
# ========================================================================


class TestConv2d:
    """컨볼루션 핵심 기능 검증"""

    def test_pad_matrix(self):
        """제로 패딩 적용"""
        mod = _load_module()
        result = mod.pad_matrix([[1, 2], [3, 4]], 1)
        expected = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
        assert result == expected, f"패딩 결과 오류: {result}"

    def test_conv2d_basic(self):
        """기본 컨볼루션 연산"""
        mod = _load_module()
        image = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
        kernel = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = mod.conv2d(image, kernel)
        assert result == [[18, 21], [30, 33]], f"컨볼루션 결과 오류: {result}"

    def test_conv2d_output_size(self):
        """컨볼루션 출력 크기 검증 (5x5 이미지, 3x3 커널 → 3x3 출력)"""
        mod = _load_module()
        image = [[0] * 5 for _ in range(5)]
        kernel = [[0] * 3 for _ in range(3)]
        result = mod.conv2d(image, kernel)
        assert len(result) == 3, f"출력 행 수 오류: {len(result)}"
        assert len(result[0]) == 3, f"출력 열 수 오류: {len(result[0])}"


# ========================================================================
# TestActivation — ReLU 및 Flatten 검증 (2개)
# ========================================================================


class TestActivation:
    """활성화 함수 및 변환 검증"""

    def test_relu(self):
        """ReLU: 음수를 0으로 변환"""
        mod = _load_module()
        result = mod.relu([[1, -2, 3], [-4, 5, -6]])
        assert result == [[1, 0, 3], [0, 5, 0]], f"ReLU 결과 오류: {result}"

    def test_flatten(self):
        """2D 행렬을 1D 리스트로 변환 (행 우선)"""
        mod = _load_module()
        result = mod.flatten([[1, 2, 3], [4, 5, 6]])
        assert result == [1, 2, 3, 4, 5, 6], f"Flatten 결과 오류: {result}"


# ========================================================================
# TestStats — 통계 계산 검증 (1개)
# ========================================================================


class TestStats:
    """통계 계산 검증"""

    def test_compute_stats(self):
        """min, max, mean 계산"""
        mod = _load_module()
        result = mod.compute_stats([[1, 2, 3], [4, 5, 6]])
        assert result["min"] == 1, f"min 오류: {result['min']}"
        assert result["max"] == 6, f"max 오류: {result['max']}"
        assert abs(result["mean"] - 3.5) < 1e-9, f"mean 오류: {result['mean']}"


# ========================================================================
# TestFeatures — 특징 추출 검증 (2개)
# ========================================================================


class TestFeatures:
    """특징 추출 기능 검증"""

    def test_extract_features(self):
        """다중 커널 특징맵 추출"""
        mod = _load_module()
        image = [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
        kernels = {
            "edge_h": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        }
        result = mod.extract_features(image, kernels)
        assert "edge_h" in result, "edge_h 키 없음"
        # conv2d 후 relu 적용 결과 검증
        assert result["edge_h"] == [[2, 2, 2], [0, 0, 0], [0, 0, 0]]

    def test_find_strongest_feature(self):
        """가장 강한 특징 커널 탐색"""
        mod = _load_module()
        image = [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ]
        kernels = {
            "edge_h": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        }
        result = mod.find_strongest_feature(image, kernels)
        # edge_h relu sum=6, sharpen relu sum=13
        assert result == "sharpen", f"기대: sharpen, 결과: {result}"


# ========================================================================
# TestMain — 전체 파이프라인 검증 (2개)
# ========================================================================


class TestMain:
    """main() 전체 파이프라인 검증"""

    def test_main_feature_sums(self, data_path):
        """특징맵 합계 검증"""
        mod = _load_module()
        result = mod.main(data_path)

        sums = result["feature_sums"]
        # img_01 (십자 패턴)
        assert sums["img_01"]["edge_h"] == 6
        assert sums["img_01"]["edge_v"] == 6
        assert sums["img_01"]["sharpen"] == 13

        # img_02 (블록 패턴)
        assert sums["img_02"]["edge_h"] == 0
        assert sums["img_02"]["sharpen"] == 8

        # img_03 (수직선 패턴)
        assert sums["img_03"]["edge_v"] == 9

    def test_main_strongest(self, data_path):
        """최강 특징 필터 검증"""
        mod = _load_module()
        result = mod.main(data_path)

        strongest = result["strongest_features"]
        assert strongest["img_01"] == "sharpen"
        assert strongest["img_02"] == "sharpen"
        assert strongest["img_03"] == "edge_v"
