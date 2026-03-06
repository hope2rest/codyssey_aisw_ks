"""
Q1 CV — pytest 검증 (21개 테스트, 전체 통과 시 합격)

검증 방식: AST 구조 분석 + importlib 모듈 import 후 기능 검증
제출물: conv2d.py, counter.py, metrics.py, main.py, result_q1.json (5파일)
"""
import ast
import importlib
import json
import os
import re
import sys

import numpy as np
import pytest

# ─── 모듈 레벨 변수 (fixture에서 설정) ───

_SUBMISSION_DIR = None
_MISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_MISSION_DIR, "data")
_IMAGES_DIR = os.path.join(_DATA_DIR, "images")
_LABELS_PATH = os.path.join(_DATA_DIR, "labels.json")


@pytest.fixture(autouse=True, scope="module")
def _configure(submission_dir):
    """submission_dir fixture로 모듈 경로 설정"""
    global _SUBMISSION_DIR
    _SUBMISSION_DIR = submission_dir


# ─── 공통 헬퍼 ───


def _import_module(module_name):
    """제출 디렉토리에서 특정 모듈을 import (캐시 제거 후)"""
    if _SUBMISSION_DIR not in sys.path:
        sys.path.insert(0, _SUBMISSION_DIR)
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _parse_ast(filename):
    """제출물 파일을 AST로 파싱"""
    path = os.path.join(_SUBMISSION_DIR, filename)
    assert os.path.isfile(path), f"{filename} 파일 없음: {path}"
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ast.parse(source, filename=path), source


# ========================================================================
# TestStructure — 코드 구조 검증 (4개)
# ========================================================================


class TestStructure:
    """제출물 코드 구조 검증"""

    def test_conv2d_functions(self):
        """conv2d.py에 필수 함수 7개가 정의되어 있는지 확인 (5점)"""
        tree, _ = _parse_ast("conv2d.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"conv2d", "to_grayscale", "compute_edge_magnitude",
                     "flip_horizontal", "flip_vertical",
                     "adjust_brightness", "normalize_image"}
        missing = required - func_names
        assert not missing, f"conv2d.py 누락 함수: {missing}"

    def test_no_filter2d(self):
        """conv2d.py에서 cv2.filter2D를 사용하지 않는지 확인 (5점)"""
        _, source = _parse_ast("conv2d.py")
        assert "filter2D" not in source, "filter2D 사용 감지 — conv2d 직접 구현 필요"

    def test_counter_functions(self):
        """counter.py에 필수 함수와 하이퍼파라미터 정의 확인 (5점)"""
        tree, source = _parse_ast("counter.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"count_boxes", "count_boxes_augmented",
                     "ensemble_count", "extract_bounding_boxes"}
        missing = required - func_names
        assert not missing, f"counter.py 누락 함수: {missing}"
        assert "THRESHOLD" in source, "THRESHOLD 변수 미정의"
        assert "MIN_AREA" in source, "MIN_AREA 변수 미정의"

    def test_metrics_functions(self):
        """metrics.py에 필수 함수 7개가 정의되어 있는지 확인 (5점)"""
        tree, _ = _parse_ast("metrics.py")
        func_names = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        required = {"compute_metrics", "find_worst_case",
                     "get_failure_reasons", "get_why_learning_based",
                     "compare_methods", "create_detection_log",
                     "generate_weekly_report"}
        missing = required - func_names
        assert not missing, f"metrics.py 누락 함수: {missing}"


# ========================================================================
# TestConv2d — conv2d 및 이미지 처리 기능 검증 (5개)
# ========================================================================


class TestConv2d:
    """conv2d 모듈 기능 검증"""

    def test_conv2d_identity(self):
        """단위 커널(identity)에 대한 conv2d 결과 검증 (5점)"""
        mod = _import_module("conv2d")
        image = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=np.float64)
        kernel = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.float64)
        result = mod.conv2d(image, kernel)
        assert result.shape == (1, 1), f"shape 오류: {result.shape}"
        assert abs(result[0, 0] - 5.0) < 1e-9, f"값 오류: {result[0, 0]}"

    def test_conv2d_sobel(self):
        """Sobel 커널 적용 결과 검증 (5점)"""
        mod = _import_module("conv2d")
        image = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 255, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]], dtype=np.float64)
        result_x = mod.conv2d(image, mod.SOBEL_X)
        result_y = mod.conv2d(image, mod.SOBEL_Y)
        assert result_x.shape == (3, 3), f"Sobel X shape 오류: {result_x.shape}"
        assert result_y.shape == (3, 3), f"Sobel Y shape 오류: {result_y.shape}"
        assert abs(result_x[1, 1]) < 1e-9, f"Sobel X 중심값: {result_x[1, 1]}"
        assert abs(result_y[1, 1]) < 1e-9, f"Sobel Y 중심값: {result_y[1, 1]}"

    def test_grayscale(self):
        """그레이스케일 변환 검증 (5점)"""
        mod = _import_module("conv2d")
        rgb = np.zeros((2, 2, 3), dtype=np.float64)
        rgb[0, 0] = [255, 0, 0]
        rgb[0, 1] = [0, 255, 0]
        rgb[1, 0] = [0, 0, 255]
        rgb[1, 1] = [255, 255, 255]
        gray = mod.to_grayscale(rgb)
        assert abs(gray[0, 0] - 0.299 * 255) < 1e-6, "Red 변환 오류"
        assert abs(gray[0, 1] - 0.587 * 255) < 1e-6, "Green 변환 오류"
        assert abs(gray[1, 0] - 0.114 * 255) < 1e-6, "Blue 변환 오류"
        assert abs(gray[1, 1] - 255.0) < 1e-6, "White 변환 오류"

    def test_flip_operations(self):
        """좌우/상하 반전 검증 (5점)"""
        mod = _import_module("conv2d")
        image = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.float64)
        h_flip = mod.flip_horizontal(image)
        assert h_flip[0, 0] == 3 and h_flip[0, 2] == 1, \
            f"좌우 반전 오류: {h_flip}"
        v_flip = mod.flip_vertical(image)
        assert v_flip[0, 0] == 4 and v_flip[1, 0] == 1, \
            f"상하 반전 오류: {v_flip}"

    def test_brightness_and_normalize(self):
        """밝기 조절 및 정규화 검증 (5점)"""
        mod = _import_module("conv2d")
        image = np.array([[100, 200],
                          [50, 150]], dtype=np.float64)
        bright = mod.adjust_brightness(image, 2.0)
        assert bright[0, 0] == 200, f"밝기 증가 오류: {bright[0, 0]}"
        assert bright[0, 1] == 255, f"클리핑 오류: {bright[0, 1]}"

        normed = mod.normalize_image(image)
        assert abs(normed.min()) < 1e-6, f"정규화 최솟값 오류: {normed.min()}"
        assert abs(normed.max() - 255.0) < 1e-6, f"정규화 최댓값 오류: {normed.max()}"


# ========================================================================
# TestCounter — counter.py 기능 검증 (3개)
# ========================================================================


class TestCounter:
    """박스 카운팅 파이프라인 검증"""

    def test_ensemble_count(self):
        """중앙값 기반 앙상블 카운팅 검증 (5점)"""
        mod = _import_module("counter")
        assert mod.ensemble_count([1, 2, 3, 4, 5]) == 3, "홀수 개 중앙값 오류"
        assert mod.ensemble_count([1, 2, 4, 5]) == 3, "짝수 개 중앙값 오류"
        assert mod.ensemble_count([5, 1, 3]) == 3, "정렬 후 중앙값 오류"

    def test_count_boxes_augmented(self):
        """증강 앙상블 카운팅이 정수를 반환하는지 확인 (5점)"""
        mod = _import_module("counter")
        img_path = os.path.join(_IMAGES_DIR, "easy_01.png")
        result = mod.count_boxes_augmented(img_path)
        assert isinstance(result, int), f"반환값이 정수가 아닙니다: {type(result)}"
        assert result >= 0, f"카운팅 결과가 음수: {result}"

    def test_bounding_boxes_format(self):
        """바운딩 박스 추출 결과 형식 검증 (5점)"""
        mod = _import_module("counter")
        img_path = os.path.join(_IMAGES_DIR, "easy_01.png")
        bboxes = mod.extract_bounding_boxes(img_path)
        assert isinstance(bboxes, list), "반환값이 list가 아닙니다"
        assert len(bboxes) > 0, "바운딩 박스가 검출되지 않았습니다"
        for bbox in bboxes:
            for key in ("x_min", "y_min", "x_max", "y_max", "area"):
                assert key in bbox, f"바운딩 박스에 {key} 키 없음"
            assert bbox["x_min"] <= bbox["x_max"], "x_min > x_max"
            assert bbox["y_min"] <= bbox["y_max"], "y_min > y_max"
            assert bbox["area"] > 0, "area가 0 이하"


# ========================================================================
# TestMetrics — metrics.py 기능 검증 (4개)
# ========================================================================


class TestMetrics:
    """성능 지표 및 분석 검증"""

    def test_compute_metrics(self):
        """MAE/Accuracy 계산 검증 (5점)"""
        mod = _import_module("metrics")
        preds = {"easy_01": 3, "easy_02": 5, "easy_03": 5}
        labels = {"easy_01": 3, "easy_02": 4, "easy_03": 6}
        result = mod.compute_metrics(preds, labels, "easy")
        assert isinstance(result, dict), "반환값이 dict가 아닙니다"
        assert "mae" in result, "mae 키 없음"
        assert "accuracy" in result, "accuracy 키 없음"
        assert abs(result["mae"] - 2/3) < 0.01, f"MAE 오류: {result['mae']}"
        assert abs(result["accuracy"] - 1/3) < 0.01, f"Accuracy 오류: {result['accuracy']}"

    def test_find_worst_case(self):
        """worst case 이미지 찾기 검증 (5점)"""
        mod = _import_module("metrics")
        preds = {"hard_01": 2, "hard_02": 3, "hard_03": 1}
        labels = {"hard_01": 5, "hard_02": 7, "hard_03": 8}
        result = mod.find_worst_case(preds, labels, "hard")
        assert result == "hard_03", f"기대: hard_03, 결과: {result}"

    def test_compare_methods(self):
        """기본 vs 증강 방식 비교 검증 (5점)"""
        mod = _import_module("metrics")
        base = {"easy_01": 3, "easy_02": 5}
        aug = {"easy_01": 3, "easy_02": 4}
        labels = {"easy_01": 3, "easy_02": 4}
        result = mod.compare_methods(base, aug, labels)
        assert "easy" in result, "easy 카테고리 없음"
        easy = result["easy"]
        for key in ("base_mae", "augmented_mae", "mae_improvement",
                    "base_accuracy", "augmented_accuracy"):
            assert key in easy, f"{key} 키 없음"
        assert easy["augmented_mae"] <= easy["base_mae"], \
            "증강 MAE가 기본보다 높음"

    def test_detection_log_and_weekly_report(self):
        """검출 로그 및 주간 보고서 검증 (5점)"""
        mod = _import_module("metrics")
        preds = {"easy_01": 3, "easy_02": 5}
        labels = {"easy_01": 3, "easy_02": 4}

        log = mod.create_detection_log(preds, labels, "2024-01-01")
        assert log["date"] == "2024-01-01", "날짜 오류"
        assert log["total_images"] == 2, "이미지 수 오류"
        assert isinstance(log["results"], list), "results가 list가 아님"
        assert "daily_accuracy" in log, "daily_accuracy 없음"
        assert log["daily_accuracy"] == 0.5, f"정확도 오류: {log['daily_accuracy']}"

        logs = []
        for i in range(7):
            p = {"easy_01": 3, "easy_02": 4 + (i % 2)}
            lg = mod.create_detection_log(p, labels, f"2024-01-0{i+1}")
            logs.append(lg)
        report = mod.generate_weekly_report(logs)
        for key in ("week_start", "week_end", "total_images_processed",
                    "average_daily_accuracy", "best_day", "worst_day"):
            assert key in report, f"주간 보고서에 {key} 없음"
        assert report["week_start"] == "2024-01-01"
        assert report["week_end"] == "2024-01-07"


# ========================================================================
# TestResult — result_q1.json 결과 검증 (5개)
# ========================================================================


class TestResult:
    """result_q1.json 최종 결과 검증"""

    @staticmethod
    def _load_result():
        path = os.path.join(_SUBMISSION_DIR, "result_q1.json")
        assert os.path.isfile(path), f"result_q1.json 없음: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_valid_images_only(self):
        """유효 이미지만 분석 (test_01 제외) (5점)"""
        result = self._load_result()
        with open(_LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        valid = [
            n for n in labels
            if os.path.isfile(os.path.join(_IMAGES_DIR, f"{n}.png"))
        ]
        preds = result.get("predictions", {})
        extra = set(preds.keys()) - set(valid)
        assert not extra, f"존재하지 않는 이미지 포함: {extra}"

    def test_augmented_predictions(self):
        """증강 카운팅 결과가 포함되어 있는지 확인 (5점)"""
        result = self._load_result()
        assert "predictions_augmented" in result, "predictions_augmented 없음"
        aug = result["predictions_augmented"]
        assert len(aug) > 0, "predictions_augmented가 비어 있음"
        for name, count in aug.items():
            assert isinstance(count, int), f"{name}: 정수가 아닙니다"

    def test_method_comparison(self):
        """방법 비교 결과 확인 (5점)"""
        result = self._load_result()
        assert "method_comparison" in result, "method_comparison 없음"
        comp = result["method_comparison"]
        for cat in ("easy", "medium", "hard"):
            assert cat in comp, f"{cat} 카테고리 없음"
            for key in ("base_mae", "augmented_mae", "mae_improvement"):
                assert key in comp[cat], f"{cat}.{key} 없음"

    def test_weekly_report(self):
        """주간 보고서 결과 확인 (5점)"""
        result = self._load_result()
        assert "weekly_report" in result, "weekly_report 없음"
        report = result["weekly_report"]
        for key in ("week_start", "week_end", "total_images_processed",
                    "average_daily_accuracy", "best_day", "worst_day"):
            assert key in report, f"weekly_report.{key} 없음"
        assert report["total_images_processed"] > 0, "처리된 이미지 수 0"

    def test_failure_reasons(self):
        """failure_reasons 3개 이상, 한국어, 20자+ (5점)"""
        result = self._load_result()
        kr = re.compile(r"[가-힣]")
        fr = result.get("failure_reasons", [])
        assert len(fr) >= 3, f"{len(fr)}개 제출 (최소 3개 필요)"
        for i, r in enumerate(fr):
            assert len(r) >= 20, f"항목 {i+1}: {len(r)}자 (20자 이상 필요)"
            assert kr.search(r), f"항목 {i+1}: 한국어 미포함"

    def test_why_learning_based(self):
        """why_learning_based 30~200자, 한국어 (5점)"""
        result = self._load_result()
        kr = re.compile(r"[가-힣]")
        wlb = result.get("why_learning_based", "")
        assert 30 <= len(wlb) <= 200, f"길이: {len(wlb)}자 (30~200자 필요)"
        assert kr.search(wlb), "한국어 미포함"
