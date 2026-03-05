import json
import time


# ─── Part A: MAC 연산 기본 ───


def load_data(filepath):
    # TODO: JSON 파일을 읽어 딕셔너리로 반환


def mac(a, b):
    # TODO: 두 개의 2D 리스트에 대해 MAC 연산 수행 (같은 위치 곱한 뒤 합산)


def normalize_labels(labels):
    # TODO: 딕셔너리의 키를 모두 소문자로 변환한 새 딕셔너리 반환


def is_close(a, b, epsilon=1e-6):
    # TODO: 두 수의 차이가 epsilon 미만이면 True 반환


def find_best_match(pattern, filters):
    # TODO: 패턴과 가장 높은 MAC 점수를 가진 필터 이름 반환


# ─── Part B: 성능 벤치마크 및 시간 복잡도 분석 ───


def measure_mac_time(n, repeat=5):
    # TODO: NxN 크기 패턴에 대한 MAC 연산 평균 시간 측정 (초 단위 float 반환)


def analyze_complexity(sizes, times):
    # TODO: 크기별 시간 데이터로 시간 복잡도 분석
    # TODO: size_pairs, time_ratios, estimated_order 포함 딕셔너리 반환


# ─── Part C: 실패 진단 ───


def diagnose_failure(scores, best_match, expected_label, filter_names):
    # TODO: 실패 원인을 분류하여 category와 reason 반환
    # TODO: category는 "data_schema", "numerical", "logic", "none" 중 하나


# ─── Part D: 개념 설명 ───


def get_mac_explanation():
    # TODO: MAC 연산의 정의와 AI에서의 중요성 설명 (한국어, 50자 이상)


def get_normalization_reason():
    # TODO: 라벨 키를 표준화하는 이유 설명 (한국어, 30자 이상)


def get_epsilon_reason():
    # TODO: 허용오차(epsilon) 기반 비교가 필요한 이유 설명 (한국어, 30자 이상)


# ─── 전체 파이프라인 ───


def main(data_path):
    # TODO: 데이터 로드 → 점수 계산 → 최적 매칭
    # TODO: 성능 벤치마크 (sizes=[1,2,4,8,16,32])
    # TODO: 실패 진단 (각 패턴별)
    # TODO: 개념 설명 포함
    # TODO: 전체 결과 딕셔너리 반환
