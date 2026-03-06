import json
import time


def load_data(filepath):
    # TODO: JSON 파일을 읽어 딕셔너리로 반환


def mac(a, b):
    # TODO: 두 개의 2D 리스트에 대해 MAC 연산 수행


def normalize_labels(labels):
    # TODO: 딕셔너리의 키를 모두 소문자로 변환


def is_close(a, b, epsilon=1e-6):
    # TODO: 두 수의 차이가 epsilon 미만이면 True 반환


def find_best_match(pattern, filters):
    # TODO: 패턴과 가장 높은 MAC 점수를 가진 필터 이름 반환


def measure_mac_time(n, repeat=5):
    # TODO: NxN 크기 MAC 연산의 평균 실행 시간 측정 (초)


def analyze_complexity(sizes, times):
    # TODO: 크기별 시간 데이터로 시간 복잡도 분석


def diagnose_failure(scores, best_match, expected_label, filter_names):
    # TODO: 실패 원인을 data_schema / numerical / logic / none 으로 분류


def main(data_path):
    # TODO: 전체 분석 파이프라인 실행
