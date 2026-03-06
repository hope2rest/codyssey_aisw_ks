import json


def load_data(filepath):
    # TODO: JSON 파일을 읽어 딕셔너리로 반환


def pad_matrix(matrix, pad_size):
    # TODO: 2D 행렬의 상하좌우에 pad_size만큼 0을 추가


def conv2d(image, kernel):
    # TODO: 패딩 없이 stride=1로 2D 컨볼루션 수행


def relu(matrix):
    # TODO: 음수 값을 0으로 변환한 새 행렬 반환


def flatten(matrix):
    # TODO: 2D 행렬을 행 우선 순서로 1D 리스트로 변환


def compute_stats(matrix):
    # TODO: min, max, mean 계산하여 딕셔너리로 반환


def extract_features(image, kernels):
    # TODO: 각 커널로 conv2d + relu 적용한 특징맵 딕셔너리 반환


def find_strongest_feature(image, kernels):
    # TODO: relu 후 합이 가장 큰 커널 이름 반환


def main(data_path):
    # TODO: 전체 특징 추출 파이프라인 실행
