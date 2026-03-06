import numpy as np


def conv2d(image, kernel):
    # TODO: NumPy만으로 2D 컨볼루션 구현 (valid 모드)


def to_grayscale(rgb):
    # TODO: 그레이스케일 변환 구현 (0.299*R + 0.587*G + 0.114*B)


def compute_edge_magnitude(gray):
    # TODO: Sobel 엣지 크기 계산 구현 (sqrt(Gx^2 + Gy^2))


def flip_horizontal(image):
    # TODO: 이미지 좌우 반전


def flip_vertical(image):
    # TODO: 이미지 상하 반전


def adjust_brightness(image, factor):
    # TODO: 밝기 조절 (factor 곱한 뒤 0~255 클리핑)


def normalize_image(image):
    # TODO: Min-Max 정규화 (0~255 범위)
