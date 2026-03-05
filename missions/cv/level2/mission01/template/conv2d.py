"""conv2d.py — NumPy 기반 2D 컨볼루션 및 엣지 검출 모듈"""

import numpy as np


# Sobel 3x3 커널
SOBEL_X = np.array([
    [-1,  0,  1],
    [-2,  0,  2],
    [-1,  0,  1]
], dtype=np.float64)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float64)


def conv2d(image, kernel):
    """
    2D 컨볼루션 (valid 모드, NumPy만 사용).

    Parameters
    ----------
    image  : 2D ndarray (H x W)
    kernel : 2D ndarray (kH x kW)

    Returns
    -------
    output : 2D ndarray ((H-kH+1) x (W-kW+1))
    """
    # TODO: NumPy만으로 2D 컨볼루션 구현
    # - cv2.filter2D 등 외부 편의 함수 사용 금지
    # - 커널을 180도 뒤집은 뒤 슬라이딩 윈도우 곱·합산
    pass


def to_grayscale(rgb):
    """
    RGB 이미지를 그레이스케일로 변환.
    공식: gray = 0.299*R + 0.587*G + 0.114*B

    Parameters
    ----------
    rgb : 3D ndarray (H x W x 3)

    Returns
    -------
    gray : 2D ndarray (H x W)
    """
    # TODO: 그레이스케일 변환 구현
    pass


def compute_edge_magnitude(gray):
    """
    Sobel 커널로 수평/수직 엣지를 검출하고 엣지 크기를 계산.
    edge_magnitude = sqrt(Gx^2 + Gy^2)

    Parameters
    ----------
    gray : 2D ndarray (H x W, 그레이스케일)

    Returns
    -------
    magnitude : 2D ndarray
    """
    # TODO: Sobel Gx, Gy를 conv2d로 계산
    # TODO: edge_magnitude = sqrt(Gx^2 + Gy^2)
    pass
