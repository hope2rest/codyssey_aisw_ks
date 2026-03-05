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

# 가우시안 블러 커널 (노이즈 제거용)
GAUSS3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0


def conv2d(image, kernel):
    """2D 컨볼루션 (valid 모드, NumPy stride_tricks 활용)."""
    kH, kW = kernel.shape
    iH, iW = image.shape
    oH = iH - kH + 1
    oW = iW - kW + 1

    k_flip = kernel[::-1, ::-1]

    shape = (oH, oW, kH, kW)
    strides = (image.strides[0], image.strides[1],
               image.strides[0], image.strides[1])
    windows = np.lib.stride_tricks.as_strided(
        image, shape=shape, strides=strides
    )

    output = np.einsum('ijkl,kl->ij', windows, k_flip)
    return output


def to_grayscale(rgb):
    """gray = 0.299*R + 0.587*G + 0.114*B"""
    return (0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2])


def pad_to(arr, target_h, target_w):
    """valid 모드로 줄어든 배열을 원본 크기로 복원 (edge 패딩)."""
    ph = target_h - arr.shape[0]
    pw = target_w - arr.shape[1]
    return np.pad(arr,
                  ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                  mode='edge')


def compute_edge_magnitude(gray):
    """가우시안 블러 → Sobel Gx/Gy → edge_magnitude = sqrt(Gx^2 + Gy^2)"""
    h, w = gray.shape

    blurred = conv2d(gray, GAUSS3)
    blurred = pad_to(blurred, h, w)

    Gx = conv2d(blurred, SOBEL_X)
    Gy = conv2d(blurred, SOBEL_Y)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    magnitude = pad_to(magnitude, h, w)

    return magnitude
