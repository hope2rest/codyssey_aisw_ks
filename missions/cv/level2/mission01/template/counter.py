"""counter.py — 박스 카운팅 파이프라인 모듈"""

import numpy as np
from PIL import Image

from conv2d import to_grayscale, compute_edge_magnitude


# 하이퍼파라미터 (명시적 변수 정의)
THRESHOLD = 30      # 엣지 이진화 임계값
MIN_AREA = 100      # 최소 연결 컴포넌트 면적 (노이즈 제거)


def count_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    """
    이미지에서 박스 개수를 카운팅하는 파이프라인.

    1. 이미지 로드 (PIL)
    2. 그레이스케일 변환 (conv2d.to_grayscale)
    3. 엣지 검출 (conv2d.compute_edge_magnitude)
    4. 이진화 (threshold)
    5. Connected Component 분석
    6. 최소 면적 필터 (노이즈 제거)

    Parameters
    ----------
    image_path : str — 이미지 파일 경로
    threshold  : float — 이진화 임계값
    min_area   : int — 최소 면적

    Returns
    -------
    count : int — 검출된 박스 개수
    """
    # TODO: 이미지 로드 (PIL, RGB 변환)

    # TODO: 그레이스케일 변환

    # TODO: 엣지 검출

    # TODO: 이진화 (threshold 적용)

    # TODO: Connected Component 분석 (scipy.ndimage.label 또는 BFS/DFS)

    # TODO: 최소 면적 필터 적용 후 객체 수 반환

    pass
