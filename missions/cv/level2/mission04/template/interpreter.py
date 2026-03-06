"""interpreter.py - 모델 해석 모듈"""
import numpy as np
from sklearn.cluster import KMeans


def get_feature_importance(model, feature_names):
    """모델의 coef_ 속성에서 절댓값 기준 feature importance를 추출합니다.

    Returns:
        list[dict]: [{"feature": str, "importance": float}, ...]
                     절댓값 내림차순 정렬
    """
    # TODO: 구현하세요
    pass


def get_pca_variance(pca):
    """PCA 객체의 explained_variance_ratio_를 반환합니다.

    Returns:
        list[dict]: [{"component": int, "variance_ratio": float}, ...]
    """
    # TODO: 구현하세요
    pass


def cluster_features(X_scaled, n_clusters=3):
    """K-Means(n_clusters=3, random_state=42)로 클러스터링합니다.

    Returns:
        dict: {"labels": list, "cluster_counts": dict, "inertia": float}
    """
    # TODO: 구현하세요
    pass
