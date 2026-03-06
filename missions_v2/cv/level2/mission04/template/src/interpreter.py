import numpy as np
from sklearn.cluster import KMeans


def get_feature_importance(model, feature_names):
    # TODO: coef_ 절댓값 기준 feature importance 추출 (내림차순)


def get_pca_variance(pca):
    # TODO: PCA explained_variance_ratio_ 반환


def cluster_features(X_scaled, n_clusters=3):
    # TODO: K-Means(n_clusters=3, random_state=42) 클러스터링
