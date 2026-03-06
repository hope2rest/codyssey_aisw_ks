"""interpreter.py - 모델 해석 모듈"""
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


def get_feature_importance(model, feature_names):
    """모델의 coef_ 속성에서 절댓값 기준 feature importance를 추출합니다."""
    coefs = model.coef_.flatten()
    abs_coefs = np.abs(coefs)
    indices = np.argsort(abs_coefs)[::-1]
    result = []
    for idx in indices:
        result.append({
            "feature": feature_names[idx],
            "importance": round(float(abs_coefs[idx]), 4),
        })
    return result


def get_pca_variance(pca):
    """PCA 객체의 explained_variance_ratio_를 반환합니다."""
    result = []
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        result.append({
            "component": i + 1,
            "variance_ratio": round(float(ratio), 4),
        })
    return result


def cluster_features(X_scaled, n_clusters=3):
    """K-Means(n_clusters=3, random_state=42)로 클러스터링합니다."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    counts = Counter(labels.tolist())
    cluster_counts = {str(k): counts[k] for k in sorted(counts.keys())}
    return {
        "labels": labels.tolist(),
        "cluster_counts": cluster_counts,
        "inertia": round(float(kmeans.inertia_), 4),
    }
