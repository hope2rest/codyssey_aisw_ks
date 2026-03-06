"""K-Means 클러스터 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_cluster_scatter(X_pca, cluster_labels, output_path):
    """K-Means 클러스터 2D 산점도를 저장한다."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.Set1(np.linspace(0, 0.5, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = [cl == label for cl in cluster_labels]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=f"Cluster {label}", alpha=0.6,
                   edgecolors="k", linewidths=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means 클러스터링 결과", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
