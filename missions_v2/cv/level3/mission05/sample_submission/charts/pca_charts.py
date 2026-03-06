"""PCA 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_pca_scatter(X_pca, y, output_path):
    """PCA 2D 산점도를 저장한다."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4CAF50" if label == 0 else "#F44336" for label in y]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D 산점도 (녹색=안전, 빨간색=위험)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_pca_variance(variance_ratios, output_path):
    """PCA 분산 설명 비율 바 차트를 저장한다."""
    components = [item["component"] for item in variance_ratios]
    ratios = [item["variance_ratio"] for item in variance_ratios]
    cumulative = np.cumsum(ratios)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(components, ratios, color="#2196F3", alpha=0.7, label="개별 분산")
    ax.plot(components, cumulative, "ro-", label="누적 분산")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% 기준선")
    ax.set_xlabel("주성분")
    ax.set_ylabel("분산 설명 비율")
    ax.set_title("PCA 분산 설명 비율", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
