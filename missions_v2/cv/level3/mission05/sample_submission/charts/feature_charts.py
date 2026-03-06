"""Feature Importance 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_feature_importance(importance_list, output_path):
    """Feature Importance 수평 바 차트를 저장한다."""
    features = [item["feature"] for item in importance_list]
    values = [item["importance"] for item in importance_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("중요도 (|coefficient|)")
    ax.set_title("Feature Importance", fontsize=14)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
