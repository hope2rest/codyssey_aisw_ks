"""리스크 분포 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_risk_distribution(distribution, output_path):
    """안전/주의/위험 분포 파이 차트를 저장한다."""
    labels = list(distribution.keys())
    sizes = list(distribution.values())
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12}
    )
    ax.set_title("신규 고객 리스크 분포", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_model_comparison(logistic_metrics, ridge_metrics, output_path):
    """모델별 성능 비교 바 차트를 저장한다."""
    metrics = ["accuracy", "precision", "recall", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Macro"]
    logistic_vals = [logistic_metrics[m] for m in metrics]
    ridge_vals = [ridge_metrics[m] for m in metrics]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], logistic_vals, width, label="Logistic", color="#2196F3")
    bars2 = ax.bar([i + width/2 for i in x], ridge_vals, width, label="Ridge", color="#FF5722")
    ax.set_ylabel("점수")
    ax.set_title("모델 성능 비교", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
