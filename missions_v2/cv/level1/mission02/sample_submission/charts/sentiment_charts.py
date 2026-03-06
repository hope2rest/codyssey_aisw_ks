"""감성 분석 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_sentiment_distribution(positive_count, negative_count, output_path):
    """긍정/부정 분포 파이 차트를 저장한다."""
    labels = ["긍정", "부정"]
    sizes = [positive_count, negative_count]
    colors = ["#4CAF50", "#F44336"]
    explode = (0.05, 0.05)

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12}
    )
    ax.set_title("감성 분석 결과 분포", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_sentiment_metrics_chart(metrics, output_path):
    """감성 분석 평가 지표 바 차트를 저장한다."""
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["precision"],
              metrics["recall"], metrics["f1"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=["#2196F3", "#FF9800", "#9C27B0", "#009688"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("점수")
    ax.set_title("감성 분석 평가 지표", fontsize=14)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
