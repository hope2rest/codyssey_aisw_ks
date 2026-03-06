"""추천 결과 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_similarity_heatmap(sim_matrix, titles, output_path):
    """도서 유사도 행렬 히트맵을 저장한다."""
    n = min(len(titles), 15)
    sub_matrix = sim_matrix[:n, :n]
    sub_titles = [t[:12] for t in titles[:n]]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sub_titles, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sub_titles, fontsize=8)
    ax.set_title("도서 유사도 히트맵", fontsize=14)
    plt.colorbar(im, ax=ax, label="코사인 유사도")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_recommendation_chart(recommendations, target_title, output_path):
    """추천 도서 상위 5건의 유사도를 바 차트로 저장한다."""
    titles = [r["title"][:20] for r in recommendations]
    sims = [r["similarity"] for r in recommendations]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(titles)))
    bars = ax.barh(titles[::-1], sims[::-1], color=colors[::-1])
    ax.set_title(f'"{target_title[:20]}" 추천 도서 Top 5', fontsize=13)
    ax.set_xlabel("유사도")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, sims[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
