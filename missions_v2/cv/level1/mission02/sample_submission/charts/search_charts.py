"""검색 결과 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_search_results_chart(search_results, output_path):
    """검색 쿼리별 상위 3건의 유사도를 바 차트로 저장한다."""
    n = len(search_results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]
    for ax, sr in zip(axes, search_results):
        titles = [item["title"][:20] for item in sr["top3"]]
        sims = [item["similarity"] for item in sr["top3"]]
        bars = ax.barh(titles, sims, color="#4CAF50")
        ax.set_title(f'검색: "{sr["query"][:30]}"', fontsize=11)
        ax.set_xlabel("유사도")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, sims):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
