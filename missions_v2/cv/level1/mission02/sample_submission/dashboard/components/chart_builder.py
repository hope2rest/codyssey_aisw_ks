"""차트 생성 유틸리티"""
import os
import sys

_COMP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_COMP_DIR))
_CHARTS_DIR = os.path.join(_PROJECT_DIR, "charts")

if _CHARTS_DIR not in sys.path:
    sys.path.insert(0, _CHARTS_DIR)


def generate_all_charts(result, sim_matrix, titles, output_dir):
    """모든 차트를 생성하여 output_dir에 저장한다."""
    from search_charts import save_search_results_chart
    from recommend_charts import save_similarity_heatmap, save_recommendation_chart
    from sentiment_charts import save_sentiment_distribution, save_sentiment_metrics_chart

    os.makedirs(output_dir, exist_ok=True)

    save_search_results_chart(
        result["search_results"],
        os.path.join(output_dir, "search_results.png")
    )
    save_similarity_heatmap(
        sim_matrix, titles,
        os.path.join(output_dir, "similarity_heatmap.png")
    )
    save_recommendation_chart(
        result["recommendation"]["top5_similar"],
        result["recommendation"]["target_book"]["title"],
        os.path.join(output_dir, "recommendation_top5.png")
    )
    save_sentiment_distribution(
        result["positive_count"], result["negative_count"],
        os.path.join(output_dir, "sentiment_distribution.png")
    )
    save_sentiment_metrics_chart(
        {
            "accuracy": result["sentiment_accuracy"],
            "precision": result["sentiment_precision"],
            "recall": result["sentiment_recall"],
            "f1": result["sentiment_f1"],
        },
        os.path.join(output_dir, "sentiment_metrics.png")
    )

    return [
        "search_results.png",
        "similarity_heatmap.png",
        "recommendation_top5.png",
        "sentiment_distribution.png",
        "sentiment_metrics.png",
    ]
