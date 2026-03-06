"""차트 생성 유틸리티"""
import os
import sys

_COMP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_COMP_DIR))
_CHARTS_DIR = os.path.join(_PROJECT_DIR, "charts")

if _CHARTS_DIR not in sys.path:
    sys.path.insert(0, _CHARTS_DIR)


def generate_all_charts(result, X_pca, y, cluster_labels, output_dir):
    """모든 차트를 생성하여 output_dir에 저장한다."""
    from risk_charts import save_risk_distribution, save_model_comparison
    from feature_charts import save_feature_importance
    from pca_charts import save_pca_scatter, save_pca_variance
    from cluster_charts import save_cluster_scatter

    os.makedirs(output_dir, exist_ok=True)

    save_risk_distribution(
        result["new_customer_predictions"]["risk_distribution"],
        os.path.join(output_dir, "risk_distribution.png")
    )
    save_model_comparison(
        result["model_logistic"], result["model_ridge"],
        os.path.join(output_dir, "model_comparison.png")
    )
    save_feature_importance(
        result["feature_importance"],
        os.path.join(output_dir, "feature_importance.png")
    )
    save_pca_scatter(
        X_pca, y,
        os.path.join(output_dir, "pca_scatter.png")
    )
    save_pca_variance(
        result["pca"]["variance_ratios"],
        os.path.join(output_dir, "pca_variance.png")
    )
    save_cluster_scatter(
        X_pca, cluster_labels,
        os.path.join(output_dir, "cluster_scatter.png")
    )

    return [
        "risk_distribution.png",
        "model_comparison.png",
        "feature_importance.png",
        "pca_scatter.png",
        "pca_variance.png",
        "cluster_scatter.png",
    ]
