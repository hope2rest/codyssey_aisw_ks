"""모델 해석 페이지 모듈"""


def render_model_analysis(result):
    """모델 해석 정보를 렌더링 가능한 데이터로 변환한다."""
    return {
        "feature_importance": result["feature_importance"],
        "pca": {
            "n_components": result["pca"]["n_components_selected"],
            "total_variance": result["pca"]["total_variance_explained"],
            "variance_ratios": result["pca"]["variance_ratios"],
        },
        "clustering": result["clustering"],
    }
