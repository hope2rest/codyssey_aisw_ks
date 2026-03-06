"""데이터 개요 페이지 모듈"""


def render_overview(preprocessing_info):
    """전처리 정보를 렌더링 가능한 데이터로 변환한다."""
    return {
        "rows": preprocessing_info["original_shape"][0],
        "cols": preprocessing_info["original_shape"][1],
        "missing_before": preprocessing_info["missing_values_before"],
        "missing_after": preprocessing_info["missing_values_after"],
        "scaled_mean_abs_max": preprocessing_info.get("scaled_mean_abs_max", 0),
    }
