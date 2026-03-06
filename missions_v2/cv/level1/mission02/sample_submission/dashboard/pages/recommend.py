"""추천 페이지 모듈"""


def render_recommendations(recommendation):
    """추천 결과를 렌더링 가능한 데이터로 변환한다."""
    return {
        "target": recommendation["target_book"]["title"],
        "similar": [
            {"title": r["title"], "category": r["category"],
             "similarity": round(r["similarity"], 4)}
            for r in recommendation["top5_similar"]
        ],
        "same_category": [
            {"title": r["title"], "similarity": round(r["similarity"], 4)}
            for r in recommendation["same_category_top3"]
        ],
    }
