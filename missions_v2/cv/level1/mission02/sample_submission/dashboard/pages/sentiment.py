"""감성 분석 페이지 모듈"""


def render_sentiment_summary(result):
    """감성 분석 요약을 렌더링 가능한 데이터로 변환한다."""
    return {
        "total_reviews": result["total_reviews"],
        "positive_count": result["positive_count"],
        "negative_count": result["negative_count"],
        "metrics": {
            "accuracy": result["sentiment_accuracy"],
            "precision": result["sentiment_precision"],
            "recall": result["sentiment_recall"],
            "f1": result["sentiment_f1"],
        },
    }
