"""리스크 예측 페이지 모듈"""


def render_predictions(new_customer_predictions):
    """고객 판정 결과를 렌더링 가능한 데이터로 변환한다."""
    return {
        "total": new_customer_predictions["total_customers"],
        "distribution": new_customer_predictions["risk_distribution"],
        "predictions": new_customer_predictions["predictions"],
    }
