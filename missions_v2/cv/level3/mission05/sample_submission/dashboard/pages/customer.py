"""신규 고객 판정 페이지 모듈"""


def classify_single_customer(customer_data, threshold_config):
    """단일 고객의 리스크 등급을 판정한다."""
    prob = customer_data.get("risk_probability", 0)
    if prob >= threshold_config["default_threshold"]:
        return "위험"
    elif prob >= threshold_config["conservative_threshold"]:
        return "주의"
    return "안전"
