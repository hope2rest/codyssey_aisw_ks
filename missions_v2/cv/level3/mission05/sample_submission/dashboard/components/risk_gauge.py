"""리스크 게이지 컴포넌트"""


def get_risk_color(risk_level):
    """리스크 등급에 따른 색상을 반환한다."""
    colors = {"안전": "#4CAF50", "주의": "#FF9800", "위험": "#F44336"}
    return colors.get(risk_level, "#9E9E9E")


def format_risk_summary(predictions):
    """리스크 판정 결과 요약을 반환한다."""
    summary = {"안전": 0, "주의": 0, "위험": 0}
    for p in predictions:
        level = p.get("risk_level", "")
        if level in summary:
            summary[level] += 1
    return summary
