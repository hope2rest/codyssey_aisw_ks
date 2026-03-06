"""고객 정보 입력 폼 컴포넌트"""

FEATURE_LABELS = {
    "age": "나이",
    "annual_income": "연소득",
    "debt_ratio": "부채 비율",
    "credit_score": "신용 점수",
    "employment_years": "근속 연수",
    "loan_amount": "대출 금액",
    "num_credit_lines": "신용 계좌 수",
    "recent_inquiries": "최근 조회 수",
    "payment_history": "납부 이력 점수",
}


def validate_customer_input(data):
    """고객 입력 데이터를 검증한다."""
    errors = []
    if data.get("age", 0) < 18 or data.get("age", 0) > 100:
        errors.append("나이는 18~100 범위여야 합니다.")
    if data.get("credit_score", 0) < 0 or data.get("credit_score", 0) > 850:
        errors.append("신용 점수는 0~850 범위여야 합니다.")
    return len(errors) == 0, errors
