"""검색 입력 컴포넌트"""


def validate_query(query):
    """검색 쿼리를 검증한다."""
    if not query or not query.strip():
        return False, "검색어를 입력하세요."
    if len(query.strip()) < 2:
        return False, "검색어는 2자 이상이어야 합니다."
    return True, ""
