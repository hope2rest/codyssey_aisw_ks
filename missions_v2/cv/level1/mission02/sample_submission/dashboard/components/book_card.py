"""도서 카드 컴포넌트"""


def format_book_card(title, similarity, category=None, rank=None):
    """도서 정보를 카드 형태의 딕셔너리로 반환한다."""
    card = {"title": title, "similarity": round(similarity, 4)}
    if category:
        card["category"] = category
    if rank is not None:
        card["rank"] = rank
    return card
