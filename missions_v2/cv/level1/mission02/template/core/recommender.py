"""
Part B: 도서 추천 서비스
"""
import numpy as np

try:
    from .search_engine import cosine_similarity
except ImportError:
    from search_engine import cosine_similarity


def compute_book_similarity(tfidf_matrix: np.ndarray) -> np.ndarray:
    """모든 도서 쌍의 코사인 유사도를 계산하여 (N x N) 유사도 행렬을 반환한다."""
    # TODO: 구현하세요
    pass


def recommend_books(book_index: int, similarity_matrix: np.ndarray,
                    metadata: list, top_k: int = 5) -> list:
    """지정된 도서와 유사도가 높은 상위 top_k권을 추천한다 (자기 자신 제외)."""
    # TODO: 구현하세요
    pass


def recommend_by_category(book_index: int, similarity_matrix: np.ndarray,
                          metadata: list, top_k: int = 3) -> list:
    """동일 카테고리 내에서 유사도가 높은 상위 top_k권을 추천한다."""
    # TODO: 구현하세요
    pass
