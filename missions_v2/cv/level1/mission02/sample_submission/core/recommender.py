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
    N = tfidf_matrix.shape[0]
    sim_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, N):
            s = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s
    return sim_matrix


def recommend_books(book_index: int, similarity_matrix: np.ndarray,
                    metadata: list, top_k: int = 5) -> list:
    """지정된 도서와 유사도가 높은 상위 top_k권을 추천한다 (자기 자신 제외)."""
    N = similarity_matrix.shape[0]
    sims = []
    for i in range(N):
        if i == book_index:
            continue
        sims.append((i, similarity_matrix[book_index, i]))
    sims.sort(key=lambda x: (-x[1], x[0]))
    results = []
    for idx, sim in sims[:top_k]:
        results.append({
            "index": idx,
            "title": metadata[idx]["title"],
            "similarity": round(float(sim), 6),
            "category": metadata[idx]["category"],
        })
    return results


def recommend_by_category(book_index: int, similarity_matrix: np.ndarray,
                          metadata: list, top_k: int = 3) -> list:
    """동일 카테고리 내에서 유사도가 높은 상위 top_k권을 추천한다."""
    target_category = metadata[book_index]["category"]
    N = similarity_matrix.shape[0]
    sims = []
    for i in range(N):
        if i == book_index:
            continue
        if metadata[i]["category"] == target_category:
            sims.append((i, similarity_matrix[book_index, i]))
    sims.sort(key=lambda x: (-x[1], x[0]))
    results = []
    for idx, sim in sims[:top_k]:
        results.append({
            "index": idx,
            "title": metadata[idx]["title"],
            "similarity": round(float(sim), 6),
        })
    return results
