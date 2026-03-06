"""
Part D: 도서 검색 및 추천 서비스 파이프라인
"""
import json
import os
import sys

import numpy as np

try:
    from .search_engine import preprocess, build_tfidf_matrix, search, compute_tf, compute_idf, cosine_similarity
    from .recommender import compute_book_similarity, recommend_books, recommend_by_category
    from .sentiment import rule_based_predict, compute_metrics
except ImportError:
    from search_engine import preprocess, build_tfidf_matrix, search, compute_tf, compute_idf, cosine_similarity
    from recommender import compute_book_similarity, recommend_books, recommend_by_category
    from sentiment import rule_based_predict, compute_metrics


def main(data_dir: str) -> dict:
    """전체 서비스 파이프라인 실행"""
    # TODO: 구현하세요
    # 1. 데이터 로드 (stopwords.txt, books.txt, book_metadata.json, queries.txt)
    # 2. Part A: TF-IDF 검색 (build_tfidf_matrix, search)
    # 3. Part B: 도서 추천 (compute_book_similarity, recommend_books, recommend_by_category)
    # 4. Part C: 감성 분석 (rule_based_predict, compute_metrics)
    # 5. 결과 딕셔너리 반환
    pass


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data")
    result = main(data_dir)

    output_dir = os.path.join(base_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result_q2.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("result_q2.json saved.")
