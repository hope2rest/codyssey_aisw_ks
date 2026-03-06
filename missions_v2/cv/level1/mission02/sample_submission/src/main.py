"""
Part D: 도서 검색 및 추천 서비스 파이프라인
"""
import json
import os
import sys

import numpy as np

# 직접 실행과 패키지 import 모두 지원
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
    # Load stopwords
    with open(os.path.join(data_dir, "stopwords.txt"), "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())

    # Load documents (books.txt)
    with open(os.path.join(data_dir, "books.txt"), "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    # Load metadata
    with open(os.path.join(data_dir, "book_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load queries
    with open(os.path.join(data_dir, "queries.txt"), "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    # Part A: TF-IDF search
    tfidf_matrix, vocab, tokenized_docs = build_tfidf_matrix(documents, stopwords)
    N, V = tfidf_matrix.shape

    # Compute IDF for search (recompute from tokenized_docs)
    idf = compute_idf(tokenized_docs, vocab)

    search_results = []
    for query in queries:
        top3 = search(query, tfidf_matrix, vocab, idf, stopwords, top_k=3)
        for item in top3:
            item["title"] = metadata[item["doc_index"]]["title"]
        search_results.append({"query": query, "top3": top3})

    # Part B: Recommendation
    sim_matrix = compute_book_similarity(tfidf_matrix)
    top5_similar = recommend_books(0, sim_matrix, metadata, top_k=5)
    same_category_top3 = recommend_by_category(0, sim_matrix, metadata, top_k=3)

    recommendation = {
        "target_book": {"index": 0, "title": metadata[0]["title"]},
        "top5_similar": top5_similar,
        "same_category_top3": same_category_top3,
    }

    # Part C: Sentiment analysis
    with open(os.path.join(data_dir, "sentiment_dict.json"), "r", encoding="utf-8") as f:
        sentiment_dict = json.load(f)

    with open(os.path.join(data_dir, "reviews.txt"), "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    reviews = []
    labels = []
    for line in lines:
        parts = line.rsplit("\t", 1)
        if len(parts) == 2:
            reviews.append(parts[0])
            labels.append(int(float(parts[1])))

    predictions = [rule_based_predict(r, sentiment_dict) for r in reviews]
    metrics = compute_metrics(predictions, labels)

    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count

    result = {
        "vocab_size": V,
        "tfidf_matrix_shape": [N, V],
        "search_results": search_results,
        "recommendation": recommendation,
        "sentiment_accuracy": metrics["accuracy"],
        "sentiment_precision": metrics["precision"],
        "sentiment_recall": metrics["recall"],
        "sentiment_f1": metrics["f1"],
        "total_reviews": len(reviews),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }

    return result


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "..", "data")
    result = main(data_dir)

    output_dir = os.path.join(base_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result_q2.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("result_q2.json saved.")
