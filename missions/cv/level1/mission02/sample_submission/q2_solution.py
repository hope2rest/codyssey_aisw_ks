"""
문항 2: TF-IDF 문서 검색 + 규칙 기반 감성 분석 (정답 코드)
"""
import json
import os
import re
import unicodedata
from collections import Counter

import numpy as np


def preprocess(text: str, stopwords: set) -> list:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\uAC00-\uD7A3a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens


def cosine_similarity(a, b) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search(query_text, tfidf_matrix, vocab, word2idx, idf, stopwords, top_n=3):
    q_tokens = preprocess(query_text, stopwords)
    V = len(vocab)
    q_tf = np.zeros(V, dtype=np.float64)
    if q_tokens:
        total = len(q_tokens)
        counts = Counter(q_tokens)
        for word, cnt in counts.items():
            if word in word2idx:
                q_tf[word2idx[word]] = cnt / total
    q_tfidf = q_tf * idf

    N = tfidf_matrix.shape[0]
    sims = [(i, cosine_similarity(q_tfidf, tfidf_matrix[i])) for i in range(N)]
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:top_n]


def rule_based_predict(text: str, sentiment_dict: dict) -> int:
    positive = sentiment_dict["positive"]
    negative = sentiment_dict["negative"]
    negation = set(sentiment_dict["negation"])
    intensifier = sentiment_dict["intensifier"]

    tokens = text.split()
    total_score = 0.0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Check if negation
        if token in negation and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            score = positive.get(next_token, negative.get(next_token, 0.0))
            total_score += score * -1
            i += 2
            continue
        # Check if intensifier
        if token in intensifier and i + 1 < len(tokens):
            multiplier = intensifier[token]
            next_token = tokens[i + 1]
            score = positive.get(next_token, negative.get(next_token, 0.0))
            total_score += score * multiplier
            i += 2
            continue
        # Regular token
        score = positive.get(token, negative.get(token, 0.0))
        total_score += score
        i += 1

    return 1 if total_score > 0 else 0


def compute_sentiment_metrics(predictions: list, labels: list) -> dict:
    n = len(predictions)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / n if n > 0 else 0.0

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def main(data_dir: str) -> dict:
    # Load stopwords
    with open(os.path.join(data_dir, "stopwords.txt"), "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())

    # Load documents
    with open(os.path.join(data_dir, "documents.txt"), "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f if line.strip()]

    # Load queries
    with open(os.path.join(data_dir, "queries.txt"), "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    # Preprocess documents
    tokenized_docs = [preprocess(doc, stopwords) for doc in documents]
    N = len(documents)

    # Build vocab
    vocab_set = set()
    for tokens in tokenized_docs:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    V = len(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}

    # TF matrix
    tf_matrix = np.zeros((N, V), dtype=np.float64)
    for i, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue
        total = len(tokens)
        counts = Counter(tokens)
        for word, cnt in counts.items():
            if word in word2idx:
                tf_matrix[i, word2idx[word]] = cnt / total

    # IDF
    df = np.sum(tf_matrix > 0, axis=0)
    idf = np.log((N + 1) / (df + 1)) + 1

    # TF-IDF
    tfidf_matrix = tf_matrix * idf

    # Search
    search_results = []
    for query in queries:
        top3 = search(query, tfidf_matrix, vocab, word2idx, idf, stopwords)
        search_results.append({
            "query": query,
            "top3": [
                {"doc_index": int(idx), "similarity": round(float(sim), 6)}
                for idx, sim in top3
            ]
        })

    # Part B: Sentiment analysis
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
    metrics = compute_sentiment_metrics(predictions, labels)

    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count

    return {
        "vocab_size": V,
        "tfidf_matrix_shape": [N, V],
        "search_results": search_results,
        "sentiment_accuracy": metrics["accuracy"],
        "sentiment_precision": metrics["precision"],
        "sentiment_recall": metrics["recall"],
        "sentiment_f1": metrics["f1"],
        "total_reviews": len(reviews),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    result = main(data_dir)
    with open(os.path.join(base_dir, "result_q2.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("result_q2.json saved.")
