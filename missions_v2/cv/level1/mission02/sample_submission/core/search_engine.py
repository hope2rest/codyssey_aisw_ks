"""
Part A: TF-IDF 기반 도서 검색 엔진
"""
import re
import unicodedata
from collections import Counter

import numpy as np


def preprocess(text: str, stopwords: set) -> list:
    """NFC 정규화 -> 소문자 -> 특수문자 제거 -> 토큰화 -> 불용어/짧은 토큰 제거"""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\uAC00-\uD7A3a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens


def compute_tf(tokens: list) -> dict:
    """TF(t, d) = count(t in d) / total_words(d)"""
    if not tokens:
        return {}
    total = len(tokens)
    counts = Counter(tokens)
    return {word: cnt / total for word, cnt in counts.items()}


def compute_idf(tokenized_docs: list, vocab: list) -> np.ndarray:
    """Smooth IDF: log((N+1)/(df(t)+1)) + 1"""
    N = len(tokenized_docs)
    V = len(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(V, dtype=np.float64)
    for tokens in tokenized_docs:
        seen = set(tokens)
        for word in seen:
            if word in word2idx:
                df[word2idx[word]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1
    return idf


def build_tfidf_matrix(documents: list, stopwords: set) -> tuple:
    """
    전처리된 문서들로부터 TF-IDF 행렬을 생성한다.
    반환: (tfidf_matrix, vocab_list, doc_tokens_list)
    """
    tokenized_docs = [preprocess(doc, stopwords) for doc in documents]
    N = len(documents)

    # Build vocab (sorted)
    vocab_set = set()
    for tokens in tokenized_docs:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    V = len(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}

    # TF matrix
    tf_matrix = np.zeros((N, V), dtype=np.float64)
    for i, tokens in enumerate(tokenized_docs):
        tf = compute_tf(tokens)
        for word, val in tf.items():
            if word in word2idx:
                tf_matrix[i, word2idx[word]] = val

    # IDF
    idf = compute_idf(tokenized_docs, vocab)

    # TF-IDF
    tfidf_matrix = tf_matrix * idf

    return tfidf_matrix, vocab, tokenized_docs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 (영벡터면 0.0)"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search(query_text: str, tfidf_matrix, vocab, idf, stopwords, top_k=3) -> list:
    """쿼리에 대해 상위 top_k 문서 검색"""
    q_tokens = preprocess(query_text, stopwords)
    V = len(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}

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
    return [{"doc_index": idx, "similarity": round(sim, 6)} for idx, sim in sims[:top_k]]
