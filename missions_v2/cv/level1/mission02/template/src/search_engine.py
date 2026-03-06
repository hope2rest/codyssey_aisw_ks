"""
Part A: TF-IDF 기반 도서 검색 엔진
"""
import re
import unicodedata
from collections import Counter

import numpy as np


def preprocess(text: str, stopwords: set) -> list:
    """NFC 정규화 -> 소문자 -> 특수문자 제거 -> 토큰화 -> 불용어/짧은 토큰 제거"""
    # TODO: 구현하세요
    pass


def compute_tf(tokens: list) -> dict:
    """TF(t, d) = count(t in d) / total_words(d)"""
    # TODO: 구현하세요
    pass


def compute_idf(tokenized_docs: list, vocab: list) -> np.ndarray:
    """Smooth IDF: log((N+1)/(df(t)+1)) + 1"""
    # TODO: 구현하세요
    pass


def build_tfidf_matrix(documents: list, stopwords: set) -> tuple:
    """
    전처리된 문서들로부터 TF-IDF 행렬을 생성한다.
    반환: (tfidf_matrix, vocab_list, doc_tokens_list)
    """
    # TODO: 구현하세요
    pass


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 (영벡터면 0.0)"""
    # TODO: 구현하세요
    pass


def search(query_text: str, tfidf_matrix, vocab, idf, stopwords, top_k=3) -> list:
    """쿼리에 대해 상위 top_k 문서 검색"""
    # TODO: 구현하세요
    pass
