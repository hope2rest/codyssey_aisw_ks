"""
문항 2: TF-IDF 문서 검색 + 규칙 기반 감성 분석

NumPy만 사용하여 구현하세요.
"""
import json
import os
import re
import unicodedata
from collections import Counter

import numpy as np


# ──────────────────────────────────────────────
# Part A: TF-IDF 문서 검색
# ──────────────────────────────────────────────

def preprocess(text: str, stopwords: set) -> list:
    """텍스트 전처리: NFC 정규화 → 소문자 → 특수문자 제거 → 토큰화 → 불용어/짧은 토큰 제거"""
    # TODO: 구현하세요
    pass


def cosine_similarity(a, b) -> float:
    """두 벡터 간 코사인 유사도 (영벡터면 0.0)"""
    # TODO: 구현하세요
    pass


def search(query_text, tfidf_matrix, vocab, word2idx, idf, stopwords, top_n=3):
    """쿼리에 대해 상위 top_n 문서 검색"""
    # TODO: 구현하세요
    pass


# ──────────────────────────────────────────────
# Part B: 규칙 기반 감성 분석
# ──────────────────────────────────────────────

def rule_based_predict(text: str, sentiment_dict: dict) -> int:
    """규칙 기반 감성 예측 (1=긍정, 0=부정)"""
    # TODO: 구현하세요
    pass


def compute_sentiment_metrics(predictions: list, labels: list) -> dict:
    """Accuracy, Precision, Recall, F1 계산"""
    # TODO: 구현하세요
    pass


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main(data_dir: str) -> dict:
    """전체 파이프라인 실행 및 result_q2.json 저장"""
    # TODO: 구현하세요
    pass


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    result = main(data_dir)
    with open(os.path.join(base_dir, "result_q2.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
