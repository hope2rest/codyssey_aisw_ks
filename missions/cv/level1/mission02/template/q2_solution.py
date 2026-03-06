import json
import os
import re
import unicodedata
from collections import Counter

import numpy as np


def preprocess(text, stopwords):
    # TODO: NFC 정규화 → 소문자 → 특수문자 제거 → 토큰화 → 불용어/짧은 토큰 제거


def cosine_similarity(a, b):
    # TODO: 두 벡터 간 코사인 유사도 (영벡터면 0.0)


def search(query_text, tfidf_matrix, vocab, word2idx, idf, stopwords, top_n=3):
    # TODO: 쿼리에 대해 상위 top_n 문서 검색


def rule_based_predict(text, sentiment_dict):
    # TODO: 규칙 기반 감성 예측 (부정어/강조어 처리, 1=긍정, 0=부정)


def compute_sentiment_metrics(predictions, labels):
    # TODO: Accuracy, Precision, Recall, F1 계산


def main(data_dir):
    # TODO: 전체 파이프라인 실행 및 result_q2.json 저장


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    result = main(data_dir)
    with open(os.path.join(base_dir, "result_q2.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
