## 문항 2 정답지 — TF-IDF 기반 도서 검색 및 추천 서비스

### 정답 코드

#### search_engine.py

```python
"""Part A: TF-IDF 기반 도서 검색 엔진"""
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
    """전처리된 문서들로부터 TF-IDF 행렬을 생성한다."""
    tokenized_docs = [preprocess(doc, stopwords) for doc in documents]
    N = len(documents)
    vocab_set = set()
    for tokens in tokenized_docs:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    V = len(vocab)
    word2idx = {w: i for i, w in enumerate(vocab)}
    tf_matrix = np.zeros((N, V), dtype=np.float64)
    for i, tokens in enumerate(tokenized_docs):
        tf = compute_tf(tokens)
        for word, val in tf.items():
            if word in word2idx:
                tf_matrix[i, word2idx[word]] = val
    idf = compute_idf(tokenized_docs, vocab)
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
```

#### recommender.py

```python
"""Part B: 도서 추천 서비스"""
import numpy as np
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


def recommend_books(book_index, similarity_matrix, metadata, top_k=5):
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


def recommend_by_category(book_index, similarity_matrix, metadata, top_k=3):
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
```

#### sentiment.py

```python
"""Part C: 규칙 기반 감성 분석"""


def rule_based_predict(text: str, sentiment_dict: dict) -> int:
    """규칙 기반 감성 예측 (부정어/강조어 처리, 1=긍정, 0=부정)"""
    positive = sentiment_dict["positive"]
    negative = sentiment_dict["negative"]
    negation = set(sentiment_dict["negation"])
    intensifier = sentiment_dict["intensifier"]
    tokens = text.split()
    total_score = 0.0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in negation and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            score = positive.get(next_token, negative.get(next_token, 0.0))
            total_score += score * -1
            i += 2
            continue
        if token in intensifier and i + 1 < len(tokens):
            multiplier = intensifier[token]
            next_token = tokens[i + 1]
            score = positive.get(next_token, negative.get(next_token, 0.0))
            total_score += score * multiplier
            i += 2
            continue
        score = positive.get(token, negative.get(token, 0.0))
        total_score += score
        i += 1
    return 1 if total_score > 0 else 0


def compute_metrics(predictions: list, labels: list) -> dict:
    """Accuracy, Precision, Recall, F1 계산"""
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
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | search_engine.py 필수 함수 6개 정의 | 5점 | AST 자동 |
| 2 | recommender.py 필수 함수 3개 정의 | 3점 | AST 자동 |
| 3 | sentiment.py 필수 함수 2개 정의 | 2점 | AST 자동 |
| 4 | main.py에 main 함수 정의 | 2점 | AST 자동 |
| 5 | 전처리: 소문자 변환, 특수문자 제거, 불용어 제거, 길이 1 이하 제거 | 5점 | import 자동 |
| 6 | TF 계산: count(t) / total_words | 5점 | import 자동 |
| 7 | Smooth IDF: log((N+1)/(df+1)) + 1 | 5점 | import 자동 |
| 8 | TF-IDF 행렬 shape (30 x vocab_size) | 5점 | import 자동 |
| 9 | 코사인 유사도 범위 (0~1), 영벡터 처리 | 5점 | import 자동 |
| 10 | 검색 결과 top_k 반환 (유사도 내림차순) | 5점 | import 자동 |
| 11 | 도서 유사도 행렬 (NxN), 대각선 1.0 | 5점 | import 자동 |
| 12 | 도서 추천 (자기 자신 제외, top_k) | 5점 | import 자동 |
| 13 | 카테고리별 추천 (동일 카테고리 내) | 5점 | import 자동 |
| 14 | 감성 분석: 부정어/강조어 처리 | 5점 | import 자동 |
| 15 | 평가 지표: accuracy, precision, recall, f1 | 5점 | import 자동 |
| 16 | result_q2.json 필수 키 확인 | 3점 | JSON 자동 |
| 17 | 검색 결과 구조 (query, top3) | 5점 | JSON 자동 |
| 18 | 추천 결과 구조 (target_book, top5_similar, same_category_top3) | 5점 | JSON 자동 |
| 19 | 감성 분석 accuracy > 0.5 | 5점 | JSON 자동 |
| 20 | 전체 리뷰 수 = 40 | 5점 | JSON 자동 |

- Pass 기준: 총 100점 중 100점 (20개 전체 정답)
- AI 트랩: Smooth IDF 수식 오류 (log 대신 log10), 코사인 유사도 영벡터 나눗셈, 부정어 처리 순서, 강조어 배수 적용 위치, 전처리 시 유니코드 NFC 정규화 누락

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `books.txt` | 텍스트 | 한 줄에 하나의 도서 설명 (30권) |
| `book_metadata.json` | JSON | title, author, category 메타데이터 |
| `stopwords.txt` | 텍스트 | 불용어 목록 |
| `reviews.txt` | TSV | 텍스트\t레이블 (40개 리뷰) |
| `sentiment_dict.json` | JSON | positive/negative/negation/intensifier 사전 |
| TF-IDF 행렬 | `np.ndarray` | (문서 수 x 단어 수) 실수 배열 |
| 코사인 유사도 | `float` | 0.0 ~ 1.0 범위 |
| 감성 점수 | `int` | 1(긍정) 또는 0(부정) |

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 텍스트 전처리 (토큰화, 불용어) | test_preprocess |
| TF-IDF 계산 (Smooth IDF) | test_compute_tf, test_compute_idf, test_build_tfidf_matrix |
| 코사인 유사도 | test_cosine_similarity, test_cosine_zero_vector |
| 정보 검색 (쿼리 기반 검색) | test_search |
| 유사도 기반 추천 | test_book_similarity_matrix, test_recommend_books, test_recommend_by_category |
| 규칙 기반 감성 분석 | test_sentiment_positive, test_sentiment_negative, test_sentiment_negation |
| 분류 평가 지표 | test_compute_metrics |
| 파이프라인 통합 | test_result_structure, test_search_results, test_recommendation, test_sentiment_accuracy |
