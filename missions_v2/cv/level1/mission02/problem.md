## 문항: TF-IDF 문서 검색

### 문제

20개의 한국어 기술 문서에 대해 TF-IDF 기반 문서 검색을 구현하고, 30개의 고객 리뷰에 대해 규칙 기반 감성 분석을 수행하는 프로그램을 구현하세요.

### 제공 데이터

```
data/
├── documents.txt          # 한국어 기술 문서 (한 줄에 하나, 20개)
├── stopwords.txt          # 불용어 목록 (한 줄에 하나)
├── queries.txt            # 검색 쿼리 (한 줄에 하나, 5개)
├── reviews.txt            # 고객 리뷰 (텍스트\t레이블, 30개)
└── sentiment_dict.json    # 감성 사전 (positive/negative/negation/intensifier)
```

- `documents.txt`: TF-IDF 검색 대상이 되는 한국어 기술 문서 코퍼스입니다. 빈 줄은 문서로 포함하지 않습니다.
- `stopwords.txt`: 전처리 시 제거할 불용어 목록입니다.
- `queries.txt`: 문서 검색에 사용할 5개의 검색 쿼리입니다.
- `reviews.txt`: 탭 구분(`텍스트\t레이블`)으로 구성된 30개 리뷰입니다. 레이블은 1(긍정) 또는 0(부정)입니다.
- `sentiment_dict.json`: 긍정/부정 단어, 부정어, 강조어를 포함한 감성 분석 사전입니다.

### 구현 요구사항

#### Part A: TF-IDF 문서 검색

#### 1. `preprocess(text: str, stopwords: set) -> list[str]`
- 유니코드 NFC 정규화를 적용합니다.
- 모든 텍스트를 소문자로 변환합니다.
- `re`를 사용하여 한글, 영문, 숫자를 제외한 모든 문자를 제거합니다.
- 공백 기준으로 토큰화합니다.
- 불용어를 제거하고, 길이 1 이하인 토큰을 제거합니다.

#### 2. TF(Term Frequency) 계산
- `TF(t, d) = count(t in d) / total_words(d)`

#### 3. IDF(Inverse Document Frequency) 계산 (Smooth IDF)
- `IDF(t) = log((N + 1) / (df(t) + 1)) + 1`
- `log`는 자연로그(`np.log`)

#### 4. TF-IDF 행렬 생성
- NumPy 배열로 (문서 수 x 단어 수) 크기의 행렬을 생성합니다.
- 단어(열) 순서는 사전순(알파벳/한글순)으로 정렬합니다.

#### 5. `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`
- `cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b))`
- 영벡터(norm=0)이면 유사도 0.0을 반환합니다.

#### 6. `search(query_text: str, ...) -> list`
- 쿼리에 동일한 전처리 및 TF-IDF 변환을 적용합니다.
- 쿼리의 IDF는 기존 코퍼스 기준으로 계산합니다 (쿼리를 코퍼스에 추가하지 않음).
- 각 문서와의 코사인 유사도를 계산하여 상위 3개를 반환합니다.

#### Part B: 규칙 기반 감성 분석

#### 7. `rule_based_predict(text: str, sentiment_dict: dict) -> int`
- 텍스트를 공백으로 토큰화합니다.
- 각 토큰의 감성 점수를 positive/negative 사전에서 조회합니다 (positive 우선).
- **부정어 처리**: 부정어(`negation`) 바로 다음 토큰의 감성 점수에 -1을 곱합니다.
- **강조어 처리**: 강조어(`intensifier`) 바로 다음 토큰의 감성 점수에 해당 배수를 곱합니다.
- 전체 감성 점수 합 > 0이면 1(긍정), 아니면 0(부정)을 반환합니다.

#### 8. `compute_sentiment_metrics(predictions: list, labels: list) -> dict`
- Accuracy: 정확히 맞춘 수 / 전체 수
- Precision (긍정): TP / (TP + FP)
- Recall (긍정): TP / (TP + FN)
- F1 (긍정): 2 * Precision * Recall / (Precision + Recall)
- TP: 예측=1 & 실제=1, FP: 예측=1 & 실제=0, FN: 예측=0 & 실제=1

#### 9. `main(data_dir: str) -> dict`
- Part A (TF-IDF)와 Part B (감성 분석)를 실행하고 `result_q2.json`을 저장합니다.

### 출력 형식

`result_q2.json` 파일로 다음 구조를 저장합니다:

```json
{
  "vocab_size": 정수,
  "tfidf_matrix_shape": [정수, 정수],
  "search_results": [
    {
      "query": "쿼리 텍스트",
      "top3": [
        {"doc_index": 정수, "similarity": 실수},
        {"doc_index": 정수, "similarity": 실수},
        {"doc_index": 정수, "similarity": 실수}
      ]
    }
  ],
  "sentiment_accuracy": 실수,
  "sentiment_precision": 실수,
  "sentiment_recall": 실수,
  "sentiment_f1": 실수,
  "total_reviews": 정수,
  "positive_count": 정수,
  "negative_count": 정수
}
```

- 모든 유사도 및 지표는 소수점 이하 6자리로 반올림합니다.

### 제약 사항
- NumPy만 사용하여 모든 수치 계산을 수행합니다 (`sklearn`, `scipy` 사용 금지).
- 정규표현식은 `re` 모듈을 사용합니다.
- Smooth IDF 수식을 정확히 적용해야 합니다.
- 쿼리의 IDF는 기존 코퍼스 기준으로 계산합니다.
- 빈 줄은 문서로 포함하지 않습니다.

### 제출 방식
- `q2_solution.py`, `result_q2.json` 총 2개 파일을 zip으로 묶어 제출합니다.
- `template/q2_solution.py`의 `# TODO` 부분을 채우세요.
