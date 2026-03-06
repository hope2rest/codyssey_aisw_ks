## 문항: 도서 검색 및 추천 서비스

### 문제

도서관 소장 도서 데이터에 대해 TF-IDF 기반 검색 엔진을 구현하고, 유사도 기반 도서 추천과 사용자 리뷰 감성 분석을 수행하는 서비스를 구현하세요.

### 제공 데이터

```
data/
├── books.txt                  # 도서 설명 텍스트 (한 줄에 하나, 30권)
├── book_metadata.json         # 도서 메타데이터 (제목, 저자, 카테고리)
├── stopwords.txt              # 불용어 목록 (한 줄에 하나)
├── queries.txt                # 검색 쿼리 (한 줄에 하나, 5개)
├── reviews.txt                # 사용자 리뷰 (텍스트\t레이블, 40개)
└── sentiment_dict.json        # 감성 사전 (positive/negative/negation/intensifier)
```

- `books.txt`: TF-IDF 검색 대상이 되는 30권의 도서 설명 텍스트입니다. 빈 줄은 문서로 포함하지 않습니다.
- `book_metadata.json`: `[{"title": str, "author": str, "category": str}, ...]` 형태의 도서 메타데이터입니다. 순서는 `books.txt`와 동일합니다.
- `stopwords.txt`: 전처리 시 제거할 한국어 불용어 목록입니다.
- `queries.txt`: 도서 검색에 사용할 5개의 검색 쿼리입니다.
- `reviews.txt`: 탭 구분(`텍스트\t레이블`)으로 구성된 40개 도서 리뷰입니다. 레이블은 1(긍정) 또는 0(부정)입니다.
- `sentiment_dict.json`: 긍정/부정 단어, 부정어, 강조어를 포함한 감성 분석 사전입니다.

### 프로젝트 구조

| 폴더/파일 | 역할 | 핵심 함수 |
|-----------|------|----------|
| `core/search_engine.py` | 전처리, TF-IDF 행렬, 코사인 유사도, 검색 | `preprocess()`, `compute_tf()`, `compute_idf()`, `build_tfidf_matrix()`, `cosine_similarity()`, `search()` |
| `core/recommender.py` | 도서 간 유사도 기반 추천 | `compute_book_similarity()`, `recommend_books()`, `recommend_by_category()` |
| `core/sentiment.py` | 규칙 기반 감성 분석, 평가 지표 | `rule_based_predict()`, `compute_metrics()` |
| `core/main.py` | 전체 서비스 파이프라인 실행 | `main()` |
| `charts/search_charts.py` | 검색 결과 시각화 | `save_search_results_chart()` |
| `charts/recommend_charts.py` | 추천 결과 시각화 | `save_similarity_heatmap()`, `save_recommendation_chart()` |
| `charts/sentiment_charts.py` | 감성 분석 시각화 | `save_sentiment_distribution()`, `save_sentiment_metrics_chart()` |
| `dashboard/app.py` | Streamlit 대시보드 메인 앱 | — |
| `dashboard/pages/` | 검색, 추천, 감성 분석 페이지 | `render_search_results()`, `render_recommendations()`, `render_sentiment_summary()` |
| `dashboard/components/` | 재사용 UI 컴포넌트 | `validate_query()`, `format_book_card()`, `generate_all_charts()` |

### 구현 요구사항

#### Part A: search_engine.py - TF-IDF 검색 엔진

#### 1. `preprocess(text: str, stopwords: set) -> list[str]`
- 유니코드 NFC 정규화를 적용합니다.
- 모든 텍스트를 소문자로 변환합니다.
- `re`를 사용하여 한글, 영문, 숫자를 제외한 모든 문자를 제거합니다.
- 공백 기준으로 토큰화합니다.
- 불용어를 제거하고, 길이 1 이하인 토큰을 제거합니다.

#### 2. `compute_tf(tokens: list) -> dict`
- `TF(t, d) = count(t in d) / total_words(d)`

#### 3. `compute_idf(documents: list, vocab: list) -> np.ndarray`
- Smooth IDF: `IDF(t) = log((N + 1) / (df(t) + 1)) + 1`
- `log`는 자연로그(`np.log`)

#### 4. `build_tfidf_matrix(documents: list, stopwords: set) -> tuple`
- 전처리된 문서들로부터 TF-IDF 행렬을 생성합니다.
- 단어(열) 순서는 사전순(알파벳/한글순)으로 정렬합니다.
- NumPy 배열로 (문서 수 × 단어 수) 크기의 행렬을 생성합니다.
- 반환: `(tfidf_matrix, vocab_list, doc_tokens_list)`

#### 5. `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`
- `cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b))`
- 영벡터(norm=0)이면 유사도 0.0을 반환합니다.

#### 6. `search(query_text: str, tfidf_matrix, vocab, idf, stopwords, top_k=3) -> list`
- 쿼리에 동일한 전처리 및 TF-IDF 변환을 적용합니다.
- 쿼리의 IDF는 기존 코퍼스 기준으로 계산합니다 (쿼리를 코퍼스에 추가하지 않음).
- 각 문서와의 코사인 유사도를 계산하여 상위 `top_k`개를 반환합니다.
- 반환: `[{"doc_index": int, "similarity": float, "title": str}, ...]`

#### Part B: recommender.py - 도서 추천 서비스

#### 7. `compute_book_similarity(tfidf_matrix: np.ndarray) -> np.ndarray`
- 모든 도서 쌍의 코사인 유사도를 계산하여 (N × N) 유사도 행렬을 반환합니다.
- 대각선은 1.0입니다.

#### 8. `recommend_books(book_index: int, similarity_matrix: np.ndarray, metadata: list, top_k: int = 5) -> list`
- 지정된 도서와 유사도가 높은 상위 `top_k`권을 추천합니다 (자기 자신 제외).
- 반환: `[{"index": int, "title": str, "similarity": float, "category": str}, ...]`

#### 9. `recommend_by_category(book_index: int, similarity_matrix: np.ndarray, metadata: list, top_k: int = 3) -> list`
- 동일 카테고리 내에서 유사도가 높은 상위 `top_k`권을 추천합니다.
- 반환: `[{"index": int, "title": str, "similarity": float}, ...]`

#### Part C: sentiment.py - 감성 분석

#### 10. `rule_based_predict(text: str, sentiment_dict: dict) -> int`
- 텍스트를 공백으로 토큰화합니다.
- 각 토큰의 감성 점수를 positive/negative 사전에서 조회합니다 (positive 우선).
- **부정어 처리**: 부정어(`negation`) 바로 다음 토큰의 감성 점수에 -1을 곱합니다.
- **강조어 처리**: 강조어(`intensifier`) 바로 다음 토큰의 감성 점수에 해당 배수를 곱합니다.
- 전체 감성 점수 합 > 0이면 1(긍정), 아니면 0(부정)을 반환합니다.

#### 11. `compute_metrics(predictions: list, labels: list) -> dict`
- Accuracy: 정확히 맞춘 수 / 전체 수
- Precision (긍정): TP / (TP + FP)
- Recall (긍정): TP / (TP + FN)
- F1 (긍정): 2 × Precision × Recall / (Precision + Recall)
- TP: 예측=1 & 실제=1, FP: 예측=1 & 실제=0, FN: 예측=0 & 실제=1

#### Part D: main.py - 서비스 파이프라인

#### 12. `main(data_dir: str) -> dict`
- Part A (TF-IDF 검색), Part B (도서 추천), Part C (감성 분석)를 실행합니다.
- 5개 쿼리 각각에 대해 상위 3개 검색 결과를 반환합니다.
- 첫 번째 도서에 대해 전체 추천과 카테고리별 추천을 수행합니다.
- 40개 리뷰에 대해 감성 분석 후 평가 지표를 계산합니다.
- `result_q2.json` 파일로 결과를 저장합니다.

#### Part E: charts/ - 시각화 모듈

#### 13. `save_search_results_chart(search_results, output_path)`
- 검색 쿼리별 상위 3건의 유사도를 수평 바 차트로 저장합니다.
- `matplotlib`을 사용하여 PNG 파일로 저장합니다.

#### 14. `save_similarity_heatmap(sim_matrix, titles, output_path)`
- 도서 간 유사도 행렬을 히트맵으로 저장합니다.

#### 15. `save_recommendation_chart(recommendations, target_title, output_path)`
- 추천 도서 상위 5건의 유사도를 수평 바 차트로 저장합니다.

#### 16. `save_sentiment_distribution(positive_count, negative_count, output_path)`
- 긍정/부정 분포를 파이 차트로 저장합니다.

#### 17. `save_sentiment_metrics_chart(metrics, output_path)`
- 감성 분석 평가 지표(Accuracy, Precision, Recall, F1)를 바 차트로 저장합니다.

#### Part F: dashboard/ - 대시보드 서비스

#### 18. `dashboard/app.py`
- Streamlit 기반 대시보드 메인 앱을 구현합니다.
- 검색, 추천, 감성 분석 3개 탭으로 구성합니다.
- `result_q2.json`을 로드하여 결과를 시각적으로 표시합니다.

#### 19. `dashboard/pages/`
- `search.py`: 검색 결과를 렌더링 가능한 데이터로 변환합니다.
- `recommend.py`: 추천 결과를 렌더링 가능한 데이터로 변환합니다.
- `sentiment.py`: 감성 분석 요약을 렌더링 가능한 데이터로 변환합니다.

#### 20. `dashboard/components/`
- `search_bar.py`: 검색 쿼리 검증 컴포넌트입니다.
- `book_card.py`: 도서 정보 카드 형태 컴포넌트입니다.
- `chart_builder.py`: 모든 차트를 일괄 생성하는 유틸리티입니다.

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
        {"doc_index": 정수, "similarity": 실수, "title": "문자열"},
        {"doc_index": 정수, "similarity": 실수, "title": "문자열"},
        {"doc_index": 정수, "similarity": 실수, "title": "문자열"}
      ]
    }
  ],
  "recommendation": {
    "target_book": {"index": 0, "title": "문자열"},
    "top5_similar": [
      {"index": 정수, "title": "문자열", "similarity": 실수, "category": "문자열"}
    ],
    "same_category_top3": [
      {"index": 정수, "title": "문자열", "similarity": 실수}
    ]
  },
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

### 제출 폴더 구조

다음 폴더 구조를 zip으로 묶어 제출합니다.

```
submission/
├── core/
│   ├── search_engine.py
│   ├── recommender.py
│   ├── sentiment.py
│   └── main.py
├── dashboard/
│   ├── app.py
│   ├── pages/
│   │   ├── search.py
│   │   ├── recommend.py
│   │   └── sentiment.py
│   ├── components/
│   │   ├── search_bar.py
│   │   ├── book_card.py
│   │   └── chart_builder.py
│   └── assets/
│       └── style.css
├── charts/
│   ├── search_charts.py
│   ├── recommend_charts.py
│   └── sentiment_charts.py
├── config/
│   └── config.json
├── output/
│   ├── result_q2.json
│   └── charts/
│       ├── search_results.png
│       ├── similarity_heatmap.png
│       ├── recommendation_top5.png
│       ├── sentiment_distribution.png
│       └── sentiment_metrics.png
└── requirements.txt
```

- `template/` 디렉토리의 각 파일의 `# TODO` 부분을 채우세요.
- `streamlit run dashboard/app.py`로 대시보드를 실행할 수 있습니다.
