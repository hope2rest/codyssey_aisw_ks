## 문항 2 정답지 — TF-IDF 기반 도서 검색 및 추천 서비스

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `books.txt` | 텍스트 | 한 줄에 하나의 도서 설명 (30권) |
| `book_metadata.json` | JSON | title, author, category 메타데이터 |
| `stopwords.txt` | 텍스트 | 불용어 목록 |
| `reviews.txt` | TSV | 텍스트\t레이블 (40개 리뷰) |
| `sentiment_dict.json` | JSON | positive/negative/negation/intensifier 사전 |
| TF-IDF 행렬 | `np.ndarray` | (문서 수 × 단어 수) 실수 배열 |
| 코사인 유사도 | `float` | 0.0 ~ 1.0 범위 |
| 감성 점수 | `int` | 1(긍정) 또는 0(부정) |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | search_engine.py 필수 함수 6개 정의 (preprocess, compute_tf, compute_idf, build_tfidf_matrix, cosine_similarity, search) | AST 자동 |
| 2 | recommender.py 필수 함수 3개 정의 (compute_book_similarity, recommend_books, recommend_by_category) | AST 자동 |
| 3 | sentiment.py 필수 함수 2개 정의 (rule_based_predict, compute_metrics) | AST 자동 |
| 4 | main.py에 main 함수 정의 | AST 자동 |
| 5 | 전처리: 소문자 변환, 특수문자 제거, 불용어 제거, 길이 1 이하 제거 | import 자동 |
| 6 | TF 계산: count(t) / total_words | import 자동 |
| 7 | Smooth IDF: log((N+1)/(df+1)) + 1 | import 자동 |
| 8 | TF-IDF 행렬 shape (30 × vocab_size) | import 자동 |
| 9 | 코사인 유사도 범위 (0~1), 영벡터 처리 | import 자동 |
| 10 | 검색 결과 top_k 반환 (유사도 내림차순) | import 자동 |
| 11 | 도서 유사도 행렬 (N×N), 대각선 1.0 | import 자동 |
| 12 | 도서 추천 (자기 자신 제외, top_k) | import 자동 |
| 13 | 카테고리별 추천 (동일 카테고리 내) | import 자동 |
| 14 | 감성 분석: 부정어/강조어 처리 | import 자동 |
| 15 | 평가 지표: accuracy, precision, recall, f1 | import 자동 |
| 16 | result_q2.json 필수 키 확인 | JSON 자동 |
| 17 | 검색 결과 구조 (query, top3) | JSON 자동 |
| 18 | 추천 결과 구조 (target_book, top5_similar, same_category_top3) | JSON 자동 |
| 19 | 감성 분석 accuracy > 0.5 | JSON 자동 |
| 20 | 전체 리뷰 수 = 40 | JSON 자동 |

- Pass 기준: 20개 전체 통과
- AI 트랩: Smooth IDF 수식 오류 (log 대신 log10), 코사인 유사도 영벡터 나눗셈, 부정어 처리 순서, 강조어 배수 적용 위치, 전처리 시 유니코드 NFC 정규화 누락

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
