## 문항 2: TF-IDF 직접 구현 및 코사인 유사도 기반 문서 검색

### 문제

20개의 한국어 기술 문서에 대해 TF-IDF 벡터화 및 코사인 유사도로 상위 3개 문서를 검색하는 시스템을 구현하세요.
- 문서 데이터는 `data/documents.txt`에 한 줄에 하나의 문서(총 20줄)로 저장되어 있으며, 불용어는 `data/stopwords.txt`, 검색 쿼리는 `data/queries.txt`(총 5개)에 저장되어 있습니다.
- TF-IDF는 단어의 문서 내 빈도(TF)와 전체 문서 집합에서의 희소성(IDF)을 결합하여 단어의 중요도를 수치화하는 기법이며, 코사인 유사도는 두 벡터 간의 방향 유사성을 -1에서 1 사이의 값으로 측정하는 방법입니다.

### 입력 파일 구조

| 파일 | 설명 |
|------|------|
| `data/documents.txt` | 한 줄에 하나의 문서 (총 20줄) |
| `data/stopwords.txt` | 한 줄에 하나의 불용어 |
| `data/queries.txt` | 한 줄에 하나의 검색 쿼리 (총 5개) |

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
  ]
}
```

- 모든 유사도는 소수점 이하 6자리로 반올림합니다.

### 구현 요구사항

#### 1. `preprocess(text, stopwords) -> list[str]`
- 텍스트에 유니코드 NFC 정규화를 적용합니다.
- 모든 텍스트를 소문자로 변환합니다.
- 정규표현식(`re`)을 사용하여 한글, 영문, 숫자를 제외한 모든 문자를 제거합니다.
- 공백 기준으로 토큰화합니다.
- `stopwords.txt`에 포함된 토큰을 제거합니다.
- 길이 1 이하인 토큰을 제거합니다.

#### 2. TF(Term Frequency) 계산
- 각 문서에서 각 단어의 등장 빈도를 해당 문서의 전체 단어 수로 나눈 값을 계산합니다.
- `TF(t, d) = count(t in d) / total_words(d)`

#### 3. IDF(Inverse Document Frequency) 계산 (Smooth IDF)
- Smooth IDF 수식을 적용하여 계산합니다.
- `IDF(t) = log((N + 1) / (df(t) + 1)) + 1`
- `N`은 전체 문서 수, `df(t)`는 단어 t가 등장하는 문서 수, `log`는 자연로그(`np.log`)입니다.

#### 4. TF-IDF 행렬 생성
- NumPy 배열로 (문서 수 x 단어 수) 크기의 행렬을 생성합니다.
- 단어(열) 순서는 전체 코퍼스의 단어를 사전순(알파벳/한글순)으로 정렬합니다.

#### 5. `cosine_similarity(a, b) -> float`
- 두 벡터의 코사인 유사도를 계산합니다.
- `cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b))`
- 두 벡터 중 하나라도 영벡터(norm=0)이면 유사도는 0.0으로 반환합니다.

#### 6. `search(query_text, top_n=3) -> list`
- 쿼리 문자열에 동일한 전처리 및 TF-IDF 변환을 적용합니다.
- 각 문서와의 코사인 유사도를 계산하여 상위 3개 문서의 인덱스와 유사도를 반환합니다.

### 제약 사항
- NumPy만 사용하여 모든 수치 계산을 수행합니다 (`sklearn`, `scipy` 사용 금지).
- 정규표현식은 `re` 모듈을 사용합니다.
- Smooth IDF 수식을 정확히 적용해야 합니다 (일반 IDF가 아닌 위 수식).
- 쿼리의 IDF는 기존 코퍼스 기준으로 계산합니다 (쿼리를 코퍼스에 추가하지 않음).
- 빈 줄은 문서로 포함하지 않습니다.

### 제출 방식
- `q2_solution.py`와 `result_q2.json` 두 파일을 제출합니다.
- `template/q2_solution.py`의 `# TODO` 부분을 채우세요.
