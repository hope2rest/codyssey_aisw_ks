## 문항 4: 고객 리뷰 감성 분석 (규칙 기반 + ML 파이프라인 + SHAP 해석)

### 문제

고객 리뷰 데이터(`reviews.csv`)에 대해 규칙 기반과 ML 기반 두 가지 방식의 감성 분석 시스템을 구축하고, 성능을 비교 분석하여 SHAP 해석과 비즈니스 요약까지 도출하시오.
- 데이터 위치: `data/reviews.csv` (text, label 열), `data/sentiment_dict.json` (감성 사전)
- 감성 사전 구조: `positive`(긍정 단어-점수), `negative`(부정 단어-점수), `negation`(부정어 리스트), `intensifier`(강조어-배수)

> **주의:** `reviews.csv`의 label 분포는 **불균형 상태**입니다.

### 데이터 구조

`sentiment_dict.json` 예시:
```json
{
  "positive": {"좋다": 1.0, "훌륭하다": 1.5, ...},
  "negative": {"나쁘다": -1.0, "불만": -1.5, ...},
  "negation": ["안", "않", "못", "없"],
  "intensifier": {"매우": 1.5, "정말": 1.3, "너무": 1.2, ...}
}
```

`result_q4.json` 출력 구조:
```json
{
  "rule_based": {
    "accuracy": 실수, "precision_pos": 실수, "recall_pos": 실수,
    "precision_neg": 실수, "recall_neg": 실수, "f1_macro": 실수
  },
  "ml_based": {
    "accuracy": 실수, "precision_pos": 실수, "recall_pos": 실수,
    "precision_neg": 실수, "recall_neg": 실수, "f1_macro": 실수
  },
  "shap_top5_positive": [{"word": "문자열", "shap_value": 실수}, ...],
  "shap_top5_negative": [{"word": "문자열", "shap_value": 실수}, ...],
  "business_summary": "3문장 이내 비기술적 요약"
}
```

### 구현 요구사항

#### Part A: 규칙 기반 감성 분석

#### 1. 감성 사전 로드 및 전처리
- `sentiment_dict.json`을 로드하여 positive, negative, negation, intensifier 사전을 준비
- `reviews.csv`를 로드하고 결측치를 처리

#### 2. `rule_based_predict(text, sentiment_dict) -> int` 규칙 기반 감성 점수 산출
- (a) 텍스트를 토큰 단위로 분리
- (b) 각 토큰의 감성 점수를 사전에서 조회
- (c) **부정어 처리**: 부정어 바로 다음 토큰의 감성 점수에 **-1을 곱함**
- (d) **강조어 처리**: 강조어 바로 다음 토큰의 감성 점수에 해당 **배수를 곱함**
- (e) 전체 감성 점수 = 모든 토큰 감성 점수의 합
- (f) 감성 점수 > 0 이면 긍정(1), 아니면 부정(0) 반환

#### Part B: ML 기반 감성 분석

#### 3. 데이터 분할
- train(70%) / test(30%), `random_state=42`

#### 4. 불균형 처리
- train 데이터의 클래스 불균형을 해소 (오버샘플링 등)
- test 데이터는 원본 유지

#### 5. TF-IDF 벡터화
- `sklearn`의 `TfidfVectorizer` 사용
- `fit_transform`은 **train 데이터에서만** 수행, test 데이터에는 `transform`만 적용

#### 6. 모델 학습 및 예측
- `LogisticRegression(random_state=42)`으로 학습 및 예측

#### Part C: 비교 분석 및 SHAP 해석

#### 7. `compute_metrics(y_true, y_pred) -> dict` 성능 비교
- 두 접근법의 test 데이터 성능을 다음 지표로 비교: Accuracy, Precision (긍정/부정 각각), Recall (긍정/부정 각각), F1-score (Macro 평균)

#### 8. SHAP 해석
- `shap.LinearExplainer`를 사용하여 ML 모델의 SHAP 값 계산
- 긍정 예측에 가장 기여한 상위 5개 단어와 SHAP 값 추출
- 부정 예측에 가장 기여한 상위 5개 단어와 SHAP 값 추출

#### 9. 비즈니스 요약
- 분석 결과를 영업팀/마케팅팀이 이해할 수 있는 **3문장 이내**의 비기술적 언어로 요약
- 요약에 '긍정'/'부정' 키워드를 반드시 포함

### 제약 사항
- 규칙 기반 파트는 `sklearn` 사용 금지 (NumPy + 감성 사전만)
- ML 파트에서 `TfidfVectorizer`의 **fit은 train 데이터에서만** 수행
- 불균형 처리는 **train 데이터에서만** 수행 (test 데이터는 원본 유지)
- `shap` 라이브러리 사용 필수
- 모든 수치는 **소수점 이하 4자리**로 반올림

### 제출 방식
- `q4_solution.py`와 `result_q4.json` 두 파일을 제출
- 두 파일 모두 제출해야 채점이 진행됩니다
