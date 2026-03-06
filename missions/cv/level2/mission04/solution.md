## 문항 4: 풀이 가이드

### 채점 기준 (config.yaml)

| ID | 설명 | 배점 |
|----|------|------|
| 1 | rule_based 6개 지표 완성 + accuracy 검증 (오차 < 0.03) | 10pts |
| 2 | ML accuracy (허용 오차 < 0.01) | 15pts |
| 3 | ML F1 macro (허용 오차 < 0.01) | 15pts |
| 4 | SHAP 부호 정확 (positive 5개 양수, negative 5개 음수) | 15pts |
| 5 | 데이터 누수 방지 (fit_transform + transform 패턴) | 15pts |
| 6 | 비즈니스 요약 ('긍정'/'부정' 포함, 20자 이상) | 15pts |
| 7 | ml_based 6개 지표 완성 | 15pts |

Pass 기준: 7개 모두 충족

### AI Trap 주의사항

1. **데이터 누수**: `fit_transform`은 학습 데이터에만 사용하고, 테스트 데이터에는 `transform`만 적용해야 한다. 전체 데이터에 `fit_transform`을 사용하면 데이터 누수가 발생하여 채점 항목 5번이 실패한다.

2. **오버샘플링**: 클래스 불균형 처리를 반드시 train 데이터에서만 수행해야 한다. 소수 클래스(부정=0)를 다수 클래스(긍정=1) 수만큼 오버샘플링한다. `random_state=42` 사용.

3. **business_summary**: 반드시 '긍정'과 '부정' 키워드를 모두 포함해야 한다. 누락 시 채점 항목 6번이 실패한다.

### 풀이 흐름

#### Part A: 규칙 기반

1. `reviews.csv` 로드 후 NaN label 필터링, NFC 유니코드 정규화 수행
2. `sentiment_dict.json` 로드
3. 규칙 기반 예측 함수 구현:
   - 텍스트를 `split()`으로 토큰화
   - while 루프로 순회하면서 부정어/강조어 처리 (부정어 만나면 다음 토큰 점수에 -1 곱, 강조어 만나면 다음 토큰 점수에 배수 곱, 둘 다 `i += 2`로 건너뜀)
   - positive 사전 우선 조회, 없으면 negative 사전 조회
   - 점수 합 > 0 이면 1(긍정), 아니면 0(부정)
4. 전체 데이터에 적용하여 예측값 저장 (test 인덱스 추출용)

#### Part B: ML 기반

5. `train_test_split(test_size=0.30, random_state=42)`로 분할
6. train에서 소수 클래스 오버샘플링:
   ```python
   neg_oversampled = neg_df.sample(n=len(pos_df), replace=True, random_state=42)
   bal = pd.concat([pos_df, neg_oversampled]).reset_index(drop=True)
   ```
7. TF-IDF: `TfidfVectorizer(sublinear_tf=False, smooth_idf=True)`
   - `fit_transform(bal["text"])` -> train
   - `transform(X_te)` -> test
8. `LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=1000)` 학습

#### Part C: 비교 분석

9. 성능 지표 계산 (sklearn metrics 사용):
   - accuracy, precision(pos_label=1), recall(pos_label=1), precision(pos_label=0), recall(pos_label=0), f1(average="macro")
   - 규칙 기반은 test 인덱스에 해당하는 예측만 추출하여 비교
   - 모든 값 소수점 4자리 반올림

10. SHAP 분석:
    ```python
    exp = shap.LinearExplainer(model, X_train_tfidf)
    sv = exp.shap_values(X_test_tfidf)
    mean_shap = np.mean(sv, axis=0).flatten()
    ```
    - `np.argsort(mean_shap)[::-1][:5]` -> 긍정 상위 5개 (양수 SHAP)
    - `np.argsort(mean_shap)[:5]` -> 부정 상위 5개 (음수 SHAP)

11. 비즈니스 요약: SHAP 상위 단어를 활용하여 3문장 이내로 작성. '긍정'/'부정' 키워드 필수 포함.

### 기대 출력 (참고값)

```json
{
  "rule_based": {
    "accuracy": 0.9547,
    "precision_pos": 1.0,
    "recall_pos": 0.9469,
    "precision_neg": 0.7639,
    "recall_neg": 1.0,
    "f1_macro": 0.9194
  },
  "ml_based": {
    "accuracy": 0.9947,
    "precision_pos": 0.9969,
    "recall_pos": 0.9969,
    "precision_neg": 0.9818,
    "recall_neg": 0.9818,
    "f1_macro": 0.9893
  }
}
```

### 검증 로직 요약 (validator.py)

- **항목 1**: rule_based에 6개 지표 존재 + accuracy가 참조값과 오차 0.03 이내
- **항목 2**: ml_based accuracy가 참조값과 오차 0.01 이내
- **항목 3**: ml_based f1_macro가 참조값과 오차 0.01 이내
- **항목 4**: shap_top5_positive 5개 모두 양수, shap_top5_negative 5개 모두 음수
- **항목 5**: 코드에 `fit_transform`과 `.transform(` 패턴이 모두 존재
- **항목 6**: business_summary에 '긍정', '부정' 모두 포함, 20자 이상
- **항목 7**: ml_based에 6개 지표 모두 존재
