## 문항: 금융 리스크 예측 + 모델 해석

### 문제

대출 고객 데이터에 대해 ML 기반 리스크 예측 모델을 구축하고, PCA/K-Means/Feature Importance로 모델을 해석하는 파이프라인을 구현하세요.
데이터는 `data/loan_data.csv`에 저장되어 있습니다 (200행, 11열, 헤더 포함).

### loan_data.csv 구조

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `loan_id` | str | 대출 고유 ID (L0001 ~ L0200) |
| `age` | int | 나이 (20~65) |
| `annual_income` | int | 연소득 (결측 포함) |
| `debt_ratio` | float | 부채 비율 (결측 포함) |
| `credit_score` | int | 신용 점수 (결측 포함) |
| `employment_years` | int | 근속 연수 (결측 포함) |
| `loan_amount` | int | 대출 금액 |
| `interest_rate` | float | 이자율 |
| `num_credit_lines` | int | 신용 거래 수 |
| `payment_history_score` | int | 납부 이력 점수 (결측 포함) |
| `risk_label` | int | 0=안전, 1=위험 (15% 불균형) |

### 프로젝트 구조

| 파일 | 역할 | 핵심 함수 |
|------|------|----------|
| `preprocessor.py` | 데이터 로드, 결측 처리, 인코딩, 스케일링 | `load_data()`, `handle_missing()`, `encode_categoricals()`, `scale_features()` |
| `model.py` | 학습/테스트 분할, PCA, 모델 학습, 평가 | `split_data()`, `apply_pca()`, `train_model()`, `evaluate_model()` |
| `interpreter.py` | 모델 해석, PCA 분석, 클러스터링 | `get_feature_importance()`, `get_pca_variance()`, `cluster_features()` |
| `main.py` | 전체 파이프라인 실행, 결과 저장 | `main()` |

### 구현 요구사항

#### Part A: preprocessor.py - 데이터 전처리

1. `load_data(csv_path)` - CSV 파일을 pandas DataFrame으로 로드합니다. `loan_id` 컬럼을 제거하고 (X, y) 튜플을 반환합니다. X는 feature DataFrame, y는 `risk_label` Series입니다.
2. `handle_missing(X)` - 수치형 결측값을 해당 컬럼의 중앙값(median)으로 대체합니다. 처리된 DataFrame을 반환합니다.
3. `encode_categoricals(X)` - 범주형 컬럼이 있으면 Label Encoding을 적용합니다. 본 데이터셋에는 범주형이 없으므로 그대로 반환합니다.
4. `scale_features(X)` - `StandardScaler`로 모든 feature를 표준화합니다. (scaled_array, scaler) 튜플을 반환합니다.

#### Part B: model.py - 모델 학습 및 평가

5. `split_data(X, y)` - `train_test_split`으로 70/30 분할합니다 (`random_state=42`, `stratify=y`). (X_train, X_test, y_train, y_test) 튜플을 반환합니다.
6. `apply_pca(X_train, X_test, n_components=0.95)` - PCA를 학습 데이터에 fit하고, 학습/테스트 데이터를 모두 transform합니다. (X_train_pca, X_test_pca, pca) 튜플을 반환합니다.
7. `train_model(X_train, y_train, model_type="logistic")` - `model_type="logistic"`이면 `LogisticRegression(random_state=42, max_iter=1000)`, `model_type="ridge"`이면 `RidgeClassifier(random_state=42)`를 학습합니다. 학습된 모델을 반환합니다.
8. `evaluate_model(model, X_test, y_test)` - accuracy, precision, recall, f1_score(macro)를 계산합니다. `{"accuracy": float, "precision": float, "recall": float, "f1_macro": float}` 딕셔너리를 반환합니다.

#### Part C: interpreter.py - 모델 해석

9. `get_feature_importance(model, feature_names)` - 모델의 `coef_` 속성에서 절댓값 기준 feature importance를 추출합니다. `[{"feature": str, "importance": float}, ...]` 리스트를 절댓값 내림차순으로 반환합니다.
10. `get_pca_variance(pca)` - PCA 객체의 `explained_variance_ratio_`를 반환합니다. `[{"component": int, "variance_ratio": float}, ...]` 리스트를 반환합니다.
11. `cluster_features(X_scaled, n_clusters=3)` - K-Means(n_clusters=3, random_state=42)로 클러스터링합니다. `{"labels": list, "cluster_counts": dict, "inertia": float}` 딕셔너리를 반환합니다.

#### Part D: main.py - 파이프라인 실행

12. `main()` - 위의 모듈을 순서대로 호출하여 전체 파이프라인을 실행합니다.
13. LogisticRegression과 RidgeClassifier 두 모델을 각각 학습/평가합니다.
14. `result_q4.json` 파일로 결과를 저장합니다.

### 출력 형식

`result_q4.json` 파일로 다음 구조를 저장합니다:

```json
{
  "preprocessing": {
    "original_shape": [행수, 열수],
    "missing_values_before": 정수,
    "missing_values_after": 정수,
    "scaled_mean_abs_max": 실수
  },
  "model_logistic": {
    "accuracy": 실수,
    "precision": 실수,
    "recall": 실수,
    "f1_macro": 실수
  },
  "model_ridge": {
    "accuracy": 실수,
    "precision": 실수,
    "recall": 실수,
    "f1_macro": 실수
  },
  "pca": {
    "n_components_selected": 정수,
    "total_variance_explained": 실수,
    "variance_ratios": [{"component": 정수, "variance_ratio": 실수}, ...]
  },
  "feature_importance": [{"feature": 문자열, "importance": 실수}, ...],
  "clustering": {
    "n_clusters": 3,
    "cluster_counts": {"0": 정수, "1": 정수, "2": 정수},
    "inertia": 실수
  }
}
```

### 제약 사항

- `sklearn`의 `StandardScaler`, `PCA`, `LogisticRegression`, `RidgeClassifier`, `KMeans`, `train_test_split` 사용 가능
- `LogisticRegression`은 `random_state=42, max_iter=1000`으로 설정
- `RidgeClassifier`는 `random_state=42`로 설정
- `PCA`는 `n_components=0.95` (95% 분산 설명)
- `KMeans`는 `n_clusters=3, random_state=42`
- `train_test_split`은 `test_size=0.3, random_state=42, stratify=y`
- 모든 실수값은 `round(..., 4)`로 반올림

### 제출 방식

- `preprocessor.py`, `model.py`, `interpreter.py`, `main.py`, `result_q4.json` 총 5개 파일을 zip으로 묶어 제출합니다.
- `template/` 디렉토리의 각 파일의 `# TODO` 부분을 채우세요.
