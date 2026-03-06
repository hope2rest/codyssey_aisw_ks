## 문항 5 정답지 — 금융 리스크 예측 모델 고도화 및 예측 시스템

### 데이터 타입

| 항목 | 타입 | 설명 |
|------|------|------|
| `loan_data.csv` | CSV | 200행, 9 feature + risk_label (불균형 85:15) |
| `new_customers.csv` | CSV | 20행, 9 feature (레이블 없음) |
| `threshold_config.json` | JSON | default_threshold=0.5, conservative_threshold=0.3 |
| StandardScaler | sklearn | 표준화 (mean=0, std=1) |
| PCA | sklearn | n_components=0.95 (95% 분산 설명) |
| LogisticRegression | sklearn | random_state=42, max_iter=1000 |
| RidgeClassifier | sklearn | random_state=42 |
| KMeans | sklearn | n_clusters=3, random_state=42 |
| 리스크 등급 | `str` | "안전", "주의", "위험" |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | preprocessor.py 필수 함수 4개 정의 (load_data, handle_missing, encode_categoricals, scale_features) | AST 자동 |
| 2 | model.py 필수 함수 4개 정의 (split_data, apply_pca, train_model, evaluate_model) | AST 자동 |
| 3 | interpreter.py 필수 함수 3개 정의 (get_feature_importance, get_pca_variance, cluster_features) | AST 자동 |
| 4 | predictor.py 필수 함수 4개 정의 (load_new_customers, predict_risk, classify_risk_level, generate_report) | AST 자동 |
| 5 | 데이터 로드 (loan_id 제거, X/y 분리) | import 자동 |
| 6 | 결측 처리 (중앙값 대체) | import 자동 |
| 7 | StandardScaler 표준화 | import 자동 |
| 8 | train_test_split (70/30, stratify, random_state=42) | import 자동 |
| 9 | PCA 적용 (95% 분산 설명) | import 자동 |
| 10 | LogisticRegression 학습 및 평가 | import 자동 |
| 11 | RidgeClassifier 학습 및 평가 | import 자동 |
| 12 | Feature Importance (절댓값 내림차순) | import 자동 |
| 13 | PCA Variance Ratio 반환 | import 자동 |
| 14 | KMeans 클러스터링 (n_clusters=3) | import 자동 |
| 15 | 신규 고객 데이터 로드 및 전처리 | import 자동 |
| 16 | 리스크 확률 예측 (predict_proba) | import 자동 |
| 17 | 3단계 리스크 등급 분류 (안전/주의/위험) | import 자동 |
| 18 | 고객별 판정 리포트 생성 | import 자동 |
| 19 | result_q5.json 필수 키 확인 (preprocessing, model_logistic, model_ridge, best_model, pca, feature_importance, clustering, new_customer_predictions) | JSON 자동 |
| 20 | 모델 accuracy > 0.7 | JSON 자동 |
| 21 | PCA 분산 설명 > 0.9 | JSON 자동 |
| 22 | 클러스터 수 = 3 | JSON 자동 |
| 23 | 신규 고객 예측 20명, 리스크 분포 포함 | JSON 자동 |

- Pass 기준: 23개 전체 통과
- AI 트랩: 불균형 데이터에서 stratify 누락, PCA를 test에도 fit (data leakage), RidgeClassifier에 predict_proba 없음 (decision_function→sigmoid 변환 필요), 최적 모델 선택 기준 f1_macro (accuracy 아님), conservative_threshold와 default_threshold 적용 순서 혼동

### 학습 목표 매핑

| 학습 목표 | 검증 테스트 |
|-----------|-----------|
| 데이터 전처리 (결측, 스케일링) | test_load_data, test_handle_missing, test_scale_features |
| 데이터 분할 (stratified) | test_split_data |
| PCA 차원 축소 | test_apply_pca |
| 분류 모델 학습/평가 | test_train_logistic, test_train_ridge, test_evaluate_model |
| 모델 해석 (Feature Importance) | test_feature_importance |
| 클러스터링 (KMeans) | test_cluster_features |
| 신규 고객 리스크 판정 | test_load_new_customers, test_predict_and_classify, test_generate_report |
| 파이프라인 통합 | test_result_structure, test_model_accuracy, test_pca_variance, test_clustering, test_new_customer_predictions |
