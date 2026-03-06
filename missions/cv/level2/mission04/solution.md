## 문항 4 정답지 — 금융 리스크 예측 + 모델 해석

### 정답 파일 구성

| 파일 | 주요 내용 |
|------|----------|
| `preprocessor.py` | CSV 로드, 결측값 중앙값 대체, StandardScaler 스케일링 |
| `model.py` | train_test_split(70/30, stratify), PCA(0.95), LogisticRegression, RidgeClassifier |
| `interpreter.py` | coef_ 기반 Feature Importance, PCA variance ratio, K-Means(k=3) |
| `main.py` | 전체 파이프라인 실행, result_q4.json 저장 |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | preprocessor.py 필수 함수 4개 | AST 분석 |
| 2 | model.py 필수 함수 4개 | AST 분석 |
| 3 | interpreter.py 필수 함수 3개 | AST 분석 |
| 4 | main.py에 main() 함수 | AST 분석 |
| 5-8 | preprocessor 기능 검증 | import + 수치 검증 |
| 9-12 | model 기능 검증 | import + 수치 검증 |
| 13-15 | interpreter 기능 검증 | import + 수치 검증 |
| 16-18 | result_q4.json 정량적 검증 | JSON 값 확인 |

### AI Trap 주의사항

1. **데이터 누수**: PCA의 `fit`은 학습 데이터에서만 수행, 테스트 데이터에는 `transform`만 적용
2. **PCA n_components**: 0.95는 정수가 아닌 분산 비율 기준 (자동 컴포넌트 수 결정)
3. **K-Means 재현성**: `random_state=42, n_init=10` 필수
4. **Feature Importance**: PCA 변환된 공간이 아닌 원본 feature 공간에서 계수 추출

### 풀이 흐름

1. `load_data()`: CSV 로드, loan_id 제거, (X, y) 분리
2. `handle_missing()`: 수치형 결측값을 컬럼별 median으로 대체
3. `encode_categoricals()`: 범주형 없으므로 그대로 반환
4. `scale_features()`: StandardScaler fit_transform
5. `split_data()`: train_test_split(test_size=0.3, random_state=42, stratify=y)
6. `apply_pca()`: PCA(n_components=0.95) fit on train, transform both
7. `train_model("logistic")`: LogisticRegression(random_state=42, max_iter=1000)
8. `train_model("ridge")`: RidgeClassifier(random_state=42)
9. `evaluate_model()`: accuracy, precision, recall, f1_macro 계산
10. `get_feature_importance()`: 원본 feature에 대해 LogisticRegression 재학습 후 coef_ 절댓값 추출
11. `get_pca_variance()`: explained_variance_ratio_ 추출
12. `cluster_features()`: KMeans(n_clusters=3, random_state=42) 적용

### 검증 로직 요약 (validator.py)

- **TestStructure (4개)**: AST 기반 필수 함수 존재 확인
- **TestPreprocessor (4개)**: 데이터 로드, 결측 처리, 인코딩, 스케일링 기능 검증
- **TestModel (4개)**: split 비율, PCA 차원 축소, Logistic/Ridge accuracy >= 0.7
- **TestInterpreter (3개)**: Feature Importance 형식/정렬, PCA variance 합 >= 0.95, K-Means 클러스터 수 확인
- **TestResult (3개)**: result_q4.json 구조, 전처리 값 정확성, 모델 성능 범위 검증
