## 문항 5 정답지 — 금융 리스크 예측 서비스

### 정답 코드

#### preprocessor.py

```python
"""preprocessor.py - 데이터 전처리 모듈"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(csv_path):
    """CSV 파일을 로드하고 (X, y) 튜플을 반환합니다."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["loan_id"])
    y = df["risk_label"]
    X = df.drop(columns=["risk_label"])
    return X, y


def handle_missing(X):
    """수치형 결측값을 해당 컬럼의 중앙값(median)으로 대체합니다."""
    X = X.copy()
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    return X


def encode_categoricals(X):
    """범주형 컬럼이 있으면 Label Encoding을 적용합니다."""
    return X.copy()


def scale_features(X):
    """StandardScaler로 모든 feature를 표준화합니다."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled, scaler
```

#### model.py

```python
"""model.py - 모델 학습 및 평가 모듈"""
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(X, y):
    """train_test_split으로 70/30 분할합니다."""
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def apply_pca(X_train, X_test, n_components=0.95):
    """PCA를 학습 데이터에 fit하고, 학습/테스트 데이터를 transform합니다."""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def train_model(X_train, y_train, model_type="logistic"):
    """모델을 학습합니다."""
    if model_type == "logistic":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == "ridge":
        model = RidgeClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """모델 성능을 평가합니다."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
    }
```

#### interpreter.py

```python
"""interpreter.py - 모델 해석 모듈"""
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


def get_feature_importance(model, feature_names):
    """모델의 coef_ 속성에서 절댓값 기준 feature importance를 추출합니다."""
    coefs = model.coef_.flatten()
    abs_coefs = np.abs(coefs)
    indices = np.argsort(abs_coefs)[::-1]
    result = []
    for idx in indices:
        result.append({
            "feature": feature_names[idx],
            "importance": round(float(abs_coefs[idx]), 4),
        })
    return result


def get_pca_variance(pca):
    """PCA 객체의 explained_variance_ratio_를 반환합니다."""
    result = []
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        result.append({
            "component": i + 1,
            "variance_ratio": round(float(ratio), 4),
        })
    return result


def cluster_features(X_scaled, n_clusters=3):
    """K-Means(n_clusters=3, random_state=42)로 클러스터링합니다."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    counts = Counter(labels.tolist())
    cluster_counts = {str(k): counts[k] for k in sorted(counts.keys())}
    return {
        "labels": labels.tolist(),
        "cluster_counts": cluster_counts,
        "inertia": round(float(kmeans.inertia_), 4),
    }
```

#### predictor.py

```python
"""predictor.py - 신규 고객 리스크 판정 서비스"""
import numpy as np
import pandas as pd


def load_new_customers(csv_path, scaler):
    """신규 고객 CSV를 로드하고 동일한 전처리를 적용합니다."""
    df = pd.read_csv(csv_path)
    loan_ids = df["loan_id"].tolist()
    X = df.drop(columns=["loan_id"])
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    X_scaled = scaler.transform(X)
    return loan_ids, X_scaled


def predict_risk(model, X_new):
    """학습된 모델로 신규 고객의 리스크 확률을 예측합니다."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_new)[:, 1]
    else:
        decisions = model.decision_function(X_new)
        probabilities = 1 / (1 + np.exp(-decisions))
    return probabilities


def classify_risk_level(probabilities, threshold_config):
    """확률 기반으로 리스크 등급을 분류합니다."""
    default_t = threshold_config["default_threshold"]
    conservative_t = threshold_config["conservative_threshold"]
    levels = []
    for p in probabilities:
        if p >= default_t:
            levels.append("위험")
        elif p >= conservative_t:
            levels.append("주의")
        else:
            levels.append("안전")
    return levels


def generate_report(loan_ids, probabilities, risk_levels):
    """고객별 판정 결과를 리포트로 생성합니다."""
    report = []
    for lid, prob, level in zip(loan_ids, probabilities, risk_levels):
        report.append({
            "loan_id": lid,
            "risk_probability": round(float(prob), 4),
            "risk_level": level
        })
    return report
```

#### charts/risk_charts.py

```python
"""리스크 분포 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_risk_distribution(distribution, output_path):
    """안전/주의/위험 분포 파이 차트를 저장한다."""
    labels = list(distribution.keys())
    sizes = list(distribution.values())
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12}
    )
    ax.set_title("신규 고객 리스크 분포", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_model_comparison(logistic_metrics, ridge_metrics, output_path):
    """모델별 성능 비교 바 차트를 저장한다."""
    metrics = ["accuracy", "precision", "recall", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Macro"]
    logistic_vals = [logistic_metrics[m] for m in metrics]
    ridge_vals = [ridge_metrics[m] for m in metrics]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width/2 for i in x], logistic_vals, width, label="Logistic", color="#2196F3")
    bars2 = ax.bar([i + width/2 for i in x], ridge_vals, width, label="Ridge", color="#FF5722")
    ax.set_ylabel("점수")
    ax.set_title("모델 성능 비교", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
```

#### charts/feature_charts.py

```python
"""Feature Importance 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_feature_importance(importance_list, output_path):
    """Feature Importance 수평 바 차트를 저장한다."""
    features = [item["feature"] for item in importance_list]
    values = [item["importance"] for item in importance_list]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("중요도 (|coefficient|)")
    ax.set_title("Feature Importance", fontsize=14)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
```

#### charts/pca_charts.py

```python
"""PCA 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_pca_scatter(X_pca, y, output_path):
    """PCA 2D 산점도를 저장한다."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4CAF50" if label == 0 else "#F44336" for label in y]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, edgecolors="k", linewidths=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2D 산점도 (녹색=안전, 빨간색=위험)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_pca_variance(variance_ratios, output_path):
    """PCA 분산 설명 비율 바 차트를 저장한다."""
    components = [item["component"] for item in variance_ratios]
    ratios = [item["variance_ratio"] for item in variance_ratios]
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(components, ratios, color="#2196F3", alpha=0.7, label="개별 분산")
    ax.plot(components, cumulative, "ro-", label="누적 분산")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% 기준선")
    ax.set_xlabel("주성분")
    ax.set_ylabel("분산 설명 비율")
    ax.set_title("PCA 분산 설명 비율", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
```

#### charts/cluster_charts.py

```python
"""K-Means 클러스터 시각화 모듈"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_cluster_scatter(X_pca, cluster_labels, output_path):
    """K-Means 클러스터 2D 산점도를 저장한다."""
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.Set1(np.linspace(0, 0.5, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = [cl == label for cl in cluster_labels]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=f"Cluster {label}", alpha=0.6,
                   edgecolors="k", linewidths=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means 클러스터링 결과", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
```

### 정답 체크리스트

| 번호 | 체크 항목 | 배점 | 검증 방법 |
|------|----------|------|----------|
| 1 | preprocessor.py 필수 함수 4개 정의 | 2점 | AST 자동 |
| 2 | model.py 필수 함수 4개 정의 | 2점 | AST 자동 |
| 3 | interpreter.py 필수 함수 3개 정의 | 2점 | AST 자동 |
| 4 | predictor.py 필수 함수 4개 정의 | 2점 | AST 자동 |
| 5 | main.py에 main() 함수 정의 | 2점 | AST 자동 |
| 6 | 데이터 로드 (loan_id 제거, X/y 분리, shape=(200,9)) | 4점 | import 자동 |
| 7 | 결측 처리 (중앙값 대체, 결측 0건) | 4점 | import 자동 |
| 8 | 인코딩 처리 (DataFrame 반환, shape 유지) | 2점 | import 자동 |
| 9 | StandardScaler 표준화 (평균 ~0) | 4점 | import 자동 |
| 10 | train_test_split (70/30, stratify) | 4점 | import 자동 |
| 11 | PCA 적용 (95% 분산, 차원 축소) | 4점 | import 자동 |
| 12 | LogisticRegression 학습 및 평가 (accuracy >= 0.7) | 4점 | import 자동 |
| 13 | RidgeClassifier 학습 및 평가 (accuracy >= 0.7) | 4점 | import 자동 |
| 14 | Feature Importance (9개, 절댓값 내림차순) | 3점 | import 자동 |
| 15 | PCA Variance Ratio (총 >= 0.95) | 3점 | import 자동 |
| 16 | KMeans 클러스터링 (3개, labels=200, inertia>0) | 3점 | import 자동 |
| 17 | 신규 고객 데이터 로드 (20행, NaN 없음) | 3점 | import 자동 |
| 18 | 리스크 확률 예측 + 등급 분류 (안전/주의/위험) | 3점 | import 자동 |
| 19 | 고객별 판정 리포트 생성 | 2점 | import 자동 |
| 20 | result_q5.json 필수 키 확인 | 2점 | JSON 자동 |
| 21 | 전처리 결과값 검증 (shape, missing) | 3점 | JSON 자동 |
| 22 | 모델 성능 범위 검증 (0~1, accuracy >= 0.7) | 3점 | JSON 자동 |
| 23 | 신규 고객 예측 20명, 리스크 분포 포함 | 3점 | JSON 자동 |
| 24 | dashboard/ 폴더 존재 | 2점 | 구조 자동 |
| 25 | charts/ 폴더 존재 | 2점 | 구조 자동 |
| 26 | dashboard/app.py 존재 | 2점 | 구조 자동 |
| 27 | dashboard/pages/ 필수 파일 존재 (overview.py, prediction.py, analysis.py, customer.py) | 3점 | 구조 자동 |
| 28 | dashboard/components/ 필수 파일 존재 (input_form.py, risk_gauge.py, chart_builder.py) | 3점 | 구조 자동 |
| 29 | charts/ 필수 모듈 존재 (risk_charts.py, feature_charts.py, pca_charts.py, cluster_charts.py) | 3점 | 구조 자동 |
| 30 | risk_charts.py 필수 함수 확인 | 2점 | AST 자동 |
| 31 | feature_charts.py 필수 함수 확인 | 2점 | AST 자동 |
| 32 | pca_charts.py 필수 함수 확인 | 2점 | AST 자동 |
| 33 | cluster_charts.py 필수 함수 확인 | 2점 | AST 자동 |
| 34 | 차트 생성 통합 테스트 (PNG 파일 생성 및 크기 > 0) | 4점 | import 자동 |

- Pass 기준: 총 100점 중 100점 (34개 전체 정답)
- AI 트랩: 불균형 데이터에서 stratify 누락, PCA를 test에도 fit (data leakage), RidgeClassifier에 predict_proba 없음 (decision_function -> sigmoid 변환 필요), 최적 모델 선택 기준 f1_macro (accuracy 아님), conservative_threshold와 default_threshold 적용 순서 혼동, matplotlib backend 미설정

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
| 대시보드 구조 설계 | test_dashboard_folder_exists, test_charts_folder_exists, test_dashboard_app_exists |
| 대시보드 페이지/컴포넌트 | test_dashboard_pages_exist, test_dashboard_components_exist |
| 시각화 모듈 (matplotlib) | test_risk_charts_functions, test_feature_charts_functions, test_pca_charts_functions, test_cluster_charts_functions, test_chart_generation |
