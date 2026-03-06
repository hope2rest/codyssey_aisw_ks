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
