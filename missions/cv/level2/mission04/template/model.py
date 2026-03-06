"""model.py - 모델 학습 및 평가 모듈"""
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(X, y):
    """train_test_split으로 70/30 분할합니다.

    - random_state=42, stratify=y

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # TODO: 구현하세요
    pass


def apply_pca(X_train, X_test, n_components=0.95):
    """PCA를 학습 데이터에 fit하고, 학습/테스트 데이터를 transform합니다.

    Returns:
        tuple: (X_train_pca, X_test_pca, pca)
    """
    # TODO: 구현하세요
    pass


def train_model(X_train, y_train, model_type="logistic"):
    """모델을 학습합니다.

    - model_type="logistic": LogisticRegression(random_state=42, max_iter=1000)
    - model_type="ridge": RidgeClassifier(random_state=42)

    Returns:
        학습된 모델 객체
    """
    # TODO: 구현하세요
    pass


def evaluate_model(model, X_test, y_test):
    """모델 성능을 평가합니다.

    Returns:
        dict: {"accuracy": float, "precision": float, "recall": float, "f1_macro": float}
    """
    # TODO: 구현하세요
    pass
