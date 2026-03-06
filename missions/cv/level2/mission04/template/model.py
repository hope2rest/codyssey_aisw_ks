from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(X, y):
    # TODO: train_test_split으로 70/30 분할 (random_state=42, stratify=y)


def apply_pca(X_train, X_test, n_components=0.95):
    # TODO: PCA fit → transform, (X_train_pca, X_test_pca, pca) 반환


def train_model(X_train, y_train, model_type="logistic"):
    # TODO: logistic이면 LogisticRegression, ridge이면 RidgeClassifier 학습


def evaluate_model(model, X_test, y_test):
    # TODO: accuracy, precision, recall, f1_macro 계산
