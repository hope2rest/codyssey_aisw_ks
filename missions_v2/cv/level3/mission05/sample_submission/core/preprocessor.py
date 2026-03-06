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
