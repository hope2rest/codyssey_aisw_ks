"""preprocessor.py - 데이터 전처리 모듈"""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(csv_path):
    """CSV 파일을 로드하고 (X, y) 튜플을 반환합니다.

    - loan_id 컬럼을 제거합니다.
    - X: feature DataFrame, y: risk_label Series
    """
    # TODO: 구현하세요
    pass


def handle_missing(X):
    """수치형 결측값을 해당 컬럼의 중앙값(median)으로 대체합니다.

    Returns:
        DataFrame: 결측값이 처리된 DataFrame
    """
    # TODO: 구현하세요
    pass


def encode_categoricals(X):
    """범주형 컬럼이 있으면 Label Encoding을 적용합니다.

    본 데이터셋에는 범주형이 없으므로 그대로 반환합니다.

    Returns:
        DataFrame: 인코딩 처리된 DataFrame
    """
    # TODO: 구현하세요
    pass


def scale_features(X):
    """StandardScaler로 모든 feature를 표준화합니다.

    Returns:
        tuple: (scaled_array, scaler)
    """
    # TODO: 구현하세요
    pass
