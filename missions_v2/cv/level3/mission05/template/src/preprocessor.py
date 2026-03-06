import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(csv_path):
    # TODO: CSV 로드, loan_id 제거, (X, y) 튜플 반환


def handle_missing(X):
    # TODO: 수치형 결측값을 해당 컬럼의 중앙값으로 대체


def encode_categoricals(X):
    # TODO: 범주형 컬럼 Label Encoding (본 데이터셋은 범주형 없으므로 그대로 반환)


def scale_features(X):
    # TODO: StandardScaler로 표준화, (scaled_array, scaler) 반환
