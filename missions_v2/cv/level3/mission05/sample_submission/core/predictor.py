"""predictor.py - 신규 고객 리스크 판정 서비스"""
import json
import numpy as np
import pandas as pd


def load_new_customers(csv_path, scaler):
    """신규 고객 CSV를 로드하고 동일한 전처리를 적용합니다."""
    df = pd.read_csv(csv_path)
    loan_ids = df["loan_id"].tolist()
    X = df.drop(columns=["loan_id"])

    # 결측값 중앙값 대체
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # 학습 데이터와 동일한 스케일러 적용
    X_scaled = scaler.transform(X)
    return loan_ids, X_scaled


def predict_risk(model, X_new):
    """학습된 모델로 신규 고객의 리스크 확률을 예측합니다."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_new)[:, 1]
    else:
        # RidgeClassifier는 predict_proba가 없으므로 decision_function 사용
        decisions = model.decision_function(X_new)
        # sigmoid로 확률 변환
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
