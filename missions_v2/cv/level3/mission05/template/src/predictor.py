"""predictor.py - 신규 고객 리스크 판정 서비스"""
import json
import numpy as np
import pandas as pd


def load_new_customers(csv_path, scaler):
    """신규 고객 CSV를 로드하고 동일한 전처리를 적용합니다."""
    # TODO: CSV 로드, loan_id 분리, 결측값 처리, scaler.transform 적용
    pass


def predict_risk(model, X_new):
    """학습된 모델로 신규 고객의 리스크 확률을 예측합니다."""
    # TODO: predict_proba 또는 decision_function으로 확률 반환
    pass


def classify_risk_level(probabilities, threshold_config):
    """확률 기반으로 리스크 등급을 분류합니다."""
    # TODO: default_threshold, conservative_threshold 기준으로 안전/주의/위험 분류
    pass


def generate_report(loan_ids, probabilities, risk_levels):
    """고객별 판정 결과를 리포트로 생성합니다."""
    # TODO: [{loan_id, risk_probability, risk_level}, ...] 형태 반환
    pass
