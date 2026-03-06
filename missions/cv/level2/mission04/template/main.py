"""main.py - 전체 파이프라인 실행"""
import json
import os

from preprocessor import load_data, handle_missing, encode_categoricals, scale_features
from model import split_data, apply_pca, train_model, evaluate_model
from interpreter import get_feature_importance, get_pca_variance, cluster_features


def main():
    """전체 ML 파이프라인을 실행하고 result_q4.json을 저장합니다."""
    # TODO: 구현하세요
    #
    # 1. 데이터 로드 및 전처리
    # 2. 스케일링
    # 3. train/test split
    # 4. PCA 적용
    # 5. LogisticRegression 학습/평가
    # 6. RidgeClassifier 학습/평가
    # 7. Feature Importance (LogisticRegression 기준, 원본 feature 이름 사용)
    # 8. PCA variance ratio
    # 9. K-Means 클러스터링 (스케일링된 전체 데이터 사용)
    # 10. result_q4.json 저장
    pass


if __name__ == "__main__":
    main()
