import json
import os

from preprocessor import load_data, handle_missing, encode_categoricals, scale_features
from model import split_data, apply_pca, train_model, evaluate_model
from interpreter import get_feature_importance, get_pca_variance, cluster_features


def main():
    # TODO: 전체 ML 파이프라인 실행 및 result_q4.json 저장
    # 1. 데이터 로드 및 전처리
    # 2. 스케일링
    # 3. train/test split
    # 4. PCA 적용
    # 5. LogisticRegression 학습/평가
    # 6. RidgeClassifier 학습/평가
    # 7. Feature Importance
    # 8. PCA variance ratio
    # 9. K-Means 클러스터링
    # 10. result_q4.json 저장


if __name__ == "__main__":
    main()
