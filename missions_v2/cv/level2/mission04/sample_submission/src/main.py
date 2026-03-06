"""main.py - 전체 파이프라인 실행"""
import json
import os
import numpy as np

from preprocessor import load_data, handle_missing, encode_categoricals, scale_features
from model import split_data, apply_pca, train_model, evaluate_model
from interpreter import get_feature_importance, get_pca_variance, cluster_features


def main():
    """전체 ML 파이프라인을 실행하고 result_q4.json을 저장합니다."""
    # 데이터 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    mission_dir = os.path.dirname(project_dir)
    csv_path = os.path.join(mission_dir, "data", "loan_data.csv")

    # 1. 데이터 로드
    X, y = load_data(csv_path)
    original_shape = list(X.shape)
    missing_before = int(X.isnull().sum().sum())

    # 2. 결측 처리
    X = handle_missing(X)
    missing_after = int(X.isnull().sum().sum())

    # 3. 인코딩
    X = encode_categoricals(X)

    # feature 이름 저장
    feature_names = list(X.columns)

    # 4. 스케일링
    X_scaled, scaler = scale_features(X)
    scaled_mean_abs_max = round(float(np.max(np.abs(np.mean(X_scaled, axis=0)))), 4)

    # 5. train/test split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 6. PCA
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=0.95)

    # 7. LogisticRegression 학습/평가
    logistic_model = train_model(X_train_pca, y_train, model_type="logistic")
    logistic_metrics = evaluate_model(logistic_model, X_test_pca, y_test)

    # 8. RidgeClassifier 학습/평가
    ridge_model = train_model(X_train_pca, y_train, model_type="ridge")
    ridge_metrics = evaluate_model(ridge_model, X_test_pca, y_test)

    # 9. Feature Importance (원본 feature에 대해 — 스케일링된 데이터로 재학습)
    logistic_full = train_model(X_train, y_train, model_type="logistic")
    importance = get_feature_importance(logistic_full, feature_names)

    # 10. PCA variance
    pca_variance = get_pca_variance(pca)
    total_var = round(float(sum(pca.explained_variance_ratio_)), 4)

    # 11. K-Means 클러스터링 (스케일링된 전체 데이터)
    clustering = cluster_features(X_scaled, n_clusters=3)

    # 12. 결과 저장
    result = {
        "preprocessing": {
            "original_shape": original_shape,
            "missing_values_before": missing_before,
            "missing_values_after": missing_after,
            "scaled_mean_abs_max": scaled_mean_abs_max,
        },
        "model_logistic": logistic_metrics,
        "model_ridge": ridge_metrics,
        "pca": {
            "n_components_selected": int(pca.n_components_),
            "total_variance_explained": total_var,
            "variance_ratios": pca_variance,
        },
        "feature_importance": importance,
        "clustering": {
            "n_clusters": 3,
            "cluster_counts": clustering["cluster_counts"],
            "inertia": clustering["inertia"],
        },
    }

    output_path = os.path.join(project_dir, "output", "result_q4.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"result_q4.json saved to {output_path}")
    return result


if __name__ == "__main__":
    main()
