"""금융 리스크 예측 서비스 대시보드"""
import os
import sys
import json

import streamlit as st

# 프로젝트 경로 설정
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_APP_DIR)
_CORE_DIR = os.path.join(_PROJECT_DIR, "core")
_CHARTS_DIR = os.path.join(_PROJECT_DIR, "charts")

if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
if _CHARTS_DIR not in sys.path:
    sys.path.insert(0, _CHARTS_DIR)

st.set_page_config(page_title="금융 리스크 예측 서비스", layout="wide")
st.title("금융 리스크 예측 서비스")

# 결과 파일 로드
result_path = os.path.join(_PROJECT_DIR, "output", "result_q5.json")
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    st.success("결과 로드 완료")
else:
    st.error("result_q5.json을 찾을 수 없습니다. main.py를 먼저 실행하세요.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["개요", "모델 성능", "모델 해석", "고객 판정"])

with tab1:
    st.header("데이터 개요")
    prep = result["preprocessing"]
    col1, col2, col3 = st.columns(3)
    col1.metric("원본 데이터", f'{prep["original_shape"][0]}행 × {prep["original_shape"][1]}열')
    col2.metric("결측값 (처리 전)", prep["missing_values_before"])
    col3.metric("결측값 (처리 후)", prep["missing_values_after"])

with tab2:
    st.header("모델 성능 비교")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistic Regression")
        for k, v in result["model_logistic"].items():
            st.metric(k, f"{v:.4f}")
    with col2:
        st.subheader("Ridge Classifier")
        for k, v in result["model_ridge"].items():
            st.metric(k, f"{v:.4f}")
    st.info(f'최적 모델: **{result["best_model"]}**')

with tab3:
    st.header("모델 해석")
    st.subheader("Feature Importance")
    for item in result["feature_importance"][:5]:
        st.write(f"- **{item['feature']}**: {item['importance']:.4f}")
    st.subheader("PCA")
    st.write(f'선택된 주성분 수: {result["pca"]["n_components_selected"]}')
    st.write(f'총 분산 설명 비율: {result["pca"]["total_variance_explained"]:.4f}')
    st.subheader("클러스터링")
    st.write(f'Inertia: {result["clustering"]["inertia"]:.4f}')
    st.json(result["clustering"]["cluster_counts"])

with tab4:
    st.header("신규 고객 리스크 판정")
    ncp = result["new_customer_predictions"]
    col1, col2, col3 = st.columns(3)
    col1.metric("총 고객", ncp["total_customers"])
    dist = ncp["risk_distribution"]
    col2.metric("안전", dist.get("안전", 0))
    col3.metric("위험", dist.get("위험", 0))
    st.dataframe(ncp["predictions"])
