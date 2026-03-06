"""도서 검색 및 추천 서비스 대시보드"""
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

st.set_page_config(page_title="도서 검색 및 추천 서비스", layout="wide")
st.title("도서 검색 및 추천 서비스")

# 결과 파일 로드
result_path = os.path.join(_PROJECT_DIR, "output", "result_q2.json")
if os.path.exists(result_path):
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    st.success(f"결과 로드 완료 — 문서 {result['tfidf_matrix_shape'][0]}건, 리뷰 {result['total_reviews']}건")
else:
    st.error("result_q2.json을 찾을 수 없습니다. main.py를 먼저 실행하세요.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["검색", "추천", "감성 분석"])

with tab1:
    st.header("도서 검색 결과")
    for sr in result["search_results"]:
        st.subheader(f'🔍 "{sr["query"]}"')
        for rank, item in enumerate(sr["top3"], 1):
            st.write(f"{rank}. **{item['title']}** (유사도: {item['similarity']:.4f})")

with tab2:
    st.header("도서 추천")
    rec = result["recommendation"]
    st.subheader(f'📖 기준 도서: {rec["target_book"]["title"]}')
    col1, col2 = st.columns(2)
    with col1:
        st.write("**유사 도서 Top 5**")
        for r in rec["top5_similar"]:
            st.write(f"- {r['title']} ({r['category']}) — {r['similarity']:.4f}")
    with col2:
        st.write("**같은 카테고리 Top 3**")
        for r in rec["same_category_top3"]:
            st.write(f"- {r['title']} — {r['similarity']:.4f}")

with tab3:
    st.header("감성 분석 결과")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("정확도", f"{result['sentiment_accuracy']:.4f}")
        st.metric("F1 점수", f"{result['sentiment_f1']:.4f}")
    with col2:
        st.metric("긍정 리뷰", result["positive_count"])
        st.metric("부정 리뷰", result["negative_count"])
