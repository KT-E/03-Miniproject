"""Streamlit dashboard: review input + analysis + aggregate visualization.

Run from the project root with:

    export OPENAI_API_KEY="sk-..."
    streamlit run src/dashboard.py

This is the Mission 5 deliverable from the notebooks, refactored to reuse
``agent1_review`` / ``agent2_insight`` / ``db`` instead of redefining the
agents inline in the Streamlit script.
"""

from __future__ import annotations

import os
import sys

# Allow `streamlit run src/dashboard.py` (executed as a plain script, not a
# package) to still import the sibling modules as `src.xxx`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src import config, db
from src.agent1_review import analyze_review, build_review_graph
from src.agent2_insight import build_insight_graph, suggest_improvements


def set_korean_font() -> None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()
st.set_page_config(page_title="상품 리뷰 분석 Agent", layout="wide")


@st.cache_resource
def get_review_app():
    return build_review_graph()


@st.cache_resource
def get_insight_app():
    return build_insight_graph()


db.init_db()

st.title("상품 리뷰 분석 Agent 대시보드")

col1, col2 = st.columns(2)

with col1:
    st.subheader("리뷰 입력")
    review = st.text_area("리뷰 내용", placeholder="여기에 리뷰를 입력하세요. (20자 이상)")
    current_len = len(review.strip())
    st.caption(f"{current_len}자 (최소 20자 필요)")
    score = st.radio("별점", [1, 2, 3, 4, 5], horizontal=True)
    run_btn = st.button("분석 실행", type="primary")

with col2:
    st.subheader("Agent 분석 결과")

    if run_btn:
        review_stripped = review.strip()
        if len(review_stripped) < 20:
            st.warning(f"리뷰는 20자 이상 작성해주세요. (현재 {len(review_stripped)}자)")
        else:
            with st.spinner("Agent 1이 리뷰를 분석하고 있습니다..."):
                result = analyze_review(get_review_app(), review_stripped, score=score)
                db.insert_analyzed_review(result)

            st.success("분석 및 저장이 완료되었습니다!")

            aspects = result.get("aspect", []) or []
            labels = result.get("label", []) or []

            st.write(f"**리뷰:** {result.get('review', 'N/A')}")
            st.write(f"**별점:** {result.get('score', 'N/A')}")

            if aspects and labels and len(aspects) == len(labels):
                st.table(pd.DataFrame({
                    "속성": aspects,
                    "감성": ["긍정" if l == 1 else "부정" for l in labels],
                }))
            else:
                st.warning("리뷰에서 분석 대상 속성이 추출되지 않았습니다.")

            negative = [a for a, l in zip(aspects, labels) if l == 0]
            if negative:
                with st.spinner("Agent 2(전문가 그룹)가 개선 방안을 도출하고 있습니다..."):
                    report = suggest_improvements(get_insight_app(), result["review"], aspects, labels)
                st.divider()
                st.subheader("전문가 개선 제안")
                st.text(report)
    else:
        st.info("리뷰를 입력하고 '분석 실행'을 눌러주세요.")

st.divider()

df = db.load_reviews()
if df.empty:
    st.info("아직 분석된 리뷰가 없습니다.")
    st.stop()

st.sidebar.header("필터")
score_min, score_max = st.sidebar.slider("별점 범위", 1, 5, (1, 5))
all_aspects = sorted({a for lst in df["aspect_parsed"] for a in lst})
selected_aspects = st.sidebar.multiselect("포함할 속성", all_aspects, default=all_aspects)
keyword = st.sidebar.text_input("리뷰 키워드 검색", "")

mask = df["score"].between(score_min, score_max)
if selected_aspects:
    mask &= df["aspect_parsed"].apply(lambda lst: any(a in lst for a in selected_aspects) if lst else False)
if keyword:
    mask &= df["review"].fillna("").str.contains(keyword, case=False)

df_filtered = df[mask].reset_index(drop=True)
if df_filtered.empty:
    st.warning("선택하신 필터 조건에 해당하는 리뷰가 없습니다.")
    st.stop()

summary_df = db.build_aspect_summary(df_filtered, config.ASPECT_LIST)
total_reviews = len(df_filtered)
avg_score = round(df_filtered["score"].mean(), 2) if total_reviews > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("총 리뷰 수", f"{total_reviews:,}")
k2.metric("평균 별점", f"{avg_score} / 5")
if not summary_df.empty and summary_df["언급횟수"].sum() > 0:
    best = summary_df.sort_values("긍정비율(%)", ascending=False).iloc[0]
    worst = summary_df.sort_values("긍정비율(%)", ascending=True).iloc[0]
    k3.metric("최고 평가 속성", best["aspect"], f"{best['긍정비율(%)']}%")
    k4.metric("최저 평가 속성", worst["aspect"], f"{worst['긍정비율(%)']}%")

st.divider()
st.header("리뷰 분석 결과 시각화")
tab1, tab2, tab3 = st.tabs(["속성별 긍/부정", "별점 분포", "속성별 긍정비율"])

with tab1:
    if summary_df["언급횟수"].sum() == 0:
        st.info("표시할 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(summary_df))
        ax.bar(x - 0.2, summary_df["긍정"], width=0.4, label="긍정", color="#4C9AFF")
        ax.bar(x + 0.2, summary_df["부정"], width=0.4, label="부정", color="#FF6B6B")
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["aspect"], rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        st.dataframe(summary_df, use_container_width=True)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df_filtered, x="score", order=[1, 2, 3, 4, 5], ax=ax, color="#4C9AFF")
    st.pyplot(fig)

with tab3:
    if summary_df["언급횟수"].sum() == 0:
        st.info("표시할 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(summary_df["aspect"], summary_df["긍정비율(%)"], color="#36B37E")
        ax.set_xlim(0, 100)
        st.pyplot(fig)

st.divider()
st.header("리뷰 분석 결과 건별 조회")
show_df = df_filtered[["id", "review", "aspect_parsed", "label_parsed", "score", "updated_at"]].rename(
    columns={"aspect_parsed": "aspect", "label_parsed": "label"}
)
st.dataframe(show_df, use_container_width=True, height=300)
