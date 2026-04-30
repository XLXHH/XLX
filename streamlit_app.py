import os
from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import streamlit as st

from demo import (
    normalize_stock_code,
    normalize_adjust,
    get_stock_history,
    get_stock_fund_flow,
    merge_history_and_fund,
    build_summary,
)
from stock_analysis_report import run_analysis_from_data


st.set_page_config(page_title="股票分析看板", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);}
    .block-container {padding-top: 1.2rem;}
    .card {
        background: #ffffff;
        border: 1px solid #e8edf7;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }
    .muted {color: #64748b; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def to_excel_bytes(dataframes: dict) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in dataframes.items():
            if df is None:
                continue
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    buffer.seek(0)
    return buffer.read()


def run_pipeline(stock_code: str, start_date: str, end_date: str, adjust: str):
    hist_df = get_stock_history(stock_code, start_date, end_date, adjust)
    fund_df = get_stock_fund_flow(stock_code)
    merged_df = merge_history_and_fund(hist_df, fund_df)
    summary_df = build_summary(stock_code, hist_df, fund_df)
    return hist_df, fund_df, merged_df, summary_df


def render_filter_table(df: pd.DataFrame, title: str):
    st.subheader(title)
    if df is None or df.empty:
        st.info("暂无数据")
        return

    show_df = df.copy()
    if "日期" in show_df.columns:
        show_df["日期"] = pd.to_datetime(show_df["日期"], errors="coerce")
        min_d = show_df["日期"].min()
        max_d = show_df["日期"].max()
        if pd.notna(min_d) and pd.notna(max_d):
            d1, d2 = st.date_input(
                f"{title} 日期筛选",
                value=(min_d.date(), max_d.date()),
                key=f"date_{title}",
            )
            if isinstance(d1, date) and isinstance(d2, date):
                show_df = show_df[(show_df["日期"] >= pd.Timestamp(d1)) & (show_df["日期"] <= pd.Timestamp(d2))]

    st.dataframe(show_df, use_container_width=True, height=360)


st.title("股票爬取与分析看板")

default_end = date.today()
default_start = default_end - timedelta(days=365 * 2)

col_a, col_b = st.columns([3, 1])
with col_a:
    stock_code_input = st.text_input("股票代码", value="002324")
with col_b:
    run_btn = st.button("开始爬取并分析", type="primary", use_container_width=True)

if run_btn:
    stock_code = normalize_stock_code(stock_code_input)
    start_date = default_start.strftime("%Y%m%d")
    end_date = default_end.strftime("%Y%m%d")
    adjust = normalize_adjust("qfq")

    with st.spinner("正在爬取与分析，请稍候..."):
        hist_df, fund_df, merged_df, summary_df = run_pipeline(stock_code, start_date, end_date, adjust)

        analysis_dir = os.path.join("stock_outputs", f"analysis_{stock_code}")
        analysis_report_path = ""
        try:
            result = run_analysis_from_data(
                stock_code=stock_code,
                hist_df=hist_df,
                fund_df=fund_df,
                output_dir=analysis_dir,
            )
            analysis_report_path = result.get("report_path", "")
        except Exception as e:
            st.warning(f"分析报告生成失败：{repr(e)}")

        st.session_state["stock_code"] = stock_code
        st.session_state["hist_df"] = hist_df
        st.session_state["fund_df"] = fund_df
        st.session_state["merged_df"] = merged_df
        st.session_state["summary_df"] = summary_df
        st.session_state["analysis_report_path"] = analysis_report_path
        st.session_state["analysis_dir"] = analysis_dir


if "summary_df" in st.session_state:
    stock_code = st.session_state["stock_code"]
    hist_df = st.session_state["hist_df"]
    fund_df = st.session_state["fund_df"]
    merged_df = st.session_state["merged_df"]
    summary_df = st.session_state["summary_df"]
    analysis_report_path = st.session_state.get("analysis_report_path", "")
    analysis_dir = st.session_state.get("analysis_dir", "")

    st.success(f"股票 {stock_code} 数据已更新。")
    st.markdown(
        f"""
        <div class="card">
            <b>本次分析对象：</b>{stock_code}&nbsp;&nbsp;|&nbsp;&nbsp;
            <b>历史行情记录：</b>{len(hist_df)} 条&nbsp;&nbsp;|&nbsp;&nbsp;
            <b>资金流记录：</b>{len(fund_df)} 条
        </div>
        """,
        unsafe_allow_html=True,
    )

    dl_bytes = to_excel_bytes(
        {
            "摘要": summary_df,
            "历史行情": hist_df,
            "资金流": fund_df,
            "合并表": merged_df,
        }
    )
    st.download_button(
        label="下载本次数据（Excel）",
        data=dl_bytes,
        file_name=f"{stock_code}_streamlit_分析数据.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    tab1, tab2, tab3 = st.tabs(["分析报告", "行情与资金图表", "数据筛选查看"])

    with tab1:
        if analysis_report_path and os.path.exists(analysis_report_path):
            with open(analysis_report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("暂无分析报告，请先执行爬取分析。")

    with tab2:
        c1, c2 = st.columns(2)
        if hist_df is not None and not hist_df.empty and "日期" in hist_df.columns:
            p = hist_df.copy()
            p["日期"] = pd.to_datetime(p["日期"], errors="coerce")
            if "收盘" in p.columns:
                with c1:
                    st.line_chart(p.set_index("日期")[["收盘"]], height=280)
            if "MA20" in p.columns and "MA60" in p.columns:
                cols = [x for x in ["收盘", "MA20", "MA60"] if x in p.columns]
                with c2:
                    st.line_chart(p.set_index("日期")[cols], height=280)

        if fund_df is not None and not fund_df.empty and "日期" in fund_df.columns:
            f = fund_df.copy()
            f["日期"] = pd.to_datetime(f["日期"], errors="coerce")
            flow_col = None
            for c in f.columns:
                if "主力" in str(c) and "净流入" in str(c):
                    flow_col = c
                    break
            if flow_col:
                st.line_chart(f.set_index("日期")[[flow_col]], height=300)

        if analysis_dir and os.path.isdir(analysis_dir):
            st.caption(f"分析图已输出到目录：`{analysis_dir}`")

    with tab3:
        render_filter_table(hist_df, "历史行情数据")
        render_filter_table(fund_df, "资金流数据")
        render_filter_table(merged_df, "合并数据")
