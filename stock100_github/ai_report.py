"""Render the original AI prose in the dashboard's visual style."""
import re

import streamlit as st

TITLES = {1: "研判结论", 2: "证据与冲突", 3: "关键价位与转折", 4: "行情路径", 5: "买卖计划", 6: "反证与风险", 7: "后续观察"}
ALIASES = {1: r"结论|先回答|当前行情|当前判断", 2: r"证据|冲突", 3: r"价位|转折", 4: r"路径|怎么走|情景", 5: r"空仓|持有|买卖|操作", 6: r"反证|风险", 7: r"观察|盯|监控|跟踪"}


def split_report(text):
    """Keep unrecognized text; never derive prices/actions/probabilities from prose."""
    blocks = {}
    current = 0
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        clean = re.sub(r"^\s*#{1,3}\s+", "", line).strip().strip("*").strip()
        match = re.match(r"^([一二三四五六七1-7])[、.．]\s*(.+)$", clean) if not fenced else None
        if match:
            n = int(match[1]) if match[1].isdigit() else "一二三四五六七".index(match[1])+1
            heading = match[2].strip()
            # Nested numbered paragraphs should remain inside their original section.
            if len(clean) < 110 and re.search(ALIASES[n], heading) and (current == 0 or n > current):
                current = n
                blocks.setdefault(current, [])
                # Preserve nonstandard headings containing additional model content.
                if heading.rstrip("：:") != TITLES[n]:
                    blocks[current].append(f"**{heading}**")
                continue
        blocks.setdefault(current, []).append(line)
    return {n: "\n".join(lines).strip() for n, lines in blocks.items() if "\n".join(lines).strip()}


REPORT_STYLE = """<style>
.st-key-ai_report{margin-top:26px}
.st-key-ai_report [data-testid="stVerticalBlockBorderWrapper"]{border-color:#d6e5f3!important}
.st-key-ai_report [data-testid="stTabs"] [data-baseweb="tab-list"]{gap:10px;padding:6px;background:#e5eff9;border-radius:12px}
.st-key-ai_report [role="tab"]{padding:9px 17px;border-radius:9px;color:#496681}
.st-key-ai_report [role="tab"][aria-selected="true"]{background:#fff;color:#215f9c;box-shadow:0 2px 8px #214b7620}
.st-key-report_summary{background:linear-gradient(115deg,#e3f0ff,#fff 80%);border:1px solid #bed6ef;border-left:5px solid #397fb8;border-radius:16px;padding:23px 26px;margin-bottom:18px}
.st-key-report_evidence,.st-key-report_levels,.st-key-report_paths,.st-key-report_actions,.st-key-report_risk,.st-key-report_watch,.st-key-report_fallback{background:#fff;border:1px solid #d6e5f3;border-radius:16px;padding:22px 26px;margin-top:14px;box-shadow:0 4px 18px #244e7308}
.st-key-report_levels{border-top:3px solid #5d95bc}
.st-key-report_actions{border-top:3px solid #418e83}
.st-key-report_risk{border-top:3px solid #d59864}
.st-key-report_watch{border-top:3px solid #9383bc}
.st-key-ai_report h3{font-size:1.25rem;color:#204668;margin:0 0 12px;padding:0}
.st-key-ai_report p,.st-key-ai_report li{color:#304f68;line-height:1.9}
.st-key-ai_report [data-testid="stMarkdownContainer"] blockquote{background:#edf5fc;border-left:3px solid #6a9bc3;padding:10px 14px;border-radius:0 8px 8px 0}
.st-key-ai_report table{font-size:.94rem;width:100%}
.st-key-ai_report th{background:#eaf2fa;color:#244c72}
.st-key-ai_report td{vertical-align:top}
@media(max-width:700px){.st-key-report_summary,.st-key-report_evidence,.st-key-report_levels,.st-key-report_paths,.st-key-report_actions,.st-key-report_risk,.st-key-report_watch{padding:17px 15px}}
</style>"""


def render_report(text, code, quote_date):
    blocks = split_report(text)
    st.markdown(REPORT_STYLE, unsafe_allow_html=True)
    with st.container(key="ai_report"):
        st.subheader("AI 行情研判")
        st.caption(f"{code} · 行情截至 {quote_date} · 条件研判，概率未经统计校准")
        if not any(n > 0 for n in blocks):
            with st.container(key="report_fallback"):
                st.markdown(text)
        else:
            with st.container(key="report_summary"):
                st.markdown("### 研判结论")
                if blocks.get(0):
                    st.markdown(blocks[0])
                if blocks.get(1):
                    st.markdown(blocks[1])
                elif not blocks.get(0):
                    st.caption("模型未单独提供结论，详情见下方报告。")
            groups = [("证据与价位", [(2, "report_evidence"), (3, "report_levels")]),
                      ("行情路径", [(4, "report_paths")]),
                      ("买卖计划", [(5, "report_actions")]),
                      ("风险与跟踪", [(6, "report_risk"), (7, "report_watch")])]
            visible = [(label, [(n, key) for n, key in sections if n in blocks]) for label, sections in groups]
            visible = [(label, sections) for label, sections in visible if sections]
            if visible:
                for tab, (_, sections) in zip(st.tabs([label for label, _ in visible]), visible):
                    with tab:
                        for n, key in sections:
                            with st.container(key=key):
                                st.markdown(f"### {TITLES[n]}")
                                st.markdown(blocks[n])
        with st.expander("完整报告原文"):
            st.markdown(text)
        st.download_button("下载完整分析报告", text, f"{code}_analysis.md", key="report_download")
