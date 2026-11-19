from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import os
import re
import uuid
from urllib.parse import urlsplit
import pandas as pd
import requests
import streamlit as st
from online import LOCK
from pipeline import collector, history, login, fund_fetch
from research import config
from ai_prompt import SYSTEM_PROMPT, PROMPT_VERSION
from ai_evidence import build_ai_packet
from ai_news import enrich_latest, fetch_document
from ai_report import render_report
from access_control import require_access
from key_codec import decode_key, KeyCodecError
from ai_requests import AnalysisRequests
from context_extra import collect_context
from financial_statements import collect_statements, display_rows

try:
    import ai_local as ai_config
except ModuleNotFoundError as exc:
    if exc.name != "ai_local":
        raise
    import ai_defaults as ai_config

base, model, key = ai_config.BASE_URL, ai_config.MODEL, ai_config.API_KEY

st.set_page_config(page_title="股票100%", page_icon="📈", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"],.stApp{background:linear-gradient(150deg,#edf6ff,#f4f9ff 55%,#e8f2fc);color:#193855}
[data-testid="stHeader"]{background:transparent}
[data-testid="stPlotlyChart"],[data-testid="stMetric"]{background:#fff;border:1px solid #d7e6f5;border-radius:16px;box-shadow:0 5px 20px #274f7d0b}
.block-container{max-width:1180px;padding-top:2.7rem}
.hero{display:flex;align-items:center;gap:15px;background:linear-gradient(110deg,#ffffff,#edf6ff);border:1px solid #d2e3f2;border-radius:18px;padding:22px 28px;margin-bottom:24px}
.hero h1{color:#193b60;margin:0;padding:0;font-size:2rem;line-height:1.2}
.rise{display:inline-flex;align-items:flex-end;gap:4px;height:32px}.rise i{width:8px;background:#d74747;border-radius:3px 3px 0 0}
.rise i:nth-child(1){height:12px}.rise i:nth-child(2){height:21px}.rise i:nth-child(3){height:30px}
.st-key-quote_cards [data-testid="stColumn"]:nth-child(1){--wash:#e6f1ff;--ink:#184d87;--edge:#4581c2}
.st-key-quote_cards [data-testid="stColumn"]:nth-child(2){--wash:#e1f4ee;--ink:#155c50;--edge:#39927c}
.st-key-quote_cards [data-testid="stColumn"]:nth-child(3){--wash:#f2eafb;--ink:#633e83;--edge:#9465b5}
.st-key-quote_cards [data-testid="stMetric"]{background:linear-gradient(115deg,var(--wash),#fff);border-left:4px solid var(--edge);padding:15px 20px;border-radius:14px}
.st-key-quote_cards [data-testid="stMetricLabel"],.st-key-quote_cards [data-testid="stMetricLabel"] p,.st-key-quote_cards [data-testid="stMetricValue"]{color:var(--ink);font-weight:650}
</style><section class="hero"><span class="rise" aria-hidden="true"><i></i><i></i><i></i></span><h1>股票100%</h1></section>""",unsafe_allow_html=True)
require_access()
code = st.text_input("沪深 A 股代码", "600519", max_chars=6)
def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))


@st.cache_data(ttl=86400, max_entries=100, show_spinner=False)
def cached_document(url):
    return fetch_document(url)


@st.cache_resource
def analysis_requests():
    return AnalysisRequests(Path(__file__).parent / "logs" / "ai_requests.jsonl")


if st.session_state.get("answer") and st.session_state.get("answer_version") != PROMPT_VERSION:
    st.session_state.pop("answer", None)

if st.button("抓取最新资料"):
    st.session_state.pop("packet",None)
    st.session_state.pop("answer",None)
    st.session_state.pop("request_audit", None)
    if not re.fullmatch(r"[036]\d{5}",code):
        st.error("请输入六位沪深 A 股代码")
        st.stop()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    end = now.date() if now.hour >= 17 else now.date()-timedelta(days=1)
    packet = {"股票":code,"采集时间":now.isoformat(),"缺失与限制":[]}
    with st.spinner("正在查询行情、东方财富资金流和最近3个月巨潮公告……"), LOCK:
        try:
            bs = login()
            try:
                for label,symbol in [("个股前复权日线",("sh." if code.startswith("6") else "sz.")+code),("沪深300日线","sh.000300")]:
                    p = history(bs,symbol,end-timedelta(days=400),end)
                    packet[label] = records(p.tail(120 if symbol!="sh.000300" else 60))
                    if symbol != "sh.000300":
                        st.session_state["chart_history"] = (code, records(p))
                with st.spinner("正在查询行业对照及近四个季度经营数据……"):
                    packet.update(collect_context(bs, ("sh." if code.startswith("6") else "sz.")+code, end-timedelta(days=160), end))
            finally:
                bs.logout()
        except Exception as exc:
            packet["缺失与限制"].append("行情查询异常："+type(exc).__name__)
        with st.spinner("正在补充收入与现金流报表金额……"):
            statements = collect_statements(code, end)
            packet.setdefault("公司经营数据", {})["报表金额补充"] = statements
        v = collector(config(Path(__file__).with_name("training_config.json")))
        try:
            packet["东财主力净流入_元"] = records(fund_fetch(v,code).tail(60))
        except Exception as exc:
            packet["缺失与限制"].append("资金流未取得："+type(exc).__name__)
        try:
            start_news = (pd.Timestamp(now.date()) - pd.DateOffset(months=3)).date() + timedelta(days=1)
            packet["公告查询范围"] = f"{start_news} 至 {now.date()}（含首尾，滚动3个月）"
            raw_news = v.get_cninfo_announcements_raw(code,start_news.strftime("%Y%m%d"),now.strftime("%Y%m%d"))
            news = v.normalize_cninfo_announcements(raw_news,code)
            packet["巨潮查询状态"] = "成功"
            packet["巨潮公告"] = records(news.filter(items=["公告时间","公告标题","公告详情链接","PDF下载链接"]))
            if not news.empty:
                with st.spinner("正在读取最新最多5条公告PDF；失败时保留标题与链接……"):
                    packet["巨潮公告正文"] = enrich_latest(packet["巨潮公告"], fetch=cached_document)
            if news.empty:
                packet["缺失与限制"].append("巨潮查询成功，所选近3个月返回 0 条公告。")
        except Exception as exc:
            packet["巨潮查询状态"] = "失败"
            packet["缺失与限制"].append("公告未取得："+type(exc).__name__)
    if not packet.get("个股前复权日线"):
        st.error("没有取得个股行情，请稍后重新抓取。")
    else:
        st.session_state.packet = packet

p = st.session_state.get("packet")
if p and p["股票"] == code:
    st.subheader(f"{code} · 日线截止 {p['个股前复权日线'][-1]['date'][:10]}")
    prices = pd.DataFrame(p["个股前复权日线"]).set_index("date")
    prices.index = pd.to_datetime(prices.index)
    close = pd.to_numeric(prices.close)
    latest = close.iloc[-1]
    change = latest / close.iloc[-2] - 1 if len(close)>1 and close.iloc[-2]>0 else None
    a,b,c = st.container(key="quote_cards").columns(3)
    a.metric("最近收盘 · 元", f"{latest:.2f}", None if change is None else f"{change:+.2%}")
    b.metric("资金流记录", f"{len(p.get('东财主力净流入_元',[]))} 天")
    c.metric("近3个月取得公告", f"{len(p['巨潮公告'])} 条" if "巨潮公告" in p else "未取得")
    import plotly.graph_objects as go
    import numpy as np
    def draw(fig, dates, unit, bounds, key):
        ticks=np.unique(np.linspace(0,len(dates)-1,min(8,len(dates)),dtype=int))
        fig.update_layout(height=405,margin=dict(l=12,r=18,t=28,b=12),showlegend=key.startswith("price"),legend=dict(orientation="h",y=1.08),plot_bgcolor="white",
            dragmode="zoom",hovermode="closest",paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(type="category",categoryorder="array",categoryarray=dates,
                       range=[-.5,len(dates)-.5],tickmode="array",tickvals=[dates[i] for i in ticks],
                       ticktext=[dates[i][5:] if len(dates)<80 else dates[i] for i in ticks],showgrid=False),
            yaxis=dict(title=unit,range=bounds,gridcolor="rgba(125,145,165,.17)",fixedrange=False))
        st.plotly_chart(fig,width="stretch",key=key,config={"displayModeBar":True,
            "displaylogo":False,"doubleClick":"reset","modeBarButtons":[["zoomIn2d","zoomOut2d","resetScale2d"]]})
    price_tab,flow_tab,news_tab=st.tabs(["行情走势","主力资金","近期公告"])
    with price_tab:
        period=st.segmented_control("行情区间",["全部行情","最近一周","最近一个月","最近一年"],default="全部行情") or "全部行情"
        cache=st.session_state.get("chart_history")
        raw=cache[1] if cache and cache[0]==code else p["个股前复权日线"]
        series=pd.DataFrame(raw).assign(date=lambda d:pd.to_datetime(d.date)).set_index("date").close
        series=pd.to_numeric(series,errors="coerce").dropna().sort_index()
        series=series[~series.index.duplicated(keep="last")]
        if len(series):
            means={n:series.rolling(n).mean() for n in (5,20)}
            delta={"最近一周":pd.DateOffset(weeks=1),"最近一个月":pd.DateOffset(months=1),"最近一年":pd.DateOffset(years=1)}.get(period)
            if delta is not None:
                series=series[series.index>series.index.max()-delta]
            means={n:m.reindex(series.index) for n,m in means.items()}
            dates=series.index.strftime("%Y-%m-%d").tolist()
            visible=pd.concat([series,*means.values()]).dropna()
            lo,hi=visible.min(),visible.max()
            pad=max((hi-lo)*.17,.01)
            st.caption(f"前复权收盘价 · {dates[0]} 至 {dates[-1]} · {len(series)} 个有效交易日 ｜ 拖选放大，双击复位")
            fig=go.Figure(go.Scatter(x=dates,y=series,mode="lines+markers" if len(series)<25 else "lines",
                name="收盘价",line=dict(color="#256DB0",width=2.8),marker=dict(size=5),hovertemplate="%{x}<br>收盘 %{y:.2f} 元<extra></extra>"))
            for n,color in [(5,"#CC8A2C"),(20,"#8664AE")]:
                fig.add_scatter(x=dates,y=means[n],mode="lines",name=f"{n} 日均线",line=dict(color=color,width=1.7),hovertemplate="%{y:.2f} 元")
            last=series.iloc[-1]
            tag="最新 · 最高" if last==series.max() else "最新 · 最低" if last==series.min() else "最新"
            fig.add_annotation(x=dates[-1],y=last,text=f"{tag} {last:.2f} 元",showarrow=True,ax=-54,ay=-25,bgcolor="#216497",borderpad=4,font=dict(color="white",size=13))
            for idx,label,offset,color in [(series.idxmax(),"最高", -16,"#9E443F"),(series.idxmin(),"最低",16,"#196B61")]:
                if series.loc[idx]==last or (series.min()==series.max() and label=="最低"):
                    continue
                pos=series.index.get_loc(idx)
                fig.add_annotation(x=dates[pos],y=series.loc[idx],text=f"{label} {series.loc[idx]:.2f} 元",
                    showarrow=False,yshift=offset,xanchor="left" if pos==0 else "right" if pos==len(series)-1 else "center",
                    bgcolor="white",font=dict(color=color,size=12))
            draw(fig,dates,"收盘价（元）",[lo-pad,hi+pad],f"price_{code}_{period}")
    with flow_tab:
        flow=pd.DataFrame(p.get("东财主力净流入_元",[]))
        if len(flow):
            flow=flow.sort_values("date").drop_duplicates("date",keep="last")
            net=pd.to_numeric(flow.main_net,errors="coerce").dropna()/1e8
            if len(net):
                dates=pd.to_datetime(flow.loc[net.index,"date"]).dt.strftime("%Y-%m-%d").tolist()
                threshold=net.abs().quantile(.70)
                st.caption(f"按本股最近 {len(net)} 条有效记录计算：净额绝对值超过第 70 百分位（{threshold:.3f} 亿元）时标注金额。红色为流入，绿色为流出。")
                large=net.abs()>threshold
                fig=go.Figure(go.Bar(x=dates,y=net,marker_color=["#CC5650" if v>=0 else "#218877" for v in net],
                    text=[(f"{v:+.2f}亿" if abs(v)>=1 else f"{v*10000:+.0f}万") if flag else "" for v,flag in zip(net,large)],
                    textposition="outside",textangle=0,textfont=dict(size=15,color="#213F59"),cliponaxis=False,
                    hovertemplate="%{x}<br>主力净额 %{y:+.3f} 亿元<extra></extra>"))
                fig.update_layout(uniformtext=dict(minsize=15,mode="show"),bargap=.23,barcornerradius=4)
                fig.add_hline(y=0,line_color="#899DA8",line_width=1)
                lo,hi=min(0,net.min()),max(0,net.max())
                pad=max((hi-lo)*.26,.001)
                draw(fig,dates,"净额（亿元）",[lo-pad,hi+pad],f"flow_{code}")
        else:
            st.info("未取得资金流，暂时无法绘图。")
    with news_tab:
        news = pd.DataFrame(p.get("巨潮公告",[]))
        if len(news):
            st.dataframe(news,hide_index=True,column_config={"公告详情链接":st.column_config.LinkColumn("原文",display_text="阅读公告")})
        else:
            st.info("巨潮接口在近3个月返回 0 条公告。" if p.get("巨潮查询状态")=="成功" else "本次未取得公告，请查看下方采集状态。")
        for notice in p.get("巨潮公告正文", []):
            with st.expander(f"{notice['正文状态']} · {notice.get('公告标题', '')}"):
                st.caption(notice.get("正文限制", ""))
                if notice.get("正文摘录"):
                    st.text(notice["正文摘录"])
    with st.expander("行业对照与公司经营数据", expanded=True):
        context = p.get("行业对照", {})
        st.write("所属行业", context.get("行业") or "未取得")
        st.caption(context.get("说明", ""))
        comparison = pd.DataFrame(context.get("对照", []))
        if len(comparison):
            st.dataframe(comparison, hide_index=True)
            st.caption("收益率以小数表示，0.05表示5%。")
        for issue in context.get("问题", []):
            st.warning(issue)
        business = p.get("公司经营数据", {})
        statements = business.get("报表金额补充", {})
        if statements:
            st.caption("报表金额来源为东方财富，展示单位为亿元，AI资料保留人民币元。下列金额按年初至报告期末累计。")
            for kind, status in statements.get("接口状态", {}).items():
                if status.get("状态") != "已取得":
                    st.warning(f"{kind} {status.get('状态', '未取得')}")
            money = pd.DataFrame(display_rows(statements))
            if not money.empty:
                income_tab, cash_tab = st.tabs(["收入与利润", "现金流量"])
                with income_tab:
                    fields = ["报告期", "利润表披露日", "营业总收入（亿元）", "营业收入（亿元）", "归母净利润（亿元）", "扣非归母净利润（亿元）", "营业利润（亿元）", "营业成本（亿元）"]
                    st.dataframe(money[fields], hide_index=True, width="stretch")
                with cash_tab:
                    fields = ["报告期", "现金流表披露日", "经营活动现金流量净额（亿元）", "投资活动现金流量净额（亿元）", "筹资活动现金流量净额（亿元）", "现金及现金等价物净增加额（亿元）", "销售商品提供劳务收到的现金（亿元）", "购建长期资产支付的现金（亿元）"]
                    st.dataframe(money[fields], hide_index=True, width="stretch")
                st.caption("空白表示该字段未取得。主营收入、营业收入和营业总收入分别保留。单季度推导与同口径累计同比随数据发送给AI。")
            else:
                st.info("本次未取得可用的报表金额，经营分析将使用已取得的其他指标。")
        else:
            st.info("请重新采集以补充收入与现金流金额。")
        st.caption("BaoStock原始经营指标")
        st.json({k: v for k, v in business.items() if k != "报表金额补充"}, expanded=False)
    for issue in p["缺失与限制"]:
        st.warning(issue)
    ai_packet = build_ai_packet(p, prepared_news=p.get("巨潮公告正文"))
    # Bodies are attached only to the selected notices, never send an extra duplicate list.
    ai_packet.pop("巨潮公告正文", None)
    if "公告筛选说明" in ai_packet:
        st.caption(ai_packet["公告筛选说明"] + " 来源下载保留全部公告。")
    st.caption(f"当前模型：{model} · 详细研判，涵盖证据、转折与买卖条件。")
    blob = json.dumps(ai_packet,ensure_ascii=False,separators=(",",":"))
    st.download_button("下载本次来源数据",json.dumps(p,ensure_ascii=False),f"{code}_sources.json","application/json")
    entered_key = st.text_input("编码后的 DeepSeek 密钥", type="password", key="analysis_api_key",
                                placeholder="粘贴 D1. 或 K1. 开头的编码；本地已配置密钥可留空")
    st.caption("先在本机运行“编码密钥.cmd”。编码内容仅用于本次会话，程序会还原后调用 AI。")
    if st.button("发送以上资料并分析",type="primary"):
        key_error = None
        try:
            request_key = decode_key(entered_key) if entered_key.strip() else str(key or "").strip()
        except KeyCodecError as exc:
            request_key = ""
            key_error = str(exc)
        u = urlsplit(base.strip())
        if key_error:
            st.error(key_error)
        elif not request_key:
            st.error("请先在上方填写编码后的 DeepSeek 密钥。")
        elif u.scheme!="https" or not u.hostname or u.username or u.password or u.query or u.fragment or not model.strip():
            st.error("AI服务地址或模型配置无效，请检查配置文件。")
        else:
            prompt = SYSTEM_PROMPT
            request_body = {"model":model.strip(),"messages":[{"role":"system","content":prompt},{"role":"user","content":blob}]}
            if u.hostname == "api.deepseek.com":
                request_body.update(thinking={"type": "enabled" if getattr(ai_config, "THINKING_ENABLED", True) else "disabled"},
                                    reasoning_effort=getattr(ai_config, "REASONING_EFFORT", "high"),
                                    max_tokens=getattr(ai_config, "MAX_TOKENS", 32768))
            owner = st.session_state.setdefault("analysis_owner_id", uuid.uuid4().hex)
            with st.spinner("等待在线 AI 分析……同一批资料的重复点击将复用请求。"):
                result = analysis_requests().run(owner, base.strip().rstrip("/")+"/chat/completions", request_body,
                                                 request_key, (10,getattr(ai_config, "REQUEST_TIMEOUT_SECONDS", 300)))
            st.session_state.request_audit = result["audit"]
            if result["status"] == "completed":
                st.session_state.answer = result["answer"]
                st.session_state.answer_version = PROMPT_VERSION
                if result["reused"]:
                    st.info("已复用本批资料的分析结果，没有再次调用AI。")
            elif result["status"] == "pending":
                st.info("本批资料的请求仍在进行，没有重复提交。稍后再次点击可取回结果。")
            else:
                st.error(result["error"])
    if st.session_state.get("request_audit"):
        audit = st.session_state.request_audit
        with st.expander("本次调用与费用记录"):
            st.write(f"请求时间：{audit['started_at']} ｜ 状态：{audit['status']}")
            st.code(audit.get("provider_request_id", audit["local_request_id"]), language=None)
            usage = audit.get("usage", {})
            x,y,z = st.columns(3)
            x.metric("输入 tokens", usage.get("prompt_tokens", "未返回"))
            y.metric("输出 tokens（含思考）", usage.get("completion_tokens", "未返回"))
            z.metric("总 tokens", usage.get("total_tokens", "未返回"))
            cost = audit.get("cost_estimate")
            if cost:
                st.write(f"空闲／高峰价格估算：¥{cost['空闲时段估算元']:.4f}／¥{cost['高峰时段估算元']:.4f}")
                st.caption("按2026-11-18官方价格与返回用量估算，具体时段与账单以DeepSeek为准。")
            st.download_button("下载本次调用记录", json.dumps(audit, ensure_ascii=False, indent=2), f"{code}_request.json", "application/json")
            st.caption("本次记录不含密钥或正文。同一浏览器会话、服务未重启时拦截同批重复请求；重新采集可能产生新请求。")
    if st.session_state.get("answer"):
        render_report(st.session_state.answer, code, p["个股前复权日线"][-1]["date"][:10])








