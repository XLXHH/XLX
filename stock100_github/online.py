"""Frozen model inference with fresh data. No online fitting."""
import json
import re
import threading
from pathlib import Path
import joblib
import pandas as pd
from events import event_report
from pipeline import collector, bs_rows, fund_fetch, history, login
from research import FLOW, VERSION, context_features, make_features

LOCK = threading.Lock()


def load_model(c):
    payload = joblib.load(Path(c["root"]) / "models/model.joblib")
    if payload["version"] != VERSION:
        raise ValueError("模型特征版本不一致，请用匹配代码重新训练")
    return payload


def fetch_features(code, end, c):
    if not re.fullmatch(r"[036]\d{5}", code):
        raise ValueError("请输入沪深A股6位股票代码")
    symbol = ("sh." if code.startswith("6") else "sz.") + code
    root = Path(c["root"])
    pool = pd.read_csv(root / "universe.csv")
    baskets = json.loads((root / "baskets.json").read_text(encoding="utf-8"))
    start = (pd.Timestamp(end)-pd.Timedelta(days=550)).date()
    issues = []
    with LOCK:
        bs = login()
        try:
            market = history(bs, "sh.000300", start, end)
            prices = history(bs, symbol, start, end)
            matches = pool.loc[pool.code == symbol]
            if len(matches):
                industry = matches.industry.iloc[0]
            else:
                info = bs_rows(bs.query_stock_industry(code=symbol))
                industry = info.industry.iloc[0] if len(info) else "未知"
            peers = []
            for peer in baskets.get(industry, []):
                try:
                    peers.append(prices if peer == symbol else history(bs, peer, start, end))
                except Exception:
                    peers.append(pd.DataFrame(columns=market.columns))
                    issues.append(f"行业篮子成员{peer}获取失败")
        finally:
            bs.logout()
        v5 = collector(c)
        funds = pd.DataFrame()
        try:
            funds = fund_fetch(v5, code)
            funds = funds[funds.date <= pd.Timestamp(end)]
            if funds.empty or funds.date.max() < market.date.max():
                issues.append("资金流未更新至行情日，当前缺失值不会填零")
        except Exception as exc:
            issues.append(f"资金流抓取失败({type(exc).__name__})，仅使用基础模型")
        raw_news = pd.DataFrame()
        try:
            raw_news = v5.get_stock_announcements(code, (pd.Timestamp(end)-pd.Timedelta(days=c["news_days"])).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d"))
            if raw_news.empty:
                issues.append("最近一周公告为空，不能区分无公告和上游接口无结果")
            else:
                # Bounded primary-document enrichment, never thousands of PDFs per request.
                v5.CNINFO_DOWNLOAD_PDFS = True
                v5.CNINFO_EXTRACT_TEXT = True
                selected = raw_news[raw_news["公告标题"].str.contains("业绩|回购|增持|减持|合同|中标|立案|逾期", na=False)].head(3)
                if len(selected):
                    enriched = v5.download_and_read_cninfo_announcements(selected.copy(), code)
                    raw_news = pd.concat([enriched, raw_news[~raw_news.index.isin(selected.index)]], ignore_index=True)
        except Exception as exc:
            issues.append(f"公告或正文获取失败({type(exc).__name__})，保留已取得元数据")
    calendar = pd.DatetimeIndex(market.date)
    ctx = context_features(market, peers, calendar)
    x = make_features(prices, funds, ctx, calendar)
    if x.empty or x.date.iloc[-1] != market.date.iloc[-1]:
        raise ValueError("该股最新行情不完整、停牌或历史不足250交易日")
    if (pd.Timestamp(end)-x.date.iloc[-1]).days > 10:
        raise ValueError("行情已过期，停止推断")
    if x.industry20.tail(1).isna().any():
        issues.append("行业篮子覆盖不足，行业特征由训练中位数补全，本次行业解释不可用")
    return {"x": x.tail(1), "prices": prices, "funds": funds, "market": market, "context": ctx.reset_index(names="date"), "news_raw": raw_news,
            "events": event_report(raw_news), "industry": industry, "issues": issues, "code": symbol}


def infer(payload, data, c):
    rows, explanations = [], {}
    x = data["x"]
    price = float(x.close.iloc[0])
    for h in c["horizons"]:
        models = payload["models"].get(h)
        if models is None:
            continue
        available = x[FLOW].notna().all(axis=1).iloc[0]
        use_flow = models["flow_verified"] and available and models["flow"] is not None
        model = models["flow"] if use_flow else models["base"]
        p = model.predict(x).iloc[0]
        detail = payload["report"]["horizons"][str(h)]
        ci = detail["base_advantage_ci"]
        verified = bool(ci["low"] is not None and ci["low"] > 0)
        rows.append({"期限": h, "上涨概率": p.p, "下跌或持平概率": 1-p.p,
                     "上涨条件收益": p.up, "下跌条件损失": p.down,
                     "上涨条件均价": price*(1+p.up), "下跌条件均价": price*(1-p.down),
                     "期望收益": p.ev, "扣假设成本收益": p.ev-c["round_trip_cost_bps"]/10000,
                     "收益风险比": p.up/p.down if p.down > 0 else None,
                     "价格P10": price*(1+p.q10), "价格P50": price*(1+p.q50), "价格P90": price*(1+p.q90),
                     "模型": "资金流联合模型" if use_flow else "长期基础模型",
                     "资金流状态": detail.get("flow", {}).get("status", "不可用") if available else "最新特征不足",
                     "验证": "基础概率优于常数基准（历史测试）" if verified else "优势未验证"})
        explanations[h] = model.explain(x)
    return pd.DataFrame(rows), explanations
