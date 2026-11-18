"""Small, deterministic context to support the language model's analysis."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_prompt import PROMPT_VERSION


def value(number):
    return round(float(number), 6) if pd.notna(number) and np.isfinite(number) else None


def frame(rows, columns):
    df = pd.DataFrame(rows)
    if df.empty or "date" not in df:
        return pd.DataFrame(columns=["date", *columns])
    df["date"] = pd.to_datetime(df.date, errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    for name in columns:
        df[name] = pd.to_numeric(df[name], errors="coerce") if name in df else np.nan
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def period_meta(df):
    return {"记录数": len(df), "起始日": df.date.iloc[0].strftime("%Y-%m-%d") if len(df) else None,
            "截止日": df.date.iloc[-1].strftime("%Y-%m-%d") if len(df) else None}


def summarize(packet):
    p = frame(packet.get("个股前复权日线", []), ["open", "high", "low", "close", "volume", "amount"])
    f = frame(packet.get("东财主力净流入_元", []), ["main_net"])
    m = frame(packet.get("沪深300日线", []), ["close"])
    out = {"口径": "本地计算，只对实际有效数据计算；收益率为小数、金额为元；缺失不填零", "行情": period_meta(p), "资金": period_meta(f), "大盘": period_meta(m)}
    if p.empty:
        out["限制"] = ["个股日线缺失，不能输出价格方案"]
        return out
    price = p.close.where(p.close > 0)
    out["行情"]["最新收盘"] = value(price.iloc[-1])
    out["行情"]["无效收盘记录数"] = int(price.isna().sum())
    out["行情"]["零成交量记录数"] = int(p.volume.eq(0).sum())
    out["行情"]["均线"] = {}
    for n in (5, 10, 20, 60):
        avg = price.rolling(n).mean()
        out["行情"]["均线"][f"MA{n}"] = {"数值": value(avg.iloc[-1]), "5条记录前数值": value(avg.iloc[-6]) if len(avg) >= 6 else None}
    out["行情"]["涨跌幅"] = {f"{n}条记录收益": value(price.iloc[-1]/price.iloc[-n-1]-1) if len(price) > n else None for n in (1, 5, 10, 20, 60)}
    out["行情"]["量能"] = {"当日成交量": value(p.volume.iloc[-1]), "前20条记录平均成交量_不含当日": value(p.volume.shift().rolling(20).mean().iloc[-1])}
    av = p.volume.shift().rolling(20).mean().iloc[-1]
    out["行情"]["量能"]["当日量比_对前20条"] = value(p.volume.iloc[-1]/av) if pd.notna(av) and av > 0 else None
    tr = pd.concat([(p.high-p.low).where((p.high >= p.low) & (p.low > 0)), (p.high-price.shift()).abs(), (p.low-price.shift()).abs()], axis=1).max(axis=1)
    out["行情"]["ATR14_简单平均真实波幅"] = value(tr.rolling(14).mean().iloc[-1])
    out["行情"]["波动说明"] = "ATR为价格单位的波幅，不是涨跌预测；按有效记录滚动，遇停牌/日期缺口须降低可比性"
    levels = {}
    for n in (5, 20, 60):
        prior = p.iloc[:-1].tail(n)
        if len(prior) < n:
            continue
        hi, lo = prior.high.where(prior.high > 0), prior.low.where(prior.low > 0)
        if hi.notna().any() and lo.notna().any():
            levels[f"前{n}条_不含当日"] = {"最高价": value(hi.max()), "最高价日期": prior.loc[hi.idxmax(), "date"].strftime("%Y-%m-%d"),
                                             "最低价": value(lo.min()), "最低价日期": prior.loc[lo.idxmin(), "date"].strftime("%Y-%m-%d")}
    out["行情"]["候选结构价位_不是自动买卖信号"] = levels
    # CSI300 comparisons are aligned to the same dates, never compare two unequal windows.
    aligned = p[["date", "close"]].merge(m[["date", "close"]], on="date", suffixes=("_stock", "_market"))
    out["大盘"]["共同日期"] = period_meta(aligned)
    out["大盘"]["相对收益"] = {}
    for n in (5, 20):
        if len(aligned) > n and (aligned[["close_stock", "close_market"]].iloc[[-1, -n-1]] > 0).all().all():
            sr = aligned.close_stock.iloc[-1]/aligned.close_stock.iloc[-n-1]-1
            mr = aligned.close_market.iloc[-1]/aligned.close_market.iloc[-n-1]-1
            out["大盘"]["相对收益"][f"最近{n}个共同日期间隔"] = {"个股": value(sr), "沪深300": value(mr), "差值": value(sr-mr)}
    # Keep missing sessions visible: rolling sums require a flow record for each price date.
    common = p[["date", "amount"]].merge(f[["date", "main_net"]], how="left", on="date")
    out["资金"]["与行情同日窗口"] = {}
    for n in (1, 5, 20):
        recent = common.tail(n)
        covered = len(recent) == n and recent.main_net.notna().all()
        net = value(recent.main_net.sum()) if covered else None
        amount = recent.amount.sum() if len(recent) == n and recent.amount.notna().all() and (recent.amount > 0).all() else None
        out["资金"]["与行情同日窗口"][f"最近{n}条行情记录"] = {"完整覆盖": bool(covered), "净流入合计": net,
            "净流入为正天数": int(recent.main_net.gt(0).sum()) if covered else None,
            "净流入合计除成交额合计": value(net/amount) if covered and amount is not None and amount > 0 else None}
    available = f.loc[(f.date <= p.date.iloc[-1]) & f.main_net.notna()]
    out["资金"]["净额绝对值70分位_行情截止以前的已提供记录"] = value(available.main_net.abs().quantile(.70)) if len(available) else None
    out["限制"] = ["指标基于已发送的有限日线，并非全上市历史", "前复权历史价位不等同实时委托报价", "当前无行业、估值、个人持仓成本及风险预算信息"]
    return out


def build_ai_packet(packet, prepared_news=None, analysis_time=None):
    data = deepcopy(packet)
    data.pop("巨潮公告正文", None)
    if "巨潮公告" in data:
        notices = sorted(data["巨潮公告"], key=lambda n: n.get("公告时间") or "", reverse=True)
        latest = notices[:5]
        # Only attach text to selected records with the SAME title, timestamp and URL.
        def identity(n):
            return tuple(n.get(k) for k in ("公告标题", "公告时间", "公告详情链接"))
        bodies = {identity(n): n for n in (prepared_news or [])}
        data["巨潮公告"] = [{**n, **bodies.get(identity(n), {}), "证据编号": f"公告{i}",
                              "正文状态": bodies.get(identity(n), {}).get("正文状态", "未读取正文")}
                             for i, n in enumerate(latest, 1)]
        data["公告筛选说明"] = f"三个月取得{len(notices)}条，仅发送最新{len(latest)}条；未发送的不代表不重要。链接本身不等于已读取。"
    data["分析时间"] = analysis_time or datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    data["分析设定"] = {"提示词版本": PROMPT_VERSION, "观察周期": "依据结构自行选择，不强制固定期限", "持仓与风险预算": "未知，分别分析空仓者与持有者", "行情口径": "BaoStock前复权收盘日线；不是盘中实时报价", "公告口径": "巨潮原文PDF文本提取，成功时附摘录；缺失时仅标题与链接"}
    data["本地计算摘要"] = summarize(data)
    return data
