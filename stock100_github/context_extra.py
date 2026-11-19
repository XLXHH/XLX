import pandas as pd
import streamlit as st
from pipeline import bs_rows, history

@st.cache_data(ttl=21600, max_entries=100, show_spinner=False)
def collect_context(_bs, symbol, start, end):
    bs = _bs
    result = {"说明": "同一BaoStock行业分类按代码选取最多5只非ST样本，计算等权收益；可能包含多个细分行业", "问题": []}
    try:
        mapping = bs_rows(bs.query_stock_industry())
        own = mapping[mapping.code == symbol]
        industry = own.industry.iloc[0] if len(own) else None
        result["行业"] = industry
        peers = mapping[(mapping.industry == industry) & (mapping.code != symbol)].sort_values("code").head(5) if industry else mapping.iloc[:0]
        peers = peers[~peers.code_name.str.contains('ST|退', case=False, regex=True, na=False)]
        series = []
        for row in peers.itertuples():
            try:
                p = history(bs, row.code, start, end)
                s = p.set_index("date").close.pct_change(fill_method=None).rename(row.code)
                series.append(s)
            except Exception:
                result["问题"].append(f"同行{row.code}行情未取得")
        if len(series) >= 3:
            returns = pd.concat(series, axis=1).sort_index()
            good = returns.notna().sum(axis=1) >= max(3, int(len(peers)*.8+.999))
            daily = returns.mean(axis=1).where(good)
            own_prices = history(bs, symbol, start, end).set_index("date").close
            rows = []
            for n in (5,20,60):
                window = daily.tail(n)
                dates = daily.index
                if len(dates)>n and window.notna().all() and dates[-n-1] in own_prices.index and dates[-1] in own_prices.index:
                    sector = float((1+window).prod()-1)
                    stock = float(own_prices.loc[dates[-1]]/own_prices.loc[dates[-n-1]]-1)
                    rows.append({"交易日窗口":n,"起始日":str(dates[-n-1].date()),"截止日":str(dates[-1].date()),"个股收益":stock,"同行等权收益":sector,"超额收益":stock-sector})
            if not rows:
                result["问题"].append("共同日期覆盖不足，未生成对照窗口")
            result["对照"] = rows
            result["样本"] = peers[["code","code_name"]].to_dict("records")
        else:
            result["问题"].append("有效同行少于3只，无法形成行业对照")
    except Exception as exc:
        result["问题"].append("行业查询失败："+type(exc).__name__)
    reports = {}
    # The latest four fully ended quarters; unavailable disclosures remain missing.
    quarter = pd.Period(pd.Timestamp(end),freq="Q")-1
    for i in range(4):
        q = quarter-i
        item = {"报告期":str(q),"披露数据":{}}
        for label,method in [("盈利",bs.query_profit_data),("成长",bs.query_growth_data),("现金流",bs.query_cash_flow_data)]:
            try:
                df = bs_rows(method(code=symbol,year=q.year,quarter=q.quarter))
                if not df.empty:
                    published = pd.to_datetime(df.pubDate,errors="coerce")
                    df = df[published <= pd.Timestamp(end)]
                    item["披露数据"][label] = df.to_dict("records")
            except Exception as exc:
                item["披露数据"][label] = {"获取状态":type(exc).__name__}
        reports[str(q)] = item
    return {"行业对照":result,"公司经营数据":{"最近四个已结束季度":reports,"口径说明":"字段来自BaoStock原始财务接口，按pubDate过滤。部分指标为年初至报告期累计或TTM，必须核对口径后比较，不直接将累计值作单季度环比。金额单位元，比例单位按字段文档核对。未披露和未取得均不能填零。"}}
