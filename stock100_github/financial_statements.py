"""Free Eastmoney statement amounts with dates and explicit cumulative periods."""
from datetime import datetime
import math
import re
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

ENDPOINT = "https://datacenter-web.eastmoney.com/api/data/v1/get"
REPORTS = {"利润表": "RPT_DMSK_FN_INCOME", "现金流量表": "RPT_DMSK_FN_CASHFLOW"}
FIELDS = {
    "利润表": {
        "营业总收入": "TOTAL_OPERATE_INCOME",
        "营业收入": "OPERATE_INCOME",
        "归母净利润": "PARENT_NETPROFIT",
        "扣非归母净利润": "DEDUCT_PARENT_NETPROFIT",
        "营业利润": "OPERATE_PROFIT",
        "营业成本": "OPERATE_COST",
    },
    "现金流量表": {
        "经营活动现金流量净额": "NETCASH_OPERATE",
        "投资活动现金流量净额": "NETCASH_INVEST",
        "筹资活动现金流量净额": "NETCASH_FINANCE",
        "现金及现金等价物净增加额": "CCE_ADD",
        "销售商品提供劳务收到的现金": "SALES_SERVICES",
        "购建长期资产支付的现金": "CONSTRUCT_LONG_ASSET",
    },
}


def amount(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def day(value):
    try:
        date = pd.Timestamp(value)
        if pd.isna(date):
            return None
        if date.tzinfo is not None:
            date = date.tz_convert("Asia/Shanghai").tz_localize(None)
        return date.normalize()
    except (ValueError, TypeError):
        return None


@st.cache_data(ttl=21600, max_entries=200, show_spinner=False)
def fetch_rows(code, report):
    # Only this stock's latest 16 report records, not the entire historical market.
    response = requests.get(
        ENDPOINT,
        params={
            "reportName": report,
            "columns": "ALL",
            "filter": f'(SECURITY_CODE="{code}")',
            "sortColumns": "REPORT_DATE",
            "sortTypes": "-1",
            "pageSize": "16",
            "pageNumber": "1",
        },
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"},
        timeout=(6, 18),
        allow_redirects=False,
    )
    if response.status_code != 200:
        raise ValueError(f"财务接口HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise ValueError("财务接口缺少result字段")
    rows = payload["result"].get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("财务接口没有返回记录")
    return rows


def normalize_rows(rows, code, kind, cutoff):
    records = {}
    excluded = 0
    for row in rows:
        if not isinstance(row, dict) or str(row.get("SECURITY_CODE", "")) != code:
            excluded += 1
            continue
        report_date, notice_date = day(row.get("REPORT_DATE")), day(row.get("NOTICE_DATE"))
        if report_date is None or notice_date is None or report_date > cutoff or notice_date > cutoff:
            excluded += 1
            continue
        if (report_date.month, report_date.day) not in {(3, 31), (6, 30), (9, 30), (12, 31)}:
            excluded += 1
            continue
        values = {name: amount(row.get(field)) for name, field in FIELDS[kind].items()}
        record = {
            "报告期": report_date.strftime("%Y-%m-%d"),
            "披露日期": notice_date.strftime("%Y-%m-%d"),
            "来源": "东方财富财务数据中心",
            "来源接口": REPORTS[kind],
            "金额单位": "人民币元",
            "期间口径": "本年年初至报告期末累计",
            "累计金额": values,
            "缺失字段": [name for name, number in values.items() if number is None],
        }
        key = record["报告期"]
        if key not in records or record["披露日期"] > records[key]["披露日期"]:
            records[key] = record
    return records, excluded


def derived_values(records, date_key):
    current = records.get(date_key)
    if current is None:
        return None
    period = pd.Period(date_key, freq="Q")
    prior_key = (period-1).end_time.strftime("%Y-%m-%d")
    previous = records.get(prior_key) if period.quarter > 1 else None
    prior_year = records.get((period-4).end_time.strftime("%Y-%m-%d"))
    quarterly, growth = {}, {}
    for name, current_amount in current["累计金额"].items():
        quarter_value = None
        if current_amount is not None:
            if period.quarter == 1:
                quarter_value = current_amount
            elif previous and previous["累计金额"].get(name) is not None:
                quarter_value = amount(round(current_amount-previous["累计金额"][name], 2))
        quarterly[name] = quarter_value
        year_value = prior_year["累计金额"].get(name) if prior_year else None
        # Percentage growth is ambiguous for zero/negative bases. Keep amounts instead.
        growth[name] = amount(current_amount/year_value-1) if current_amount is not None and year_value is not None and year_value > 0 else None
    return {
        "单季度推导金额": quarterly,
        "推导依据": f"{date_key}为一季报，累计即单季度" if period.quarter == 1 else f"{date_key}累计减{prior_key}累计；缺任一期金额则为空",
        "累计同比增长率_小数": growth,
        "同比比较期": (period-4).end_time.strftime("%Y-%m-%d"),
        "同比说明": "同一来源、同字段、同累计期间比较；上年金额非正时不计算增长率",
    }


def collect_statements(symbol, end, fetch=None):
    code = symbol.split(".")[-1]
    if not re.fullmatch(r"[036]\d{5}", code):
        raise ValueError("需要沪深A股六位代码")
    cutoff = day(end)
    if cutoff is None:
        raise ValueError("无效财务数据截止日")
    fetch = fetch or fetch_rows
    result = {
        "数据源": "东方财富财务数据中心公开接口",
        "查询时间": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "数据截止日": cutoff.strftime("%Y-%m-%d"),
        "金额单位": "人民币元",
        "来源页面": f"https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/Index?type=web&code={symbol.replace('.', '')}",
        "字段对应": FIELDS,
        "接口状态": {},
        "最近四个报告期": [],
        "口径说明": [
            "营业总收入、营业收入与BaoStock主营收入分别保留，不互相替代空值",
            "原表为年初至报告期末累计；单季度金额由同年相邻累计值相减，增长率为小数",
            "现金流负值保留原符号；空值不填零，不用现金流比率倒推金额",
            "按报告期与披露日期过滤。来源为当前快照，可能包含重述，不能保证还原历史发布版本",
        ],
    }
    tables = {}
    for kind, report in REPORTS.items():
        try:
            rows = fetch(code, report)
            records, excluded = normalize_rows(rows, code, kind, cutoff)
            tables[kind] = records
            result["接口状态"][kind] = {"状态": "已取得" if records else "没有截止日前的有效记录", "返回条数": len(rows), "有效条数": len(records), "剔除条数": excluded}
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            tables[kind] = {}
            result["接口状态"][kind] = {"状态": "未取得", "异常类型": type(exc).__name__}
    dates = sorted(set().union(*(table.keys() for table in tables.values())), reverse=True)[:4]
    for date_key in dates:
        item = {"报告期": date_key, "季度": str(pd.Period(date_key, freq="Q"))}
        for kind, table in tables.items():
            if date_key in table:
                item[kind] = {**table[date_key], **derived_values(table, date_key)}
            else:
                item[kind] = {"状态": "本报告期未取得"}
        result["最近四个报告期"].append(item)
    return result


def display_rows(statements):
    """Display yuan amounts as 100M yuan; the AI/source packet retains yuan."""
    rows = []
    for report in statements.get("最近四个报告期", []):
        income, cash = report.get("利润表", {}), report.get("现金流量表", {})
        row = {"报告期": report["报告期"], "利润表披露日": income.get("披露日期"), "现金流表披露日": cash.get("披露日期")}
        for kind, table in [("利润表", income), ("现金流量表", cash)]:
            for name in FIELDS[kind]:
                number = table.get("累计金额", {}).get(name)
                row[name+"（亿元）"] = number/1e8 if number is not None else None
        rows.append(row)
    return rows
