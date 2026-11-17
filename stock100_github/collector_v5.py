import math
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

try:
    import akshare as ak
except ImportError as e:
    raise ImportError("缺少 akshare，请先执行: pip install -U akshare") from e

try:
    import baostock as bs
except ImportError as e:
    raise ImportError("缺少 baostock，请先执行: pip install -U baostock") from e

try:
    import fitz  # PyMuPDF，优先用于公告 PDF 文本提取
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None



# =========================================================
# 配置区
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "stock_outputs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CACHE_DIR = os.path.join(BASE_DIR, "stock_cache")
FUND_FLOW_CACHE_DIR = os.path.join(CACHE_DIR, "eastmoney_fund_flow")
ANNOUNCEMENT_DIR = os.path.join(BASE_DIR, "stock_announcements")
os.makedirs(FUND_FLOW_CACHE_DIR, exist_ok=True)
os.makedirs(ANNOUNCEMENT_DIR, exist_ok=True)

REQUEST_TIMEOUT = 20
MAX_RETRY = 3
RETRY_SLEEP_MIN = 1.5
RETRY_SLEEP_MAX = 5.0

# 东财历史资金流专用配置（仅此模块使用 Cookie；BaoStock/巨潮/其他 AKShare 请求不使用该 Cookie）
# 1) 用浏览器正常访问/登录东方财富后，打开开发者工具 -> Network -> 任意东财请求 -> Request Headers -> Cookie
# 2) 将 Cookie: 后面的整串内容原样粘贴到下面双引号内。
# 3) Cookie 属于敏感登录信息，请只保存在你自己的本机脚本中，不要提交到 Git/网盘或发给他人。
EASTMONEY_COOKIE = os.environ.get("EASTMONEY_COOKIE", "")

# 固定使用一个浏览器 UA，和 Cookie 的浏览器身份保持稳定，不再随机切换 UA。
EASTMONEY_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/152.0.0.0 Safari/537.36"
)

# 东财失败后使用长退避，避免连续撞接口。单位：秒。
EASTMONEY_RETRY_DELAYS = [8, 25, 60]

# Cookie 直连接口失败后，再尝试 AKShare 一次作为包装层兜底。
FUND_FLOW_RETRY = 1

# 东财资金流历史窗口配置。
# 当前 push2his 在 2026 年实际常被服务端限制为约 120 个交易日。
# 程序会先请求超大 lmt，再尝试用 end 日期向前探测分段；
# 如果服务端忽略 end，程序会识别重复窗口并停止，避免伪造“已补齐”。
EASTMONEY_FUND_FLOW_LIMIT = 100000
EASTMONEY_MAX_BACKFILL_SEGMENTS = 30
EASTMONEY_SEGMENT_SLEEP_MIN = 1.5
EASTMONEY_SEGMENT_SLEEP_MAX = 3.5

# 巨潮公告 PDF 下载与正文提取。
CNINFO_DOWNLOAD_PDFS = True
CNINFO_EXTRACT_TEXT = True
CNINFO_PDF_SLEEP_MIN = 0.25
CNINFO_PDF_SLEEP_MAX = 0.75
CNINFO_MAX_PDF_MB = 80
CNINFO_TEXT_PREVIEW_CHARS = 1200
CNINFO_PDF_BASE = "http://static.cninfo.com.cn/"

# 巨潮单页 30 条，避免一次请求过大。
CNINFO_PAGE_SIZE = 30

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

CNINFO_CATEGORY_MAP = {
    "年报": "category_ndbg_szsh",
    "半年报": "category_bndbg_szsh",
    "一季报": "category_yjdbg_szsh",
    "三季报": "category_sjdbg_szsh",
    "业绩预告": "category_yjygjxz_szsh",
    "权益分派": "category_qyfpxzcs_szsh",
    "董事会": "category_dshgg_szsh",
    "监事会": "category_jshgg_szsh",
    "股东大会": "category_gddh_szsh",
    "日常经营": "category_rcjy_szsh",
    "公司治理": "category_gszl_szsh",
    "中介报告": "category_zj_szsh",
    "首发": "category_sf_szsh",
    "增发": "category_zf_szsh",
    "股权激励": "category_gqjl_szsh",
    "配股": "category_pg_szsh",
    "解禁": "category_jj_szsh",
    "公司债": "category_gszq_szsh",
    "可转债": "category_kzzq_szsh",
    "其他融资": "category_qtrz_szsh",
    "股权变动": "category_gqbd_szsh",
    "补充更正": "category_bcgz_szsh",
    "澄清致歉": "category_cqdq_szsh",
    "风险提示": "category_fxts_szsh",
    "特别处理和退市": "category_tbclts_szsh",
    "退市整理期": "category_tszlq_szsh",
}


# =========================================================
# 日志 / 通用调用
# =========================================================


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{now}] {msg}"
    print(text)
    with open(os.path.join(LOG_DIR, "stock_collect.log"), "a", encoding="utf-8") as f:
        f.write(text + "\n")


@dataclass
class CallResult:
    value: Any
    success: bool
    source: str = ""
    error: str = ""
    retries: int = 0


def retry_sleep(attempt: int):
    low = RETRY_SLEEP_MIN * attempt
    high = RETRY_SLEEP_MAX * attempt
    sec = min(random.uniform(low, high), 12.0)
    log(f"退避等待 {sec:.2f} 秒")
    time.sleep(sec)


def safe_call(
    func: Callable,
    name: str,
    default=None,
    retry: int = MAX_RETRY,
    empty_is_error: bool = True,
) -> CallResult:
    """统一调用器：None/空表默认视为失败并继续重试。"""
    last_error = ""
    for attempt in range(1, retry + 1):
        try:
            log(f"{name}: 第 {attempt}/{retry} 次尝试")
            value = func()
            if value is None:
                raise ValueError("返回 None")
            if isinstance(value, pd.DataFrame) and value.empty and empty_is_error:
                raise ValueError("返回空 DataFrame")
            log(f"{name}: 成功")
            return CallResult(value=value, success=True, source=name, retries=attempt)
        except Exception as e:
            last_error = repr(e)
            log(f"{name}: 失败: {last_error}")
            if attempt < retry:
                retry_sleep(attempt)
    return CallResult(
        value=default,
        success=False,
        source=name,
        error=last_error,
        retries=retry,
    )


# =========================================================
# 股票代码 / 日期 / 字段标准化
# =========================================================


def normalize_stock_code(code: str) -> str:
    code = str(code).strip().lower()
    for prefix in ["sh", "sz", "bj"]:
        code = code.replace(prefix, "")
    code = "".join(c for c in code if c.isdigit())
    return code.zfill(6)


def detect_market(code: str) -> str:
    """返回 sh / sz / bj。"""
    code = normalize_stock_code(code)
    if code.startswith(("4", "8", "9")):
        return "bj"
    if code.startswith(("0", "1", "2", "3")):
        return "sz"
    return "sh"


def normalize_adjust(adjust: str) -> str:
    adjust = str(adjust).strip().lower()
    if adjust in ["qfq", "前复权"]:
        return "qfq"
    if adjust in ["hfq", "后复权"]:
        return "hfq"
    if adjust in ["none", "不复权", ""]:
        return ""
    return "qfq"


def baostock_adjust_flag(adjust: str) -> str:
    # BaoStock: 1=后复权, 2=前复权, 3=不复权
    adjust = normalize_adjust(adjust)
    return {"hfq": "1", "qfq": "2", "": "3"}[adjust]


def yyyymmdd_to_iso(date_str: str) -> str:
    dt = datetime.strptime(str(date_str), "%Y%m%d")
    return dt.strftime("%Y-%m-%d")


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    exclude = {
        "日期", "股票代码", "代码", "名称", "股票名称", "简称",
        "板块", "行业", "市场", "上市时间", "数据源", "公告标题",
        "公告链接", "公告时间", "交易状态", "是否ST",
    }
    for col in out.columns:
        if col not in exclude:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            except Exception:
                pass
    return out


def normalize_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {
        "date": "日期",
        "Date": "日期",
        "code": "股票代码",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "preclose": "前收盘_原始",
        "volume": "成交量",
        "amount": "成交额",
        "turn": "换手率",
        "pctChg": "涨跌幅",
        "peTTM": "市盈率_TTM",
        "pbMRQ": "市净率_MRQ",
        "psTTM": "市销率_TTM",
        "pcfNcfTTM": "市现率_TTM",
        "tradestatus": "交易状态",
        "isST": "是否ST",
        "adjustflag": "复权标记",
    }
    out = out.rename(columns=rename_map)
    if "日期" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={"index": "日期"})
    if "日期" not in out.columns:
        return pd.DataFrame()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates("日期", keep="last")
    return out.reset_index(drop=True)


def normalize_fund_flow_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" in out.columns and "日期" not in out.columns:
        out = out.rename(columns={"date": "日期"})
    if "Date" in out.columns and "日期" not in out.columns:
        out = out.rename(columns={"Date": "日期"})
    if "日期" not in out.columns:
        return pd.DataFrame()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    return out


# =========================================================
# 指标计算
# =========================================================


def add_analysis_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).sort_values("日期")

    if "收盘" in out.columns:
        out["前收盘"] = out["收盘"].shift(1)
        out["日收益率"] = out["收盘"].pct_change()
        for n in [5, 10, 20, 60, 120, 250]:
            out[f"MA{n}"] = out["收盘"].rolling(n).mean()
        out["收盘_相对MA5"] = out["收盘"] / out["MA5"] - 1
        out["收盘_相对MA20"] = out["收盘"] / out["MA20"] - 1
        out["收盘_相对MA60"] = out["收盘"] / out["MA60"] - 1
        for n in [5, 10, 20, 60]:
            out[f"{n}日涨跌幅"] = out["收盘"] / out["收盘"].shift(n) - 1
        out["20日波动率"] = out["日收益率"].rolling(20).std()
        out["60日波动率"] = out["日收益率"].rolling(60).std()

    if "成交量" in out.columns:
        out["成交量_MA5"] = out["成交量"].rolling(5).mean()
        out["成交量_MA20"] = out["成交量"].rolling(20).mean()
        out["量比_相对20日"] = out["成交量"] / out["成交量_MA20"]

    if "成交额" in out.columns:
        out["成交额_MA5"] = out["成交额"].rolling(5).mean()
        out["成交额_MA20"] = out["成交额"].rolling(20).mean()

    if all(c in out.columns for c in ["最高", "最低", "收盘"]):
        out["日内振幅_计算"] = (out["最高"] - out["最低"]) / out["收盘"]

    if all(c in out.columns for c in ["收盘", "开盘"]):
        out["实体涨跌幅"] = out["收盘"] / out["开盘"] - 1

    return out.reset_index(drop=True)


def calc_continue_positive_days(series: pd.Series) -> pd.Series:
    result, count = [], 0
    for value in series:
        if pd.notna(value) and value > 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=series.index)


def add_fund_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().sort_values("日期")
    main_cols = [c for c in out.columns if "主力" in str(c) and "净流入" in str(c) and "净额" in str(c)]
    if not main_cols:
        main_cols = [c for c in out.columns if "主力" in str(c) and "净流入" in str(c)]
    if main_cols:
        main_col = main_cols[0]
        out[main_col] = pd.to_numeric(out[main_col], errors="coerce")
        out["主力净流入_MA3"] = out[main_col].rolling(3).mean()
        out["主力净流入_MA5"] = out[main_col].rolling(5).mean()
        out["主力连续流入天数"] = calc_continue_positive_days(out[main_col])
    return out


# =========================================================
# 历史行情：BaoStock 主源，AKShare 备用
# =========================================================


def get_history_baostock(stock_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    code = normalize_stock_code(stock_code)
    market = detect_market(code)

    # BaoStock公开示例主要覆盖沪深。北交所直接交给 AKShare 备用，避免错误前缀。
    if market == "bj":
        raise ValueError("北交所股票跳过 BaoStock，使用 AKShare 备用源")

    bs_code = f"{market}.{code}"
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login_result.error_code} {login_result.error_msg}")

    try:
        fields = (
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
            "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
        )
        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=yyyymmdd_to_iso(start_date),
            end_date=yyyymmdd_to_iso(end_date),
            frequency="d",
            adjustflag=baostock_adjust_flag(adjust),
        )
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock 查询失败: {rs.error_code} {rs.error_msg}")

        rows: List[List[str]] = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if df.empty:
        return df
    df = normalize_history_dataframe(df)
    df["股票代码"] = code
    df["数据源"] = "BaoStock"
    df = clean_numeric_columns(df)
    return df


def get_history_akshare(stock_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    code = normalize_stock_code(stock_code)
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=normalize_adjust(adjust),
        timeout=REQUEST_TIMEOUT,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = normalize_history_dataframe(df)
    df["股票代码"] = code
    df["数据源"] = "AKShare-东方财富"
    df = clean_numeric_columns(df)
    return df


def history_quality(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return -999.0
    required = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
    coverage = sum(c in df.columns for c in required) / len(required)
    null_rate = 1.0
    existing = [c for c in required if c in df.columns]
    if existing:
        null_rate = float(df[existing].isna().mean().mean())
    return coverage * 100 + min(len(df), 1000) * 0.02 - null_rate * 30


def get_stock_history(stock_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    """
    稳定优先：BaoStock -> 如失败/质量低则 AKShare。
    不把不同提供商的复权K线逐行混合，避免复权算法差异造成价格断层。
    """
    code = normalize_stock_code(stock_code)

    bs_result = safe_call(
        lambda: get_history_baostock(code, start_date, end_date, adjust),
        f"{code} 历史行情(BaoStock)",
        default=pd.DataFrame(),
        retry=2,
    )
    bs_df = bs_result.value if bs_result.success else pd.DataFrame()

    # BaoStock数据足够完整时直接使用，减少外部请求。
    if history_quality(bs_df) >= 90 and len(bs_df) >= 20:
        log(f"{code} 历史行情采用 BaoStock，{len(bs_df)} 条")
        return add_analysis_indicators(bs_df)

    ak_result = safe_call(
        lambda: get_history_akshare(code, start_date, end_date, adjust),
        f"{code} 历史行情(AKShare)",
        default=pd.DataFrame(),
        retry=2,
    )
    ak_df = ak_result.value if ak_result.success else pd.DataFrame()

    candidates = [("BaoStock", bs_df), ("AKShare", ak_df)]
    candidates = [(name, df) for name, df in candidates if df is not None and not df.empty]
    if not candidates:
        return pd.DataFrame()

    source, best = max(candidates, key=lambda x: history_quality(x[1]))
    log(f"{code} 历史行情最终采用 {source}，{len(best)} 条，质量分 {history_quality(best):.1f}")
    return add_analysis_indicators(best)


# =========================================================
# 资金流：东方财富历史资金流 + 分段探测 + 本地长期缓存
# =========================================================


_EASTMONEY_SESSION: Optional[requests.Session] = None


def get_eastmoney_session() -> requests.Session:
    """
    复用同一个东财 Session。
    只有这个 Session 使用脚本顶部 EASTMONEY_COOKIE；
    BaoStock、巨潮和其他模块不使用该 Cookie。
    """
    global _EASTMONEY_SESSION
    if _EASTMONEY_SESSION is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": EASTMONEY_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://data.eastmoney.com/zjlx/detail.html",
            "Origin": "https://data.eastmoney.com",
            "Connection": "keep-alive",
        })
        cookie = EASTMONEY_COOKIE.strip()
        if cookie.lower().startswith("cookie:"):
            cookie = cookie.split(":", 1)[1].strip()
        if cookie:
            session.headers["Cookie"] = cookie
        _EASTMONEY_SESSION = session
    return _EASTMONEY_SESSION


def validate_fund_flow_df(df: pd.DataFrame) -> bool:
    """不仅判断非空，还要求真正存在有效资金流数值。"""
    if df is None or df.empty:
        return False
    candidates = [
        "主力净流入-净额", "主力净流入", "净额",
        "超大单净流入-净额", "大单净流入-净额",
        "中单净流入-净额", "小单净流入-净额",
    ]
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                return True
    return False


def _fund_cache_path(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    return os.path.join(FUND_FLOW_CACHE_DIR, f"{code}_eastmoney_fund_flow.csv")


def load_fund_flow_cache(stock_code: str) -> pd.DataFrame:
    path = _fund_cache_path(stock_code)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = normalize_fund_flow_dataframe(df)
        if not df.empty:
            log(f"{normalize_stock_code(stock_code)} 读取本地资金流缓存: {len(df)} 条")
        return df
    except Exception as e:
        log(f"{normalize_stock_code(stock_code)} 读取资金流缓存失败: {repr(e)}")
        return pd.DataFrame()


def save_fund_flow_cache(stock_code: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    code = normalize_stock_code(stock_code)
    path = _fund_cache_path(code)
    out = df.copy()
    if "日期" in out.columns:
        out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期")
        out = out.drop_duplicates(subset=["日期"], keep="last")
        out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"{code} 已更新资金流本地缓存: {path}，{len(out)} 条")


def _parse_eastmoney_fund_flow_json(js: Dict[str, Any]) -> pd.DataFrame:
    data = (js or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame()

    rows = [item.split(",") for item in klines]
    cols = [
        "日期", "主力净流入-净额", "小单净流入-净额", "中单净流入-净额",
        "大单净流入-净额", "超大单净流入-净额", "主力净流入-净占比",
        "小单净流入-净占比", "中单净流入-净占比", "大单净流入-净占比",
        "超大单净流入-净占比", "收盘价", "涨跌幅", "_1", "_2",
    ]
    parsed = [row[:len(cols)] + [""] * max(0, len(cols) - len(row)) for row in rows]
    df = pd.DataFrame(parsed, columns=cols)
    df = df.drop(columns=["_1", "_2"], errors="ignore")
    return df


def get_fund_flow_em_direct(
    stock_code: str,
    end_date: Optional[str] = None,
    limit: int = EASTMONEY_FUND_FLOW_LIMIT,
) -> pd.DataFrame:
    """
    直接请求东财 push2his 历史资金流。

    end_date 会作为“向前分段探测”参数传给服务端。
    当前端点可能忽略该参数；上层会检测重复窗口并自动停止。
    """
    code = normalize_stock_code(stock_code)
    market = detect_market(code)
    market_map = {"sh": 1, "sz": 0, "bj": 0}

    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "lmt": str(limit),
        "klt": "101",
        "secid": f"{market_map[market]}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "ut": "b2884a393a59ad64002292a3e90d46a5",
        "_": int(time.time() * 1000),
    }
    if end_date:
        params["end"] = str(end_date).replace("-", "")

    session = get_eastmoney_session()
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return _parse_eastmoney_fund_flow_json(r.json())


def _eastmoney_segment_call(
    stock_code: str,
    end_date: Optional[str],
    attempts: int,
) -> CallResult:
    code = normalize_stock_code(stock_code)
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            label = end_date or "最新"
            log(f"{code} 东财资金流窗口 end={label}: 第 {attempt}/{attempts} 次尝试")
            df = get_fund_flow_em_direct(code, end_date=end_date)
            df = normalize_fund_flow_dataframe(df)
            if not validate_fund_flow_df(df):
                raise ValueError("返回数据为空或资金流字段无有效数值")
            log(
                f"{code} 东财资金流窗口 end={label}: "
                f"成功 {len(df)} 条，{df['日期'].min().date()} ~ {df['日期'].max().date()}"
            )
            return CallResult(
                value=df,
                success=True,
                source="东方财富-Cookie直连",
                retries=attempt,
            )
        except Exception as e:
            last_error = repr(e)
            log(f"{code} 东财资金流窗口 end={end_date or '最新'}: 失败: {last_error}")
            if attempt < attempts:
                idx = min(attempt - 1, len(EASTMONEY_RETRY_DELAYS) - 1)
                base = EASTMONEY_RETRY_DELAYS[idx]
                sec = base + random.uniform(0.8, 4.0)
                log(f"{code} 东财资金流: 长退避 {sec:.2f} 秒")
                time.sleep(sec)

    return CallResult(
        value=pd.DataFrame(),
        success=False,
        source="东方财富-Cookie直连",
        error=last_error,
        retries=attempts,
    )


def get_fund_flow_em_segmented(
    stock_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    东财历史资金流向前分段探测。

    流程：
    1. 第一次请求 requested end_date，lmt=100000；
    2. 若最早日期仍晚于 requested start_date，则把 cursor 移到最早日期前一天；
    3. 再次请求更早 end；
    4. 如果返回窗口与前一次完全相同，说明服务端忽略 end / 存在约120日硬上限，
       立即停止，不伪造完整历史；
    5. 多段结果按日期合并去重。
    """
    code = normalize_stock_code(stock_code)
    start_ts = pd.Timestamp(datetime.strptime(start_date, "%Y%m%d").date())
    requested_end_ts = pd.Timestamp(datetime.strptime(end_date, "%Y%m%d").date())

    cursor = requested_end_ts
    frames: List[pd.DataFrame] = []
    seen_signatures = set()

    for seg in range(1, EASTMONEY_MAX_BACKFILL_SEGMENTS + 1):
        cursor_str = cursor.strftime("%Y%m%d")
        result = _eastmoney_segment_call(
            code,
            end_date=cursor_str,
            attempts=4 if seg == 1 else 2,
        )
        if not result.success:
            break

        raw_df = result.value.copy()
        raw_df["日期"] = pd.to_datetime(raw_df["日期"], errors="coerce")
        raw_df = raw_df.dropna(subset=["日期"]).sort_values("日期")
        if raw_df.empty:
            break

        signature = (
            raw_df["日期"].min().strftime("%Y-%m-%d"),
            raw_df["日期"].max().strftime("%Y-%m-%d"),
            len(raw_df),
        )
        if signature in seen_signatures:
            log(
                f"{code} 东财分段探测发现重复窗口 {signature}；"
                "服务端很可能忽略 end 参数或存在固定历史窗口，停止继续向前请求"
            )
            break
        seen_signatures.add(signature)

        # 即使服务端返回了 cursor 之后的数据，也只保留本段截止日以内。
        part = raw_df[raw_df["日期"] <= cursor].copy()
        if part.empty:
            log(
                f"{code} 东财 end={cursor_str} 返回的数据全部晚于该日期；"
                "说明 end 参数未生效，停止分段"
            )
            break

        frames.append(part)
        earliest = part["日期"].min()

        if earliest <= start_ts:
            log(f"{code} 东财资金流已向前覆盖到请求开始日期 {start_date}")
            break

        next_cursor = earliest - timedelta(days=1)
        if next_cursor >= cursor:
            break

        log(
            f"{code} 东财资金流尚未覆盖请求起点；"
            f"下一段尝试 end={next_cursor.strftime('%Y%m%d')}"
        )
        cursor = next_cursor
        time.sleep(random.uniform(
            EASTMONEY_SEGMENT_SLEEP_MIN,
            EASTMONEY_SEGMENT_SLEEP_MAX,
        ))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = normalize_fund_flow_dataframe(out)
    out = out.drop_duplicates(subset=["日期"], keep="last").sort_values("日期")
    out = out[
        (out["日期"] >= start_ts) &
        (out["日期"] <= requested_end_ts)
    ].copy()
    return out.reset_index(drop=True)


def _merge_fund_flow_cache(
    stock_code: str,
    fresh_df: pd.DataFrame,
) -> pd.DataFrame:
    cached = load_fund_flow_cache(stock_code)
    frames = [df for df in [cached, fresh_df] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = normalize_fund_flow_dataframe(merged)
    if merged.empty:
        return merged

    merged = merged.drop_duplicates(subset=["日期"], keep="last").sort_values("日期")
    save_fund_flow_cache(stock_code, merged)
    return merged.reset_index(drop=True)


def get_stock_fund_flow(
    stock_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    历史资金流：
    1. 东方财富 Cookie + 固定 Session；
    2. lmt=100000，并尝试 end 日期向前分段；
    3. 分段无效时明确记录服务端窗口限制；
    4. 每次成功结果与本地 CSV 缓存合并，保证未来不会因窗口滚动丢掉旧数据；
    5. 只有直连完全失败且没有可用缓存时，才让 AKShare 再兜底一次。
    """
    code = normalize_stock_code(stock_code)
    market = detect_market(code)

    fresh = pd.DataFrame()

    if EASTMONEY_COOKIE.strip():
        fresh = get_fund_flow_em_segmented(code, start_date, end_date)
    else:
        log(f"{code} 未配置 EASTMONEY_COOKIE，跳过 Cookie 直连")

    merged_cache = _merge_fund_flow_cache(code, fresh)

    # 直连和缓存都没有数据时，再尝试 AKShare 一次。
    if merged_cache.empty:
        result = safe_call(
            lambda: ak.stock_individual_fund_flow(stock=code, market=market),
            f"{code} AKShare-东财历史资金流兜底",
            default=pd.DataFrame(),
            retry=FUND_FLOW_RETRY,
        )
        if result.success and result.value is not None and not result.value.empty:
            fallback = normalize_fund_flow_dataframe(result.value)
            if validate_fund_flow_df(fallback):
                fallback["股票代码"] = code
                fallback["数据源"] = "AKShare-东方财富"
                merged_cache = _merge_fund_flow_cache(code, fallback)

    if merged_cache.empty:
        log(f"{code} 历史资金流不可用")
        return pd.DataFrame()

    merged_cache["日期"] = pd.to_datetime(merged_cache["日期"], errors="coerce")
    req_start = pd.Timestamp(datetime.strptime(start_date, "%Y%m%d").date())
    req_end = pd.Timestamp(datetime.strptime(end_date, "%Y%m%d").date())

    out = merged_cache[
        (merged_cache["日期"] >= req_start) &
        (merged_cache["日期"] <= req_end)
    ].copy()

    if out.empty:
        log(
            f"{code} 本地资金流缓存存在，但请求区间 "
            f"{start_date}~{end_date} 内没有记录"
        )
        return pd.DataFrame()

    out["股票代码"] = code
    if "数据源" not in out.columns:
        out["数据源"] = "东方财富/本地缓存"
    else:
        out["数据源"] = out["数据源"].fillna("东方财富/本地缓存")

    out = clean_numeric_columns(out)
    out = add_fund_flow_indicators(out)

    earliest = pd.to_datetime(out["日期"]).min()
    latest = pd.to_datetime(out["日期"]).max()
    if earliest > req_start + timedelta(days=10):
        log(
            f"{code} 注意：东财资金流目前仅覆盖 "
            f"{earliest.date()} ~ {latest.date()}，"
            f"未覆盖请求起点 {req_start.date()}。"
            "如果分段日志显示重复窗口，则这是东财服务端历史窗口限制，"
            "不是程序漏页。"
        )

    log(f"{code} 历史资金流最终可用: {len(out)} 条")
    return out.reset_index(drop=True)


# =========================================================
# 巨潮资讯：规则 orgId 直查优先，映射表仅作兜底
# =========================================================

_CNINFO_ORGID_CACHE: Dict[str, str] = {}


def cninfo_headers() -> Dict[str, str]:
    """
    巨潮使用独立请求头，不使用东方财富 Cookie。

    当前公开可用实现仍普遍使用 HTTP 版公告接口，因此这里保持：
    - HTTP query endpoint
    - HTTP Origin
    - HTTP Referer
    三者协议一致，避免 HTTPS 入口直接返回 403。
    """
    return {
        "User-Agent": USER_AGENTS[0],
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "http://www.cninfo.com.cn",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
        "Host": "www.cninfo.com.cn",
    }


def get_cninfo_market_params(stock_code: str) -> Dict[str, str]:
    """
    巨潮公告查询的交易所参数。
    深市/创业板: column=szse, plate=sz
    沪市/科创板: column=sse,  plate=sh
    北交所: 为兼容公开前端接口，优先留空；orgId 仍按 bj 规则构造。
    """
    code = normalize_stock_code(stock_code)
    market = detect_market(code)

    if market == "sh":
        return {"column": "sse", "plate": "sh"}
    if market == "sz":
        return {"column": "szse", "plate": "sz"}
    return {"column": "", "plate": ""}


def guess_cninfo_org_id(stock_code: str) -> str:
    """
    根据股票代码构造巨潮常见 orgId。

    注意：该规则对大量股票有效，但不是所有股票都保证正确；
    如果查询结果为 0，程序会再尝试巨潮股票映射表进行校正。
    """
    code = normalize_stock_code(stock_code)
    market = detect_market(code)

    if market == "sh":
        return f"gssh0{code}"
    if market == "bj":
        return f"gsbj0{code}"
    return f"gssz0{code}"


def get_cninfo_org_id_from_mapping(
    stock_code: str,
    session: requests.Session,
) -> str:
    """
    仅作为兜底获取真实 orgId。

    之前 HTTPS:
        https://www.cninfo.com.cn/new/data/szse_stock.json
    在部分网络环境会直接 403。

    这里改为 HTTP 版本，并做内存缓存。
    如果仍失败，不影响程序继续使用规则 orgId。
    """
    code = normalize_stock_code(stock_code)

    if code in _CNINFO_ORGID_CACHE:
        return _CNINFO_ORGID_CACHE[code]

    url = "http://www.cninfo.com.cn/new/data/szse_stock.json"

    try:
        r = session.get(
            url,
            headers={
                "User-Agent": USER_AGENTS[0],
                "Accept": "application/json,text/plain,*/*",
                "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
            },
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        r.raise_for_status()
        js = r.json()

        mapping = {}
        for item in js.get("stockList", []):
            item_code = str(item.get("code", "")).zfill(6)
            org_id = str(item.get("orgId", "")).strip()
            if item_code and org_id:
                mapping[item_code] = org_id

        _CNINFO_ORGID_CACHE.update(mapping)

        org_id = mapping.get(code, "")
        if org_id:
            log(f"{code} 巨潮映射表获取真实 orgId: {org_id}")
        return org_id

    except Exception as e:
        log(f"{code} 巨潮 orgId 映射表兜底失败: {repr(e)}")
        return ""


def _cninfo_query_once(
    session: requests.Session,
    code: str,
    org_id: str,
    start_iso: str,
    end_iso: str,
    keyword: str,
    category_value: str,
    page_num: int,
) -> Dict[str, Any]:
    """
    请求巨潮公告单页。

    重点：
    1. 使用 HTTP 版公开前端 AJAX 接口；
    2. column / plate 根据股票市场明确填写；
    3. 巨潮模块完全不使用 EASTMONEY_COOKIE；
    4. 非 200 时记录响应摘要，便于判断 403/限流/网关错误。
    """
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    market_params = get_cninfo_market_params(code)

    payload = {
        "pageNum": str(page_num),
        "pageSize": str(CNINFO_PAGE_SIZE),
        "tabName": "fulltext",
        "column": market_params["column"],
        "stock": f"{code},{org_id}",
        "searchkey": keyword,
        "secid": "",
        "plate": market_params["plate"],
        "category": category_value,
        "trade": "",
        "seDate": f"{start_iso}~{end_iso}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }

    r = session.post(
        url,
        data=payload,
        headers=cninfo_headers(),
        timeout=REQUEST_TIMEOUT,
        allow_redirects=False,
    )

    if r.status_code != 200:
        preview = (r.text or "").replace(chr(10), " ").replace(chr(13), " ")[:300]
        location = r.headers.get("Location", "")
        raise requests.HTTPError(
            f"CNINFO HTTP {r.status_code}; "
            f"Location={location!r}; "
            f"Body={preview!r}",
            response=r,
        )

    try:
        js = r.json()
    except Exception as e:
        preview = (r.text or "")[:300]
        raise ValueError(f"巨潮返回非 JSON: {preview}") from e

    if not isinstance(js, dict) or js.get("totalAnnouncement") is None or "announcements" not in js:
        raise ValueError("巨潮返回缺少公告计数或列表字段")
    if int(js["totalAnnouncement"]) > 0 and not js["announcements"]:
        raise ValueError("巨潮报告有公告但未返回记录")
    return js


def get_cninfo_announcements_raw(
    stock_code: str,
    start_date: str,
    end_date: str,
    keyword: str = "",
    category: str = "",
) -> pd.DataFrame:
    """
    巨潮公告查询策略：

    1. 不再把 szse_stock.json 放在必经路径；
    2. 先按股票代码规则构造 orgId 直接查询；
    3. 若返回 0 条，再尝试 HTTP 版股票映射表获取真实 orgId；
    4. 若真实 orgId 与规则值不同，则重新查询；
    5. 分页低频抓取公告。

    巨潮模块完全独立，不使用 EASTMONEY_COOKIE。
    """
    code = normalize_stock_code(stock_code)

    category_value = ""
    if category:
        if category not in CNINFO_CATEGORY_MAP:
            raise ValueError(f"不支持的巨潮公告类别: {category}")
        category_value = CNINFO_CATEGORY_MAP[category]

    start_iso = yyyymmdd_to_iso(start_date)
    end_iso = yyyymmdd_to_iso(end_date)

    with requests.Session() as session:
        session.headers.update(cninfo_headers())

        guessed_org_id = guess_cninfo_org_id(code)
        log(f"{code} 巨潮先使用规则 orgId: {guessed_org_id}")

        first_json = _cninfo_query_once(
            session=session,
            code=code,
            org_id=guessed_org_id,
            start_iso=start_iso,
            end_iso=end_iso,
            keyword=keyword,
            category_value=category_value,
            page_num=1,
        )

        total = int(first_json.get("totalAnnouncement") or 0)
        active_org_id = guessed_org_id

        # 如果规则 orgId 查询不到，再尝试映射表。
        if total <= 0:
            mapped_org_id = get_cninfo_org_id_from_mapping(code, session)

            if mapped_org_id and mapped_org_id != guessed_org_id:
                log(
                    f"{code} 巨潮规则 orgId 返回 0 条，"
                    f"改用映射 orgId: {mapped_org_id}"
                )
                first_json = _cninfo_query_once(
                    session=session,
                    code=code,
                    org_id=mapped_org_id,
                    start_iso=start_iso,
                    end_iso=end_iso,
                    keyword=keyword,
                    category_value=category_value,
                    page_num=1,
                )
                total = int(first_json.get("totalAnnouncement") or 0)
                active_org_id = mapped_org_id

        if total <= 0:
            log(
                f"{code} 巨潮公告查询返回 0 条 "
                f"(orgId={active_org_id}, {start_iso}~{end_iso})"
            )
            return pd.DataFrame()

        log(
            f"{code} 巨潮公告查询成功: 共 {total} 条，"
            f"orgId={active_org_id}"
        )

        total_pages = math.ceil(total / CNINFO_PAGE_SIZE)
        frames = []

        # 第一页直接复用第一次请求结果，避免重复请求。
        first_announcements = first_json.get("announcements") or []
        if first_announcements:
            frames.append(pd.DataFrame(first_announcements))

        for page in range(2, total_pages + 1):
            # 巨潮分页不要并发，主动降低访问频率。
            time.sleep(random.uniform(0.6, 1.2))

            js = _cninfo_query_once(
                session=session,
                code=code,
                org_id=active_org_id,
                start_iso=start_iso,
                end_iso=end_iso,
                keyword=keyword,
                category_value=category_value,
                page_num=page,
            )

            announcements = js.get("announcements") or []
            if announcements:
                frames.append(pd.DataFrame(announcements))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _safe_filename(text_value: str, max_len: int = 100) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(text_value or "")).strip(" ._")
    name = re.sub(r"\s+", " ", name)
    return (name or "announcement")[:max_len]


def _cninfo_pdf_url(adjunct_url: str) -> str:
    adjunct = str(adjunct_url or "").strip()
    if not adjunct:
        return ""
    if adjunct.startswith("http://") or adjunct.startswith("https://"):
        return adjunct
    return CNINFO_PDF_BASE + adjunct.lstrip("/")


def normalize_cninfo_announcements(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.rename(
        columns={
            "secCode": "代码",
            "secName": "简称",
            "announcementTitle": "公告标题",
            "announcementTime": "公告时间",
            "announcementId": "公告ID",
            "orgId": "组织ID",
            "adjunctUrl": "附件相对路径",
        }
    )

    required = ["代码", "简称", "公告标题", "公告时间", "公告ID", "组织ID"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"巨潮返回字段变化，缺少: {missing}; "
            f"实际字段: {list(out.columns)}"
        )

    if "附件相对路径" not in out.columns:
        out["附件相对路径"] = ""

    out["公告标题"] = (
        out["公告标题"]
        .astype(str)
        .str.replace(r"<[^>]+>", "", regex=True)
    )

    out["公告时间"] = pd.to_datetime(
        out["公告时间"],
        unit="ms",
        utc=True,
        errors="coerce",
    )
    out["公告时间"] = (
        out["公告时间"]
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )

    out["公告详情链接"] = out.apply(
        lambda r: (
            "https://www.cninfo.com.cn/new/disclosure/detail"
            f"?stockCode={r['代码']}"
            f"&announcementId={r['公告ID']}"
            f"&orgId={r['组织ID']}"
            f"&announcementTime={r['公告时间']}"
        ),
        axis=1,
    )
    out["PDF下载链接"] = out["附件相对路径"].apply(_cninfo_pdf_url)

    out["股票代码"] = normalize_stock_code(stock_code)
    out["数据源"] = "巨潮资讯"

    return out[
        [
            "股票代码",
            "代码",
            "简称",
            "公告标题",
            "公告时间",
            "公告ID",
            "组织ID",
            "公告详情链接",
            "PDF下载链接",
            "数据源",
        ]
    ].sort_values(
        "公告时间",
        ascending=False,
    ).reset_index(drop=True)


def _download_cninfo_pdf(
    url: str,
    target_path: str,
    session: requests.Session,
) -> str:
    """下载公告 PDF；已存在则复用。返回 PDF 状态。"""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 100:
        return "CACHED"

    urls = [url]
    if url.startswith("http://"):
        urls.append("https://" + url[len("http://"):])

    last_error = ""
    for candidate in urls:
        try:
            with session.get(
                candidate,
                headers={
                    "User-Agent": USER_AGENTS[0],
                    "Accept": "application/pdf,*/*",
                    "Referer": "http://www.cninfo.com.cn/",
                },
                timeout=max(REQUEST_TIMEOUT, 45),
                stream=True,
            ) as r:
                r.raise_for_status()

                length = r.headers.get("Content-Length")
                if length:
                    mb = int(length) / 1024 / 1024
                    if mb > CNINFO_MAX_PDF_MB:
                        return f"SKIP_TOO_LARGE_{mb:.1f}MB"

                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                total = 0
                first_chunk = b""
                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        if not first_chunk:
                            first_chunk = chunk[:16]
                        total += len(chunk)
                        if total > CNINFO_MAX_PDF_MB * 1024 * 1024:
                            f.close()
                            try:
                                os.remove(target_path)
                            except Exception:
                                pass
                            return "SKIP_TOO_LARGE_STREAM"
                        f.write(chunk)

                if total <= 100 or not first_chunk.startswith(b"%PDF"):
                    try:
                        os.remove(target_path)
                    except Exception:
                        pass
                    last_error = "响应不是有效 PDF"
                    continue

                return "OK"

        except Exception as e:
            last_error = repr(e)

    return f"FAILED:{last_error[:160]}"


def _extract_pdf_text(pdf_path: str) -> tuple[str, str]:
    """
    从公告 PDF 提取文本。
    优先 PyMuPDF，失败后回退 pypdf。
    不自动 OCR：扫描件会标记 EMPTY_TEXT，避免高成本误识别。
    """
    errors = []

    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                pages.append(page.get_text("text") or "")
            doc.close()
            text_value = "\n".join(pages).strip()
            if text_value:
                return text_value, "OK_PYMUPDF"
            errors.append("PyMuPDF_EMPTY")
        except Exception as e:
            errors.append(f"PyMuPDF:{repr(e)[:120]}")

    if PdfReader is not None:
        try:
            reader = PdfReader(pdf_path)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            text_value = "\n".join(pages).strip()
            if text_value:
                return text_value, "OK_PYPDF"
            errors.append("pypdf_EMPTY")
        except Exception as e:
            errors.append(f"pypdf:{repr(e)[:120]}")

    if fitz is None and PdfReader is None:
        return "", "NO_TEXT_LIBRARY"

    return "", "EMPTY_TEXT|" + "|".join(errors)[:220]


def download_and_read_cninfo_announcements(
    announcement_df: pd.DataFrame,
    stock_code: str,
) -> pd.DataFrame:
    """
    下载巨潮公告 PDF，并把正文提取为同名 TXT。

    Excel 只保存正文预览与路径，完整正文保存到 TXT，
    避免超过 Excel 单元格 32767 字符限制。
    """
    if announcement_df is None or announcement_df.empty:
        return pd.DataFrame()

    code = normalize_stock_code(stock_code)
    out = announcement_df.copy()

    for col in [
        "PDF本地路径",
        "PDF下载状态",
        "正文文本路径",
        "正文提取状态",
        "正文字数",
        "正文预览",
    ]:
        if col not in out.columns:
            out[col] = ""

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": USER_AGENTS[0],
            "Accept-Language": "zh-CN,zh;q=0.9",
        })

        total = len(out)
        for pos, idx in enumerate(out.index, start=1):
            title = str(out.at[idx, "公告标题"] or "")
            pdf_url = str(out.at[idx, "PDF下载链接"] or "")
            ann_id = str(out.at[idx, "公告ID"] or "")
            ann_time = pd.to_datetime(out.at[idx, "公告时间"], errors="coerce")

            if not pdf_url:
                out.at[idx, "PDF下载状态"] = "NO_URL"
                out.at[idx, "正文提取状态"] = "SKIPPED_NO_PDF"
                continue

            year = str(ann_time.year) if pd.notna(ann_time) else "unknown_year"
            day = ann_time.strftime("%Y%m%d") if pd.notna(ann_time) else "unknown_date"
            safe_title = _safe_filename(title, 90)
            safe_id = _safe_filename(ann_id, 30)

            folder = os.path.join(ANNOUNCEMENT_DIR, code, year)
            filename = f"{day}_{safe_id}_{safe_title}.pdf"
            pdf_path = os.path.abspath(os.path.join(folder, filename))
            txt_path = os.path.splitext(pdf_path)[0] + ".txt"

            log(f"{code} 公告文件 {pos}/{total}: {title[:55]}")

            if CNINFO_DOWNLOAD_PDFS:
                pdf_status = _download_cninfo_pdf(pdf_url, pdf_path, session)
            else:
                pdf_status = "DOWNLOAD_DISABLED"

            out.at[idx, "PDF下载状态"] = pdf_status
            if pdf_status in {"OK", "CACHED"}:
                out.at[idx, "PDF本地路径"] = pdf_path
            else:
                out.at[idx, "正文提取状态"] = "SKIPPED_PDF_FAILED"
                time.sleep(random.uniform(CNINFO_PDF_SLEEP_MIN, CNINFO_PDF_SLEEP_MAX))
                continue

            if CNINFO_EXTRACT_TEXT:
                # 已经有 TXT 时直接复用，避免每次重新解析大 PDF。
                if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
                    try:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            body = f.read()
                        text_status = "CACHED_TEXT"
                    except Exception:
                        body, text_status = _extract_pdf_text(pdf_path)
                else:
                    body, text_status = _extract_pdf_text(pdf_path)

                if body:
                    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(body)
                    out.at[idx, "正文文本路径"] = txt_path
                    out.at[idx, "正文字数"] = len(body)
                    out.at[idx, "正文预览"] = body[:CNINFO_TEXT_PREVIEW_CHARS]
                else:
                    out.at[idx, "正文字数"] = 0
                    out.at[idx, "正文预览"] = ""

                out.at[idx, "正文提取状态"] = text_status
            else:
                out.at[idx, "正文提取状态"] = "TEXT_EXTRACTION_DISABLED"

            time.sleep(random.uniform(CNINFO_PDF_SLEEP_MIN, CNINFO_PDF_SLEEP_MAX))

    return out


def get_stock_announcements(
    stock_code: str,
    start_date: str,
    end_date: str,
    keyword: str = "",
    category: str = "",
) -> pd.DataFrame:
    code = normalize_stock_code(stock_code)

    result = safe_call(
        lambda: get_cninfo_announcements_raw(
            code,
            start_date,
            end_date,
            keyword,
            category,
        ),
        f"{code} 巨潮公告",
        default=pd.DataFrame(),
        retry=2,
        empty_is_error=False,
    )

    if not result.success:
        return pd.DataFrame()

    if result.value is None or result.value.empty:
        return pd.DataFrame()

    metadata_df = normalize_cninfo_announcements(result.value, code)

    if CNINFO_DOWNLOAD_PDFS:
        metadata_df = download_and_read_cninfo_announcements(metadata_df, code)

    return metadata_df


# =========================================================
# 汇总 / 合并 / Excel
# =========================================================


def merge_history_and_fund(hist_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()
    hist = hist_df.copy()
    hist["日期"] = pd.to_datetime(hist["日期"], errors="coerce")
    if fund_df is None or fund_df.empty or "日期" not in fund_df.columns:
        return hist

    fund = fund_df.copy()
    fund["日期"] = pd.to_datetime(fund["日期"], errors="coerce")
    keep = ["日期", "股票代码"]
    keep += [
        c for c in fund.columns
        if any(k in str(c) for k in ["主力", "超大单", "大单", "中单", "小单", "净流入"])
    ]
    keep = list(dict.fromkeys(c for c in keep if c in fund.columns))
    return pd.merge(hist, fund[keep], on=["日期", "股票代码"], how="left")


def build_summary(
    stock_code: str,
    hist_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    announcement_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    code = normalize_stock_code(stock_code)
    summary: Dict[str, Any] = {
        "股票代码": code,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "历史行情状态": "OK" if hist_df is not None and not hist_df.empty else "MISSING",
        "历史资金流状态": "OK" if fund_df is not None and not fund_df.empty else "UNAVAILABLE",
        "巨潮公告状态": "OK" if announcement_df is not None and not announcement_df.empty else "EMPTY_OR_UNAVAILABLE",
        "巨潮公告数量": 0 if announcement_df is None else len(announcement_df),
    }

    if hist_df is not None and not hist_df.empty:
        hist = hist_df.sort_values("日期")
        latest = hist.iloc[-1]
        summary["行情数据源"] = latest.get("数据源", "")
        summary["历史数据开始日期"] = hist["日期"].min()
        summary["历史数据结束日期"] = hist["日期"].max()
        summary["历史交易日数量"] = len(hist)
        for col in [
            "收盘", "涨跌幅", "换手率", "市盈率_TTM", "市净率_MRQ", "市销率_TTM",
            "MA20", "MA60", "20日涨跌幅", "60日涨跌幅", "20日波动率", "量比_相对20日",
        ]:
            if col in hist.columns:
                summary[f"最新_{col}"] = latest.get(col)

    if fund_df is not None and not fund_df.empty:
        fund = fund_df.sort_values("日期")
        latest = fund.iloc[-1]
        summary["历史资金流数据源"] = latest.get("数据源", "")
        summary["资金流开始日期"] = fund["日期"].min()
        summary["资金流结束日期"] = fund["日期"].max()
        summary["资金流交易日数量"] = len(fund)

        if hist_df is not None and not hist_df.empty:
            hist_dates = set(pd.to_datetime(hist_df["日期"], errors="coerce").dropna().dt.date)
            fund_dates = set(pd.to_datetime(fund["日期"], errors="coerce").dropna().dt.date)
            if hist_dates:
                coverage = len(hist_dates & fund_dates) / len(hist_dates)
                summary["资金流对行情交易日覆盖率"] = coverage
                summary["历史资金流状态"] = "OK" if coverage >= 0.95 else "PARTIAL"

        for col in fund.columns:
            if any(k in str(col) for k in ["主力净流入", "超大单净流入", "大单净流入"]):
                summary[f"最新资金_{col}"] = latest.get(col)

    if announcement_df is not None and not announcement_df.empty:
        summary["最新公告时间"] = announcement_df["公告时间"].max()
        summary["最新公告标题"] = announcement_df.sort_values("公告时间", ascending=False).iloc[0]["公告标题"]
        if "PDF下载状态" in announcement_df.columns:
            summary["公告PDF可用数量"] = int(
                announcement_df["PDF下载状态"].isin(["OK", "CACHED"]).sum()
            )
        if "正文提取状态" in announcement_df.columns:
            summary["公告正文已提取数量"] = int(
                announcement_df["正文提取状态"].astype(str).str.startswith(
                    ("OK_", "CACHED_TEXT")
                ).sum()
            )

    return pd.DataFrame([summary])


def safe_sheet_name(name: str) -> str:
    for ch in ['\\', '/', '*', '?', ':', '[', ']']:
        name = name.replace(ch, "_")
    return name[:31]


def auto_adjust_excel_width(filepath: str):
    try:
        from openpyxl import load_workbook

        wb = load_workbook(filepath)
        for ws in wb.worksheets:
            for col_cells in ws.columns:
                col_letter = col_cells[0].column_letter
                max_len = max((len(str(c.value)) for c in col_cells if c.value is not None), default=0)
                ws.column_dimensions[col_letter].width = min(max_len + 2, 40)
        wb.save(filepath)
    except Exception as e:
        log(f"Excel列宽调整失败: {repr(e)}")


def save_stock_excel(
    stock_code: str,
    summary_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    announcement_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> str:
    code = normalize_stock_code(stock_code)
    filename = f"{code}_股票研究数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=safe_sheet_name("摘要"), index=False)
        hist_df.to_excel(writer, sheet_name=safe_sheet_name("历史行情_BaoStock优先"), index=False)
        fund_df.to_excel(writer, sheet_name=safe_sheet_name("东财历史资金流"), index=False)
        announcement_df.to_excel(writer, sheet_name=safe_sheet_name("巨潮公告"), index=False)
        merged_df.to_excel(writer, sheet_name=safe_sheet_name("行情资金合并"), index=False)

    auto_adjust_excel_width(filepath)
    log(f"已保存: {filepath}")
    return filepath


# =========================================================
# 主流程
# =========================================================


def collect_one_stock(stock_code: str, start_date: str, end_date: str, adjust: str) -> Dict[str, Any]:
    code = normalize_stock_code(stock_code)
    log("=" * 80)
    log(f"开始采集 {code}: {start_date} ~ {end_date}, adjust={adjust}")

    hist_df = get_stock_history(code, start_date, end_date, adjust)
    log(f"{code} 历史行情: {len(hist_df)} 条")

    # 核心行情成功后再请求非核心源，避免失败时浪费请求。
    fund_df = get_stock_fund_flow(code, start_date, end_date)
    log(f"{code} 历史资金流: {len(fund_df)} 条")

    announcement_df = get_stock_announcements(code, start_date, end_date)
    log(f"{code} 巨潮公告: {len(announcement_df)} 条")

    merged_df = merge_history_and_fund(hist_df, fund_df)
    summary_df = build_summary(code, hist_df, fund_df, announcement_df)

    filepath = save_stock_excel(
        code,
        summary_df,
        hist_df,
        fund_df,
        announcement_df,
        merged_df,
    )

    # 成功标准：历史行情是核心；资金流和公告均为增强项。
    success = hist_df is not None and not hist_df.empty
    return {
        "stock_code": code,
        "filepath": filepath,
        "success": success,
        "hist_count": len(hist_df),
        "fund_count": len(fund_df),
        "announcement_count": len(announcement_df),
    }


def main():
    print("\n====== 股票公开数据采集工具：BaoStock + AKShare + 巨潮 ======\n")
    print("核心行情: BaoStock 优先 / AKShare 备用（BaoStock 不使用 Cookie）")
    print("历史资金流: 东财 Cookie + 分段探测 + 本地缓存（仅此模块使用 Cookie）")
    print("公告: 巨潮元数据 + PDF 下载 + 正文 TXT 提取")
    print("仅采集与保存数据，不生成图表或分析报告")
    print("多个股票用英文逗号分隔，例如: 000001,600519,300750\n")

    stock_codes_input = input("请输入股票代码: ").strip()
    start_date = input("请输入开始日期，例如 20230101: ").strip()
    end_date = input("请输入结束日期，例如 20260917: ").strip()
    adjust = input("复权方式 qfq/hfq/none，默认 qfq: ").strip() or "qfq"
    adjust = normalize_adjust(adjust)

    codes = [normalize_stock_code(x) for x in stock_codes_input.split(",") if x.strip()]
    codes = list(dict.fromkeys(codes))
    if not codes:
        log("没有有效股票代码")
        return

    success_files, failed_codes = [], []
    for idx, code in enumerate(codes, start=1):
        log(f"进度 {idx}/{len(codes)}: {code}")
        try:
            result = collect_one_stock(code, start_date, end_date, adjust)
            if result["success"]:
                success_files.append(result["filepath"])
            else:
                failed_codes.append(code)
        except Exception as e:
            log(f"{code} 未捕获异常: {repr(e)}")
            log(traceback.format_exc())
            failed_codes.append(code)

        if idx < len(codes):
            time.sleep(random.uniform(2, 5))

    log("====== 采集完成 ======")
    for path in success_files:
        log(f"成功文件: {path}")
    if failed_codes:
        log(f"历史行情失败股票: {failed_codes}")


if __name__ == "__main__":
    main()

