import os
import time
import random
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Any, List, Dict, Optional

import pandas as pd
import akshare as ak
import requests
from stock_analysis_report import run_analysis_from_data


# =========================================================
# 配置区
# =========================================================

OUTPUT_DIR = "stock_outputs"
LOG_DIR = "logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MAX_RETRY = 5
MIN_SLEEP = 2
MAX_SLEEP = 6
BACKOFF_BASE = 1.8
MAX_BACKOFF_SLEEP = 20
PARALLEL_SOURCE_WORKERS = 6
FUND_FLOW_RETRY = 5
FUND_FLOW_SLEEP_MIN = 3
FUND_FLOW_SLEEP_MAX = 9

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


# =========================================================
# 日志工具
# =========================================================

def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{now}] {msg}"
    print(text)

    log_file = os.path.join(LOG_DIR, "stock_collect.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")


@contextmanager
def temporary_requests_headers(extra_headers: Optional[Dict[str, str]] = None):
    """
    临时注入 requests 请求头（AKShare 底层多使用 requests）。
    """
    original_request = requests.sessions.Session.request
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://quote.eastmoney.com/",
    }
    if extra_headers:
        headers.update(extra_headers)

    def patched_request(self, method, url, **kwargs):
        req_headers = kwargs.get("headers") or {}
        merged_headers = headers.copy()
        merged_headers.update(req_headers)
        kwargs["headers"] = merged_headers
        kwargs.setdefault("timeout", 30)
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = patched_request
    try:
        yield
    finally:
        requests.sessions.Session.request = original_request


def random_sleep(min_s=MIN_SLEEP, max_s=MAX_SLEEP):
    sleep_time = random.uniform(min_s, max_s)
    log(f"等待 {sleep_time:.2f} 秒...")
    time.sleep(sleep_time)


def calc_retry_sleep(attempt: int, sleep_min: float, sleep_max: float) -> float:
    """
    递增等待策略：每次重试等待递增，且不超过 20 秒。
    """
    min_s = max(0.5, float(sleep_min))
    max_s = min(float(sleep_max), MAX_BACKOFF_SLEEP)
    if max_s < min_s:
        max_s = min_s

    # 线性递增，避免等待时间忽大忽小
    step = max((max_s - min_s) / 4.0, 0.5)
    sleep_time = min(min_s + (max(attempt, 1) - 1) * step, MAX_BACKOFF_SLEEP)
    return sleep_time


@dataclass
class CallResult:
    value: Any
    success: bool
    error: str = ""
    source: str = ""
    retries: int = 0


def safe_call(func: Callable, name: str, default=None, retry: int = MAX_RETRY, sleep_min=MIN_SLEEP, sleep_max=MAX_SLEEP):
    """
    通用安全请求：
    - 自动重试
    - 自动延迟
    - 失败后返回默认值
    """
    for i in range(1, retry + 1):
        try:
            log(f"{name}：第 {i}/{retry} 次尝试")
            result = func()

            if result is None:
                raise ValueError(f"{name} 返回 None")

            if isinstance(result, pd.DataFrame) and result.empty:
                log(f"{name} 返回空 DataFrame")
                return result

            log(f"{name} 成功")
            return result

        except Exception as e:
            log(f"{name} 失败：{repr(e)}")
            if i < retry:
                sleep_time = calc_retry_sleep(i, sleep_min, sleep_max)
                log(f"{name} 触发退避等待 {sleep_time:.2f} 秒...")
                time.sleep(sleep_time)
            else:
                log(f"{name} 最终失败，已跳过")
                return default


def safe_call_detail(
    func: Callable,
    name: str,
    default=None,
    retry: int = MAX_RETRY,
    sleep_min=MIN_SLEEP,
    sleep_max=MAX_SLEEP
) -> CallResult:
    last_error = ""
    for i in range(1, retry + 1):
        try:
            log(f"{name}：第 {i}/{retry} 次尝试")
            result = func()

            if result is None:
                raise ValueError(f"{name} 返回 None")

            if isinstance(result, pd.DataFrame) and result.empty:
                log(f"{name} 返回空 DataFrame")
                return CallResult(value=result, success=True, retries=i)

            log(f"{name} 成功")
            return CallResult(value=result, success=True, retries=i)

        except Exception as e:
            last_error = repr(e)
            log(f"{name} 失败：{last_error}")
            if i < retry:
                sleep_time = calc_retry_sleep(i, sleep_min, sleep_max)
                log(f"{name} 触发退避等待 {sleep_time:.2f} 秒...")
                time.sleep(sleep_time)
            else:
                log(f"{name} 最终失败，已跳过")

    return CallResult(value=default, success=False, error=last_error, retries=retry)


def safe_call_fallback(
    tasks: List[Dict[str, Any]],
    default=None,
    retry: int = MAX_RETRY,
    sleep_min=MIN_SLEEP,
    sleep_max=MAX_SLEEP
) -> CallResult:
    """
    顺序尝试多个数据源，前一个失败后自动尝试下一个
    tasks: [{"name": "...", "func": callable}, ...]
    """
    last_error = ""
    for idx, task in enumerate(tasks, start=1):
        source_name = task["name"]
        func = task["func"]
        log(f"切换数据源 {idx}/{len(tasks)}：{source_name}")

        result = safe_call_detail(
            func=func,
            name=source_name,
            default=default,
            retry=retry,
            sleep_min=sleep_min,
            sleep_max=sleep_max
        )
        result.source = source_name

        if result.success and not (isinstance(result.value, pd.DataFrame) and result.value.empty):
            return result

        last_error = result.error

    return CallResult(value=default, success=False, error=last_error, retries=retry)


def score_dataframe_quality(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> float:
    """
    评分越高表示数据质量越好，用于并发多源选优。
    """
    if df is None or df.empty:
        return -1.0

    required_cols = required_cols or []
    row_score = min(len(df), 5000) * 0.01

    if required_cols:
        ok_cols = sum(1 for c in required_cols if c in df.columns)
        col_score = (ok_cols / len(required_cols)) * 30
    else:
        col_score = 0.0

    null_rate = float(df.isna().mean().mean()) if not df.empty else 1.0
    null_penalty = null_rate * 10

    return row_score + col_score - null_penalty


def safe_call_parallel_sources(
    tasks: List[Dict[str, Any]],
    default=None,
    retry: int = MAX_RETRY,
    sleep_min=MIN_SLEEP,
    sleep_max=MAX_SLEEP,
    required_cols: Optional[List[str]] = None
) -> CallResult:
    """
    并发请求多数据源，按质量评分选最优结果。
    """
    if not tasks:
        return CallResult(value=default, success=False, error="no tasks")

    best_result = CallResult(value=default, success=False, error="all sources failed")
    best_score = -10**9
    worker_count = min(PARALLEL_SOURCE_WORKERS, len(tasks))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for task in tasks:
            source_name = task["name"]
            func = task["func"]
            future = executor.submit(
                safe_call_detail,
                func,
                source_name,
                default,
                retry,
                sleep_min,
                sleep_max
            )
            futures[future] = source_name

        for future in as_completed(futures):
            source_name = futures[future]
            try:
                result: CallResult = future.result()
                result.source = source_name

                if not result.success:
                    if result.error:
                        log(f"{source_name} 并发源失败：{result.error}")
                    continue

                if isinstance(result.value, pd.DataFrame):
                    score = score_dataframe_quality(result.value, required_cols=required_cols)
                    log(f"{source_name} 并发源评分：{score:.2f}，行数：{len(result.value)}")
                else:
                    score = 1.0

                if score > best_score:
                    best_score = score
                    best_result = result
            except Exception as e:
                log(f"{source_name} 并发执行异常：{repr(e)}")

    if best_result.success and isinstance(best_result.value, pd.DataFrame):
        if best_result.value.empty:
            return CallResult(value=default, success=False, error="all sources empty")
        log(f"并发选优结果：{best_result.source}，评分：{best_score:.2f}")
        return best_result

    return CallResult(value=default, success=False, error=best_result.error)


# =========================================================
# 基础工具
# =========================================================

def normalize_stock_code(code: str) -> str:
    """
    支持：
    000001
    sz000001
    sh600519
    600519
    """
    code = str(code).strip().lower()
    code = code.replace("sh", "").replace("sz", "").replace("bj", "")
    code = "".join([c for c in code if c.isdigit()])
    return code.zfill(6)


def normalize_adjust(adjust: str) -> str:
    """
    AKShare:
    qfq = 前复权
    hfq = 后复权
    ""  = 不复权
    """
    adjust = str(adjust).strip().lower()
    if adjust in ["qfq", "前复权"]:
        return "qfq"
    if adjust in ["hfq", "后复权"]:
        return "hfq"
    if adjust in ["none", "不复权", ""]:
        return ""
    return "qfq"


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    安全转换数字列，避免 FutureWarning
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    exclude_cols = {
        "日期", "股票代码", "代码", "名称", "股票名称",
        "板块", "行业", "市场", "上市时间"
    }

    for col in df.columns:
        if col not in exclude_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                pass

    return df


def normalize_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一历史行情字段，避免不同数据源列名不一致导致后续报错。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    rename_map = {
        "date": "日期",
        "Date": "日期",
        "open": "开盘",
        "Open": "开盘",
        "high": "最高",
        "High": "最高",
        "low": "最低",
        "Low": "最低",
        "close": "收盘",
        "Close": "收盘",
        "volume": "成交量",
        "Volume": "成交量",
        "amount": "成交额",
        "Amount": "成交额",
    }
    df = df.rename(columns=rename_map)

    if "日期" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "日期", "date": "日期"})

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"]).sort_values("日期")
    else:
        log("历史行情字段标准化失败：缺少 日期 列")
        return pd.DataFrame()

    return df


def normalize_fund_flow_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一资金流字段，至少保证存在日期列。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    rename_map = {
        "date": "日期",
        "Date": "日期",
    }
    df = df.rename(columns=rename_map)

    if "日期" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "日期", "date": "日期"})

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"]).sort_values("日期")
    else:
        log("资金流字段标准化失败：缺少 日期 列")
        return pd.DataFrame()

    return df


def safe_sheet_name(name: str) -> str:
    """
    Excel sheet 名不能超过31字符，且不能包含特殊字符
    """
    for ch in ['\\', '/', '*', '?', ':', '[', ']']:
        name = name.replace(ch, "_")
    return name[:31]


# =========================================================
# 分析指标
# =========================================================

def add_analysis_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"])
        df = df.sort_values("日期")

    if "收盘" in df.columns:
        df["前收盘"] = df["收盘"].shift(1)
        df["日收益率"] = df["收盘"].pct_change()

        df["MA5"] = df["收盘"].rolling(5).mean()
        df["MA10"] = df["收盘"].rolling(10).mean()
        df["MA20"] = df["收盘"].rolling(20).mean()
        df["MA60"] = df["收盘"].rolling(60).mean()
        df["MA120"] = df["收盘"].rolling(120).mean()
        df["MA250"] = df["收盘"].rolling(250).mean()

        df["收盘_相对MA5"] = df["收盘"] / df["MA5"] - 1
        df["收盘_相对MA20"] = df["收盘"] / df["MA20"] - 1
        df["收盘_相对MA60"] = df["收盘"] / df["MA60"] - 1

        df["5日涨跌幅"] = df["收盘"] / df["收盘"].shift(5) - 1
        df["10日涨跌幅"] = df["收盘"] / df["收盘"].shift(10) - 1
        df["20日涨跌幅"] = df["收盘"] / df["收盘"].shift(20) - 1
        df["60日涨跌幅"] = df["收盘"] / df["收盘"].shift(60) - 1

        df["20日波动率"] = df["日收益率"].rolling(20).std()
        df["60日波动率"] = df["日收益率"].rolling(60).std()

    if "成交量" in df.columns:
        df["成交量_MA5"] = df["成交量"].rolling(5).mean()
        df["成交量_MA20"] = df["成交量"].rolling(20).mean()
        df["量比_相对20日"] = df["成交量"] / df["成交量_MA20"]

    if "成交额" in df.columns:
        df["成交额_MA5"] = df["成交额"].rolling(5).mean()
        df["成交额_MA20"] = df["成交额"].rolling(20).mean()

    if all(col in df.columns for col in ["最高", "最低", "收盘"]):
        df["日内振幅_计算"] = (df["最高"] - df["最低"]) / df["收盘"]

    if all(col in df.columns for col in ["收盘", "开盘"]):
        df["实体涨跌幅"] = df["收盘"] / df["开盘"] - 1

    return df


def add_fund_flow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.sort_values("日期")

    # 兼容不同字段名
    main_cols = [c for c in df.columns if "主力" in c and "净流入" in c]
    amount_cols = [c for c in df.columns if "成交额" in c]

    if main_cols:
        main_col = main_cols[0]

        df["主力净流入_MA3"] = df[main_col].rolling(3).mean()
        df["主力净流入_MA5"] = df[main_col].rolling(5).mean()
        df["主力连续流入天数"] = calc_continue_positive_days(df[main_col])

        if amount_cols:
            amount_col = amount_cols[0]
            df["主力净流入强度"] = df[main_col] / df[amount_col]

    return df


def calc_continue_positive_days(series: pd.Series) -> pd.Series:
    result = []
    count = 0

    for value in series:
        if pd.notna(value) and value > 0:
            count += 1
        else:
            count = 0
        result.append(count)

    return pd.Series(result, index=series.index)


# =========================================================
# 数据采集
# =========================================================

def get_stock_history(stock_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    stock_code = normalize_stock_code(stock_code)
    adjust = normalize_adjust(adjust)

    tasks = []

    def hist_em():
        return ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            timeout=30
        )

    tasks.append({"name": f"{stock_code} 历史行情(EM)", "func": hist_em})

    if hasattr(ak, "stock_zh_a_hist_163"):
        def hist_163():
            return ak.stock_zh_a_hist_163(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date
            )
        tasks.append({"name": f"{stock_code} 历史行情(163)", "func": hist_163})

    if hasattr(ak, "stock_zh_a_daily"):
        def hist_sina_daily():
            df = ak.stock_zh_a_daily(
                symbol=f"sz{stock_code}" if stock_code.startswith(("0", "3")) else f"sh{stock_code}",
                adjust=adjust if adjust else ""
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"date": "日期"})
            return df
        tasks.append({"name": f"{stock_code} 历史行情(SINA_DAILY)", "func": hist_sina_daily})

    # 统一限制为 3 路并发；若候选不足 3 路，补一个 EM 备用通道提升成功率
    selected_tasks = tasks[:3]
    if len(selected_tasks) < 3:
        def hist_em_backup():
            with temporary_requests_headers({"Referer": "https://quote.eastmoney.com/"}):
                return ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    timeout=30
                )
        selected_tasks.append({"name": f"{stock_code} 历史行情(EM_BACKUP)", "func": hist_em_backup})

    call_result = safe_call_parallel_sources(
        tasks=selected_tasks,
        default=pd.DataFrame(),
        retry=MAX_RETRY,
        sleep_min=3,
        sleep_max=8,
        required_cols=["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]
    )
    df = call_result.value

    if df is None or df.empty:
        if call_result.error:
            log(f"{stock_code} 历史行情最终错误：{call_result.error}")
        return pd.DataFrame()

    df = normalize_history_dataframe(df)
    if df.empty:
        log(f"{stock_code} 历史行情标准化后为空")
        return pd.DataFrame()

    df["股票代码"] = stock_code
    df = clean_numeric_columns(df)
    df = add_analysis_indicators(df)

    return df


def get_stock_fund_flow(stock_code: str) -> pd.DataFrame:
    stock_code = normalize_stock_code(stock_code)
    market = "sz" if stock_code.startswith(("0", "3")) else "sh"

    tasks = []

    def fund_em():
        with temporary_requests_headers({"Referer": "https://data.eastmoney.com/zjlx/detail.html"}):
            return ak.stock_individual_fund_flow(stock=stock_code, market=market)
    tasks.append({"name": f"{stock_code} 个股资金流(EM)", "func": fund_em})

    # 若存在新浪接口，作为回退源
    if hasattr(ak, "stock_individual_fund_flow_sina"):
        def fund_sina():
            with temporary_requests_headers({"Referer": "https://finance.sina.com.cn/"}):
                return ak.stock_individual_fund_flow_sina(stock=stock_code)
        tasks.append({"name": f"{stock_code} 个股资金流(SINA)", "func": fund_sina})

    # 额外接口：全市场个股资金流，按股票代码过滤（部分环境下稳定性更好）
    if hasattr(ak, "stock_fund_flow_individual"):
        def fund_individual_table():
            with temporary_requests_headers({"Referer": "https://data.eastmoney.com/zjlx/list.html"}):
                df = ak.stock_fund_flow_individual(symbol="即时")
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            code_col = None
            for c in ["股票代码", "代码", "symbol"]:
                if c in df.columns:
                    code_col = c
                    break
            if code_col is None:
                return pd.DataFrame()
            df[code_col] = df[code_col].astype(str).str.zfill(6)
            out = df[df[code_col] == stock_code].copy()
            if out.empty:
                return out
            if "日期" not in out.columns:
                out["日期"] = datetime.now().strftime("%Y-%m-%d")
            return out
        tasks.append({"name": f"{stock_code} 个股资金流(INDIV_TABLE)", "func": fund_individual_table})

    # 额外接口：主力资金流排行榜，按股票代码过滤
    if hasattr(ak, "stock_individual_fund_flow_rank"):
        def fund_rank_today():
            with temporary_requests_headers({"Referer": "https://data.eastmoney.com/zjlx/list.html"}):
                df = ak.stock_individual_fund_flow_rank(indicator="今日")
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            code_col = None
            for c in ["股票代码", "代码", "symbol"]:
                if c in df.columns:
                    code_col = c
                    break
            if code_col is None:
                return pd.DataFrame()
            df[code_col] = df[code_col].astype(str).str.zfill(6)
            out = df[df[code_col] == stock_code].copy()
            if out.empty:
                return out
            if "日期" not in out.columns:
                out["日期"] = datetime.now().strftime("%Y-%m-%d")
            return out
        tasks.append({"name": f"{stock_code} 个股资金流(RANK_TODAY)", "func": fund_rank_today})

    call_result = safe_call_parallel_sources(
        tasks=tasks,
        default=pd.DataFrame(),
        retry=FUND_FLOW_RETRY,
        sleep_min=FUND_FLOW_SLEEP_MIN,
        sleep_max=FUND_FLOW_SLEEP_MAX,
        required_cols=["日期"]
    )
    df = call_result.value

    # 并发失败后，顺序补抓兜底
    if df is None or df.empty:
        log(f"{stock_code} 个股资金流并发失败，启动顺序补抓兜底")
        fallback_result = safe_call_fallback(
            tasks=tasks,
            default=pd.DataFrame(),
            retry=max(FUND_FLOW_RETRY - 1, 3),
            sleep_min=FUND_FLOW_SLEEP_MIN,
            sleep_max=FUND_FLOW_SLEEP_MAX
        )
        df = fallback_result.value
        if (df is None or df.empty) and call_result.error:
            log(f"{stock_code} 个股资金流最终错误：{call_result.error}")
            return pd.DataFrame()

    df = normalize_fund_flow_dataframe(df)
    if df.empty:
        log(f"{stock_code} 个股资金流标准化后为空")
        return pd.DataFrame()

    df["股票代码"] = stock_code
    df = clean_numeric_columns(df)
    df = add_fund_flow_indicators(df)

    return df


# =========================================================
# 汇总分析
# =========================================================

def build_summary(stock_code: str, hist_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    stock_code = normalize_stock_code(stock_code)

    summary = {
        "股票代码": stock_code,
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "历史行情是否成功": not hist_df.empty,
        "资金流是否成功": not fund_df.empty,
    }

    if not hist_df.empty:
        if "日期" not in hist_df.columns:
            log(f"{stock_code} 汇总跳过历史日期统计：缺少 日期 列")
        else:
            hist_df = hist_df.sort_values("日期")
            latest = hist_df.iloc[-1]

            summary["历史数据开始日期"] = hist_df["日期"].min()
            summary["历史数据结束日期"] = hist_df["日期"].max()
            summary["历史交易日数量"] = len(hist_df)

            for col in [
                "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅",
                "换手率", "MA5", "MA10", "MA20", "MA60", "MA120", "MA250",
                "5日涨跌幅", "10日涨跌幅", "20日涨跌幅", "60日涨跌幅",
                "20日波动率", "60日波动率", "量比_相对20日", "收盘_相对MA20"
            ]:
                if col in hist_df.columns:
                    summary[f"最新_{col}"] = latest.get(col)

            if "收盘" in hist_df.columns and len(hist_df) >= 2:
                first_close = hist_df["收盘"].iloc[0]
                last_close = hist_df["收盘"].iloc[-1]
                if pd.notna(first_close) and first_close != 0:
                    summary["区间收盘涨跌幅"] = last_close / first_close - 1

                summary["区间最高收盘价"] = hist_df["收盘"].max()
                summary["区间最低收盘价"] = hist_df["收盘"].min()

            if "成交额" in hist_df.columns:
                summary["区间日均成交额"] = hist_df["成交额"].mean()

    if not fund_df.empty:
        fund_df = fund_df.copy()

        if "日期" in fund_df.columns:
            fund_df["日期"] = pd.to_datetime(fund_df["日期"], errors="coerce")
            fund_df = fund_df.sort_values("日期")

        latest_fund = fund_df.iloc[-1]

        for col in fund_df.columns:
            if any(key in col for key in ["主力", "超大单", "大单", "中单", "小单", "净流入", "流入强度"]):
                summary[f"最新资金_{col}"] = latest_fund.get(col)

    return pd.DataFrame([summary])


def merge_history_and_fund(hist_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并行情 + 资金流，便于直接分析
    """
    if hist_df.empty:
        return pd.DataFrame()

    hist = hist_df.copy()
    if "日期" not in hist.columns:
        log("合并失败：历史行情缺少 日期 列，返回历史原表")
        return hist
    hist["日期"] = pd.to_datetime(hist["日期"], errors="coerce")

    if fund_df.empty or "日期" not in fund_df.columns:
        return hist

    fund = fund_df.copy()
    fund["日期"] = pd.to_datetime(fund["日期"], errors="coerce")

    keep_cols = ["日期", "股票代码"]
    for col in fund.columns:
        if any(key in col for key in ["主力", "超大单", "大单", "中单", "小单", "净流入", "流入强度"]):
            keep_cols.append(col)

    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in fund.columns]))

    merged = pd.merge(
        hist,
        fund[keep_cols],
        on=["日期", "股票代码"],
        how="left"
    )

    return merged


# =========================================================
# Excel 输出
# =========================================================

def auto_adjust_excel_width(filepath: str):
    try:
        from openpyxl import load_workbook

        wb = load_workbook(filepath)

        for ws in wb.worksheets:
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter

                for cell in col_cells:
                    value = cell.value
                    if value is not None:
                        max_len = max(max_len, len(str(value)))

                ws.column_dimensions[col_letter].width = min(max_len + 2, 28)

        wb.save(filepath)

    except Exception as e:
        log(f"Excel列宽调整失败：{repr(e)}")


def save_stock_excel(
    stock_code: str,
    summary_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    merged_df: pd.DataFrame
) -> str:
    stock_code = normalize_stock_code(stock_code)

    filename = f"{stock_code}_股票分析数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=safe_sheet_name("摘要"), index=False)
        hist_df.to_excel(writer, sheet_name=safe_sheet_name("历史行情_含分析指标"), index=False)
        fund_df.to_excel(writer, sheet_name=safe_sheet_name("资金流_含分析指标"), index=False)
        merged_df.to_excel(writer, sheet_name=safe_sheet_name("行情资金合并表"), index=False)

    auto_adjust_excel_width(filepath)

    log(f"已保存：{filepath}")
    return filepath


# =========================================================
# 主采集流程
# =========================================================

def collect_one_stock(
    stock_code: str,
    start_date: str,
    end_date: str,
    adjust: str
) -> Dict[str, Any]:
    stock_code = normalize_stock_code(stock_code)

    log("=" * 80)
    log(f"开始采集：{stock_code}")
    log(f"日期范围：{start_date} ~ {end_date}")
    log(f"复权方式：{adjust}")

    hist_df = get_stock_history(stock_code, start_date, end_date, adjust)
    log(f"{stock_code} 历史行情数量：{len(hist_df)}")

    random_sleep(2, 5)

    fund_df = get_stock_fund_flow(stock_code)
    log(f"{stock_code} 资金流数量：{len(fund_df)}")

    merged_df = merge_history_and_fund(hist_df, fund_df)

    summary_df = build_summary(stock_code, hist_df, fund_df)

    filepath = save_stock_excel(
        stock_code=stock_code,
        summary_df=summary_df,
        hist_df=hist_df,
        fund_df=fund_df,
        merged_df=merged_df
    )

    analysis_dir = os.path.join(OUTPUT_DIR, f"analysis_{stock_code}")
    analysis_report_path = ""
    try:
        analysis_result = run_analysis_from_data(
            stock_code=stock_code,
            hist_df=hist_df,
            fund_df=fund_df,
            output_dir=analysis_dir,
        )
        analysis_report_path = analysis_result.get("report_path", "")
        log(f"{stock_code} 分析报告已生成：{analysis_report_path}")
    except Exception as e:
        log(f"{stock_code} 分析报告生成失败：{repr(e)}")

    # 你要求历史行情 + 资金流为必选，两个都非空才算成功
    data_ready = (not hist_df.empty) and (not fund_df.empty)

    return {
        "stock_code": stock_code,
        "filepath": filepath,
        "summary_df": summary_df,
        "success": data_ready,
        "hist_count": len(hist_df),
        "fund_count": len(fund_df),
        "analysis_report_path": analysis_report_path,
    }


def main():
    print("\n====== AKShare 股票数据采集工具：优化版 ======\n")
    print("示例股票代码：000001 / 600519 / 300750 / 601601")
    print("多个股票用英文逗号分隔，例如：000001,600519,601601")
    print("日期格式：20230101\n")

    stock_codes_input = input("请输入股票代码，多个用英文逗号分隔：").strip()
    start_date = input("请输入开始日期，例如 20230101：").strip()
    end_date = input("请输入结束日期，例如 20260430：").strip()

    adjust = input("复权方式：qfq=前复权，hfq=后复权，none/空=不复权，默认 qfq：").strip()
    if not adjust:
        adjust = "qfq"
    adjust = normalize_adjust(adjust)

    codes = [normalize_stock_code(x) for x in stock_codes_input.split(",") if x.strip()]
    codes = list(dict.fromkeys(codes))

    if not codes:
        log("没有输入有效股票代码，程序结束")
        return

    log(f"本次准备采集股票：{codes}")

    log("已移除动态代理与全市场基础信息请求，仅采集历史行情与资金流")

    success_files = []
    failed_codes = []
    all_summary = []

    for idx, code in enumerate(codes, start=1):
        log(f"\n进度：{idx}/{len(codes)}，当前股票：{code}")

        try:
            result = collect_one_stock(
                stock_code=code,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

            if result["success"]:
                success_files.append(result["filepath"])
                all_summary.append(result["summary_df"])
            else:
                failed_codes.append(code)
                log(f"{code} 三类数据均为空，标记为失败")

        except Exception as e:
            log(f"{code} 采集出现未捕获异常：{repr(e)}")
            log(traceback.format_exc())
            failed_codes.append(code)

        # 股票之间增加间隔，降低被断开的概率
        if idx < len(codes):
            random_sleep(5, 12)

    log("\n====== 采集完成 ======")

    if success_files:
        log("成功生成文件：")
        for file in success_files:
            log(file)

    if failed_codes:
        log(f"失败股票：{failed_codes}")
    else:
        log("没有完全失败的股票")


if __name__ == "__main__":
    main()