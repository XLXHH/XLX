"""Run locally: universe -> collect -> build -> train -> report. See TRAINING.md."""
from __future__ import annotations

import argparse
import ast
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research import config, context_features, json_write, make_features, train_panel


def read_cookie(c):
    value = os.environ.get("EASTMONEY_COOKIE", "").strip()
    if value:
        return value
    source = Path(c.get("v5_path", ""))
    if source.is_file():
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EASTMONEY_COOKIE" for t in node.targets):
                try:
                    return str(ast.literal_eval(node.value)).strip()
                except (ValueError, TypeError):
                    pass
    return ""


def collector(c):
    import collector_v5 as v5
    v5.EASTMONEY_COOKIE = read_cookie(c)
    v5._EASTMONEY_SESSION = None
    v5.CNINFO_DOWNLOAD_PDFS = False
    v5.CNINFO_EXTRACT_TEXT = False
    return v5


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def bs_rows(result):
    if result.error_code != "0":
        raise RuntimeError("BaoStock查询失败，错误码=" + result.error_code)
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def login():
    import baostock as bs
    status = bs.login()
    if status.error_code != "0":
        raise RuntimeError("BaoStock登录失败")
    return bs


def history(bs, code, start, end):
    raw = bs_rows(bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,amount", start_date=str(start), end_date=str(end), frequency="d", adjustflag="2"))
    if raw.empty:
        raise ValueError("行情为空")
    raw.date = pd.to_datetime(raw.date)
    for col in raw.columns.difference(["date"]):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.sort_values("date").drop_duplicates("date")
    if raw.close.isna().any() or raw.close.le(0).any():
        raise ValueError("行情含无效收盘价")
    if ((raw.high < raw[["open", "close", "low"]].max(axis=1)) | (raw.low > raw[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC关系异常")
    return raw


def fund_fetch(v5, code, old=None):
    # Bounded retries, then the batch circuit breaker takes over.
    for attempt in range(3):
        try:
            raw = v5.get_fund_flow_em_direct(code, limit=100000)
            if raw is None or raw.empty:
                raise ValueError("东财资金流为空")
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    if raw is None or raw.empty:
        raise ValueError("东财资金流为空；没有用零值填补")
    f = raw.rename(columns={"日期": "date", "主力净流入-净额": "main_net"})[["date", "main_net"]].copy()
    f.date = pd.to_datetime(f.date, errors="coerce").dt.normalize()
    f.main_net = pd.to_numeric(f.main_net, errors="coerce")
    f = f.dropna()
    if f.empty:
        raise ValueError("资金流字段无效")
    if old is not None:
        f = pd.concat([old, f], ignore_index=True)
    return f.sort_values("date").drop_duplicates("date", keep="last")


def universe(c, asof, limit):
    root = Path(c["root"])
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "universe.csv"
    if frozen.exists():
        raise ValueError("股票池已冻结。更新数据用collect；建立新股票池请在配置中换root，保留旧测试集。")
    bs = login()
    try:
        basics = bs_rows(bs.query_stock_basic())
        industry = bs_rows(bs.query_stock_industry())[["code", "industry"]].drop_duplicates("code")
        candidates = basics[(basics.type == "1") & (basics.status == "1") & basics.code.str.match(r"(?:sh\.6|sz\.[03])") & ~basics.code_name.str.contains("ST|退", case=False, regex=True)].copy()
        candidates = candidates[(pd.Timestamp(asof)-pd.to_datetime(candidates.ipoDate, errors="coerce")).dt.days >= c["minimum_listing_days"]]
        candidates = candidates.merge(industry, on="code", how="left").fillna({"industry": "未知"}).sort_values("code")
        if limit:
            candidates = candidates.head(limit)
        save_frame(root / "universe_candidates.parquet", candidates)
        rows, failures, consecutive = [], [], 0
        for i, (_, row) in enumerate(candidates.iterrows(), 1):
            dest = root / "screen" / f"{row.code}_{asof}.parquet"
            try:
                p = pd.read_parquet(dest) if dest.exists() else history(bs, row.code, (pd.Timestamp(asof)-pd.Timedelta(days=150)).date(), asof)
                save_frame(dest, p)
                recent = p.tail(60)
                liquidity = float(recent.amount.tail(20).median())
                if len(recent) >= 60 and recent.volume.gt(0).mean() >= .95 and liquidity >= c["minimum_median_amount"]:
                    rows.append({"code": row.code, "name": row.code_name, "industry": row.industry, "median_amount20": liquidity, "asof": asof})
                consecutive = 0
            except Exception as exc:
                failures.append({"code": row.code, "error_type": type(exc).__name__})
                consecutive += 1
            if i % 25 == 0:
                print(f"流动性筛选 {i}/{len(candidates)}，符合{len(rows)}，失败{len(failures)}", flush=True)
            if consecutive >= c["consecutive_failure_limit"]:
                json_write(root / "screen_failures.json", failures)
                raise RuntimeError("连续失败已暂停；排查网络后重跑，成功筛选缓存可续用")
            time.sleep(c["request_delay_seconds"])
        pool = pd.DataFrame(rows)
        if pool.empty:
            raise ValueError("没有符合筛选条件的股票")
        # Industry-stratified sampling, includes several liquidity tiers; no future returns used.
        n = min(c["pool_size"], len(pool))
        pool["liquidity_tier"] = pool.groupby("industry").median_amount20.rank(pct=True).mul(3).clip(upper=2.999).astype(int)
        rng = np.random.default_rng(c["seed"])
        selected = []
        for _, g in pool.groupby(["industry", "liquidity_tier"]):
            count = max(1, int(np.floor(n*len(g)/len(pool))))
            selected.extend(rng.choice(g.index, size=min(count, len(g)), replace=False))
        selected = list(dict.fromkeys(selected))
        if len(selected) > n:
            selected = list(rng.choice(selected, n, replace=False))
        if len(selected) < n:
            selected.extend(rng.choice(pool.index.difference(selected), n-len(selected), replace=False))
        pool = pool.loc[selected].sort_values("code")
        pool.to_csv(frozen, index=False, encoding="utf-8-sig")
        baskets = {industry: group.sort_values("median_amount20", ascending=False).head(c["industry_basket_size"]).code.tolist() for industry, group in pool.groupby("industry")}
        json_write(root / "baskets.json", baskets)
        json_write(root / "universe_metadata.json", {"asof": asof, "count": len(pool), "screened": len(candidates), "qualified": len(rows), "smoke_limit": limit, "failures": failures, "bias": "当前存续股票池、当前行业分类；历史回测有幸存者偏差，需后续前瞻验证"})
        print(f"已冻结 {len(pool)} 只股票；行业{len(baskets)}类")
    finally:
        bs.logout()


def collect(c, asof, stage, refresh, limit):
    root = Path(c["root"])
    pool = pd.read_csv(root / "universe.csv")
    if limit:
        pool = pool.head(limit)
    bs = login() if stage == "prices" else None
    v5 = collector(c) if stage == "funds" else None
    status_path = root / f"{stage}_status.json"
    import json
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    codes = list(pool.code)
    if bs:
        codes = ["sh.000300"] + codes
    consecutive = 0
    try:
        for i, code in enumerate(codes, 1):
            path = root / stage / f"{code}.parquet"
            previous = status.get(code, {})
            if path.exists() and not refresh and previous.get("asof") == asof and previous.get("status") == "ok":
                continue
            try:
                if bs:
                    # Full qfq refresh: do not splice incompatible adjustment bases.
                    data = history(bs, code, (pd.Timestamp(asof)-pd.Timedelta(days=365*c["history_years"])).date(), asof)
                else:
                    data = fund_fetch(v5, code.split(".")[-1], pd.read_parquet(path) if path.exists() else None)
                    data = data[data.date <= pd.Timestamp(asof)]
                    if data.empty or (pd.Timestamp(asof)-data.date.max()).days > 10:
                        raise ValueError("资金流为空或超过10自然日未更新")
                save_frame(path, data)
                status[code] = {"status": "ok", "asof": asof, "rows": len(data), "first": str(data.date.min().date()), "last": str(data.date.max().date())}
                consecutive = 0
            except Exception as exc:
                status[code] = {"status": "failed", "asof": asof, "error_type": type(exc).__name__}
                consecutive += 1
            json_write(status_path, status)
            print(f"{stage} {i}/{len(codes)} {code}: {status[code]['status']}", flush=True)
            if consecutive >= c["consecutive_failure_limit"]:
                raise RuntimeError("连续请求失败，已暂停。修复网络/Cookie后重跑同一命令续采。")
            time.sleep(c["request_delay_seconds"])
    finally:
        if bs:
            bs.logout()


def build(c):
    import json
    root = Path(c["root"])
    pool = pd.read_csv(root / "universe.csv")
    baskets = json.loads((root / "baskets.json").read_text(encoding="utf-8"))
    market = pd.read_parquet(root / "prices/sh.000300.parquet")
    calendar = pd.DatetimeIndex(market.date)
    prices = {code: pd.read_parquet(root / "prices" / f"{code}.parquet") for code in pool.code if (root / "prices" / f"{code}.parquet").exists()}
    frames, quality = [], []
    for industry, group in pool.groupby("industry"):
        # Keep missing members as empty frames so coverage threshold doesn't shrink silently.
        peers = [prices.get(code, pd.DataFrame(columns=market.columns)) for code in baskets[industry]]
        ctx = context_features(market, peers, calendar)
        for row in group.itertuples():
            if row.code not in prices:
                quality.append({"code": row.code, "status": "missing_prices"})
                continue
            funds_path = root / "funds" / f"{row.code}.parquet"
            funds = pd.read_parquet(funds_path) if funds_path.exists() else pd.DataFrame()
            x = make_features(prices[row.code], funds, ctx, calendar, c["horizons"])
            x["code"], x["industry"] = row.code, industry
            frames.append(x)
            quality.append({"code": row.code, "status": "ok", "feature_rows": len(x), "flow_rows": int(x.flow20.notna().sum()), "industry_missing_fraction": float(x.industry20.isna().mean()) if len(x) else 1})
    if not frames:
        raise ValueError("没有可用股票行情")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    save_frame(root / "panel.parquet", panel)
    json_write(root / "quality.json", quality)
    print(f"面板 {len(panel)} 行 / {panel.code.nunique()}只股票 / {panel.date.nunique()}个交易日")


def package_report(c):
    root = Path(c["root"])
    dest = root / "results_for_review.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["models/validation.json", "quality.json", "source_check.json", "universe_metadata.json", "prices_status.json", "funds_status.json", "universe.csv", "baskets.json"]:
            p = root / name
            if p.exists():
                z.write(p, name)
        # No cookies, raw code, joblib models, local paths, or credentials.
        safe = {k: v for k, v in c.items() if k not in ("v5_path", "root")}
        z.writestr("parameters.json", __import__("json").dumps(safe, ensure_ascii=False, indent=2))
    print(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["universe", "collect", "build", "train", "report"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("training_config.json")))
    parser.add_argument("--asof", default=(datetime.now(ZoneInfo("Asia/Shanghai"))-timedelta(days=1)).date().isoformat())
    parser.add_argument("--stage", choices=["prices", "funds"], default="prices")
    parser.add_argument("--limit", type=int, default=0, help="仅测试用；使用单独root，不能冒充全股票池验证")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    pd.Timestamp(args.asof)  # fail before network on invalid input
    c = config(args.config)
    if args.command == "universe":
        universe(c, args.asof, args.limit)
    elif args.command == "collect":
        collect(c, args.asof, args.stage, args.refresh, args.limit)
    elif args.command == "build":
        build(c)
    elif args.command == "train":
        from threadpoolctl import threadpool_limits
        root = Path(c["root"])
        out = root / "models"
        if (out / "model.joblib").exists():
            raise ValueError("模型已存在；保留测试记录，换root建立新实验，避免静默覆盖")
        with threadpool_limits(limits=4):
            train_panel(pd.read_parquet(root / "panel.parquet"), c, out)
    else:
        package_report(c)


if __name__ == "__main__":
    main()
