import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Glyph .* missing from font\\(s\\).*")


@dataclass
class ForecastResult:
    horizon: int
    p_up: float
    p_down: float
    median_ret: float
    q25: float
    q75: float


@dataclass
class AnalysisContext:
    stock_code: str
    has_flow_data: bool
    close_vs_ma20: Optional[float]
    close_vs_ma60: Optional[float]
    ret5: Optional[float]
    ret20: Optional[float]
    vol20: Optional[float]
    vol_ratio: Optional[float]
    main_flow: Optional[float]
    main_flow_ma3: Optional[float]
    flow_days: Optional[float]
    missing_notes: List[str]


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _pick_sheet(sheet_names: List[str], keywords: List[str], fallback_idx: int) -> str:
    for s in sheet_names:
        if all(k in s for k in keywords):
            return s
    return sheet_names[fallback_idx]


def _match_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    norm_map = {str(c).replace(" ", "").lower(): c for c in columns}
    for cand in candidates:
        key = cand.replace(" ", "").lower()
        if key in norm_map:
            return norm_map[key]

    for c in columns:
        c2 = str(c).replace(" ", "").lower()
        if any(k.replace(" ", "").lower() in c2 for k in candidates):
            return c
    return None


def read_workbook(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    if len(sheets) < 3:
        raise ValueError("工作表数量不足，至少需要历史行情和资金流。")

    hist_sheet = _pick_sheet(sheets, ["历史", "行情"], 1 if len(sheets) > 1 else 0)
    fund_sheet = _pick_sheet(sheets, ["资金", "流"], 2 if len(sheets) > 2 else 1)
    merged_sheet = _pick_sheet(sheets, ["合并"], min(3, len(sheets) - 1))

    hist = pd.read_excel(path, sheet_name=hist_sheet)
    fund = pd.read_excel(path, sheet_name=fund_sheet)
    merged = pd.read_excel(path, sheet_name=merged_sheet)
    return hist, fund, merged


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = _match_col(df.columns.tolist(), ["日期", "date"])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        if date_col != "日期":
            df = df.rename(columns={date_col: "日期"})
    return df


def build_indicators(hist: pd.DataFrame, fund: pd.DataFrame) -> Dict[str, float]:
    hist = normalize_df(hist)
    fund = normalize_df(fund)

    close_col = _match_col(hist.columns.tolist(), ["收盘", "close"])
    ma20_col = _match_col(hist.columns.tolist(), ["MA20", "ma20"])
    ma60_col = _match_col(hist.columns.tolist(), ["MA60", "ma60"])
    ret5_col = _match_col(hist.columns.tolist(), ["5日涨跌幅", "5日"])
    ret20_col = _match_col(hist.columns.tolist(), ["20日涨跌幅", "20日"])
    vol20_col = _match_col(hist.columns.tolist(), ["20日波动率", "波动率"])
    vr_col = _match_col(hist.columns.tolist(), ["量比_相对20日", "量比"])

    main_flow_col = _match_col(fund.columns.tolist(), ["主力净流入-净额", "主力净流入"])
    main_flow_ma3 = _match_col(fund.columns.tolist(), ["主力净流入_MA3", "MA3"])
    flow_days_col = _match_col(fund.columns.tolist(), ["主力连续流入天数", "连续"])

    latest = hist.iloc[-1] if not hist.empty else pd.Series(dtype=float)
    latest_fund = fund.iloc[-1] if not fund.empty else pd.Series(dtype=float)

    def _v(s: pd.Series, col: Optional[str], default=np.nan):
        if col is None or s.empty:
            return default
        return pd.to_numeric(pd.Series([s.get(col)]), errors="coerce").iloc[0]

    close = _v(latest, close_col)
    ma20 = _v(latest, ma20_col)
    ma60 = _v(latest, ma60_col)
    ret5 = _v(latest, ret5_col, 0.0)
    ret20 = _v(latest, ret20_col, 0.0)
    vol20 = _v(latest, vol20_col, 0.0)
    vol_ratio = _v(latest, vr_col, 1.0)
    main_flow = _v(latest_fund, main_flow_col, 0.0)
    main_flow3 = _v(latest_fund, main_flow_ma3, 0.0)
    flow_days = _v(latest_fund, flow_days_col, 0.0)

    trend_score = 0.0
    if pd.notna(close) and pd.notna(ma20) and ma20 != 0:
        trend_score += np.clip((close / ma20 - 1) * 6, -1.2, 1.2)
    if pd.notna(close) and pd.notna(ma60) and ma60 != 0:
        trend_score += np.clip((close / ma60 - 1) * 4, -1.0, 1.0)
    trend_score += np.clip(ret5 * 5 + ret20 * 2, -1.5, 1.5)
    trend_score -= np.clip(vol20 * 8, 0, 1.2)

    flow_score = 0.0
    if pd.notna(main_flow):
        flow_score += np.clip(main_flow / 1e8, -1.2, 1.2)
    if pd.notna(main_flow3):
        flow_score += np.clip(main_flow3 / 1e8, -1.0, 1.0)
    flow_score += np.clip((flow_days or 0) / 5.0, 0, 1.0)

    volume_score = np.clip((vol_ratio or 1.0) - 1.0, -0.8, 1.2)

    total_score = 0.55 * trend_score + 0.30 * flow_score + 0.15 * volume_score
    total_score = float(np.clip(total_score, -2.5, 2.5))

    return {
        "trend_score": float(trend_score),
        "flow_score": float(flow_score),
        "volume_score": float(volume_score),
        "total_score": total_score,
    }


def build_analysis_context(stock_code: str, hist: pd.DataFrame, fund: pd.DataFrame) -> AnalysisContext:
    hist = normalize_df(hist)
    fund = normalize_df(fund)

    close_col = _match_col(hist.columns.tolist(), ["收盘", "close"])
    ma20_col = _match_col(hist.columns.tolist(), ["MA20", "ma20"])
    ma60_col = _match_col(hist.columns.tolist(), ["MA60", "ma60"])
    ret5_col = _match_col(hist.columns.tolist(), ["5日涨跌幅", "5日"])
    ret20_col = _match_col(hist.columns.tolist(), ["20日涨跌幅", "20日"])
    vol20_col = _match_col(hist.columns.tolist(), ["20日波动率", "波动率"])
    vr_col = _match_col(hist.columns.tolist(), ["量比_相对20日", "量比"])
    main_flow_col = _match_col(fund.columns.tolist(), ["主力净流入-净额", "主力净流入"])
    main_flow_ma3_col = _match_col(fund.columns.tolist(), ["主力净流入_MA3", "MA3"])
    flow_days_col = _match_col(fund.columns.tolist(), ["主力连续流入天数", "连续"])

    latest = hist.iloc[-1] if not hist.empty else pd.Series(dtype=float)
    latest_fund = fund.iloc[-1] if not fund.empty else pd.Series(dtype=float)

    def _num(series: pd.Series, col: Optional[str]) -> Optional[float]:
        if col is None or series.empty:
            return None
        v = pd.to_numeric(pd.Series([series.get(col)]), errors="coerce").iloc[0]
        return None if pd.isna(v) else float(v)

    close = _num(latest, close_col)
    ma20 = _num(latest, ma20_col)
    ma60 = _num(latest, ma60_col)

    missing_notes = []
    if close_col is None:
        missing_notes.append("缺少收盘价字段，趋势判断精度降低")
    if ma20_col is None:
        missing_notes.append("缺少MA20字段，短中期趋势信号已跳过")
    if main_flow_col is None and fund.empty:
        missing_notes.append("资金流数据为空，资金维度已跳过")
    elif main_flow_col is None:
        missing_notes.append("缺少主力净流入字段，资金维度部分跳过")

    close_vs_ma20 = (close / ma20 - 1) if (close is not None and ma20 not in [None, 0]) else None
    close_vs_ma60 = (close / ma60 - 1) if (close is not None and ma60 not in [None, 0]) else None

    return AnalysisContext(
        stock_code=stock_code,
        has_flow_data=not fund.empty,
        close_vs_ma20=close_vs_ma20,
        close_vs_ma60=close_vs_ma60,
        ret5=_num(latest, ret5_col),
        ret20=_num(latest, ret20_col),
        vol20=_num(latest, vol20_col),
        vol_ratio=_num(latest, vr_col),
        main_flow=_num(latest_fund, main_flow_col),
        main_flow_ma3=_num(latest_fund, main_flow_ma3_col),
        flow_days=_num(latest_fund, flow_days_col),
        missing_notes=missing_notes,
    )


def forecast_probabilities(hist: pd.DataFrame, score: float, n_sim: int = 5000) -> List[ForecastResult]:
    hist = normalize_df(hist)
    close_col = _match_col(hist.columns.tolist(), ["收盘", "close"])
    if close_col is None or len(hist) < 40:
        raise ValueError("历史行情不足，无法进行概率评估。")

    close = pd.to_numeric(hist[close_col], errors="coerce").dropna()
    rets = close.pct_change().dropna()
    rets = rets.tail(min(180, len(rets)))
    if len(rets) < 30:
        raise ValueError("有效收益率样本不足，无法进行概率评估。")

    horizons = [5, 10, 20]
    results: List[ForecastResult] = []

    for h in horizons:
        samples = np.random.choice(rets.values, size=(n_sim, h), replace=True)
        path_ret = np.prod(1 + samples, axis=1) - 1

        base_up = float((path_ret > 0).mean())
        adjusted_up = float(np.clip(base_up + 0.06 * score, 0.05, 0.95))
        p_down = 1.0 - adjusted_up

        results.append(
            ForecastResult(
                horizon=h,
                p_up=adjusted_up,
                p_down=p_down,
                median_ret=float(np.median(path_ret)),
                q25=float(np.quantile(path_ret, 0.25)),
                q75=float(np.quantile(path_ret, 0.75)),
            )
        )
    return results


def explain_probability_reasons(
    indicators: Dict[str, float],
    forecasts: List[ForecastResult],
    ctx: AnalysisContext,
) -> Dict[int, List[str]]:
    reasons: Dict[int, List[str]] = {}
    for f in forecasts:
        rs = []
        if ctx.close_vs_ma20 is not None:
            if ctx.close_vs_ma20 > 0.03:
                rs.append("收盘价高于MA20较多，短期趋势偏强")
            elif ctx.close_vs_ma20 < -0.03:
                rs.append("收盘价低于MA20较多，短期趋势偏弱")
        if ctx.close_vs_ma60 is not None and f.horizon >= 10:
            rs.append("MA60相对位置用于中期方向修正")
        if ctx.ret5 is not None and abs(ctx.ret5) > 0.06:
            rs.append("近5日波动较大，短期概率分布更分散")
        if ctx.vol20 is not None and ctx.vol20 > 0.04:
            rs.append("20日波动率偏高，下跌尾部风险抬升")
        if ctx.vol_ratio is not None and ctx.vol_ratio > 1.2:
            rs.append("量比偏高，价格突破或反转概率上升")

        if ctx.main_flow is not None:
            if ctx.main_flow > 0:
                rs.append("主力净流入为正，对上涨概率有正向修正")
            else:
                rs.append("主力净流入为负，对上涨概率有负向修正")
        else:
            rs.append("资金流关键字段缺失，本期仅使用价格与波动率因子")

        if not rs:
            rs.append("主要基于历史收益分布进行中性估计")
        reasons[f.horizon] = rs
    return reasons


def plot_figures(hist: pd.DataFrame, fund: pd.DataFrame, forecasts: List[ForecastResult], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    hist = normalize_df(hist)
    fund = normalize_df(fund)

    date_col = _match_col(hist.columns.tolist(), ["日期", "date"]) or "日期"
    close_col = _match_col(hist.columns.tolist(), ["收盘", "close"])
    ma20_col = _match_col(hist.columns.tolist(), ["MA20", "ma20"])
    ma60_col = _match_col(hist.columns.tolist(), ["MA60", "ma60"])
    vol_col = _match_col(hist.columns.tolist(), ["成交量", "volume"])
    vr_col = _match_col(hist.columns.tolist(), ["量比_相对20日", "量比"])
    flow_col = _match_col(fund.columns.tolist(), ["主力净流入-净额", "主力净流入"])
    flow_ma3 = _match_col(fund.columns.tolist(), ["主力净流入_MA3", "MA3"])

    if close_col:
        plt.figure(figsize=(12, 5))
        plt.plot(hist[date_col], hist[close_col], label="收盘价", linewidth=1.2)
        if ma20_col:
            plt.plot(hist[date_col], hist[ma20_col], label="MA20", linewidth=1.0)
        if ma60_col:
            plt.plot(hist[date_col], hist[ma60_col], label="MA60", linewidth=1.0)
        plt.title("价格趋势与均线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_price_ma.png"), dpi=150)
        plt.close()

    if vol_col:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(hist[date_col], pd.to_numeric(hist[vol_col], errors="coerce"), alpha=0.5, label="成交量")
        ax1.set_title("成交量与量比")
        if vr_col:
            ax2 = ax1.twinx()
            ax2.plot(hist[date_col], pd.to_numeric(hist[vr_col], errors="coerce"), color="tab:red", label="量比")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "02_volume_ratio.png"), dpi=150)
        plt.close(fig)

    if flow_col and not fund.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(fund["日期"], pd.to_numeric(fund[flow_col], errors="coerce"), label="主力净流入")
        if flow_ma3:
            plt.plot(fund["日期"], pd.to_numeric(fund[flow_ma3], errors="coerce"), label="主力净流入_MA3")
        plt.title("主力资金流")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_main_flow.png"), dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    horizons = [f"{x.horizon}日" for x in forecasts]
    up_vals = [x.p_up * 100 for x in forecasts]
    down_vals = [x.p_down * 100 for x in forecasts]
    x = np.arange(len(horizons))
    w = 0.35
    plt.bar(x - w / 2, up_vals, width=w, label="上涨概率")
    plt.bar(x + w / 2, down_vals, width=w, label="下跌概率")
    plt.xticks(x, horizons)
    plt.ylim(0, 100)
    plt.title("未来涨跌概率对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_probabilities.png"), dpi=150)
    plt.close()


def write_report(
    stock_code: str,
    indicators: Dict[str, float],
    forecasts: List[ForecastResult],
    reasons_by_horizon: Dict[int, List[str]],
    ctx: AnalysisContext,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    best = max(forecasts, key=lambda x: x.p_up)
    worst = max(forecasts, key=lambda x: x.p_down)

    trend_text = "偏强" if indicators["trend_score"] > 0.4 else ("偏弱" if indicators["trend_score"] < -0.4 else "震荡")
    flow_text = "净流入占优" if indicators["flow_score"] > 0.3 else ("净流出压力" if indicators["flow_score"] < -0.3 else "中性")

    lines = [
        f"# 股票分析报告 - {stock_code}",
        "",
        "## 一、综合结论",
        f"- 趋势判断：{trend_text}",
        f"- 资金流判断：{flow_text}",
        f"- 综合评分：{indicators['total_score']:.2f}（区间 -2.5 到 2.5）",
        "",
        "## 二、未来涨跌概率（基于历史波动 bootstrap + 技术面/资金面修正）",
    ]
    for r in forecasts:
        lines.append(
            f"- 未来 {r.horizon} 日：上涨 {r.p_up*100:.1f}% / 下跌 {r.p_down*100:.1f}% / "
            f"收益中位数 {r.median_ret*100:.2f}% / 区间 [{r.q25*100:.2f}%, {r.q75*100:.2f}%]"
        )
        for reason in reasons_by_horizon.get(r.horizon, []):
            lines.append(f"  - 原因：{reason}")

    lines.extend(
        [
            "",
            "## 三、时间范围与交易提示",
            f"- 概率最高上涨时间窗：未来 {best.horizon} 个交易日（上涨概率 {best.p_up*100:.1f}%）",
            f"- 主要风险时间窗：未来 {worst.horizon} 个交易日（下跌概率 {worst.p_down*100:.1f}%）",
            "- 建议结合行业、公告、市场风险偏好做二次确认；该结果不构成投资建议。",
            "",
            "## 四、图表文件",
            "- `01_price_ma.png`：价格趋势与均线",
            "- `02_volume_ratio.png`：成交量与量比",
            "- `03_main_flow.png`：主力资金流",
            "- `04_probabilities.png`：未来涨跌概率",
        ]
    )
    if ctx.missing_notes:
        lines.extend(["", "## 五、缺失字段与跳过项"])
        for note in ctx.missing_notes:
            lines.append(f"- {note}")

    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def run_analysis_from_data(
    stock_code: str,
    hist_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    output_dir: str,
) -> Dict[str, str]:
    """
    直接使用内存中的 DataFrame 做分析，不读取 Excel 文件。
    """
    hist = normalize_df(hist_df)
    fund = normalize_df(fund_df)
    if hist.empty:
        raise ValueError("历史行情为空，无法生成分析报告。")

    indicators = build_indicators(hist, fund)
    ctx = build_analysis_context(stock_code, hist, fund)
    forecasts = forecast_probabilities(hist, indicators["total_score"])
    reasons_by_horizon = explain_probability_reasons(indicators, forecasts, ctx)
    plot_figures(hist, fund, forecasts, output_dir)
    report_path = write_report(stock_code, indicators, forecasts, reasons_by_horizon, ctx, output_dir)
    return {
        "output_dir": output_dir,
        "report_path": report_path,
    }


def main():
    parser = argparse.ArgumentParser(description="基于单票结果文件的综合分析与可视化。")
    parser.add_argument(
        "--input",
        default=r"002324_股票分析数据_20260430_155939.xlsx",
        help="输入的股票分析 xlsx 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        default=r"danalysis_002324",
        help="输出目录（图表和报告）",
    )
    args = parser.parse_args()

    hist, fund, _ = read_workbook(args.input)
    hist = normalize_df(hist)
    fund = normalize_df(fund)

    code_col = _match_col(hist.columns.tolist(), ["股票代码", "代码"])
    stock_code = str(hist.iloc[-1][code_col]).zfill(6) if code_col and not hist.empty else "UNKNOWN"

    result = run_analysis_from_data(
        stock_code=stock_code,
        hist_df=hist,
        fund_df=fund,
        output_dir=args.output_dir,
    )

    print(f"分析完成，输出目录：{args.output_dir}")
    print(f"报告文件：{result['report_path']}")


if __name__ == "__main__":
    main()
