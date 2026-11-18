"""Offline pooled training and the SAME feature functions for online inference.

No news probability adjustment without a historical, point-in-time news dataset.
Industry baskets are frozen pool proxies, not official industry indexes.
"""
from __future__ import annotations

import hashlib
import json
import time
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "pooled-v2.1"
BASE = ["ret1", "ret5", "ret20", "ret60", "ma20", "ma60", "position250", "vol20", "volume_ratio", "range", "market5", "market20", "market_vol20", "industry5", "industry20", "industry_breadth", "relative20"]
FLOW = ["flow1", "flow5", "flow20", "flow_positive20"]
LABELS = {"ret1": "1日涨跌幅", "ret5": "5日涨跌幅", "ret20": "20日涨跌幅", "ret60": "60日涨跌幅",
          "ma20": "价格偏离20日均线", "ma60": "价格偏离60日均线", "position250": "250日价格位置(0～1)",
          "vol20": "20日日收益标准差", "volume_ratio": "当日量/20日均量", "range": "日内振幅/昨收",
          "market5": "沪深300的5日收益", "market20": "沪深300的20日收益", "market_vol20": "大盘20日波动",
          "industry5": "行业篮子5日收益", "industry20": "行业篮子20日收益", "industry_breadth": "行业篮子上涨比例",
          "relative20": "个股20日收益减行业20日收益", "flow1": "当日主力净流入/成交额",
          "flow5": "5日主力净流入合计/成交额合计", "flow20": "20日主力净流入合计/成交额合计",
          "flow_positive20": "20日主力净流入为正的天数比例"}


def config(path="training_config.json"):
    p = Path(path).resolve()
    c = json.loads(p.read_text(encoding="utf-8"))
    c["root"] = str((p.parent / c["root"]).resolve())
    return c


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def stock_holdout(code, c):
    return int(hashlib.sha256(str(code).encode()).hexdigest()[:8], 16) % 100 < c["heldout_stock_percent"]


def context_features(market, peers, calendar):
    ctx = pd.DataFrame(index=calendar)
    m = market.set_index("date").close.reindex(calendar)
    ctx["market5"], ctx["market20"] = m.pct_change(5, fill_method=None), m.pct_change(20, fill_method=None)
    ctx["market_vol20"] = m.pct_change(fill_method=None).rolling(20).std()
    if peers:
        returns = pd.concat([p.set_index("date").close.reindex(calendar).pct_change(fill_method=None) for p in peers], axis=1)
        count = returns.notna().sum(axis=1)
        enough = count >= max(3, int(np.ceil(len(peers) * .8)))
        daily = returns.mean(axis=1).where(enough)
        # Rolling products do not bridge missing industry observations.
        ctx["industry5"] = (1 + daily).rolling(5).apply(np.prod, raw=True) - 1
        ctx["industry20"] = (1 + daily).rolling(20).apply(np.prod, raw=True) - 1
        ctx["industry_breadth"] = returns.gt(0).sum(axis=1).div(count).where(enough)
    else:
        ctx[["industry5", "industry20", "industry_breadth"]] = np.nan
    return ctx


def make_features(prices, funds, context, calendar, horizons=(1, 5, 10, 30)):
    p = prices.drop_duplicates("date").set_index("date").reindex(calendar)
    x = pd.DataFrame(index=calendar)
    close = p.close
    for n in (1, 5, 20, 60):
        x[f"ret{n}"] = close.pct_change(n, fill_method=None)
    for n in (20, 60):
        x[f"ma{n}"] = close / close.rolling(n).mean() - 1
    low, high = close.rolling(250).min(), close.rolling(250).max()
    x["position250"] = (close - low) / (high - low).replace(0, np.nan)
    x["vol20"] = x.ret1.rolling(20).std()
    x["volume_ratio"] = p.volume / p.volume.rolling(20).mean().replace(0, np.nan)
    x["range"] = (p.high - p.low) / close.shift(1)
    x = x.join(context)
    x["relative20"] = x.ret20 - x.industry20
    net = pd.Series(np.nan, index=calendar)
    if funds is not None and not funds.empty:
        net = funds.drop_duplicates("date").set_index("date").main_net.reindex(calendar)
    amount = p.amount.where(p.amount > 0)
    # Missing flows NEVER become zero and cannot form a matched training sample.
    net = net.where(amount.notna())
    x["flow1"] = net / amount
    for n in (5, 20):
        x[f"flow{n}"] = net.rolling(n).sum() / amount.rolling(n).sum()
    x["flow_positive20"] = net.gt(0).astype(float).where(net.notna()).rolling(20).mean()
    x["close"] = close
    valid = close.gt(0) & p.volume.gt(0)
    dates = pd.Series(calendar, index=calendar)
    for h in horizons:
        # Labels on the full exchange calendar, BEFORE any row filtering.
        x[f"y{h}"] = (close.shift(-h) / close - 1).where(valid & valid.shift(-h, fill_value=False))
        x[f"end{h}"] = dates.shift(-h)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.loc[valid & x.position250.notna()]
    x.index.name = "date"
    return x.reset_index()


def time_split(frame, h, minimum_dates=10):
    dates = np.sort(frame.date.unique())
    if len(dates) < 50:
        raise ValueError(f"只有{len(dates)}个日期；不足以独立切分")
    a, b, d = [dates[int(len(dates) * v)] for v in (.50, .67, .83)]
    end = frame[f"end{h}"]
    blocks = [frame[(frame.date < a) & (end < a)],
              frame[(frame.date >= a) & (frame.date < b) & (end < b)],
              frame[(frame.date >= b) & (frame.date < d) & (end < d)],
              frame[frame.date >= d]]
    if any(v.date.nunique() < minimum_dates for v in blocks):
        raise ValueError(f"{h}日标签清除跨段重叠后，四段日期数为{[v.date.nunique() for v in blocks]}，每段至少{minimum_dates}日")
    return blocks


def capped(frame, c):
    if len(frame) <= c["max_train_rows"]:
        return frame
    # Sampling is only inside training. Test/selection are never mixed in.
    return frame.sample(c["max_train_rows"], random_state=c["seed"]).sort_values(["date", "code"])


class Bundle:
    def fit(self, train, cal, names, h, c):
        self.names, self.h = names, h
        self.training_end = str(train.date.max().date())
        self.calibration_end = str(cal.date.max().date())
        train = capped(train, c)
        x, y = train[names], train[f"y{h}"]
        self.prob = make_pipeline(SimpleImputer(add_indicator=True, keep_empty_features=True), StandardScaler(),
                                  LogisticRegression(C=c["logistic_C"], max_iter=600, random_state=c["seed"]))
        weights = 1 / train.groupby("date").date.transform("size")
        weights = weights / weights.mean()
        self.prob.fit(x, y.gt(0), logisticregression__sample_weight=weights)
        self.cal = LogisticRegression(C=1, max_iter=300)
        cw = 1 / cal.groupby("date").date.transform("size")
        self.cal.fit(self.prob.decision_function(cal[names]).reshape(-1, 1), cal[f"y{h}"].gt(0), sample_weight=cw / cw.mean())
        if self.cal.coef_[0, 0] <= 0:
            raise ValueError("校准方向不稳定，拒绝发布")
        args = dict(max_iter=c["regressor_iterations"], max_leaf_nodes=c["regressor_leaves"],
                    min_samples_leaf=c["regressor_min_leaf"], l2_regularization=c["regressor_l2"],
                    learning_rate=.05, early_stopping=False, random_state=c["seed"])
        self.up, self.down = HistGradientBoostingRegressor(**args), HistGradientBoostingRegressor(**args)
        pos = y > 0
        if min(pos.sum(), (~pos).sum()) < 100:
            raise ValueError("上涨或下跌条件收益样本不足100")
        self.up.fit(x[pos], y[pos], sample_weight=weights[pos])
        self.down.fit(x[~pos], -y[~pos], sample_weight=weights[~pos])
        self.quantiles = [HistGradientBoostingRegressor(loss="quantile", quantile=q, **args).fit(x, y, sample_weight=weights) for q in (.1, .5, .9)]
        self.prior = float(np.average(pos, weights=weights))
        return self

    def predict(self, x):
        xx = x[self.names]
        p = self.cal.predict_proba(self.prob.decision_function(xx).reshape(-1, 1))[:, 1]
        up, down = np.maximum(0, self.up.predict(xx)), np.maximum(0, self.down.predict(xx))
        q = np.sort(np.column_stack([m.predict(xx) for m in self.quantiles]), axis=1)
        # Long-only simple returns cannot fall below -100%.
        q = np.maximum(q, -.9999)
        return pd.DataFrame({"p": p, "up": up, "down": down, "ev": p*up-(1-p)*down,
                             "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]}, index=x.index)

    def explain(self, x):
        transformed = self.prob[:-1].transform(x[self.names])
        names = list(self.prob[0].get_feature_names_out(self.names))
        values = transformed[0] * self.prob[-1].coef_[0] * self.cal.coef_[0, 0]
        return pd.DataFrame({"参数": [LABELS.get(n, n) for n in names], "对上涨对数赔率的贡献": values}).sort_values("对上涨对数赔率的贡献")


def evaluate(model, frame, h):
    pred = model.predict(frame)
    pred["actual"] = frame[f"y{h}"]
    pred["date"], pred["code"] = frame.date, frame.code
    pred["loss"] = (pred.p - pred.actual.gt(0))**2
    def dm(s):
        return float(s.groupby(frame.date).mean().mean())
    metrics = {"rows": len(frame), "dates": frame.date.nunique(), "stocks": frame.code.nunique(),
               "start": str(frame.date.min().date()), "end": str(frame.date.max().date()),
               "brier": dm(pred.loss), "baseline_brier": dm((model.prior-pred.actual.gt(0))**2),
               "accuracy": dm((pred.p.ge(.5) == pred.actual.gt(0)).astype(float)),
               "interval80_coverage": dm((pred.actual.ge(pred.q10) & pred.actual.le(pred.q90)).astype(float)),
               "ev_mae": dm((pred.ev-pred.actual).abs())}
    for side, mask, estimate, actual in [("up", pred.actual > 0, pred.up, pred.actual), ("down", pred.actual <= 0, pred.down, -pred.actual)]:
        metrics[f"conditional_{side}_mae"] = dm((estimate-actual).abs().where(mask)) if mask.any() else None
    pred["probability_bin"] = pd.cut(pred.p, bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
    calibration = pred.assign(outcome=pred.actual.gt(0).astype(float)).groupby("probability_bin").agg(n=("p", "size"), predicted=("p", "mean"), observed=("outcome", "mean")).reset_index().to_dict("records")
    metrics["calibration"] = calibration
    return metrics, pred


def block_ci(values, block, c):
    """Bootstrap date aggregates, not pseudo-independent stock rows."""
    v = np.asarray(values, float)
    if len(v) < max(30, 5*block):
        return {"mean": float(v.mean()), "low": None, "high": None, "reason": "独立测试日期不足，至少需要max(30, 5×期限)日"}
    rng = np.random.default_rng(c["seed"])
    results = []
    for _ in range(c["bootstrap_repetitions"]):
        starts = rng.integers(0, len(v)-block+1, size=int(np.ceil(len(v)/block)))
        sample = np.concatenate([v[s:s+block] for s in starts])[:len(v)]
        results.append(sample.mean())
    lo, hi = np.quantile(results, [.025, .975])
    return {"mean": float(v.mean()), "low": float(lo), "high": float(hi), "reason": "按日期移动块bootstrap；有限市场状态下的估计"}


def train_panel(panel, c, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    held = panel.code.map(lambda s: stock_holdout(s, c))
    report = {"version": VERSION, "limitations": ["当前存续股票与当前行业映射存在幸存者偏差", "资金流只代表东财按成交单分类的净额，不等同真实机构持仓", "消息解析未进入数值模型", "测试集不得反复用于调参"], "horizons": {}}
    import sklearn
    report["runtime"] = {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    report["panel"] = {"rows": len(panel), "stocks": panel.code.nunique(), "dates": panel.date.nunique(), "date_start": str(panel.date.min()), "date_end": str(panel.date.max())}
    artifacts = {}
    for h in c["horizons"]:
        started = time.perf_counter()
        print(f"训练/验证 {h} 日", flush=True)
        valid = panel.dropna(subset=[f"y{h}", f"end{h}"])
        detail = {}
        try:
            train, cal, select, test = time_split(valid.loc[~held.loc[valid.index]], h)
            base = Bundle().fit(train, cal, BASE, h, c)
            detail["base_selection"], _ = evaluate(base, select, h)
            detail["base_test"], bp = evaluate(base, test, h)
            bp.to_parquet(output / f"base_{h}_test.parquet", index=False)
            improvement = ((base.prior-bp.actual.gt(0))**2 - bp.loss).groupby(bp.date).mean()
            detail["base_advantage_ci"] = block_ci(improvement, h, c)
            unseen = valid.loc[held.loc[valid.index] & (valid.date >= test.date.min())]
            if len(unseen):
                detail["unseen_stock_test"], up = evaluate(base, unseen, h)
                up.to_parquet(output / f"unseen_{h}_test.parquet", index=False)
            artifacts[h] = {"base": base, "flow": None, "flow_verified": False}
            # Independent short-history comparison: same rows, dates, labels and settings.
            flow = valid.dropna(subset=BASE+FLOW)
            flow = flow.loc[~held.loc[flow.index]]
            try:
                ft, fc, fs, fe = time_split(flow, h)
                control = Bundle().fit(ft, fc, BASE, h, c)
                joint = Bundle().fit(ft, fc, BASE+FLOW, h, c)
                cs, _ = evaluate(control, fs, h)
                js, _ = evaluate(joint, fs, h)
                selected = js["brier"] < cs["brier"]
                cm, cp = evaluate(control, fe, h)
                jm, jp = evaluate(joint, fe, h)
                delta = (cp.loss-jp.loss).groupby(fe.date).mean()
                ci = block_ci(delta, h, c)
                # Also compare with the long-history base. A short control alone is not enough.
                bm, matched_bp = evaluate(base, fe, h)
                # Base may have seen these dates: use only its untouched final-test overlap.
                honest = fe.date >= test.date.min()
                long_ci = block_ci((matched_bp.loc[honest, "loss"]-jp.loc[honest, "loss"]).groupby(fe.loc[honest, "date"]).mean(), h, c) if honest.any() else {"low": None, "reason": "没有共同的未见测试日期"}
                verified = bool(selected and ci["low"] is not None and ci["low"] > 0 and long_ci["low"] is not None and long_ci["low"] > 0)
                detail["flow"] = {"status": "通过独立验证" if verified else "优势未验证", "selected_before_test": selected, "control_selection": cs, "joint_selection": js,
                                  "control_test": cm, "joint_test": jm, "brier_improvement_ci": ci, "versus_long_base_ci": long_ci}
                jp.to_parquet(output / f"flow_{h}_test.parquet", index=False)
                cp.to_parquet(output / f"control_{h}_test.parquet", index=False)
                # Never use held-out ticker labels to select or fit either model.
                unseen_flow = valid.loc[held.loc[valid.index] & (valid.date >= fe.date.min())].dropna(subset=BASE+FLOW)
                if len(unseen_flow):
                    detail["flow"]["unseen_stock_test"], fp = evaluate(joint, unseen_flow, h)
                    fp.to_parquet(output / f"flow_unseen_{h}_test.parquet", index=False)
                artifacts[h].update(flow=joint, flow_verified=verified)
            except ValueError as exc:
                detail["flow"] = {"status": "历史不足或拟合不稳定", "reason": str(exc)}
        except ValueError as exc:
            detail["error"] = str(exc)
        if h in artifacts:
            # Save frozen evaluated models separately, then refit fixed settings on recent data.
            joblib.dump(artifacts[h], output / f"evaluated_{h}.joblib")
            for kind, names in [("base", BASE), ("flow", BASE+FLOW)]:
                if artifacts[h][kind] is None:
                    continue
                deploy = valid.loc[~held.loc[valid.index]].dropna(subset=FLOW if kind == "flow" else [])
                ds = np.sort(deploy.date.unique())
                boundary = ds[int(len(ds)*.85)]
                dt = deploy[(deploy.date < boundary) & (deploy[f"end{h}"] < boundary)]
                dc = deploy[deploy.date >= boundary]
                try:
                    artifacts[h][kind] = Bundle().fit(dt, dc, names, h, c)
                    detail[kind+"_deployment"] = {"training_end": artifacts[h][kind].training_end, "calibration_end": artifacts[h][kind].calibration_end,
                                                 "note": "固定参数在评估完成后重拟合；上面的测试结果属于冻结评估模型，部署模型需前瞻验证"}
                except ValueError as exc:
                    detail[kind+"_deployment"] = {"note": "重拟合失败，保留原评估模型", "reason": str(exc)}
        detail["elapsed_seconds"] = round(time.perf_counter()-started, 2)
        report["horizons"][str(h)] = detail
        json_write(output / "validation.json", report)
    if not artifacts:
        raise ValueError("没有可发布期限，请查看validation.json")
    payload = {"version": VERSION, "models": artifacts, "report": report, "config": {k: v for k, v in c.items() if k != "v5_path"}, "data_end": str(panel.date.max().date())}
    joblib.dump(payload, output / "model.joblib")
    return report
