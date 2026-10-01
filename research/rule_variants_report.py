"""方向 C（2026-10-01）: 運用ルールの変種の一覧表。最良は選ばず、リスクとリターンを並べるだけ。

変種 = 上限ロット数（risk_variant_max_lots）× 保有期限（risk_variant_max_hold_days）× g（phase4_baseline_g）。
いずれも「未保有の朝は必ず買う」。上限3ロット・期限なしが現行ルール。
1. 開発期間（dev_end まで）の実際の値動きで回す（口数固定。現行ルールどおり）
2. ブロック・ブートストラップの架空の値動き（建値は開発期間の最終日の終値、5年分 × risk_bootstrap_paths 本）で回す。
   全変種に同じパスを使う。drift=zero は上昇トレンドを除いた場合。口数固定だと株価が上がった架空のパスで
   円のリスクが資金を超えて膨らむので、ブートストラップでは1ロットを「100口 × 建値」の金額に固定する（lot_jpy）
3. 各変種に、同じ値動きで「その変種の平均投入額を毎日持ち続ける」比較（const_*）を並べる。
   そろえる売買のコストは含めないので、比較はやや const に有利
出力: research/trial_log/rule_variants/summary.json と variants.csv
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from common_utils.position_simulator import PositionRules, simulate_positions, summarize  # noqa: E402
from common_utils.rule_risk import block_bootstrap_ohlc, constant_exposure_daily, exposure_stats  # noqa: E402
from generate.features_v2 import build_v2_dataset, v2_config  # noqa: E402

OUT = ROOT / "research" / "trial_log" / "rule_variants"
_STATE: dict = {}


def variants(cfg: dict, base: PositionRules) -> list[tuple[str, PositionRules, float]]:
    out = []
    for lots in cfg["risk_variant_max_lots"]:
        for hold in cfg["risk_variant_max_hold_days"]:
            r = replace(base, max_units=int(lots) * base.lot_units, max_hold_days=None if hold is None else int(hold))
            for g in cfg["phase4_baseline_g"]:
                out.append((f"lots{lots}_hold{hold or 'none'}_g{g}", r, float(g)))
    return out


def _curve_stats(daily: pd.DataFrame, rules: PositionRules, ppy: int) -> dict:
    r, eq = daily["ret"], daily["equity_jpy"]
    sd = r.std(ddof=1)
    return {"sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan,
            "ann_return_on_capital": float(r.mean() * ppy),
            "max_drawdown_share": float((eq - eq.cummax()).min() / rules.capital_jpy),
            "final_equity_share": float(eq.iloc[-1] / rules.capital_jpy)}


def measure(prices: pd.DataFrame, rules: PositionRules, g: float, ppy: int) -> dict:
    daily, trades = simulate_positions(prices, pd.Series(True, index=prices.index), g, rules)
    e = exposure_stats(daily, trades, rules)
    closed = trades.dropna(subset=["exit_date"])
    share = float(daily["invested_jpy"].mean() / rules.capital_jpy)
    out = {**_curve_stats(daily, rules, ppy), "avg_invested_share": share,
           "worst_unrealized_share": e["worst_unrealized_share_of_capital"],
           "max_invested_share": e["max_invested_share_of_capital"],
           "longest_hold_days": e["longest_hold_days"], "longest_underwater_days": e["longest_underwater_days"],
           "n_trades": int(len(trades)), "n_losing_closed": int((closed["pnl_jpy"] <= 0).sum())}
    const = constant_exposure_daily(prices, rules, share)
    cs = _curve_stats(const, rules, ppy)
    out.update({f"const_{k}": v for k, v in cs.items()})
    out["const_worst_below_start_share"] = float(min(const["equity_jpy"].min(), 0.0) / rules.capital_jpy)
    out["const_longest_underwater_days"] = exposure_stats(
        const, pd.DataFrame({"holding_days": [len(const)], "lots": [1]}), rules)["longest_underwater_days"]
    return out


def _init(src, start, cfg, base, ppy):
    yen_base = replace(base, lot_jpy=base.lot_units * start)
    _STATE.update(src=src, start=start, cfg=cfg, ppy=ppy, vs=variants(cfg, yen_base))


def _one_path(job):
    drift, seed = job
    c = _STATE["cfg"]
    path = block_bootstrap_ohlc(_STATE["src"], int(c["risk_bootstrap_days"]), float(c["risk_bootstrap_block_mean"]),
                                _STATE["start"], np.random.default_rng(seed), drift=drift)
    return [{"drift": drift, "variant": name, **measure(path, r, g, _STATE["ppy"])} for name, r, g in _STATE["vs"]]


def agg(df: pd.DataFrame, ppy: int) -> pd.Series:
    return pd.Series({
        "ann_return_median": df["ann_return_on_capital"].median(),
        "ann_return_p05": df["ann_return_on_capital"].quantile(0.05),
        "sharpe_median": df["sharpe"].median(),
        "avg_invested_median": df["avg_invested_share"].median(),
        "worst_unrealized_median": df["worst_unrealized_share"].median(),
        "worst_unrealized_p05": df["worst_unrealized_share"].quantile(0.05),
        "worst_unrealized_p01": df["worst_unrealized_share"].quantile(0.01),
        "max_drawdown_median": df["max_drawdown_share"].median(),
        "max_drawdown_p05": df["max_drawdown_share"].quantile(0.05),
        "longest_underwater_median": df["longest_underwater_days"].median(),
        "longest_underwater_p95": df["longest_underwater_days"].quantile(0.95),
        "p_hold_over_1y": (df["longest_hold_days"] > ppy).mean(),
        "p_underwater_over_1y": (df["longest_underwater_days"] > ppy).mean(),
        "p_loss_after_5y": (df["final_equity_share"] < 0).mean(),
        "const_ann_return_median": df["const_ann_return_on_capital"].median(),
        "const_max_drawdown_median": df["const_max_drawdown_share"].median(),
        "const_max_drawdown_p05": df["const_max_drawdown_share"].quantile(0.05),
        "p_rule_return_above_const": (df["ann_return_on_capital"] > df["const_ann_return_on_capital"]).mean(),
        "p_rule_drawdown_smaller_than_const": (df["max_drawdown_share"] > df["const_max_drawdown_share"]).mean(),
    })


def main() -> dict:
    cfg = v2_config()
    base = PositionRules.from_config()
    ppy = int(cfg["trading_days_per_year"])
    end = pd.Timestamp(cfg["dev_end"])
    prices = build_v2_dataset()["prices"].loc[:end]
    start = float(prices["raw_close"].dropna().iloc[-1])
    vs = variants(cfg, base)

    hist = pd.DataFrame([{"variant": n, **measure(prices, r, g, ppy)} for n, r, g in vs]).set_index("variant")

    seeds = np.random.SeedSequence(int(cfg["risk_bootstrap_seed"])).generate_state(int(cfg["risk_bootstrap_paths"]))
    jobs = [(d, int(s)) for d in cfg["risk_bootstrap_drifts"] for s in seeds]
    src = prices[["adj_open", "adj_high", "adj_low", "adj_close"]]
    with Pool(initializer=_init, initargs=(src, start, cfg, base, ppy)) as pool:
        rows = [r for chunk in pool.map(_one_path, jobs, chunksize=20) for r in chunk]
    boot = pd.DataFrame(rows)
    table = boot.groupby(["drift", "variant"]).apply(lambda df: agg(df, ppy), include_groups=False)

    OUT.mkdir(parents=True, exist_ok=True)
    flat = table.unstack("drift")
    flat.columns = [f"{d}_{m}" for m, d in flat.columns]
    hist.add_prefix("hist_").join(flat).to_csv(OUT / "variants.csv", float_format="%.4f")
    summary = {"dev_period": [str(prices.index[0].date()), str(end.date())], "bootstrap_start_price": start,
               "bootstrap_lot_jpy": base.lot_units * start, "capital_jpy": base.capital_jpy, "n_variants": len(vs),
               "n_paths": int(cfg["risk_bootstrap_paths"]), "days_per_path": int(cfg["risk_bootstrap_days"]),
               "block_mean": float(cfg["risk_bootstrap_block_mean"]),
               "history": hist.to_dict("index"),
               "bootstrap": {f"{d}|{v}": row.to_dict() for (d, v), row in table.iterrows()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return {"hist": hist, "table": table}


if __name__ == "__main__":
    res = main()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(res["hist"][["sharpe", "ann_return_on_capital", "avg_invested_share", "worst_unrealized_share", "max_drawdown_share",
                       "longest_hold_days", "longest_underwater_days", "n_losing_closed",
                       "const_ann_return_on_capital", "const_max_drawdown_share", "const_sharpe"]].round(3))
    for d in res["table"].index.get_level_values(0).unique():
        print("== drift", d)
        print(res["table"].loc[d].round(3))
