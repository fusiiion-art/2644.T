"""一定額保有の保有額を、ユーザーの含み損の限界（fixed_loss_limit_jpy）から逆算する（2026-10-01）。

保有額を選ぶのに成績は使わない。保有額1円あたりの「始めた時点からの最悪の損」L を測り、上限額 = 限界 ÷ |L| とする。
1. 実データ: 開発期間の中で開始日を fixed_rolling_start_step ごとにずらし、各開始日から開発期間の終わりまで回す
2. ブートストラップ: 建値 5,009円（開発期間の最終日の終値）から5年分 × risk_bootstrap_paths 本（トレンドなし／そのまま）
どちらも手元資金の制約は外して測る（買い足しが資金で止まると損はむしろ小さくなるので、保守的）。
毎月戻す方は、買い足しに必要な手元資金の最大値（保有額あたり）も出し、上限額で資金150万円に収まるかを確かめる。
出力: research/trial_log/fixed_amount/summary.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from common_utils.fixed_amount import fixed_amount_risk, simulate_fixed_amount  # noqa: E402
from common_utils.rule_risk import block_bootstrap_ohlc  # noqa: E402
from generate.features_v2 import build_v2_dataset, v2_config  # noqa: E402

OUT = ROOT / "research" / "trial_log" / "fixed_amount"
UNLIMITED = 1e3           # 手元資金の制約を外すための十分大きな資金（保有額は1円。買い足しは最大でも数円）
_STATE: dict = {}


def run_one(prices: pd.DataFrame, every, cost: float, ppy: int) -> dict:
    d = simulate_fixed_amount(prices, 1.0, UNLIMITED, cost, every)
    m = fixed_amount_risk(d, 1.0, ppy)
    m["max_topup_per_amount"] = float(max((d["cash_jpy"].iloc[0] - d["cash_jpy"]).max(), 0.0))
    return m


def _init(src, start, cfg):
    _STATE.update(src=src, start=start, cfg=cfg)


def _one_path(job):
    drift, seed = job
    c = _STATE["cfg"]
    path = block_bootstrap_ohlc(_STATE["src"], int(c["risk_bootstrap_days"]), float(c["risk_bootstrap_block_mean"]),
                                _STATE["start"], np.random.default_rng(seed), drift=drift)
    return [{"drift": drift, "method": name, **run_one(path, every, float(c["round_trip_cost"]), int(c["trading_days_per_year"]))}
            for name, every in c["fixed_methods"].items()]


def describe(df: pd.DataFrame, wait: int) -> dict:
    w = df["worst_below_start_per_amount"]
    return {"n": int(len(df)),
            "worst_loss_min": float(w.min()), "worst_loss_p01": float(w.quantile(0.01)),
            "worst_loss_p05": float(w.quantile(0.05)), "worst_loss_median": float(w.median()),
            "p_below_start_over_wait": float((df["longest_below_start_days"] > wait).mean()),
            "p_underwater_over_wait": float((df["longest_underwater_days"] > wait).mean()),
            "ann_return_median": float(df["ann_return_on_amount"].median()),
            "final_pnl_p05": float(df["final_pnl_per_amount"].quantile(0.05)),
            "max_topup_p95": float(df["max_topup_per_amount"].quantile(0.95)),
            "max_topup_max": float(df["max_topup_per_amount"].max())}


def main() -> dict:
    cfg = v2_config()
    from data.adjust import load_price_config
    capital = float(load_price_config()["capital_jpy"])
    ppy = int(cfg["trading_days_per_year"])
    cost = float(cfg["round_trip_cost"])
    limit = float(cfg["fixed_loss_limit_jpy"])
    wait = int(cfg["fixed_wait_limit_days"])
    end = pd.Timestamp(cfg["dev_end"])
    prices = build_v2_dataset()["prices"].loc[:end, ["adj_open", "adj_high", "adj_low", "adj_close", "raw_close"]]
    prices = prices.dropna(subset=["adj_open", "adj_high", "adj_low", "adj_close"])   # 2644 の値が無い日（開発期間に1日）

    # 1. 実データ（開始日をずらす）
    rows = []
    last = len(prices) - int(cfg["fixed_rolling_min_days"])
    for s in range(0, last + 1, int(cfg["fixed_rolling_start_step"])):
        seg = prices.iloc[s:]
        for name, every in cfg["fixed_methods"].items():
            rows.append({"start": str(seg.index[0].date()), "method": name, **run_one(seg, every, cost, ppy)})
    hist = pd.DataFrame(rows)

    # 2. ブートストラップ
    start = float(prices["raw_close"].dropna().iloc[-1])
    seeds = np.random.SeedSequence(int(cfg["risk_bootstrap_seed"])).generate_state(int(cfg["risk_bootstrap_paths"]))
    jobs = [(d, int(s)) for d in cfg["risk_bootstrap_drifts"] for s in seeds]
    with Pool(initializer=_init, initargs=(prices[["adj_open", "adj_high", "adj_low", "adj_close"]], start, cfg)) as pool:
        boot = pd.DataFrame([r for chunk in pool.map(_one_path, jobs, chunksize=20) for r in chunk])

    out = {"capital_jpy": capital, "loss_limit_jpy": limit, "wait_limit_days": wait, "bootstrap_start_price": start,
           "dev_period": [str(prices.index[0].date()), str(end.date())], "methods": {}}
    for name in cfg["fixed_methods"]:
        h = describe(hist[hist["method"] == name], wait)
        worst_start = hist[hist["method"] == name].sort_values("worst_below_start_per_amount").iloc[0]["start"]
        b = {d: describe(boot[(boot["method"] == name) & (boot["drift"] == d)], wait) for d in cfg["risk_bootstrap_drifts"]}
        stresses = {"history_worst": h["worst_loss_min"],
                    **{f"{d}_p05": b[d]["worst_loss_p05"] for d in b}, **{f"{d}_p01": b[d]["worst_loss_p01"] for d in b}}
        sizing = {k: float(min(limit / abs(v), capital)) if v < 0 else capital for k, v in stresses.items()}
        topup_ok = {k: bool(a * (1 + max(h["max_topup_max"], *(b[d]["max_topup_p95"] for d in b))) <= capital)
                    for k, a in sizing.items()}
        # 候補の保有額ごとに、損が限界を超える割合（実データは開始日の割合、ブートストラップはパスの割合）
        cands = sorted({limit, *sizing.values()})
        hm = hist[hist["method"] == name]["worst_below_start_per_amount"]
        exceed = {f"{a:.0f}": {"history_starts": float((hm * a < -limit).mean()),
                               **{d: float((boot[(boot["method"] == name) & (boot["drift"] == d)]["worst_below_start_per_amount"] * a
                                            < -limit).mean()) for d in b}} for a in cands}
        out["methods"][name] = {"history": h, "history_worst_start": worst_start, "bootstrap": b,
                                "loss_per_amount": stresses, "max_amount_jpy": sizing, "fits_capital_with_topups": topup_ok,
                                "p_loss_exceeds_limit_by_amount": exceed}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))
    hist.to_csv(OUT / "rolling_starts.csv", index=False, float_format="%.5f")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=1, default=float))
