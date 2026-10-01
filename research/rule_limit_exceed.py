"""売買ルールの上限口数ごとに、含み損が限界（fixed_loss_limit_jpy）を超える割合を出す（2026-10-01、上限口数の判断材料）。

ブートストラップの作り方は research/rule_variants_report.py と同じ（開発期間の 2644 の日次の値動きを平均 risk_bootstrap_block_mean
営業日のブロックで並べ替え、建値は開発期間の最終日の終値、risk_bootstrap_days 日 × risk_bootstrap_paths 本、同じ乱数の種）。
1ロットは「position_lot_units 口 × 建値」の金額に固定する。保有期限なし、未保有の朝は必ず買う。
変種は上限 risk_variant_max_lots × g phase4_baseline_g（いずれも rule_variants で計上済みなので、試行数は増えない）。
出力: research/trial_log/rule_limit/summary.json
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

from common_utils.position_simulator import PositionRules, simulate_positions  # noqa: E402
from common_utils.rule_risk import block_bootstrap_ohlc  # noqa: E402
from generate.features_v2 import build_v2_dataset, v2_config  # noqa: E402

OUT = ROOT / "research" / "trial_log" / "rule_limit"
_S: dict = {}


def _init(src, start, cfg, base):
    _S.update(src=src, start=start, cfg=cfg, base=base)


def _one(job):
    drift, seed = job
    c, base = _S["cfg"], _S["base"]
    path = block_bootstrap_ohlc(_S["src"], int(c["risk_bootstrap_days"]), float(c["risk_bootstrap_block_mean"]),
                                _S["start"], np.random.default_rng(seed), drift=drift)
    rows = []
    for lots in c["risk_variant_max_lots"]:
        r = replace(base, max_units=int(lots) * base.lot_units, lot_jpy=base.lot_units * _S["start"])
        for g in c["phase4_baseline_g"]:
            d, _ = simulate_positions(path, pd.Series(True, index=path.index), float(g), r)
            rows.append({"drift": drift, "lots": int(lots), "g": float(g), "worst_unrealized_jpy": float(d["unrealized_jpy"].min())})
    return rows


def main() -> dict:
    cfg = v2_config()
    base = PositionRules.from_config()
    limit = float(cfg["fixed_loss_limit_jpy"])
    prices = build_v2_dataset()["prices"].loc[:pd.Timestamp(cfg["dev_end"])]
    start = float(prices["raw_close"].dropna().iloc[-1])
    seeds = np.random.SeedSequence(int(cfg["risk_bootstrap_seed"])).generate_state(int(cfg["risk_bootstrap_paths"]))
    jobs = [(d, int(s)) for d in cfg["risk_bootstrap_drifts"] for s in seeds]
    with Pool(initializer=_init, initargs=(prices[["adj_open", "adj_high", "adj_low", "adj_close"]], start, cfg, base)) as pool:
        boot = pd.DataFrame([r for chunk in pool.map(_one, jobs, chunksize=20) for r in chunk])
    out = {"loss_limit_jpy": limit, "start_price": start, "lot_jpy": base.lot_units * start, "variants": []}
    for lots in cfg["risk_variant_max_lots"]:
        for g in cfg["phase4_baseline_g"]:
            hist, _ = simulate_positions(prices, pd.Series(True, index=prices.index), float(g),
                                         replace(base, max_units=int(lots) * base.lot_units))
            row = {"max_units": int(lots) * base.lot_units, "g": float(g),
                   "history_worst_unrealized_jpy": float(hist["unrealized_jpy"].min())}
            for d in cfg["risk_bootstrap_drifts"]:
                w = boot[(boot["drift"] == d) & (boot["lots"] == int(lots)) & (boot["g"] == float(g))]["worst_unrealized_jpy"]
                row[f"{d}_p_exceed_limit"] = float((w < -limit).mean())
                row[f"{d}_worst_unrealized_p05_jpy"] = float(w.quantile(0.05))
            out["variants"].append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    for v in main()["variants"]:
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
