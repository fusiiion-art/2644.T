"""方向 C（2026-10-01）: 運用ルールの含み損リスクを測る。ルールの変種は試さない（許容額をユーザーに聞いてから）。

1. 開発期間（dev_end まで）で「未保有の朝は必ず買う」を g = phase4_baseline_g で回し、上限到達・最長保有・
   最大含み損・水面下の期間・暦年ごとの損益を出す
2. 開発期間の最終日の生値を建値として、下落シナリオ（risk_scenario_drops）の含み損を円で出す
3. 3ロット満額に必要な金額が資金（capital_jpy）を超える価格を出す
出力: research/trial_log/rule_risk/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from common_utils.position_simulator import PositionRules, simulate_positions, summarize  # noqa: E402
from common_utils.rule_risk import averaging_down_levels, exposure_stats, scenario_loss  # noqa: E402
from generate.features_v2 import build_v2_dataset, v2_config  # noqa: E402

OUT = ROOT / "research" / "trial_log" / "rule_risk"


def main() -> dict:
    cfg = v2_config()
    rules = PositionRules.from_config()
    ppy = int(cfg["trading_days_per_year"])
    end = pd.Timestamp(cfg["dev_end"])
    prices = build_v2_dataset()["prices"].loc[:end]
    always = pd.Series(True, index=prices.index)

    by_g = {}
    for g in cfg["phase4_baseline_g"]:
        daily, trades = simulate_positions(prices, always, float(g), rules)
        st = exposure_stats(daily, trades, rules)
        st.update({k: summarize(daily, trades, ppy)[k] for k in ("sharpe", "ann_return_on_capital", "max_drawdown_jpy", "n_trades")})
        capped = trades[trades["lots"] >= rules.max_lots].copy()
        capped["worst_unrealized_jpy"] = [float(daily.loc[a:b, "unrealized_jpy"].min())
                                          for a, b in zip(capped["entry_date"], capped["exit_date"].fillna(daily.index[-1]))]
        capped["pnl_per_year_on_invested"] = capped["pnl_jpy"] / capped["max_invested_jpy"] / (capped["holding_days"] / ppy)
        st["capped_trades"] = capped.sort_values("worst_unrealized_jpy")[
            ["entry_date", "exit_date", "holding_days", "max_invested_jpy", "worst_low_ret", "worst_unrealized_jpy",
             "pnl_jpy", "pnl_per_year_on_invested"]].astype({"entry_date": str, "exit_date": str}).to_dict("records")
        yearly = daily["pnl_jpy"].groupby(daily.index.year).sum()
        st["pnl_by_year_jpy"] = {int(y): float(v) for y, v in yearly.items()}
        st["worst_unrealized_by_year_jpy"] = {int(y): float(v) for y, v in daily["unrealized_jpy"].groupby(daily.index.year).min().items()}
        by_g[f"g{g}"] = st

    price_now = float(prices["raw_close"].dropna().iloc[-1])
    levels = averaging_down_levels(rules)
    full_cost_factor = sum(lv["buy_at"] for lv in levels) * rules.lot_units
    scen = []
    for d in cfg["risk_scenario_drops"]:
        s = scenario_loss(price_now, float(d), rules)
        s["exit_level_vs_entry"] = {f"g{g}": s["avg_vs_entry"] * (1 + float(g)) for g in cfg["phase4_baseline_g"]}
        scen.append(s)
    summary = {
        "dev_period": [str(prices.index[0].date()), str(end.date())],
        "capital_jpy": rules.capital_jpy,
        "rule": {"lot_units": rules.lot_units, "max_units": rules.max_units, "add_on_drop": rules.add_on_drop,
                 "round_trip_cost": rules.round_trip_cost},
        "always_buy": by_g,
        "price_at_dev_end_raw": price_now,
        "averaging_down_levels": levels,
        "full_three_lots_cost_jpy_at_dev_end_price": full_cost_factor * price_now,
        "price_where_three_lots_exceed_capital": rules.capital_jpy / full_cost_factor,
        "raw_close_max_in_dev": float(prices["raw_close"].max()),
        "scenarios_at_dev_end_price": scen,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return summary


if __name__ == "__main__":
    s = main()
    for k, st in s["always_buy"].items():
        print(k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in st.items() if x not in ("capped_trades",)})
        for t in st["capped_trades"]:
            print("  ", t["entry_date"], t["exit_date"], t["holding_days"], round(t["max_invested_jpy"]), round(t["worst_low_ret"], 3),
                  round(t["worst_unrealized_jpy"]), round(t["pnl_jpy"]), round(t["pnl_per_year_on_invested"], 3))
    print("price", s["price_at_dev_end_raw"], "full cost", round(s["full_three_lots_cost_jpy_at_dev_end_price"]),
          "cap price", round(s["price_where_three_lots_exceed_capital"]), "max raw close", s["raw_close_max_in_dev"])
    for sc in s["scenarios_at_dev_end_price"]:
        print(sc["drop"], sc["lots"], round(sc["invested_jpy"]), round(sc["unrealized_jpy"]), round(sc["share_of_capital"], 3),
              {k: round(v, 4) for k, v in sc["exit_level_vs_entry"].items()})
