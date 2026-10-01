"""一定額保有（2026-10-01 ユーザー決定。買い増しルールから切り替え）のシミュレータとリスク指標。

- 初日の寄付で amount 円分を買う。手元資金は capital − amount − 手数料
- rebalance_every = None: 買ったまま持つ。整数 N: N 営業日ごとの寄付で評価額を amount に戻す
  （上がっていれば売り、下がっていれば手元資金の範囲で買い足す。手元資金は負にならない）
- 手数料は売買代金 × round_trip_cost / 2。配当は入れない（data/adjust.py は配当を調整しないので保守的）
- 価格は調整済み（adj_*）。分割前は口数が調整済みの単位になるだけで、円の金額は実額と同じ
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_fixed_amount(prices: pd.DataFrame, amount: float, capital: float, round_trip_cost: float,
                          rebalance_every: int | None) -> pd.DataFrame:
    o = prices["adj_open"].to_numpy(dtype=float)
    c = prices["adj_close"].to_numpy(dtype=float)
    if not (np.isfinite(o).all() and np.isfinite(c).all()):
        raise ValueError("adj_open / adj_close に欠損がある。呼び出し側で欠損日を除くこと")
    half = round_trip_cost / 2.0
    n = len(o)
    units = np.zeros(n)
    cash = np.zeros(n)
    trade = np.zeros(n)
    u = amount / o[0]
    cs = capital - amount - amount * half
    trade[0] = amount
    for i in range(n):
        if i > 0 and rebalance_every and i % int(rebalance_every) == 0:
            diff = amount - u * o[i]
            if diff > 0:
                diff = min(diff, max(cs, 0.0) / (1 + half))
            u += diff / o[i]
            cs -= diff + abs(diff) * half
            trade[i] = diff
        units[i], cash[i] = u, cs
    d = pd.DataFrame({"units": units, "value_jpy": units * c, "cash_jpy": cash, "trade_jpy": trade},
                     index=prices.index)
    d["equity_jpy"] = d["value_jpy"] + d["cash_jpy"] - capital
    d["pnl_jpy"] = d["equity_jpy"].diff().fillna(d["equity_jpy"])
    return d.astype("float64")


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return int(best)


def fixed_amount_risk(daily: pd.DataFrame, amount: float, periods_per_year: int) -> dict:
    """損益を保有額（amount）あたりで表す。ピークは始点の0から数える。"""
    eq = daily["equity_jpy"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    pnl = daily["pnl_jpy"].to_numpy(dtype=float)
    sd = pnl.std(ddof=1) if len(pnl) > 1 else np.nan
    return {
        "worst_below_start_per_amount": float(min(eq.min(), 0.0) / amount),
        "max_drawdown_per_amount": float((eq - peak).min() / amount),
        "longest_below_start_days": _longest_run(eq < 0),
        "longest_underwater_days": _longest_run(eq < peak),
        "final_pnl_per_amount": float(eq[-1] / amount),
        "ann_return_on_amount": float(pnl.mean() / amount * periods_per_year),
        "sharpe": float(pnl.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else np.nan,
    }
