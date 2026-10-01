"""運用ルール（毎朝買う・4%買い増し・上限300口・平均取得価格×(1+g) の指値）の含み損リスクを測る。

方向 C（2026-10-01 ユーザー了承）。ルールは変えず、測るだけ。
- averaging_down_levels: 買い増しがちょうど発動価格で約定したときの、各ロットの買値と平均取得価格（建値=1）
- scenario_loss: 建値からの下落率 d で、届いた分だけ買い増したときの投入額と含み損（円）
- exposure_stats: シミュレータの日次・取引記録から、上限到達日数・最長保有・最大含み損・水面下の最長日数など
- block_bootstrap_ohlc: 日次の 始値・高値・安値・終値 ÷ 前日終値 を、定常ブロック・ブートストラップ（Politis–Romano）で
  並べ替えて架空の値動きを作る。drift="zero" なら終値の対数リターンの平均を引いて上昇トレンドを除く
- constant_exposure_daily: 比較用。資金の一定割合の金額を毎日持ち続けたときの日次の時価評価
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common_utils.position_simulator import PositionRules


def averaging_down_levels(rules: PositionRules) -> list[dict]:
    levels = [{"lots": 1, "buy_at": 1.0, "avg": 1.0}]
    while levels[-1]["lots"] < rules.max_lots:
        prev = levels[-1]
        buy_at = prev["avg"] * (1 - rules.add_on_drop)
        n = prev["lots"] + 1
        levels.append({"lots": n, "buy_at": buy_at, "avg": (prev["avg"] * prev["lots"] + buy_at) / n})
    return levels


def scenario_loss(entry_price: float, drop: float, rules: PositionRules) -> dict:
    """建値 entry_price（生値）から drop だけ下がった時点の含み損。買い増しは発動価格で約定したとみなす。"""
    filled = [lv for lv in averaging_down_levels(rules) if lv["buy_at"] >= 1 - drop - 1e-12]
    last = filled[-1]
    units = rules.lot_units * last["lots"]
    invested = sum(rules.lot_units * lv["buy_at"] * entry_price for lv in filled)
    unreal = units * entry_price * (1 - drop) - invested
    avg = last["avg"]
    return {"drop": float(drop), "lots": int(last["lots"]), "units": int(units), "invested_jpy": float(invested),
            "unrealized_jpy": float(unreal), "share_of_capital": float(unreal / rules.capital_jpy),
            "avg_vs_entry": float(avg)}           # 利確に必要な価格は 建値 × avg_vs_entry × (1 + g)


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return int(best)


def exposure_stats(daily: pd.DataFrame, trades: pd.DataFrame, rules: PositionRules) -> dict:
    eq = daily["equity_jpy"]
    at_max = (daily["lots"] >= rules.max_lots).to_numpy()
    return {
        "days": int(len(daily)),
        "days_at_max_lots": int(at_max.sum()),
        "share_days_at_max_lots": float(at_max.mean()),
        "longest_run_at_max_lots": _longest_run(at_max),
        "worst_unrealized_jpy": float(daily["unrealized_jpy"].min()),
        "worst_unrealized_date": str(daily["unrealized_jpy"].idxmin().date()),
        "worst_unrealized_share_of_capital": float(daily["unrealized_jpy"].min() / rules.capital_jpy),
        "max_invested_jpy": float(daily["invested_jpy"].max()),
        "max_invested_share_of_capital": float(daily["invested_jpy"].max() / rules.capital_jpy),
        "longest_hold_days": int(trades["holding_days"].max()) if len(trades) else 0,
        "trades_reaching_max_lots": int((trades["lots"] >= rules.max_lots).sum()),
        "longest_underwater_days": _longest_run((eq < eq.cummax()).to_numpy()),
    }


def block_bootstrap_ohlc(src: pd.DataFrame, n_days: int, block_mean: float, start_price: float,
                         rng: np.random.Generator, drift: str = "asis") -> pd.DataFrame:
    """src は adj_open/high/low/close。返り値は架空の営業日 index で、分割が無いので raw_* = adj_*。"""
    prev = src["adj_close"].shift()
    ratios = (src[["adj_open", "adj_high", "adj_low", "adj_close"]].div(prev, axis=0)).dropna().to_numpy(dtype=float)
    if drift == "zero":
        ratios = ratios / np.exp(np.log(ratios[:, 3]).mean())
    elif drift != "asis":
        raise ValueError(f"drift は asis か zero: {drift}")
    n_src = len(ratios)
    jumps = rng.random(n_days) < 1.0 / float(block_mean)
    jumps[0] = True
    starts = rng.integers(0, n_src, n_days)
    t = np.arange(n_days)
    seg_start = np.maximum.accumulate(np.where(jumps, t, 0))
    idx = (starts[seg_start] + (t - seg_start)) % n_src
    r = ratios[idx]
    close = start_price * np.cumprod(r[:, 3])
    prev_c = np.r_[start_price, close[:-1]]
    out = pd.DataFrame({"adj_open": prev_c * r[:, 0], "adj_high": prev_c * r[:, 1], "adj_low": prev_c * r[:, 2],
                        "adj_close": close}, index=pd.bdate_range("2100-01-04", periods=n_days, name="Date"))
    out["raw_open"] = out["adj_open"]
    out["raw_close"] = out["adj_close"]
    return out



def constant_exposure_daily(prices: pd.DataFrame, rules: PositionRules, share: float) -> pd.DataFrame:
    """比較用: 初日の寄付から、資金 × share の金額を毎日持ち続ける（毎日その金額にそろえる）。
    コストは初日の建て（片道）だけで、そろえるための売買コストは含めない（そのぶん有利な比較になる）。"""
    amount = rules.capital_jpy * float(share)
    c = prices["adj_close"].astype(float)
    r = (c / c.shift()).fillna(c.iloc[0] / float(prices["adj_open"].iloc[0])) - 1
    pnl = amount * r
    pnl.iloc[0] -= amount * rules.round_trip_cost / 2
    d = pd.DataFrame({"units": amount / c, "lots": 1.0, "invested_jpy": amount, "pnl_jpy": pnl}, index=prices.index)
    d["equity_jpy"] = d["pnl_jpy"].cumsum()
    d["unrealized_jpy"] = d["equity_jpy"]
    d["ret"] = d["pnl_jpy"] / rules.capital_jpy
    return d.astype("float64")
