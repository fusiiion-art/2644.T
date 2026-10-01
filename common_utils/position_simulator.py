"""時価評価のポジション・シミュレータ（設計書 v2 評価設計、2026-09-30 に確定した運用ルール）。

1日の流れ（行 D = 判断日）:
- 朝（判断時刻）: 未保有で BUY なら寄付で1ロット（100口）を建て、利確指値 = 平均取得価格 × (1 + g) を置く。
  保有中なら、前日終値が 平均取得価格 × (1 − add_on_drop) 以下で上限未満のとき、寄付で1ロット買い増す
  （1日1回まで。指値は新しい平均取得価格 × (1 + g) で全口まとめて置き直す。g は建てた日の値を保つ）。
- 場中: 始値が指値以上なら始値で、日中の高値が指値に届けば指値で全口を売る（寄付で指値を超えた日は買い増さない）。
- 損切りはない。保有期限も現行ルールにはない（max_hold_days = None）。方向 C の変種として max_hold_days を
  与えると、建てた日から数えてその日数を持った次の朝に、寄付で全口を売る（その朝は買い増ししない）。
- lot_jpy を与えると1ロットを金額で決める（方向 C のブートストラップ用。現行ルールは口数固定で None）。
- 毎日の損益は未決済分の含み損益の変化も含む（時価評価）。
- 約定判定は調整済み価格（adj_*）。1ロットは生の口数で数え、調整済みの口数は lot × raw_open / adj_open
  （分割前の100口 = 調整済み200口分）。これで円の金額は実額と一致する。
- コストは往復コスト c を建て・手仕舞いの約定代金に半分ずつ課す。
パラメータは semi2644/config/config.yaml（capital_jpy と v2 節の position_* / round_trip_cost）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PositionRules:
    lot_units: int
    max_units: int
    add_on_drop: float
    round_trip_cost: float
    capital_jpy: float
    max_hold_days: int | None = None
    lot_jpy: float | None = None          # 変種のみ: 1ロットを金額で決める（口数 = lot_jpy ÷ 寄付の価格）

    @property
    def max_lots(self) -> int:
        return int(self.max_units // self.lot_units)

    @classmethod
    def from_config(cls, path: Path | None = None) -> "PositionRules":
        from data.adjust import load_price_config
        cfg = load_price_config(path)
        v2 = cfg["v2"]
        return cls(lot_units=int(v2["position_lot_units"]), max_units=int(v2["position_max_units"]),
                   add_on_drop=float(v2["position_add_on_drop"]), round_trip_cost=float(v2["round_trip_cost"]),
                   capital_jpy=float(cfg["capital_jpy"]),
                   max_hold_days=None if v2.get("position_max_hold_days") is None else int(v2["position_max_hold_days"]))


def simulate_positions(prices: pd.DataFrame, buy_signal: pd.Series, g, rules: PositionRules):
    """prices（adj_open/high/low/close と raw_open、東京営業日の index）と朝の BUY シグナルから、
    日次の時価評価（daily）と売買1往復ごとの記録（trades）を返す。"""
    idx = prices.index
    o = prices["adj_open"].to_numpy(dtype=float)
    h = prices["adj_high"].to_numpy(dtype=float)
    lo = prices["adj_low"].to_numpy(dtype=float)
    c = prices["adj_close"].to_numpy(dtype=float)
    raw_o = prices["raw_open"].to_numpy(dtype=float)
    sig = buy_signal.reindex(idx).fillna(False).astype(bool).to_numpy()
    g_arr = g.reindex(idx).to_numpy(dtype=float) if isinstance(g, pd.Series) else np.full(len(idx), float(g))
    half_cost = rules.round_trip_cost / 2.0

    units = 0.0
    avg = np.nan
    limit = np.nan
    g_pos = np.nan
    lots = 0
    realized = 0.0
    cost_cum = 0.0
    last_close = np.nan
    trade = None
    rows, trades = [], []

    def buy(i):
        nonlocal units, avg, lots, cost_cum
        q = rules.lot_jpy / o[i] if rules.lot_jpy is not None else rules.lot_units * raw_o[i] / o[i]
        avg = o[i] if units == 0 else (units * avg + q * o[i]) / (units + q)
        units += q
        lots += 1
        fee = q * o[i] * half_cost
        cost_cum += fee
        trade["cost_jpy"] += fee
        trade["lots"] = lots
        trade["max_invested_jpy"] = max(trade["max_invested_jpy"], units * avg)

    def sell_all(i, price):
        nonlocal units, avg, lots, limit, realized, cost_cum, trade
        gross = units * (price - avg)
        fee = units * price * half_cost
        realized += gross
        cost_cum += fee
        trade["cost_jpy"] += fee
        trade.update(exit_date=idx[i], exit_price=price, avg_cost=avg, units=units,
                     pnl_jpy=gross - trade["cost_jpy"], holding_days=i - trade["_entry_i"] + 1)
        trades.append(trade)
        trade = None
        units, avg, lots, limit = 0.0, np.nan, 0, np.nan

    for i in range(len(idx)):
        buys = sells = 0
        valid = np.isfinite(o[i]) and np.isfinite(h[i]) and np.isfinite(lo[i]) and np.isfinite(c[i]) and np.isfinite(raw_o[i])
        if valid:
            if units > 0:
                if o[i] >= limit:                                   # 寄付で指値を超えた
                    sell_all(i, o[i])
                    sells = 1
                elif rules.max_hold_days is not None and i - trade["_entry_i"] >= rules.max_hold_days:
                    sell_all(i, o[i])                               # 保有期限（変種のみ）: 寄付で全口
                    sells = 1
                else:
                    if np.isfinite(last_close) and last_close <= avg * (1 - rules.add_on_drop) and lots < rules.max_lots:
                        buy(i)
                        buys = 1
                        limit = avg * (1 + g_pos)
                    trade["worst_low_ret"] = min(trade["worst_low_ret"], lo[i] / avg - 1)
                    if h[i] >= limit:
                        sell_all(i, limit)
                        sells = 1
            elif sig[i] and np.isfinite(g_arr[i]) and g_arr[i] > 0:
                g_pos = g_arr[i]
                trade = {"entry_date": idx[i], "exit_date": pd.NaT, "exit_price": np.nan, "lots": 0, "units": 0.0,
                         "avg_cost": np.nan, "pnl_jpy": np.nan, "holding_days": np.nan, "cost_jpy": 0.0,
                         "max_invested_jpy": 0.0, "worst_low_ret": 0.0, "g": g_pos, "_entry_i": i}
                buy(i)
                buys = 1
                limit = avg * (1 + g_pos)
                trade["worst_low_ret"] = min(trade["worst_low_ret"], lo[i] / avg - 1)
                if h[i] >= limit:
                    sell_all(i, limit)
                    sells = 1
            last_close = c[i]
        unreal = units * (last_close - avg) if units > 0 else 0.0
        rows.append({"units": units, "lots": lots, "avg_cost": avg, "limit": limit,
                     "invested_jpy": units * avg if units > 0 else 0.0, "unrealized_jpy": unreal,
                     "realized_jpy": realized, "cost_jpy": cost_cum, "equity_jpy": realized + unreal - cost_cum,
                     "buys": buys, "sells": sells,
                     "holding_days": (i - trade["_entry_i"] + 1) if trade is not None else 0})

    daily = pd.DataFrame(rows, index=idx).astype("float64")
    daily["pnl_jpy"] = daily["equity_jpy"].diff().fillna(daily["equity_jpy"])
    daily["ret"] = daily["pnl_jpy"] / rules.capital_jpy
    if trade is not None:                                            # 未決済（時価評価のまま残す）
        trade.update(avg_cost=avg, units=units, holding_days=len(idx) - trade["_entry_i"])
        trades.append(trade)
    cols = ["entry_date", "exit_date", "exit_price", "lots", "units", "avg_cost", "pnl_jpy", "cost_jpy",
            "holding_days", "max_invested_jpy", "worst_low_ret", "g"]
    return daily, pd.DataFrame([{k: t[k] for k in cols} for t in trades], columns=cols)


def summarize(daily: pd.DataFrame, trades: pd.DataFrame, periods_per_year: int = 245) -> dict:
    """時価評価の曲線からシャープ・最大ドローダウン・保有日数・最大口数・最大投入額などを出す。"""
    r = daily["ret"]
    eq = daily["equity_jpy"]
    closed = trades.dropna(subset=["exit_date"])
    return {
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else np.nan,
        "ann_return_on_capital": float(r.mean() * periods_per_year),
        "max_drawdown_jpy": float((eq - eq.cummax()).min()),
        "final_equity_jpy": float(eq.iloc[-1]),
        "n_trades": int(len(trades)),
        "n_closed": int(len(closed)),
        "win_rate_closed": float((closed["pnl_jpy"] > 0).mean()) if len(closed) else np.nan,
        "avg_holding_days": float(trades["holding_days"].mean()) if len(trades) else np.nan,
        "max_lots": int(daily["lots"].max()),
        "max_invested_jpy": float(daily["invested_jpy"].max()),
        "worst_unrealized_jpy": float(daily["unrealized_jpy"].min()),
        "days_in_market": int((daily["units"] > 0).sum()),
    }
