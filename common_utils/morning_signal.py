"""朝の合図（2026-10-01 ユーザー決定: 一定額保有をやめ、売買ルールの合図に戻す）。

ルールはバックテストのシミュレータ（common_utils.position_simulator）と同じ。予測モデルは使わない。
- 未保有の朝: 寄付・成行で1ロット買う。約定したら「約定価格 × (1+g)」に全口の売り指値を置く
- 保有中の朝: 「平均取得価格 × (1+g)」の売り指値を全口に置いておく
- 保有中で、前日終値 ≤ 平均取得価格 × (1 − add_on_drop) かつ上限未満の朝: 寄指（寄付だけ有効な指値）で、
  売り指値より1呼値安く1ロット買い増す。寄付が売り指値以上なら売りだけが約定して、買い増さない（シミュレータと同じ）。
  買い増しが約定したら、売り指値を新しい平均取得価格 × (1+g) に置き直す
価格はすべて生値（raw_*）。指値は呼値（tick_jpy）で切り上げる。判断日 D の朝には D より前の価格だけを使う。
保有状態は、自分で記入する約定記録（date, side, units, price）から作る。
建てた後に分割（corporate_actions.yaml）があったら、約定記録に「権利落ち日,split,比率,」の行を足す。その行で口数 × 比率・
平均取得価格 ÷ 比率に直す。行が無いあいだは合図を出さない。権利落ち日の朝は、前日終値が分割前の値か取得元（Yahoo）が
分割を反映した値か見分けられないので、合図を出さない（翌朝から再開）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalRules:
    lot_units: int
    max_units: int
    add_on_drop: float
    g: float
    tick_jpy: float | None
    loss_limit_jpy: float
    loss_warn_ratio: float

    @classmethod
    def from_config(cls, path=None) -> "SignalRules":
        from data.adjust import load_price_config
        cfg = load_price_config(path)
        v2, ops = cfg["v2"], cfg["ops"]
        return cls(lot_units=int(v2["position_lot_units"]), max_units=int(v2["position_max_units"]),
                   add_on_drop=float(v2["position_add_on_drop"]), g=float(ops["g"]),
                   tick_jpy=None if ops.get("tick_jpy") is None else float(ops["tick_jpy"]),
                   loss_limit_jpy=float(v2["fixed_loss_limit_jpy"]), loss_warn_ratio=float(ops["loss_warn_ratio"]))


def round_up_to_tick(price: float, tick: float | None) -> float:
    if tick is None:
        return float(price)
    return float(math.ceil(round(price / tick, 9)) * tick)


def load_fills(path) -> pd.DataFrame:
    f = pd.read_csv(path, comment="#", skipinitialspace=True)
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f["side"] = f["side"].str.strip().str.lower()
    if not f["side"].isin(["buy", "sell", "split"]).all():
        raise ValueError("side は buy・sell・split のどれか")
    return f[["date", "side", "units", "price"]].astype({"units": float, "price": float})


def position_state(fills: pd.DataFrame, before: pd.Timestamp, lot_units: int) -> dict:
    """before（判断日 D）より前の約定から、D の朝の保有状態を作る。同じ日の約定は記入順。全口売ったら未保有に戻る。
    split の行（units = 比率）は、それまでの口数を比率倍・平均取得価格を比率分の1にする。"""
    units, avg, entry, bought = 0.0, np.nan, pd.NaT, 0.0
    past = fills[fills["date"] < pd.Timestamp(before).normalize()].sort_values("date", kind="mergesort")
    for row in past.itertuples(index=False):
        q, px = float(row.units), float(row.price)
        if row.side == "split":
            if units > 0:
                units, avg = units * q, avg / q
            continue
        if row.side == "buy":
            if units <= 0:
                entry, avg, bought = row.date, px, 0.0
            else:
                avg = (units * avg + q * px) / (units + q)
            units += q
            bought += q / lot_units                       # 買った時点の口数で数える（分割で変わらない。シミュレータと同じ）
        else:
            units -= q
            if units <= 1e-9:
                units, avg, entry, bought = 0.0, np.nan, pd.NaT, 0.0
    lots = int(math.ceil(bought - 1e-9)) if units > 0 else 0
    return {"units": units, "avg_cost": avg, "lots": lots, "entry_date": entry}


def morning_orders(state: dict, prev_close: float, rules: SignalRules) -> dict:
    lot, g = rules.lot_units, rules.g
    units = float(state.get("units", 0.0))
    if units <= 0:
        return {"action": "buy", "buy_units": lot, "buy_type": "market_at_open",
                "after_fill_limit_estimate": round_up_to_tick(prev_close * (1 + g), rules.tick_jpy)}
    avg = float(state["avg_cost"])
    limit = round_up_to_tick(avg * (1 + g), rules.tick_jpy)
    room = int(state["lots"]) < rules.max_units // lot
    if prev_close <= avg * (1 - rules.add_on_drop) and room:
        new_avg = (units * avg + lot * prev_close) / (units + lot)
        return {"action": "add_on", "sell_units": units, "sell_limit": limit,
                "buy_units": lot, "buy_type": "limit_at_open_only", "buy_below": limit,
                "buy_limit": limit - rules.tick_jpy if rules.tick_jpy else limit,
                "after_fill_limit_estimate": round_up_to_tick(new_avg * (1 + g), rules.tick_jpy)}
    return {"action": "hold", "sell_units": units, "sell_limit": limit}


def build_signal(day: pd.Timestamp, prices: pd.DataFrame, fills: pd.DataFrame, rules: SignalRules,
                 sessions: pd.DatetimeIndex, splits: list) -> dict:
    """判断日 day の朝の合図。prices は raw_close を含む日次（day 以降の行は使わない）。sessions は東証の営業日。
    splits は corporate_actions.yaml の分割（{"date": 権利落ち日, "ratio": 比率}）。"""
    day = pd.Timestamp(day).normalize()
    out = {"date": day, "status": "ok", "orders": None, "prev_date": None, "prev_close": None,
           "state": None, "unrealized_jpy": None, "loss_warning": False, "loss_limit_price": None,
           "pending_splits": []}
    if day not in sessions:
        out["status"] = "holiday"
        return out
    past = prices.loc[prices.index < day, "raw_close"].dropna()
    prev_session = sessions[sessions < day]
    if past.empty or len(prev_session) == 0 or past.index[-1] != prev_session[-1]:
        out["status"] = "stale_data"
        out["prev_date"] = None if past.empty else past.index[-1]
        return out
    out["prev_date"], out["prev_close"] = past.index[-1], float(past.iloc[-1])
    if any(pd.Timestamp(a["date"]) == day for a in splits):
        out["status"] = "split_today"                                 # 前日終値の基準（分割前か後か）が分からない
        return out
    state = position_state(fills, day, rules.lot_units)
    out["state"] = state
    if state["units"] > 0:
        noted = set(fills.loc[fills["side"] == "split", "date"])
        pending = [a for a in splits if state["entry_date"] < pd.Timestamp(a["date"]) <= day
                   and pd.Timestamp(a["date"]) not in noted]
        if pending:
            out["status"] = "split_after_entry"
            out["pending_splits"] = pending
            return out
        unreal = state["units"] * (out["prev_close"] - state["avg_cost"])
        out["unrealized_jpy"] = unreal
        out["loss_warning"] = bool(unreal <= -rules.loss_warn_ratio * rules.loss_limit_jpy)
        out["loss_limit_price"] = state["avg_cost"] - rules.loss_limit_jpy / state["units"]
    out["orders"] = morning_orders(state, out["prev_close"], rules)
    return out


def _yen(x: float) -> str:
    return f"{x:,.0f}円"


def format_report(sig: dict, rules: SignalRules) -> str:
    d = sig["date"].strftime("%Y-%m-%d")
    lines = [f"# 2644.T 朝の合図 {d}", ""]
    if sig["status"] == "holiday":
        return "\n".join(lines + ["今日は東証の休場日。注文はなし。", ""])
    if sig["status"] == "stale_data":
        last = sig["prev_date"].strftime("%Y-%m-%d") if sig["prev_date"] is not None else "なし"
        return "\n".join(lines + [f"**合図を出さない**: 前営業日の値が取れていない（最新 {last}）。データを取り直してから実行する。", ""])
    if sig["status"] == "split_today":
        return "\n".join(lines + ["**合図を出さない**: 今日は分割の権利落ち日。前日終値が分割前か後か分からないので、新しい注文は出さない。",
                                   "証券会社に置いてある注文が分割で取り消されていないか確認する。保有中なら、ops/fills.csv に"
                                   f"「{sig['date'].strftime('%Y-%m-%d')},split,比率,」の行を足して、明日の朝から再開する。", ""])
    st = sig["state"]
    if sig["status"] == "split_after_entry":
        rows = [f"{pd.Timestamp(a['date']).strftime('%Y-%m-%d')},split,{float(a['ratio']):g}," for a in sig["pending_splits"]]
        return "\n".join(lines + ["**合図を出さない**: 建てた後に分割がある。ops/fills.csv の最後に次の行を足してから、もう一度実行する"
                                   "（口数と平均取得価格は自動で直る）。", ""] + [f"    {r}" for r in rows] + [""])
    o = sig["orders"]
    lines += [f"前日終値（{sig['prev_date'].strftime('%Y-%m-%d')}）: {_yen(sig['prev_close'])}"]
    if st["units"] > 0:
        limit_line = (f"含み損が限界の{_yen(rules.loss_limit_jpy)}に届く株価: {_yen(sig['loss_limit_price'])}"
                      if sig["loss_limit_price"] > 0 else
                      f"含み損は限界の{_yen(rules.loss_limit_jpy)}に届かない（株価が0円でも損は{_yen(st['units'] * st['avg_cost'])}）")
        lines += [f"保有: {st['units']:.0f}口（{st['lots']}ロット、上限 {rules.max_units // rules.lot_units}ロット）、"
                  f"平均取得価格 {_yen(st['avg_cost'])}、建てた日 {st['entry_date'].strftime('%Y-%m-%d')}",
                  f"含み損益（前日終値）: {sig['unrealized_jpy']:+,.0f}円。{limit_line}"]
    else:
        lines += ["保有: なし"]
    lines += ["", "## 今日の注文"]
    if o["action"] == "buy":
        lines += [f"1. 寄付・成行で **{o['buy_units']}口 買う**",
                  f"2. 約定したら、約定価格 × {1 + rules.g:.2f}（呼値で切り上げ）に **{o['buy_units']}口 の売り指値**を置く"
                  f"（前日終値なら {_yen(o['after_fill_limit_estimate'])}）"]
    elif o["action"] == "hold":
        lines += [f"1. **{o['sell_units']:.0f}口 を {_yen(o['sell_limit'])} で売り指値**（置いてあればそのまま）",
                  "2. 買い増しはしない"]
    else:
        lines += [f"1. **{o['sell_units']:.0f}口 を {_yen(o['sell_limit'])} で売り指値**（置いてあればそのまま）",
                  f"2. **寄指（寄付のみ）で {o['buy_units']}口 を {_yen(o['buy_limit'])} の買い指値**（買い増し。前日終値が平均取得価格の"
                  f"{1 - rules.add_on_drop:.0%}以下のため）",
                  f"3. 買い増しが約定したら、1の売り指値を取り消し、新しい平均取得価格 × {1 + rules.g:.2f}（呼値で切り上げ）で"
                  f"全口に置き直す（寄付が前日終値なら {_yen(o['after_fill_limit_estimate'])}）",
                  "   寄付が1の売り指値以上なら、売りが約定して買い増しは約定しない。それがルールどおり"]
    if sig["loss_warning"]:
        lines += ["", f"**警告**: 含み損が限界（{_yen(rules.loss_limit_jpy)}）の{rules.loss_warn_ratio:.0%}を超えた。"]
    lines += ["", "約定したら ops/fills.csv に記入する（date,side,units,price）。本ツールは情報提供であり投資助言ではない。", ""]
    return "\n".join(lines)
