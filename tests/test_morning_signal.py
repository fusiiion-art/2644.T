"""朝の合図（2026-10-01 ユーザー決定: 売買ルールの合図に戻す。上限300口・g=2%）のテスト。

- 約定記録（自分で記入する fills）から、判断日 D の朝の保有状態を作る。D 当日以降の約定は使わない
- 前日終値と保有状態から、今日の注文を出す（未保有なら寄付で買う、4%下なら寄指で買い増す、平均取得価格×(1+g) の売り指値）
- 先読み: D 以降の価格を渡しても結果が変わらない
- 休場・データが古い・建てた後の分割では、注文を出さない
- 合図どおりに売買すると、バックテストのシミュレータ（common_utils.position_simulator）と同じ取引になる
"""
import numpy as np
import pandas as pd
import pytest

from common_utils.morning_signal import (
    SignalRules, build_signal, morning_orders, position_state, round_up_to_tick,
)
from common_utils.position_simulator import PositionRules, simulate_positions

RULES = SignalRules(lot_units=100, max_units=300, add_on_drop=0.04, g=0.02, tick_jpy=1.0,
                    loss_limit_jpy=660_000, loss_warn_ratio=0.8)


def _fills(rows):
    return pd.DataFrame(rows, columns=["date", "side", "units", "price"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_position_state_from_fills_resets_after_full_sell_and_ignores_today():
    f = _fills([("2026-10-01", "buy", 100, 5000), ("2026-10-05", "buy", 100, 4800),
                ("2026-10-08", "sell", 200, 4998), ("2026-10-09", "buy", 100, 5100), ("2026-10-13", "buy", 100, 4890)])
    s = position_state(f, pd.Timestamp("2026-10-06"), RULES.lot_units)
    assert s["units"] == 200 and s["avg_cost"] == pytest.approx(4900) and s["lots"] == 2
    assert s["entry_date"] == pd.Timestamp("2026-10-01")
    s = position_state(f, pd.Timestamp("2026-10-13"), RULES.lot_units)              # 10-13 当日の約定はまだ無いものとして扱う
    assert s["units"] == 100 and s["avg_cost"] == pytest.approx(5100) and s["entry_date"] == pd.Timestamp("2026-10-09")
    assert position_state(f.iloc[:0], pd.Timestamp("2026-10-13"), RULES.lot_units)["units"] == 0


def test_round_up_to_tick():
    assert round_up_to_tick(5099.4, 1.0) == 5100
    assert round_up_to_tick(5100.0, 5.0) == 5100
    assert round_up_to_tick(5101.0, 5.0) == 5105
    assert round_up_to_tick(5099.4, None) == pytest.approx(5099.4)


def test_flat_morning_buys_at_open_and_estimates_limit():
    o = morning_orders({"units": 0}, prev_close=5000.0, rules=RULES)
    assert o["action"] == "buy"
    assert o["buy_units"] == 100 and o["buy_type"] == "market_at_open"
    assert o["after_fill_limit_estimate"] == round_up_to_tick(5000 * 1.02, 1.0)


def test_holding_without_add_on_keeps_sell_limit():
    o = morning_orders({"units": 100, "avg_cost": 5000.0, "lots": 1}, prev_close=4850.0, rules=RULES)   # 4850 > 4800
    assert o["action"] == "hold"
    assert o["sell_units"] == 100 and o["sell_limit"] == 5100


def test_holding_with_add_on_uses_opening_only_limit_below_sell_limit():
    o = morning_orders({"units": 100, "avg_cost": 5000.0, "lots": 1}, prev_close=4800.0, rules=RULES)   # 4800 ≤ 4800
    assert o["action"] == "add_on"
    assert o["sell_limit"] == 5100 and o["buy_units"] == 100
    assert o["buy_below"] == 5100 and o["buy_limit"] == 5099                  # 寄指: 売り指値より1呼値安い
    assert o["after_fill_limit_estimate"] == round_up_to_tick((100 * 5000 + 100 * 4800) / 200 * 1.02, 1.0)


def test_no_add_on_at_the_cap():
    o = morning_orders({"units": 300, "avg_cost": 5000.0, "lots": 3}, prev_close=4000.0, rules=RULES)
    assert o["action"] == "hold" and o["sell_units"] == 300


def _prices(dates, closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"adj_open": c, "adj_high": c, "adj_low": c, "adj_close": c, "raw_open": c, "raw_close": c},
                        index=pd.DatetimeIndex(dates, name="Date"))


SESSIONS = pd.DatetimeIndex(pd.bdate_range("2026-09-21", "2026-10-30"))


def test_signal_ignores_prices_on_or_after_the_decision_day():
    dates = pd.bdate_range("2026-09-21", "2026-10-09")
    p = _prices(dates, np.linspace(5000, 5200, len(dates)))
    d = pd.Timestamp("2026-10-05")
    later = p.copy()
    later.loc[later.index >= d] = 1.0                           # 当日以降を壊しても結果は同じ
    f = _fills([("2026-09-28", "buy", 100, 5050)])
    a = build_signal(d, p, f, RULES, SESSIONS, splits=[])
    b = build_signal(d, later, f, RULES, SESSIONS, splits=[])
    assert a["orders"] == b["orders"] and a["prev_close"] == b["prev_close"]
    assert a["prev_date"] == pd.Timestamp("2026-10-02")


def test_no_orders_on_holiday_stale_data_or_split_after_entry():
    dates = pd.bdate_range("2026-09-21", "2026-10-09")
    p = _prices(dates, np.full(len(dates), 5000.0))
    f = _fills([("2026-09-28", "buy", 100, 5000)])
    holiday = build_signal(pd.Timestamp("2026-10-04"), p, f, RULES, SESSIONS, splits=[])          # 日曜
    assert holiday["status"] == "holiday" and holiday["orders"] is None
    stale = build_signal(pd.Timestamp("2026-10-07"), p.loc[:"2026-10-02"], f, RULES, SESSIONS, splits=[])
    assert stale["status"] == "stale_data" and stale["orders"] is None
    split = build_signal(pd.Timestamp("2026-10-07"), p, f, RULES, SESSIONS, splits=[pd.Timestamp("2026-10-01")])
    assert split["status"] == "split_after_entry" and split["orders"] is None
    ok = build_signal(pd.Timestamp("2026-10-07"), p, f, RULES, SESSIONS, splits=[pd.Timestamp("2026-09-01")])
    assert ok["status"] == "ok" and ok["orders"]["action"] == "hold"


def test_loss_warning_near_the_limit():
    dates = pd.bdate_range("2026-09-21", "2026-10-09")
    p = _prices(dates, np.full(len(dates), 3000.0))
    f = _fills([("2026-09-21", "buy", 100, 5000), ("2026-09-22", "buy", 100, 4800), ("2026-09-23", "buy", 100, 4600)])
    s = build_signal(pd.Timestamp("2026-10-07"), p, f, RULES, SESSIONS, splits=[])
    assert s["unrealized_jpy"] == pytest.approx(300 * 3000 - (5000 + 4800 + 4600) * 100)     # −54万円
    assert s["loss_warning"] is True                                                           # 66万円の8割を超えた


def _paper_trade(prices, rules, sessions):
    """合図どおりに注文し、その日の始値・高値で約定させる（テスト用の仮の証券会社）。"""
    fills = _fills([])
    for d in prices.index[1:]:
        sig = build_signal(d, prices, fills, rules, sessions, splits=[])
        o = sig["orders"]
        bar = prices.loc[d]
        st = position_state(fills, d, rules.lot_units)
        new = []
        if o["action"] == "buy":
            new.append((d, "buy", o["buy_units"], bar["raw_open"]))
            limit = bar["raw_open"] * (1 + rules.g)
            if bar["raw_high"] >= limit:
                new.append((d, "sell", o["buy_units"], limit))
        else:
            if bar["raw_open"] >= o["sell_limit"]:
                new.append((d, "sell", o["sell_units"], bar["raw_open"]))
            else:
                units, avg = st["units"], st["avg_cost"]
                limit = o["sell_limit"]
                if o["action"] == "add_on" and bar["raw_open"] < o["buy_below"]:
                    new.append((d, "buy", o["buy_units"], bar["raw_open"]))
                    avg = (units * avg + o["buy_units"] * bar["raw_open"]) / (units + o["buy_units"])
                    units += o["buy_units"]
                    limit = avg * (1 + rules.g)
                if bar["raw_high"] >= limit:
                    new.append((d, "sell", units, limit))
        if new:
            fills = pd.concat([fills, _fills(new)], ignore_index=True)
    return fills


def test_following_the_signals_reproduces_the_backtest_simulator():
    rng = np.random.default_rng(0)
    n = 400
    c = 5000 * np.exp(np.cumsum(0.03 * rng.standard_normal(n)))
    o = np.r_[5000, c[:-1]] * np.exp(0.01 * rng.standard_normal(n))
    h = np.maximum(o, c) * (1 + 0.015 * rng.random(n))
    lo = np.minimum(o, c) * (1 - 0.015 * rng.random(n))
    idx = pd.bdate_range("2030-01-07", periods=n, name="Date")
    prices = pd.DataFrame({"adj_open": o, "adj_high": h, "adj_low": lo, "adj_close": c,
                           "raw_open": o, "raw_high": h, "raw_close": c}, index=idx)
    rules = SignalRules(lot_units=100, max_units=300, add_on_drop=0.04, g=0.02, tick_jpy=None,
                        loss_limit_jpy=660_000, loss_warn_ratio=0.8)
    fills = _paper_trade(prices, rules, idx)

    sim_rules = PositionRules(lot_units=100, max_units=300, add_on_drop=0.04, round_trip_cost=0.0, capital_jpy=1_500_000)
    signal = pd.Series(True, index=idx)
    signal.iloc[0] = False                                       # 合図は2日目の朝から（前日終値が要るため）
    _, trades = simulate_positions(prices, signal, 0.02, sim_rules)
    sells = fills[fills["side"] == "sell"].reset_index(drop=True)
    closed = trades.dropna(subset=["exit_date"]).reset_index(drop=True)
    assert len(closed) > 20 and (closed["lots"] == 3).any()       # 買い増し・上限まで含む十分な取引があること
    assert len(sells) == len(closed)
    assert (sells["date"].to_numpy() == closed["exit_date"].to_numpy()).all()
    assert np.allclose(sells["price"], closed["exit_price"])
    assert np.allclose(sells["units"], closed["units"])


def test_config_records_the_users_operation_decision():
    """2026-10-01 のユーザー決定: 売買ルールの合図に戻す。上限300口・g=2%・含み損の限界66万円。"""
    from data.adjust import load_price_config
    cfg = load_price_config()
    assert cfg["ops"]["mode"] == "rule"
    r = SignalRules.from_config()
    assert (r.lot_units, r.max_units, r.add_on_drop, r.g) == (100, 300, 0.04, 0.02)
    assert r.loss_limit_jpy == 660_000
    assert r.g in cfg["v2"]["phase4_baseline_g"]                # 検証済みの g から選んだ


def test_report_text_for_each_action():
    from common_utils.morning_signal import format_report
    dates = pd.bdate_range("2026-09-21", "2026-10-09")
    p = _prices(dates, np.full(len(dates), 4800.0))
    d = pd.Timestamp("2026-10-07")
    flat = format_report(build_signal(d, p, _fills([]), RULES, SESSIONS, splits=[]), RULES)
    assert "寄付・成行" in flat and "100口 買う" in flat
    add = format_report(build_signal(d, p, _fills([("2026-09-28", "buy", 100, 5000)]), RULES, SESSIONS, splits=[]), RULES)
    assert "寄指" in add and "5,099円" in add and "5,100円" in add
    hold = format_report(build_signal(d, p, _fills([("2026-09-28", "buy", 100, 4900)]), RULES, SESSIONS, splits=[]), RULES)
    assert "4,998円 で売り指値" in hold and "買い増しはしない" in hold
    assert "休場" in format_report(build_signal(pd.Timestamp("2026-10-04"), p, _fills([]), RULES, SESSIONS, splits=[]), RULES)
