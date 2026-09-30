"""時価評価シミュレータ（設計書 v2 評価設計、2026-09-30 確定の運用ルール）。

- 未保有の朝に BUY なら寄付で100口。利確指値は 平均取得価格 × (1+g) で全口まとめて置く。
- 保有中の朝、前日終値が 平均取得価格 × (1−4%) 以下なら寄付で100口買い増し（1日1回、上限300口）。
- 損切り・保有期限なし。未決済の含み損益も毎日の損益に入れる。
- 約定判定は調整済み価格。分割前の100口は調整済みで200口分として扱い、円の金額を実際と一致させる。
"""
import numpy as np
import pandas as pd
import pytest

from common_utils.position_simulator import PositionRules, simulate_positions

RULES = PositionRules(lot_units=100, max_units=300, add_on_drop=0.04, round_trip_cost=0.0, capital_jpy=1_500_000)


def _prices(opens, highs=None, lows=None, closes=None, raw_factor=1.0):
    n = len(opens)
    idx = pd.bdate_range("2025-01-06", periods=n, name="Date")
    o = pd.Series(opens, index=idx, dtype=float)
    h = pd.Series(highs if highs is not None else opens, index=idx, dtype=float)
    lo = pd.Series(lows if lows is not None else opens, index=idx, dtype=float)
    c = pd.Series(closes if closes is not None else opens, index=idx, dtype=float)
    return pd.DataFrame({"adj_open": o, "adj_high": h, "adj_low": lo, "adj_close": c, "raw_open": o * raw_factor})


def _signal(p, days):
    s = pd.Series(False, index=p.index)
    s.iloc[days] = True
    return s


def test_buy_and_sell_all_at_average_limit():
    p = _prices([100, 100, 100], highs=[100, 101, 102.5])
    daily, trades = simulate_positions(p, _signal(p, [0]), 0.02, RULES)
    t = trades.iloc[0]
    assert t["exit_price"] == pytest.approx(102.0) and t["holding_days"] == 3
    assert t["pnl_jpy"] == pytest.approx(100 * 2.0)
    assert daily["units"].iloc[-1] == 0


def test_add_rule_and_lot_cap():
    """前日終値が平均取得価格の4%以上下なら買い増し。1日1回、300口まで。指値は新しい平均取得価格×(1+g)。"""
    closes = [100, 95.9, 95, 91, 87, 80, 80]
    opens = [100, 97, 95, 92, 88, 81, 80]
    p = _prices(opens, highs=opens, lows=closes, closes=closes)
    daily, trades = simulate_positions(p, _signal(p, [0]), 0.02, RULES)
    # 1日目 100 で建て、2日目の朝は前日終値 100 → 買い増しなし、3日目の朝は前日終値 95.9 ≤ 96 → 95 で買い増し
    assert daily["units"].tolist()[:3] == [100, 100, 200]
    assert daily["avg_cost"].iloc[2] == pytest.approx(97.5)
    assert daily["limit"].iloc[2] == pytest.approx(97.5 * 1.02)
    # 4日目の朝: 前日終値 95 は 97.5×0.96=93.6 より上 → 買い増しなし。5日目の朝: 前日終値 91 ≤ 93.6 → 88 で3口目
    assert daily["units"].iloc[3] == 200 and daily["units"].iloc[4] == 300
    # 上限300口に達したので、以後は下がっても買わない
    assert daily["units"].max() == 300 and (daily["buys"] <= 1).all()


def test_mark_to_market_includes_open_position():
    p = _prices([100, 98, 95, 97], closes=[99, 97, 96.5, 96])     # 前日終値が 96 を下回らない → 買い増しなし
    daily, trades = simulate_positions(p, _signal(p, [0]), 0.10, RULES)
    assert trades["exit_date"].isna().all()                                  # 未決済のまま
    assert daily["unrealized_jpy"].iloc[-1] == pytest.approx(100 * (96 - 100))
    assert daily["pnl_jpy"].sum() == pytest.approx(100 * (96 - 100))         # 日次損益の合計 = 含み損益
    assert daily["ret"].iloc[1] == pytest.approx(100 * (97 - 99) / 1_500_000)


def test_gap_above_limit_sells_at_open_and_skips_add_on():
    p = _prices([100, 104, 104], highs=[100, 105, 104], lows=[100, 104, 104], closes=[100, 104, 104])
    daily, trades = simulate_positions(p, _signal(p, [0, 1, 2]), 0.02, RULES)
    assert trades.iloc[0]["exit_price"] == pytest.approx(104.0)             # 寄付で指値を超えた → 始値で約定
    assert trades.iloc[0]["exit_date"] == p.index[1]
    assert daily["buys"].iloc[1] == 0                                        # その朝は保有中だったので BUY しない
    assert daily["units"].iloc[2] == 100                                     # 翌朝は未保有 → BUY


def test_costs_are_half_round_trip_each_side():
    rules = PositionRules(lot_units=100, max_units=300, add_on_drop=0.04, round_trip_cost=0.001, capital_jpy=1_500_000)
    p = _prices([100, 100], highs=[100, 102])
    daily, trades = simulate_positions(p, _signal(p, [0]), 0.02, rules)
    assert trades.iloc[0]["cost_jpy"] == pytest.approx(100 * 100 * 0.0005 + 100 * 102 * 0.0005)
    assert daily["pnl_jpy"].sum() == pytest.approx(100 * 2 - trades.iloc[0]["cost_jpy"])


def test_split_lot_is_100_raw_units():
    """分割前（生値が調整済みの2倍）の100口は、調整済みでは200口分。円の金額は生値で見た実額と一致。"""
    p = _prices([50, 50, 51], highs=[50, 50, 51.5], raw_factor=2.0)
    daily, trades = simulate_positions(p, _signal(p, [0]), 0.02, RULES)
    assert daily["units"].iloc[0] == 200
    assert daily["invested_jpy"].iloc[0] == pytest.approx(100 * 100.0)     # 生値 100円 × 100口
    assert trades.iloc[0]["pnl_jpy"] == pytest.approx(200 * 1.0)             # 生値で 102−100 円 × 100口


def test_missing_session_is_skipped():
    p = _prices([100, np.nan, 100, 100], highs=[100, np.nan, 101, 103])
    daily, trades = simulate_positions(p, _signal(p, [0, 1]), 0.02, RULES)
    assert daily["buys"].iloc[1] == 0 and daily["pnl_jpy"].iloc[1] == 0.0
    assert trades.iloc[0]["exit_date"] == p.index[3]


def test_rules_from_config():
    r = PositionRules.from_config()
    assert (r.lot_units, r.max_units, r.add_on_drop, r.round_trip_cost, r.capital_jpy) == (100, 300, 0.04, 0.001, 1_500_000)
