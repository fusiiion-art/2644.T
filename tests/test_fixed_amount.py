"""一定額保有（2026-10-01 ユーザー決定で買い増しルールから切り替え）のシミュレータのテスト。

- 初日の寄付で amount 円分を買う。手元資金は capital − amount − 手数料
- rebalance_every = None なら買ったまま持つ。整数 N なら N 営業日ごとの寄付で、評価額を amount に戻す
  （上がっていれば売り、下がっていれば手元資金の範囲で買い足す。手元資金は負にならない）
- 手数料は売買代金 × 往復コスト / 2
- equity_jpy は 評価額 + 手元資金 − capital（始めた時点からの損益）
"""
import numpy as np
import pandas as pd
import pytest

from common_utils.fixed_amount import fixed_amount_risk, simulate_fixed_amount


def _p(opens, closes):
    idx = pd.bdate_range("2025-01-06", periods=len(opens), name="Date")
    return pd.DataFrame({"adj_open": opens, "adj_close": closes}, index=idx, dtype=float)


def test_hold_buys_once_and_keeps_units():
    d = simulate_fixed_amount(_p([100, 90, 80], [95, 85, 120]), amount=1000, capital=2000,
                              round_trip_cost=0.002, rebalance_every=None)
    assert (d["units"] == 10).all()
    assert d["cash_jpy"].iloc[-1] == pytest.approx(999)
    assert d["equity_jpy"].tolist() == pytest.approx([-51, -151, 199])
    assert d["pnl_jpy"].sum() == pytest.approx(d["equity_jpy"].iloc[-1])


def test_rebalance_restores_amount_at_open_with_fee():
    d = simulate_fixed_amount(_p([100, 90, 50, 60], [100, 80, 55, 60]), amount=1000, capital=2000,
                              round_trip_cost=0.002, rebalance_every=2)
    assert d["units"].tolist() == pytest.approx([10, 10, 20, 20])
    assert d["cash_jpy"].iloc[2] == pytest.approx(999 - 500 - 0.5)
    assert d["equity_jpy"].iloc[2] == pytest.approx(20 * 55 + 498.5 - 2000)
    assert d["trade_jpy"].tolist() == pytest.approx([1000, 0, 500, 0])


def test_rebalance_sells_when_above_amount():
    d = simulate_fixed_amount(_p([100, 200], [100, 200]), amount=1000, capital=2000,
                              round_trip_cost=0.002, rebalance_every=1)
    assert d["units"].iloc[1] == pytest.approx(5)
    assert d["cash_jpy"].iloc[1] == pytest.approx(999 + 1000 - 1)
    assert d["equity_jpy"].iloc[1] == pytest.approx(998)


def test_rebalance_never_spends_more_cash_than_it_has():
    d = simulate_fixed_amount(_p([100, 10, 10], [100, 10, 10]), amount=1000, capital=1200,
                              round_trip_cost=0.002, rebalance_every=1)
    assert (d["cash_jpy"] >= -1e-9).all()
    assert d["cash_jpy"].iloc[1] == pytest.approx(0, abs=1e-9)
    assert d["units"].iloc[1] == pytest.approx(10 + 199 / 1.001 / 10)
    assert d["equity_jpy"].iloc[-1] >= -1200 - 1e-9          # 損は資金を超えない


def test_risk_metrics_per_amount():
    idx = pd.bdate_range("2025-01-06", periods=6)
    d = pd.DataFrame({"equity_jpy": [-10.0, -30, 5, 20, 8, 25]}, index=idx)
    d["pnl_jpy"] = d["equity_jpy"].diff().fillna(d["equity_jpy"])
    m = fixed_amount_risk(d, amount=100.0, periods_per_year=245)
    assert m["worst_below_start_per_amount"] == pytest.approx(-0.30)
    assert m["max_drawdown_per_amount"] == pytest.approx(-0.30)          # 始点0から −30
    assert m["longest_below_start_days"] == 2
    assert m["longest_underwater_days"] == 2                               # 0を下回った2日（20→8 は1日）
    assert m["final_pnl_per_amount"] == pytest.approx(0.25)
    assert m["ann_return_on_amount"] == pytest.approx(25 / 6 / 100 * 245)


def test_missing_prices_are_rejected():
    with pytest.raises(ValueError):
        simulate_fixed_amount(_p([100, np.nan, 90], [100, 95, 90]), 1000, 2000, 0.002, None)


def test_config_records_the_users_fixed_amount_decision():
    """2026-10-01 のユーザー決定（保有額84万円・買ったまま持つ・1年たっても売らない）が config と食い違わないこと。"""
    from data.adjust import load_price_config
    cfg = load_price_config()
    v2 = cfg["v2"]
    assert v2["fixed_method"] in v2["fixed_methods"] and v2["fixed_methods"][v2["fixed_method"]] is None
    assert v2["fixed_wait_action"] == "hold"
    assert 0 < v2["fixed_amount_jpy"] <= cfg["capital_jpy"]
    assert v2["fixed_amount_jpy"] >= v2["fixed_loss_limit_jpy"]       # 限界を超える確率は0ではない（5%）ことを明示的に選んだ
