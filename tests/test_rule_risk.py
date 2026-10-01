"""運用ルールの含み損リスクの測定（方向 C、2026-10-01）のテスト。ルールは変えず、測るだけ。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_utils.position_simulator import PositionRules  # noqa: E402
from common_utils.rule_risk import (  # noqa: E402
    averaging_down_levels, block_bootstrap_ohlc, exposure_stats, scenario_loss,
)

RULES = PositionRules(lot_units=100, max_units=300, add_on_drop=0.04, round_trip_cost=0.0, capital_jpy=1_500_000)


def test_averaging_down_levels_follow_the_four_percent_rule():
    lv = averaging_down_levels(RULES)
    assert [x["lots"] for x in lv] == [1, 2, 3]
    assert lv[0]["buy_at"] == 1.0 and lv[0]["avg"] == 1.0
    assert lv[1]["buy_at"] == pytest.approx(0.96)
    assert lv[1]["avg"] == pytest.approx(0.98)
    assert lv[2]["buy_at"] == pytest.approx(0.98 * 0.96)
    assert lv[2]["avg"] == pytest.approx((1 + 0.96 + 0.9408) / 3)


def test_scenario_loss_fills_only_the_lots_the_drop_reaches():
    p = 5000.0
    s = scenario_loss(p, 0.05, RULES)                 # 2本目（−4%）は届き、3本目（−5.92%）は届かない
    assert s["lots"] == 2
    assert s["invested_jpy"] == pytest.approx(100 * p + 100 * 0.96 * p)
    assert s["unrealized_jpy"] == pytest.approx(200 * 0.95 * p - s["invested_jpy"])
    deep = scenario_loss(p, 0.5, RULES)
    avg3 = (1 + 0.96 + 0.9408) / 3
    assert deep["lots"] == 3
    assert deep["unrealized_jpy"] == pytest.approx(300 * p * (0.5 - avg3))
    assert deep["share_of_capital"] == pytest.approx(deep["unrealized_jpy"] / 1_500_000)
    assert deep["avg_vs_entry"] == pytest.approx(avg3)


def test_exposure_stats_counts_cap_days_lockup_and_underwater():
    idx = pd.bdate_range("2024-01-01", periods=8)
    daily = pd.DataFrame({
        "lots": [1, 2, 3, 3, 3, 0, 1, 0],
        "invested_jpy": [100.0, 200, 300, 300, 300, 0, 100, 0],
        "unrealized_jpy": [0.0, -10, -40, -60, -20, 0, 0, 0],
        "equity_jpy": [0.0, -10, -40, -60, -20, 5, 5, 6],
    }, index=idx).astype("float64")
    trades = pd.DataFrame({"entry_date": [idx[0], idx[6]], "exit_date": [idx[5], idx[7]],
                           "lots": [3, 1], "holding_days": [6, 2], "max_invested_jpy": [300.0, 100.0],
                           "worst_low_ret": [-0.3, 0.0], "pnl_jpy": [5.0, 1.0]})
    st = exposure_stats(daily, trades, RULES)
    assert st["days_at_max_lots"] == 3
    assert st["share_days_at_max_lots"] == pytest.approx(3 / 8)
    assert st["worst_unrealized_jpy"] == -60.0
    assert st["worst_unrealized_date"] == str(idx[3].date())
    assert st["longest_hold_days"] == 6
    assert st["trades_reaching_max_lots"] == 1
    assert st["longest_underwater_days"] == 4          # 時価評価の損益が直前の最高値（0）を下回った idx1〜idx4
    assert st["max_invested_share_of_capital"] == pytest.approx(300 / 1_500_000)


def _source(n=300, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(0.002 + 0.02 * rng.standard_normal(n)))
    prev = np.r_[100.0, c[:-1]]
    o = prev * np.exp(0.005 * rng.standard_normal(n))
    h = np.maximum(o, c) * (1 + 0.01 * rng.random(n))
    lo = np.minimum(o, c) * (1 - 0.01 * rng.random(n))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"adj_open": o, "adj_high": h, "adj_low": lo, "adj_close": c}, index=idx)


def test_bootstrap_keeps_ohlc_consistent_and_starts_from_given_price():
    src = _source()
    out = block_bootstrap_ohlc(src, n_days=500, block_mean=20, start_price=5000.0, rng=np.random.default_rng(1))
    assert len(out) == 500
    assert (out["adj_high"] >= out[["adj_open", "adj_close"]].max(axis=1) - 1e-9).all()
    assert (out["adj_low"] <= out[["adj_open", "adj_close"]].min(axis=1) + 1e-9).all()
    assert (out["raw_open"] == out["adj_open"]).all() and (out["raw_close"] == out["adj_close"]).all()
    r0 = out["adj_close"].iloc[0] / 5000.0
    rc = (src["adj_close"] / src["adj_close"].shift()).dropna().to_numpy()
    assert np.isclose(rc, r0).any()                      # 1日目は元データのどれかの日の値動き


def test_bootstrap_zero_drift_removes_the_trend():
    src = _source()
    rng = np.random.default_rng(2)
    z = block_bootstrap_ohlc(src, 200000, 20, 100.0, rng, drift="zero")
    a = block_bootstrap_ohlc(src, 200000, 20, 100.0, np.random.default_rng(2), drift="asis")
    lz = np.log(z["adj_close"]).diff().dropna().mean()
    la = np.log(a["adj_close"]).diff().dropna().mean()
    src_mu = np.log(src["adj_close"] / src["adj_close"].shift()).dropna().mean()
    assert abs(lz) < 0.001                                # 標準誤差は約0.0002
    assert la == pytest.approx(src_mu, abs=0.001)


def test_bootstrap_is_reproducible_and_uses_contiguous_blocks():
    src = _source(n=50)
    a = block_bootstrap_ohlc(src, 40, 1e9, 100.0, np.random.default_rng(3))
    b = block_bootstrap_ohlc(src, 40, 1e9, 100.0, np.random.default_rng(3))
    pd.testing.assert_frame_equal(a, b)
    rc = (src["adj_close"] / src["adj_close"].shift()).dropna().to_numpy()
    got = (a["adj_close"] / a["adj_close"].shift().fillna(100.0)).to_numpy()
    start = int(np.argmin(np.abs(rc - got[0])))
    assert np.allclose(got, rc[(start + np.arange(40)) % len(rc)])   # ブロックが十分長いと元の並びのまま（端で折り返す）



def test_constant_exposure_holds_a_fixed_yen_amount_every_day():
    """比較用: 資金 × share の金額を毎日持ち続ける（毎日その金額にそろえる。そろえる売買のコストは含めない）。"""
    from common_utils.rule_risk import constant_exposure_daily
    idx = pd.bdate_range("2024-01-01", periods=3)
    p = pd.DataFrame({"adj_open": [100.0, 110, 90], "adj_close": [105.0, 100, 120]}, index=idx)
    rules = PositionRules(lot_units=100, max_units=300, add_on_drop=0.04, round_trip_cost=0.002, capital_jpy=1000)
    d = constant_exposure_daily(p, rules, share=0.5)
    pnl = 500 * np.array([0.05, 100 / 105 - 1, 0.2])
    fee = 500 * 0.001
    assert d["pnl_jpy"].tolist() == pytest.approx([pnl[0] - fee, pnl[1], pnl[2]])
    assert d["equity_jpy"].tolist() == pytest.approx(list(np.cumsum(pnl) - fee))
    assert d["ret"].tolist() == pytest.approx(list(d["pnl_jpy"] / 1000))
    assert (d["invested_jpy"] == 500).all() and (d["lots"] == 1).all()
