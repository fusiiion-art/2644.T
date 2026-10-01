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
from common_utils.rule_risk import averaging_down_levels, exposure_stats, scenario_loss  # noqa: E402

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
