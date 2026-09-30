"""v2 ラベル（設計書 v2 ラベリング設計）: 寄付で買い、買値×(1+g) の売り指値が約定するまで保有。
学習では max_hold 営業日目までに約定しなければ、その日の大引けで評価する。σ̂√5 で正規化。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common_utils.target_utils import v2_barrier_labels
from generate.features_v2 import barrier_labels, build_v2_dataset, v2_config

ROOT = Path(__file__).resolve().parents[1]
T = pd.Timestamp
SQ5 = np.sqrt(5)


def _prices(n=30, o=100.0, h=101.0, l=99.0, c=100.0):
    idx = pd.bdate_range("2025-01-06", periods=n, name="Date")
    return pd.DataFrame({"adj_open": o, "adj_high": h, "adj_low": l, "adj_close": c}, index=idx, dtype="float64")


def _label(p, g=0.02, sigma=0.01, max_hold=20):
    s = pd.Series(sigma, index=p.index)
    return v2_barrier_labels(p, s, g, max_hold=max_hold, vol_scale_days=5, fill_within=(5, 20))


def test_fill_at_limit_when_high_reaches_it():
    p = _prices()
    p.iloc[2, p.columns.get_loc("adj_high")] = 102.5          # 3日目に高値が指値 102 に届く
    lab = _label(p).iloc[0]
    assert lab["bar_ret"] == pytest.approx(np.log(1.02))
    assert lab["bar_days_to_fill"] == 3 and lab["bar_censored"] == 0
    assert lab["bar_filled_5"] == 1 and lab["bar_filled_20"] == 1
    assert lab["bar_y"] == pytest.approx(np.log(1.02) / (0.01 * SQ5))


def test_same_day_fill_after_open():
    p = _prices()
    p.iloc[0, p.columns.get_loc("adj_high")] = 102.0          # 買った日のうちに届く
    lab = _label(p).iloc[0]
    assert lab["bar_days_to_fill"] == 1 and lab["bar_ret"] == pytest.approx(np.log(1.02))


def test_gap_above_limit_fills_at_open():
    p = _prices()
    p.iloc[3, p.columns.get_loc("adj_open")] = 103.0          # 始値が指値を超えた日は始値で約定
    p.iloc[3, p.columns.get_loc("adj_high")] = 104.0
    lab = _label(p).iloc[0]
    assert lab["bar_ret"] == pytest.approx(np.log(1.03)) and lab["bar_days_to_fill"] == 4


def test_censored_at_close_of_last_day():
    p = _prices(n=40)
    p.iloc[19, p.columns.get_loc("adj_close")] = 97.0          # 20営業日目の大引け
    lab = _label(p).iloc[0]
    assert lab["bar_censored"] == 1 and np.isnan(lab["bar_days_to_fill"])
    assert lab["bar_ret"] == pytest.approx(np.log(0.97))
    assert lab["bar_filled_20"] == 0


def test_mae_is_worst_low_until_fill():
    p = _prices()
    p.iloc[1, p.columns.get_loc("adj_low")] = 90.0
    p.iloc[4, p.columns.get_loc("adj_high")] = 102.0
    p.iloc[6, p.columns.get_loc("adj_low")] = 80.0           # 約定後の安値は含めない
    lab = _label(p).iloc[0]
    assert lab["bar_mae"] == pytest.approx(np.log(0.90))


def test_unresolved_at_data_end_is_nan_but_resolved_is_kept():
    p = _prices(n=30)
    p.iloc[-2, p.columns.get_loc("adj_high")] = 105.0         # 最後から2日目に届く
    lab = _label(p)
    assert lab["bar_ret"].iloc[-1:].isna().all()               # 残り1日で未約定 → 結果不明
    assert lab["bar_ret"].iloc[-10] == pytest.approx(np.log(1.02))   # 終端前に約定済みなら確定
    assert lab["bar_censored"].iloc[0] == 1 and lab["bar_ret"].iloc[0] == pytest.approx(0.0)  # 28日目は窓の外


def test_missing_session_in_window_is_nan():
    p = _prices()
    p.iloc[2] = np.nan                                         # 欠損営業日（例: 2025-10-24）
    p.iloc[5, p.columns.get_loc("adj_high")] = 103.0
    lab = _label(p)
    assert np.isnan(lab["bar_ret"].iloc[0])
    assert lab["bar_ret"].iloc[3] == pytest.approx(np.log(1.02))


def test_g_can_be_a_series():
    p = _prices()
    p["adj_high"] = 101.5
    g = pd.Series(0.01, index=p.index)
    lab = v2_barrier_labels(p, pd.Series(0.01, index=p.index), g, max_hold=20, vol_scale_days=5, fill_within=(5, 20))
    assert lab["bar_ret"].iloc[0] == pytest.approx(np.log(1.01)) and lab["bar_g"].iloc[0] == 0.01


# ---------- 実データ ----------
@pytest.fixture(scope="module")
def cache():
    return pd.read_parquet(ROOT / "data" / "cache" / "yf_data_daily_2644t.parquet")


@pytest.fixture(scope="module")
def ds(cache):
    return build_v2_dataset(cache)


def test_barrier_uses_past_vol(ds):
    """σ̂ は判断時刻より前に既知の EWMA ボラ（semi_ewm_vol）だけ。"""
    lab = barrier_labels(ds, g_fixed=0.02)
    sig = ds["features"]["semi_ewm_vol"]
    ok = lab["bar_y"].notna()
    assert np.allclose(lab.loc[ok, "bar_y"], lab.loc[ok, "bar_ret"] / (sig[ok] * SQ5))
    assert (ds["known_at"]["semi_ewm_vol"][ok] < ds["decision_time"][ok]).all()
    vm = barrier_labels(ds, vol_multiple=0.5)
    okv = vm["bar_g"].notna()
    assert np.allclose(vm.loc[okv, "bar_g"], 0.5 * sig[okv] * SQ5)


def test_barrier_labels_do_not_use_data_after_resolution(cache):
    cut = T("2025-06-30")
    full = barrier_labels(build_v2_dataset(cache), g_fixed=0.03)
    trunc = barrier_labels(build_v2_dataset(cache[cache["Date"] <= cut]), g_fixed=0.03)
    known = trunc["bar_ret"].notna()
    assert known.sum() > 800
    cols = ["bar_ret", "bar_y", "bar_days_to_fill", "bar_censored", "bar_mae"]
    a, b = trunc.loc[known, cols], full.loc[trunc.index[known], cols]
    assert ((a - b).abs().fillna(0) < 1e-12).all().all() and (a.isna() == b.isna()).all().all()


def test_barrier_labels_are_split_adjusted(ds):
    lab = barrier_labels(ds, g_fixed=0.02)
    window = lab.loc[T("2024-09-17"):T("2024-10-08"), "bar_ret"].dropna()
    assert len(window) > 10 and (window.abs() < 0.25).all()      # 分割の −0.68 が入らない


def test_default_candidates_come_from_config():
    cfg = v2_config()
    assert cfg["barrier_max_hold_days"] == 20 and cfg["barrier_vol_scale_days"] == 5
    assert cfg["barrier_g_fixed"] and cfg["barrier_g_vol_multiples"]
