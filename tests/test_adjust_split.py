"""監査 B: 2024-10-09 の 1:2 分割。調整済み系列に段差がなく、生値は発注用に残る。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.adjust import load_price_config, raw_ohlc_from_cache, split_adjust_ohlc

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "yf_data_daily_2644t.parquet"
SPLIT = pd.Timestamp("2024-10-09")
PREV = pd.Timestamp("2024-10-08")
ACTIONS = [{"date": "2024-10-09", "type": "split", "ratio": 2.0}]


@pytest.fixture(scope="module")
def raw():
    return raw_ohlc_from_cache(pd.read_parquet(CACHE), "2644.T")


def test_split_2024_10_09(raw):
    adj = split_adjust_ohlc(raw)                      # corporate_actions.yaml を読む
    # 生値は半分になる（発注用に残す）
    assert adj.at[PREV, "raw_open"] == pytest.approx(3745.0)
    assert adj.at[SPLIT, "raw_open"] == pytest.approx(1896.0)
    # 調整済み系列に段差がない
    assert adj.at[PREV, "adj_open"] == pytest.approx(3745.0 / 2)
    assert abs(np.log(adj.at[SPLIT, "adj_close"] / adj.at[PREV, "adj_close"])) < 0.05
    # 分割後の値は変わらない
    assert adj.loc[SPLIT:, "adj_close"].equals(raw.loc[SPLIT:, "close"].rename("adj_close"))
    # 出来高は逆方向に調整
    assert adj.at[PREV, "adj_volume"] == pytest.approx(raw.at[PREV, "volume"] * 2)


def test_all_adjusted_daily_moves_within_jump_bounds(raw):
    adj = split_adjust_ohlc(raw)
    lo, hi = load_price_config()["data"]["jump_bounds"]
    ratio = (adj["adj_close"] / adj["adj_close"].shift(1)).dropna()
    assert ratio.between(lo, hi).all()


def test_no_double_adjustment_when_source_already_adjusted():
    idx = pd.bdate_range("2024-10-01", "2024-10-18")
    close = pd.Series(np.linspace(1800, 1900, len(idx)), index=idx)
    already = pd.DataFrame({"open": close, "high": close + 5, "low": close - 5, "close": close, "volume": 1000.0})
    adj = split_adjust_ohlc(already, actions=ACTIONS, jump_bounds=[0.55, 1.8])
    assert np.allclose(adj["adj_close"], already["close"])       # 二重に割らない


def test_unexplained_jump_raises():
    idx = pd.bdate_range("2023-01-02", periods=10)
    close = pd.Series(1000.0, index=idx)
    close.iloc[5:] = 400.0                                       # 分割でない −60%
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    with pytest.raises(ValueError):
        split_adjust_ohlc(df, actions=[], jump_bounds=[0.55, 1.8])


def test_index_must_be_naive_dates():
    idx = pd.date_range("2024-10-01", periods=3, tz="Asia/Tokyo")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    with pytest.raises(ValueError):
        split_adjust_ohlc(df, actions=[], jump_bounds=[0.55, 1.8])
