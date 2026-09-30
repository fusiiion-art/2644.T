"""差分キャッシュ（監査 B の原因推定・C の途中足リスク）。

- 毎回 overlap_days 分をさかのぼって取り直し、途中足などの最終行を置き換える。
- 重なり期間の過去値が変わっていたら（分割・配当の遡及調整）、全期間を取り直して新旧を混ぜない。
"""
import numpy as np
import pandas as pd
import pytest

import common_utils.data_fetcher as dfm

FIELDS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


def _yf_frame(close: pd.Series, ticker: str = "AAA") -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples([(f, ticker) for f in FIELDS], names=["Price", "Ticker"])
    data = np.column_stack([close.to_numpy()] * 5 + [np.full(len(close), 100.0)])
    return pd.DataFrame(data, index=pd.DatetimeIndex(close.index, name="Date"), columns=cols)


class FakeYahoo:
    def __init__(self):
        self.truth = None
        self.calls = []

    def download(self, tickers, start=None, interval="1d", **kwargs):
        self.calls.append(pd.Timestamp(start))
        return self.truth[self.truth.index >= pd.Timestamp(start)].copy()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    y = FakeYahoo()
    monkeypatch.setattr(dfm, "CACHE_DATA_DIR", tmp_path)
    monkeypatch.setattr(dfm.yf, "download", y.download)
    return y


SYMS = {"a": "AAA"}
DATES = pd.bdate_range("2024-01-01", periods=40)


def test_overlap_refetch_replaces_partial_last_bar(fake):
    close = pd.Series(np.arange(100.0, 140.0), index=DATES)
    first = close.iloc[:20].copy()
    first.iloc[-1] = 118.5                                   # 場中に取った途中足
    fake.truth = _yf_frame(first)
    dfm.get_yfinance_data(SYMS, "2024-01-01", "1d", "c.parquet", overlap_days=15)
    fake.truth = _yf_frame(close)
    out = dfm.get_yfinance_data(SYMS, "2024-01-01", "1d", "c.parquet", overlap_days=15)
    assert fake.calls[-1] == DATES[19] - pd.Timedelta(days=15)          # 重なりをさかのぼって取得
    assert len(fake.calls) == 2                                         # 全期間の取り直しはしない
    assert out.set_index("Date").at[DATES[19], "AAA_close"] == 119.0    # 途中足が確定値に置き換わる
    assert len(out) == 40 and out["Date"].is_monotonic_increasing


def test_revised_history_triggers_full_refetch(fake):
    raw = pd.Series(np.r_[np.full(20, 200.0), np.full(20, 100.0)], index=DATES)    # 1:2 分割（未調整）
    fake.truth = _yf_frame(raw.iloc[:30])
    dfm.get_yfinance_data(SYMS, "2024-01-01", "1d", "c.parquet", overlap_days=15)
    adjusted = raw.copy()
    adjusted.iloc[:20] = 100.0                                          # 取得元が過去を遡及調整
    fake.truth = _yf_frame(adjusted)
    out = dfm.get_yfinance_data(SYMS, "2024-01-01", "1d", "c.parquet", overlap_days=15)
    assert fake.calls[1] == DATES[29] - pd.Timedelta(days=15)           # まず重なり分だけ取得
    assert len(fake.calls) == 3 and fake.calls[2] == pd.Timestamp("2024-01-01")   # 改定を検知して全期間を取り直した
    assert (out["AAA_close"] == 100.0).all()                            # 新旧が混ざらない
    assert len(out) == 40


def test_no_revision_keeps_old_rows(fake):
    close = pd.Series(np.arange(100.0, 140.0), index=DATES)
    fake.truth = _yf_frame(close.iloc[:30])
    dfm.get_yfinance_data(SYMS, "2024-01-01", "1d", "c.parquet", overlap_days=15)
    fake.truth = _yf_frame(close)
    out = dfm.get_yfinance_data(SYMS, "2024-01-01", "1d", "c.parquet", overlap_days=15)
    assert len(fake.calls) == 2
    assert out["AAA_close"].tolist() == close.tolist()
