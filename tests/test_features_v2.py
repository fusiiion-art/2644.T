"""v2 データ層の先読みテスト（設計書 v2「テスト・規約」、監査 G-1〜G-4・A-FRED・B）。

行 D = 判断日（設計書の t+1）。判断時刻は D の 08:50 JST。特徴量は判断時刻より前に確定した値だけ。
"""
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from generate.features_v2 import (
    asof_join, build_v2_dataset, fred_source, tokyo_close_known_at, us_close_known_at, v2_config,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "yf_data_daily_2644t.parquet"
T = pd.Timestamp


@pytest.fixture(scope="module")
def cache():
    return pd.read_parquet(CACHE)


@pytest.fixture(scope="module")
def ds(cache):
    return build_v2_dataset(cache)


def _nan_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """NaN の位置が同じで、値が一致する（分割調整の ÷2 による 1e-15 程度の丸め差は許す）。"""
    a, b = a.sort_index(axis=1), b.sort_index(axis=1)
    if a.shape != b.shape or not a.index.equals(b.index) or not (a.isna() == b.isna()).all().all():
        return False
    return bool(np.allclose(a.fillna(0).to_numpy(), b.fillna(0).to_numpy(), rtol=1e-12, atol=1e-12))


# ---------- index（G-1） ----------
def test_index_is_xtks_sessions(ds):
    idx = ds["features"].index
    cal = xcals.get_calendar("XTKS")
    expected = cal.sessions_in_range(idx[0], idx[-1]).tz_localize(None) if cal.sessions_in_range(idx[0], idx[-1]).tz is not None \
        else cal.sessions_in_range(idx[0], idx[-1])
    assert idx.tz is None and (idx == idx.normalize()).all()
    assert list(idx) == list(pd.DatetimeIndex(expected))
    assert T("2024-10-14") not in idx and T("2025-07-21") not in idx      # 休場日の行が無い
    for key in ["features", "prices", "labels"]:
        assert ds[key].index.equals(idx)


def test_columns_are_float64(ds):
    for key in ["features", "prices", "labels"]:
        assert (ds[key].dtypes == "float64").all(), key


def test_history_recovered_from_2021_09(ds):
    """G-4: FRED の欠損で 2021-09〜2023-07 が消えない。"""
    f = ds["features"]
    assert f.index[0] == T(v2_config()["start"])
    usable = f.join(ds["labels"]).dropna()
    assert usable.index[0] < T("2021-12-31")
    assert len(f) >= 1150          # 現行の特徴量ファイルは 783 行
    assert len(usable) >= 1000


# ---------- 既知時刻 ----------
def test_known_at_monotonic(ds):
    """すべての特徴量で known_at_jst < decision_time_jst。"""
    dec = ds["decision_time"]
    ka = ds["known_at"]
    for col in ka.columns:
        ok = ka[col].isna() | (ka[col] < dec)
        assert ok.all(), f"{col}: {ka.index[~ok][:3].tolist()}"


def test_us_dst_boundary():
    """米国夏時間の切替週で米国終値の既知時刻（JST）が1時間ずれる。"""
    ka = us_close_known_at(pd.DatetimeIndex([T("2025-03-07"), T("2025-03-10")]))
    assert ka.iloc[0] == pd.Timestamp("2025-03-08 06:00", tz="Asia/Tokyo")     # EST
    assert ka.iloc[1] == pd.Timestamp("2025-03-11 05:00", tz="Asia/Tokyo")     # EDT


def test_close_time_change():
    """2024-11-05 から東証の大引けは 15:30。"""
    ka = tokyo_close_known_at(pd.DatetimeIndex([T("2024-11-01"), T("2024-11-05")]))
    assert ka.iloc[0] == pd.Timestamp("2024-11-01 15:00", tz="Asia/Tokyo")
    assert ka.iloc[1] == pd.Timestamp("2024-11-05 15:30", tz="Asia/Tokyo")


def test_decision_time_is_0850_jst(ds):
    dec = ds["decision_time"]
    assert (dec.dt.tz is not None) and (dec.dt.strftime("%H:%M") == v2_config()["decision_time_jst"]).all()
    assert (dec.dt.tz_convert("Asia/Tokyo").dt.normalize().dt.tz_localize(None) == dec.index).all()


# ---------- 休場日の食い違い ----------
def test_holiday_mismatch(ds, cache):
    f = ds["features"]
    sox = cache.set_index("Date")["^SOX_adj_close"].dropna()
    # 東京営業・米国休場（2025-07-04 独立記念日）: 米国の値は前のまま、当夜の変化は 0
    assert f.at[T("2025-07-07"), "sox_ret_last"] == f.at[T("2025-07-04"), "sox_ret_last"]
    assert f.at[T("2025-07-07"), "sox_ret_overnight"] == pytest.approx(0.0)
    # 米国営業・東京休場（2025-07-21 海の日）: 行は増えず、翌営業日の当夜変化が2夜分をまとめる
    assert T("2025-07-21") not in f.index
    expected = np.log(sox[T("2025-07-21")] / sox[T("2025-07-17")])
    assert f.at[T("2025-07-22"), "sox_ret_overnight"] == pytest.approx(expected)


# ---------- 未来を変えても過去は変わらない ----------
def test_shuffle_future(cache):
    cut = T("2025-06-30")
    full = build_v2_dataset(cache)
    trunc = build_v2_dataset(cache[cache["Date"] <= cut])
    rng = np.random.default_rng(0)
    shuffled = cache.copy()
    fut = shuffled["Date"] > cut
    for c in shuffled.columns.drop("Date"):
        if c.startswith("2644.T_"):
            # 2644 は分割調整の段差検証（jump_bounds）を通る範囲で撹乱する
            shuffled.loc[fut, c] = shuffled.loc[fut, c] * rng.uniform(0.9, 1.1, int(fut.sum()))
        else:
            shuffled.loc[fut, c] = rng.permutation(shuffled.loc[fut, c].to_numpy())
    shuf = build_v2_dataset(shuffled)
    upto = full["features"].index <= cut
    assert _nan_equal(full["features"][upto], trunc["features"].loc[:cut])
    assert _nan_equal(full["features"][upto], shuf["features"][upto])


def test_labels_do_not_leak_past_data_end(cache):
    cut = T("2025-06-30")
    trunc = build_v2_dataset(cache[cache["Date"] <= cut])
    lab = trunc["labels"]
    # 20営業日先を要するラベルは、最後の20行で NaN
    assert lab["fwd_oc_20"].iloc[-19:].isna().all()
    assert lab["fwd_oo_1"].iloc[-1:].isna().all()


def test_no_bfill(cache):
    """G-2: 先頭のウォームアップ行は NaN のまま（未来の値で埋めない）。"""
    ds60 = build_v2_dataset(cache[cache["Date"] <= T("2021-12-31")])
    f = ds60["features"]
    assert f["semi_mom_20"].iloc[:20].isna().all()
    assert f["semi_ret_1d"].iloc[:2].isna().all()
    full = build_v2_dataset(cache)["features"].loc[: f.index[-1]]
    assert _nan_equal(f, full)


# ---------- ラベル（B・G-1） ----------
def test_target_adjusted_prices(ds):
    lab = ds["labels"]
    assert lab.at[T("2024-10-08"), "fwd_oo_1"] == pytest.approx(np.log(1896.0 / (3745.0 / 2)))
    assert lab["fwd_oo_1"].abs().max() < 0.25          # 分割の −0.68 が入らない
    assert lab["fwd_oc_1"].abs().max() < 0.25


def test_label_starts_next_session(ds):
    """休場日（2024-10-14）の前後でもラベルは翌営業日の始値から始まる。"""
    p, lab = ds["prices"], ds["labels"]
    assert lab.at[T("2024-10-11"), "fwd_oo_1"] == pytest.approx(np.log(p.at[T("2024-10-15"), "adj_open"] / p.at[T("2024-10-11"), "adj_open"]))
    assert lab.at[T("2024-10-11"), "fwd_oo_1"] != 0.0
    assert lab.at[T("2024-10-15"), "fwd_oc_1"] == pytest.approx(np.log(p.at[T("2024-10-15"), "adj_close"] / p.at[T("2024-10-15"), "adj_open"]))
    assert lab.at[T("2024-10-15"), "gap"] == pytest.approx(np.log(p.at[T("2024-10-15"), "adj_open"] / p.at[T("2024-10-11"), "adj_close"]))


def test_prices_keep_raw_open_for_orders(ds):
    p = ds["prices"]
    assert p.at[T("2024-10-08"), "raw_open"] == pytest.approx(3745.0)
    assert p.at[T("2024-10-08"), "adj_open"] == pytest.approx(3745.0 / 2)
    assert not any(c.startswith(("adj_", "raw_")) for c in ds["features"].columns)


# ---------- as-of 結合と FRED（A-FRED） ----------
def test_asof_join_is_strictly_before():
    base = pd.DataFrame({"decision_time_jst": pd.to_datetime(["2025-01-06 08:50", "2025-01-07 08:50"]).tz_localize("Asia/Tokyo")})
    feat = pd.DataFrame({"known_at_jst": pd.to_datetime(["2025-01-06 08:50", "2025-01-06 06:00"]).tz_localize("Asia/Tokyo"),
                         "x": [2.0, 1.0]})
    out = asof_join(base, feat)
    assert out["x"].tolist() == [1.0, 2.0]     # 判断時刻ちょうどの値は使わない


def test_fred_published_before_decision():
    obs = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                        "realtime_start": pd.to_datetime(["2025-01-03", "2025-01-06"]),
                        "value": [4.1, 4.2]})
    src = fred_source(obs, "us_10y")
    # 公表日の翌日 0:00 America/New_York に既知 → JST では公表日の翌日 14:00（冬時間）
    assert src["known_at_jst"].iloc[0] == pd.Timestamp("2025-01-04 14:00", tz="Asia/Tokyo")
    with pytest.raises(ValueError):
        fred_source(obs.drop(columns="realtime_start"), "us_10y")
