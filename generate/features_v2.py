"""v2 データ層（設計書 v2「特徴量設計」「タイムライン設計」、移行計画フェーズ1）。

- index は東京営業日（XTKS）。休場日の行は作らない（監査 G-1）。
- 行 D = 判断日（設計書の t+1）。判断時刻は D の decision_time_jst（既定 08:50 JST）。
- すべての値に「日本時間の既知時刻」known_at_jst を付け、判断時刻より厳密に前に確定した値だけを
  as-of 結合で取り込む（監査 A・A-FRED）。列名の部分一致による shift は使わない。
- 価格は data/adjust.py で分割調整してから使う（監査 B）。生の始値は発注専用に prices に残す。
- bfill も全期間統計による正規化もしない。ウォームアップ行は NaN のまま（監査 G-2・G-3）。
- FRED のような疎な系列で行を落とさない（監査 G-4）。FRED は ALFRED の公表日が無い限り使わない。

数値パラメータは semi2644/config/config.yaml の v2 節で管理する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_utils.target_utils import v2_barrier_labels, v2_forward_returns  # noqa: E402
from data.adjust import load_price_config, raw_ohlc_from_cache, split_adjust_ohlc  # noqa: E402

JST = "Asia/Tokyo"
NY = "America/New_York"
DEFAULT_CACHE = ROOT / "data" / "cache" / "yf_data_daily_2644t.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "features_v2_daily_2644t.parquet"

MAIN_TICKER = "2644.T"
US_RETURNS = {"sox": "^SOX", "nvda": "NVDA", "tsm": "TSM", "mu": "MU", "ndx": "^NDX"}
US_LEVELS = {"vxn_close": "^VXN"}
TOKYO_INDEX = {"n225": "^N225"}
PRICE_COLUMNS = ["adj_open", "adj_high", "adj_low", "adj_close", "raw_open", "raw_close"]


def v2_config(path: Path | None = None) -> dict:
    cfg = load_price_config(path)
    return {**cfg["v2"], "start": cfg["data"]["start"]}


# ---------- 暦と時刻 ----------
def xtks_sessions(start, end) -> pd.DatetimeIndex:
    """東京証券取引所の営業日（tz-naive, 正規化日付）。"""
    s = pd.DatetimeIndex(xcals.get_calendar("XTKS").sessions_in_range(pd.Timestamp(start), pd.Timestamp(end)))
    if s.tz is not None:
        s = s.tz_localize(None)
    return s.normalize().rename("Date")


def _clock(dates, hhmm: str, tz: str) -> pd.Series:
    """各日付の現地時刻 hhmm を JST の tz-aware 時刻にする（夏時間は tz ライブラリに任せる）。"""
    dates = pd.DatetimeIndex(dates)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    h, m = (int(x) for x in hhmm.split(":"))
    ts = (dates + pd.Timedelta(hours=h, minutes=m)).tz_localize(tz).tz_convert(JST).as_unit("ns")
    return pd.Series(ts, index=dates.rename("Date"))


def decision_times(sessions, cfg: dict | None = None) -> pd.Series:
    cfg = cfg or v2_config()
    return _clock(sessions, cfg["decision_time_jst"], JST)


def tokyo_open_known_at(dates, cfg: dict | None = None) -> pd.Series:
    cfg = cfg or v2_config()
    return _clock(dates, cfg["tokyo_open_jst"], JST)


def tokyo_close_known_at(dates, cfg: dict | None = None) -> pd.Series:
    """東証の大引け。2024-11-05 から 15:30、それ以前は 15:00。"""
    cfg = cfg or v2_config()
    before = _clock(dates, cfg["tokyo_close_jst_before"], JST)
    after = _clock(dates, cfg["tokyo_close_jst_after"], JST)
    change = pd.Timestamp(cfg["tokyo_close_change_date"])
    return before.where(before.index < change, after)


def us_close_known_at(dates, cfg: dict | None = None) -> pd.Series:
    """米国の取引日 d の 16:00 America/New_York（JST では夏時間 d+1 05:00、冬時間 d+1 06:00）。"""
    cfg = cfg or v2_config()
    return _clock(dates, cfg["us_close_local"], NY)


# ---------- as-of 結合 ----------
def asof_join(base: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """判断時刻より厳密に前に既知になった最新の値を結合する（設計書 v2 のコード）。"""
    base = base.sort_values("decision_time_jst").copy()
    feat = feat.sort_values("known_at_jst").copy()
    base["decision_time_jst"] = base["decision_time_jst"].dt.as_unit("ns")
    feat["known_at_jst"] = feat["known_at_jst"].dt.as_unit("ns")
    return pd.merge_asof(
        base, feat,
        left_on="decision_time_jst", right_on="known_at_jst",
        direction="backward", allow_exact_matches=False,
    )


def _source(values: pd.Series, known_at: pd.Series, name: str) -> pd.DataFrame:
    known_at = known_at.reindex(values.index)
    df = pd.DataFrame({"known_at_jst": known_at.to_numpy(), name: values.to_numpy(dtype=float)})
    return df.dropna(subset=["known_at_jst"]).sort_values("known_at_jst", kind="mergesort").reset_index(drop=True)


def fred_source(obs: pd.DataFrame, name: str, cfg: dict | None = None) -> pd.DataFrame:
    """ALFRED の観測（date, realtime_start, value）を既知時刻つきの系列にする（監査 A-FRED）。

    公表時刻は分からないので、公表日の fred_known_lag_days 日後 0:00 America/New_York に既知とみなす。
    同じ時点までに公表された中で最も新しい観測日の値だけを残す（古い観測日の改定で上書きしない）。
    """
    if "realtime_start" not in obs.columns:
        raise ValueError("FRED 系列には ALFRED の realtime_start（公表日）が必要。観測日で結合すると先読みになる")
    cfg = cfg or v2_config()
    rs = pd.DatetimeIndex(pd.to_datetime(obs["realtime_start"])).normalize() + pd.Timedelta(days=int(cfg["fred_known_lag_days"]))
    df = pd.DataFrame({
        "known_at_jst": rs.tz_localize(NY).tz_convert(JST).as_unit("ns"),
        "obs_date": pd.to_datetime(obs["date"]).to_numpy(),
        name: obs["value"].astype(float).to_numpy(),
    }).sort_values(["known_at_jst", "obs_date"], kind="mergesort")
    df = df[df["obs_date"] >= df["obs_date"].cummax()]
    return df.drop(columns="obs_date").reset_index(drop=True)


def _asof(dec: pd.Series, src: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    base = pd.DataFrame({"Date": dec.index, "decision_time_jst": dec.to_numpy()})
    out = asof_join(base, src).set_index("Date").reindex(dec.index)
    return out[name].astype("float64"), out["known_at_jst"]


# ---------- 本体 ----------
def _cache_series(cache: pd.DataFrame, column: str) -> pd.Series:
    s = cache.set_index("Date")[column] if column in cache.columns else pd.Series(dtype=float)
    return s.dropna().astype(float)


def build_v2_dataset(cache: pd.DataFrame | None = None, cfg: dict | None = None) -> dict:
    """v2 の特徴量・価格・ラベルを東京営業日の index で作る。

    Returns:
        features     : 判断時刻より前に既知の特徴量（float64）
        known_at     : 各特徴量の値が確定した時刻（JST, tz-aware）。テスト・監査用
        decision_time: 各行の判断時刻（JST, tz-aware）
        prices       : 行 D 自身の分割調整済み OHLC と生値（ラベル計算と発注専用。特徴量に使わない）
        labels       : 行 D に置いたラベル（common_utils.target_utils.v2_forward_returns）
    """
    cfg = cfg or v2_config()
    cache = pd.read_parquet(DEFAULT_CACHE) if cache is None else cache.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(cache["Date"]))
    cache["Date"] = (dates.tz_localize(None) if dates.tz is not None else dates).normalize()

    sessions = xtks_sessions(cfg["start"], cache["Date"].max())
    dec = decision_times(sessions, cfg)
    feats: dict[str, pd.Series] = {}
    known: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series, known_at: pd.Series) -> None:
        feats[name], known[name] = _asof(dec, _source(values, known_at, name), name)

    # 2644 自身（行 d の値は d の大引けで確定）
    adj = split_adjust_ohlc(raw_ohlc_from_cache(cache, MAIN_TICKER))
    prices = adj.reindex(sessions)[PRICE_COLUMNS].astype("float64")
    close_ka = tokyo_close_known_at(sessions, cfg)
    c, o = prices["adj_close"], prices["adj_open"]
    lc = np.log(c)
    ret = lc.diff()
    add("semi_ret_1d", ret, close_ka)
    add("semi_intraday", np.log(c / o), close_ka)
    for w in cfg["momentum_windows"]:
        add(f"semi_mom_{w}", lc - lc.shift(w), close_ka)
    hl = int(cfg["ewma_vol_halflife"])
    add("semi_ewm_vol", ret.ewm(halflife=hl, min_periods=hl).std(), close_ka)
    ma = int(cfg["ma_window"])
    add(f"semi_ma{ma}_dev", c / c.rolling(ma, min_periods=ma).mean() - 1.0, close_ka)

    # 東京の指数（大引けで確定）
    for key, ticker in TOKYO_INDEX.items():
        ln = np.log(_cache_series(cache, f"{ticker}_adj_close").reindex(sessions))
        add(f"{key}_ret_1d", ln.diff(), close_ka)
        add(f"{key}_mom_5", ln - ln.shift(5), close_ka)

    # 米国（取引日 d の 16:00 New York で確定）。当夜の変化 = 今回と前回の判断時刻に既知だった水準の差
    prev_close_ka = pd.Series(tokyo_close_known_at(sessions, cfg).shift(1).to_numpy(), index=sessions)
    for key, ticker in US_RETURNS.items():
        px = _cache_series(cache, f"{ticker}_adj_close")
        lp = np.log(px)
        ka = us_close_known_at(px.index, cfg)
        level, level_ka = _asof(dec, _source(lp, ka, "lp"), "lp")
        feats[f"{key}_ret_overnight"], known[f"{key}_ret_overnight"] = level.diff(), level_ka
        add(f"{key}_ret_last", lp.diff(), ka)
        if key == "sox":
            for w in cfg["momentum_windows"]:
                add(f"sox_mom_{w}", lp - lp.shift(w), ka)
            # 比較用: 現行（t の大引けで判断）と同じ t-1 版の当夜変化
            old_dec = prev_close_ka.dropna()
            old_level, old_ka = _asof(old_dec, _source(lp, ka, "lp"), "lp")
            feats["sox_ret_overnight_old"] = old_level.diff().reindex(sessions)
            known["sox_ret_overnight_old"] = old_ka.reindex(sessions)
    for name, ticker in US_LEVELS.items():
        px = _cache_series(cache, f"{ticker}_close")
        add(name, px, us_close_known_at(px.index, cfg))

    features = pd.DataFrame(feats, index=sessions).astype("float64")
    known_at = pd.DataFrame(known, index=sessions)
    labels = v2_forward_returns(prices, list(cfg["label_horizons"]))
    return {"features": features, "known_at": known_at, "decision_time": dec,
            "prices": prices, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description="v2 の特徴量・価格・ラベルを作って保存する")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    ds = build_v2_dataset()
    out = ds["features"].join(ds["prices"]).join(ds["labels"])
    out.insert(0, "decision_time_jst", ds["decision_time"])
    out.to_parquet(args.output)
    print(f"saved {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()


def barrier_labels(ds: dict, g_fixed: float | None = None, vol_multiple: float | None = None,
                   cfg: dict | None = None) -> pd.DataFrame:
    """build_v2_dataset の結果から利確指値バリアのラベルを作る（g は固定率か、週次ボラ σ̂√5 の倍率のどちらか一方）。

    σ̂ は特徴量 semi_ewm_vol（判断時刻より前に既知の EWMA ボラ）を使う。
    """
    cfg = cfg or v2_config()
    if (g_fixed is None) == (vol_multiple is None):
        raise ValueError("g_fixed か vol_multiple のどちらか一方を指定する")
    sig = ds["features"]["semi_ewm_vol"]
    k = int(cfg["barrier_vol_scale_days"])
    g = float(g_fixed) if g_fixed is not None else float(vol_multiple) * sig * np.sqrt(k)
    return v2_barrier_labels(ds["prices"], sig, g, int(cfg["barrier_max_hold_days"]), k,
                             tuple(cfg["barrier_fill_within_days"]))
