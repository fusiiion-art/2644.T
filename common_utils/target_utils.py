"""v2 のラベル（設計書 v2 ラベリング設計）。行 D（判断日）に直接置く。

旧パイプラインのターゲット生成（create_targets など）は 2026-10-01 に research/legacy/common_utils/target_utils.py へ移した。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def v2_forward_returns(prices: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """v2 のラベル（設計書 v2 ラベリング設計・監査 G-1/B）。行 D（判断日 = 設計書の t+1）に直接置く。

    prices は東京営業日（XTKS）で連続した index を持ち、分割調整済みの adj_open / adj_close を含むこと。
    欠損営業日は NaN の行として残す（shift が営業日単位で正しく数えられるように）。

    fwd_oc_{h}: ln(C[D+h-1] / O[D])  D の寄付で買い、h 営業日目の大引けで評価
    fwd_oo_{h}: ln(O[D+h]   / O[D])  D の寄付で買い、h 営業日後の寄付で評価
    gap       : ln(O[D] / C[D-1])    前営業日の大引け→D の寄付（判断時刻には未知。分析専用で特徴量にしない）
    """
    o = prices["adj_open"].astype(float)
    c = prices["adj_close"].astype(float)
    out = pd.DataFrame(index=prices.index)
    for h in horizons:
        out[f"fwd_oc_{h}"] = np.log(c.shift(-(h - 1)) / o)
        out[f"fwd_oo_{h}"] = np.log(o.shift(-h) / o)
    out["gap"] = np.log(o / c.shift(1))
    return out.astype("float64")


def v2_barrier_labels(
    prices: pd.DataFrame,
    sigma: pd.Series,
    g,
    max_hold: int,
    vol_scale_days: int,
    fill_within=(5, 20),
) -> pd.DataFrame:
    """v2 の本ラベル（設計書 v2 ラベリング設計）。行 D（判断日）に直接置く。

    D の寄付（adj_open）で買い、買値 × (1 + g) の売り指値が約定するまで保有する。
    - 始値が指値以上の日（D の翌日以降）は始値で約定、日中の高値が指値に届いた日は指値で約定とみなす
      （買った日の高値は寄付後に付くので D 当日も約定しうる）。
    - max_hold 営業日目までに約定しなければ、その日の大引けで評価する（学習用の打ち切り。運用では売却日を決めない）。
    - 窓の中に欠損営業日（NaN）があるか、データ終端までに結果が出ない行は NaN。
    価格は分割調整済み（adj_*）。σ̂ は判断時刻までに既知の日次ボラ（sigma）で、y = ret / (σ̂ √vol_scale_days)。

    列: bar_g, bar_ret（ln(売値/買値)）, bar_y, bar_days_to_fill（約定まで何営業日目か。打ち切りは NaN）,
        bar_censored（1=打ち切り）, bar_mae（約定までの最大含み損 ln(安値/買値)）, bar_filled_{k}（k 日以内に約定）
    """
    idx = prices.index
    o = prices["adj_open"].to_numpy(dtype=float)
    h = prices["adj_high"].to_numpy(dtype=float)
    lo = prices["adj_low"].to_numpy(dtype=float)
    c = prices["adj_close"].to_numpy(dtype=float)
    n = len(idx)
    g_arr = (g.reindex(idx).to_numpy(dtype=float) if isinstance(g, pd.Series)
             else np.full(n, float(g)))
    sig = sigma.reindex(idx).to_numpy(dtype=float)

    ret = np.full(n, np.nan)
    days = np.full(n, np.nan)
    cens = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(o[i]) and np.isfinite(g_arr[i]) and g_arr[i] > 0):
            continue
        limit = o[i] * (1.0 + g_arr[i])
        worst = np.inf
        for j in range(i, min(i + max_hold, n)):
            if not (np.isfinite(o[j]) and np.isfinite(h[j]) and np.isfinite(lo[j]) and np.isfinite(c[j])):
                break                                   # 欠損営業日: 結果不明
            if j > i and o[j] >= limit:                 # 寄付で指値を超えた: 始値で約定（その日の安値は約定後）
                ret[i], days[i], cens[i], mae[i] = np.log(o[j] / o[i]), j - i + 1, 0.0, np.log(min(worst, o[i]) / o[i])
                break
            worst = min(worst, lo[j])
            if h[j] >= limit:                           # 日中に指値へ届いた
                ret[i], days[i], cens[i], mae[i] = np.log(limit / o[i]), j - i + 1, 0.0, np.log(worst / o[i])
                break
            if j == i + max_hold - 1:                   # 打ち切り: 最終日の大引けで評価
                ret[i], cens[i], mae[i] = np.log(c[j] / o[i]), 1.0, np.log(worst / o[i])

    out = pd.DataFrame(index=idx)
    out["bar_g"] = g_arr
    out["bar_ret"] = ret
    out["bar_y"] = ret / (sig * np.sqrt(vol_scale_days))
    out["bar_days_to_fill"] = days
    out["bar_censored"] = cens
    out["bar_mae"] = mae
    known = np.isfinite(ret)
    for k in fill_within:
        out[f"bar_filled_{k}"] = np.where(known, (np.nan_to_num(days, nan=np.inf) <= k).astype(float), np.nan)
    return out.astype("float64")
