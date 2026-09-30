"""v2 のシグナル決定ルール（設計書 v2 シグナル決定ルール）。

未保有の朝だけモデルを使う: μ̂ = ŷ · σ̂√v、BUY は μ̂ > c + k · σ̂√v。
保有中の買い増しと売りは固定ルール（common_utils.position_simulator）で、モデルは使わない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def buy_signal(y_hat: pd.Series, sigma: pd.Series, c: float, k: float, vol_scale_days: int) -> pd.Series:
    scale = sigma.reindex(y_hat.index) * np.sqrt(vol_scale_days)
    mu = y_hat * scale
    return ((mu > c + k * scale) & mu.notna() & scale.notna()).astype(bool)
