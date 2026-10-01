"""選択バイアスの補正（設計書 v2 評価設計・フェーズ4）。

- PSR（Probabilistic Sharpe Ratio）: 歪度・尖度を考慮して、真のシャープが基準値を超える確率。
- DSR（Deflated Sharpe Ratio）: 基準値を「N 回試したときに偶然出る最大シャープの期待値」にした PSR。
  Bailey & López de Prado (2014)。シャープはすべて1期間（日次）単位で扱う。
- PBO（Probability of Backtest Overfitting）: CSCV。ブロックを半分ずつ IS / OOS に分ける全組み合わせで、
  IS で一番良かった候補の OOS 順位が中央値以下になる割合。Bailey et al. (2017)。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(returns) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else np.nan


def probabilistic_sharpe(returns, sr_benchmark: float) -> float:
    """PSR(SR*) = Φ((SR − SR*)·√(T−1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²))。γ4 は超過でない尖度。"""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    sr = sharpe_per_period(r)
    if t < 3 or not np.isfinite(sr):
        return np.nan
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, trial_sharpe_var: float) -> float:
    """N 本の独立な試行（真のシャープ0、分散 V）の最大値の期待値の近似。"""
    n = int(n_trials)
    if n <= 1 or trial_sharpe_var <= 0:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(trial_sharpe_var) * ((1 - EULER_GAMMA) * z(1 - 1 / n) + EULER_GAMMA * z(1 - 1 / (n * np.e))))


def deflated_sharpe(returns, n_trials: int, trial_sharpe_var: float) -> dict:
    """returns は日次リターン。trial_sharpe_var は試行間の（日次）シャープの分散。"""
    sr0 = expected_max_sharpe(n_trials, trial_sharpe_var)
    return {"sr_per_period": sharpe_per_period(returns), "sr0_per_period": sr0,
            "n_trials": int(n_trials), "trial_sharpe_var": float(trial_sharpe_var),
            "dsr": probabilistic_sharpe(returns, sr0)}


def pbo_from_block_scores(matrices) -> dict:
    """matrices: (S ブロック × N 候補) のスコア行列のリスト（S は偶数）。各行列で C(S, S/2) 通りに
    IS / OOS を分け、IS 平均が最大の候補の OOS 平均での相対順位 ω から λ = logit(ω) を出す。
    PBO = λ ≤ 0 の割合（全行列の組み合わせをまとめて数える）。"""
    lambdas = []
    for m in matrices:
        m = np.asarray(m, dtype=float)
        s, n = m.shape
        if s % 2 or s < 2:
            raise ValueError(f"ブロック数は2以上の偶数が必要: {s}")
        for is_rows in combinations(range(s), s // 2):
            oos_rows = [i for i in range(s) if i not in is_rows]
            is_score = m[list(is_rows)].mean(axis=0)
            oos_score = m[oos_rows].mean(axis=0)
            best = int(np.argmax(is_score))
            rank = stats.rankdata(oos_score)[best]          # 1 = 最下位、同順位は平均順位
            w = rank / (n + 1)
            lambdas.append(np.log(w / (1 - w)))
    lam = np.asarray(lambdas)
    return {"pbo": float(np.mean(lam <= 0)), "n_combinations": int(len(lam)),
            "lambda_median": float(np.median(lam)) if len(lam) else np.nan}
