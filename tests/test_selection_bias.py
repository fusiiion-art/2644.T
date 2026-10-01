"""選択バイアスの補正（PSR・DSR・PBO）のテスト。設計書 v2 評価設計・フェーズ4。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common_utils.selection_bias import (  # noqa: E402
    deflated_sharpe, expected_max_sharpe, pbo_from_block_scores, probabilistic_sharpe,
)


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for n in (10, 100, 1000):
        mc = rng.standard_normal((4000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 1.0) == pytest.approx(mc, abs=0.06)


def test_expected_max_sharpe_scales_with_sd_and_grows_with_n():
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * expected_max_sharpe(100, 1.0))
    assert expected_max_sharpe(5000, 1.0) > expected_max_sharpe(100, 1.0)
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_psr_is_calibrated_under_the_null():
    """真のシャープが0の系列で PSR(0) > 0.95 になる割合は約5%。"""
    rng = np.random.default_rng(1)
    r = rng.standard_normal((3000, 250))
    hits = np.mean([probabilistic_sharpe(x, 0.0) > 0.95 for x in r])
    assert 0.03 <= hits <= 0.07


def test_psr_rises_with_sharpe_and_length():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(500)
    base = base - base.mean()                         # 標本平均を0にそろえ、加えた分だけが平均になるように
    assert probabilistic_sharpe(base + 0.1, 0.0) > probabilistic_sharpe(base, 0.0)
    long = np.tile(base + 0.05, 4)
    assert probabilistic_sharpe(long, 0.0) > probabilistic_sharpe(base + 0.05, 0.0)


def test_dsr_reduces_to_psr_with_one_trial_or_no_dispersion():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(400) * 0.01 + 0.001
    psr0 = probabilistic_sharpe(r, 0.0)
    assert deflated_sharpe(r, n_trials=1, trial_sharpe_var=0.01)["dsr"] == pytest.approx(psr0)
    assert deflated_sharpe(r, n_trials=5000, trial_sharpe_var=0.0)["dsr"] == pytest.approx(psr0)


def test_dsr_falls_as_trials_grow():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(1000) * 0.01 + 0.0008
    d = [deflated_sharpe(r, n, 0.002)["dsr"] for n in (1, 10, 100, 5000)]
    assert all(a >= b for a, b in zip(d, d[1:]))
    assert d[-1] < d[0]


def test_pbo_near_half_for_pure_noise():
    rng = np.random.default_rng(5)
    vals = [pbo_from_block_scores([rng.standard_normal((8, 40))])["pbo"] for _ in range(40)]
    assert 0.4 <= np.mean(vals) <= 0.6


def test_pbo_near_zero_when_skill_persists():
    rng = np.random.default_rng(6)
    skill = np.linspace(0, 3, 30)
    m = skill[None, :] + 0.1 * rng.standard_normal((8, 30))
    assert pbo_from_block_scores([m])["pbo"] == 0.0


def test_pbo_pools_several_matrices_and_rejects_odd_blocks():
    rng = np.random.default_rng(7)
    out = pbo_from_block_scores([rng.standard_normal((4, 20)), rng.standard_normal((4, 20))])
    assert out["n_combinations"] == 12               # C(4,2) × 2
    with pytest.raises(ValueError):
        pbo_from_block_scores([rng.standard_normal((5, 20))])
