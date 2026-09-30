"""フェーズ3: 入れ子 CPCV（設計書 v2 評価設計・テスト表）。

- test_purge_covers_label_span: 学習 fold のラベル期間（最大20営業日）がテスト fold と重ならない
- test_nested_selection: 特徴量・ハイパラ・g・k の選択が外側テスト fold のデータに触れない
- test_scaler_fit_train_only: 標準化の統計量が学習 fold だけで計算される
- test_signal_cost: c=0 と c=0.1% でシグナル数が単調に減る
"""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from common_utils.cpcv import CombinatorialPurgedCV, label_end_positions, uniqueness_weights
from common_utils.signal_rule import buy_signal
from common_utils.trainer_v2 import fit_huber, run_nested_cpcv


def _ends(n, rng, max_span=20):
    return np.minimum(np.arange(n) + rng.integers(1, max_span + 1, n) - 1, n - 1)


def test_groups_and_split_count():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=20)
    groups = cv.groups(120)
    assert len(groups) == 6 and np.array_equal(np.concatenate(groups), np.arange(120))
    splits = list(cv.split(np.arange(120)))
    assert len(splits) == 15
    for sid, train, test, gids in splits:
        assert set(test) == set(np.concatenate([groups[g] for g in gids]))
        assert not set(train) & set(test)


def test_purge_covers_label_span():
    rng = np.random.default_rng(0)
    n = 600
    end = _ends(n, rng)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=20)
    for sid, train, test, gids in cv.split(end):
        for g in gids:
            block = cv.groups(n)[g]
            a, e = block[0], end[block].max()
            overlap = (train <= e) & (end[train] >= a)          # 学習ラベル [i, end_i] とテスト [a, e] の重なり
            assert not overlap.any()
            embargo = (train > e) & (train <= e + 20)
            assert not embargo.any()


def test_paths_cover_each_group_once():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=20)
    paths = cv.paths()
    assert len(paths) == 5
    split_groups = {sid: set(g) for sid, g in enumerate(combinations(range(6), 2))}
    used = set()
    for path in paths:
        assert sorted(path) == list(range(6))
        for g, sid in path.items():
            assert g in split_groups[sid]
            used.add((sid, g))
    assert len(used) == 15 * 2


def test_uniqueness_weights():
    assert np.allclose(uniqueness_weights(np.array([0, 1, 2]), 3), [1, 1, 1])
    w = uniqueness_weights(np.array([1, 1, 2]), 3)              # 行0: [0,1]、行1: [1,1]、行2: [2,2]
    assert w[0] == pytest.approx((1 + 0.5) / 2) and w[1] == pytest.approx(0.5) and w[2] == pytest.approx(1.0)


def test_label_end_positions():
    days = np.array([1, 3, np.nan, np.nan])
    cens = np.array([0, 0, 1, np.nan])
    end = label_end_positions(days, cens, max_hold=20, n=4)
    assert end.tolist() == [0, 3, 3, 3]                         # 打ち切り・不明は max_hold まで（データ終端で切る）


def test_signal_cost():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2025-01-01", periods=500)
    y = pd.Series(rng.normal(0, 0.5, 500), index=idx)
    sig = pd.Series(0.02, index=idx)
    counts = [int(buy_signal(y, sig, c, 0.0, 5).sum()) for c in [0.0, 0.0005, 0.001, 0.002]]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]
    assert buy_signal(y, sig, 0.001, 0.5, 5).sum() <= buy_signal(y, sig, 0.001, 0.0, 5).sum()


def test_scaler_fit_train_only():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    X.iloc[100:] += 10.0                                        # テスト側だけ大きくずらす
    y = pd.Series(rng.normal(size=200))
    scaler, model = fit_huber(X.iloc[:100], y.iloc[:100], np.ones(100), alpha=0.01, epsilon=1.5, max_iter=500)
    assert np.allclose(scaler.mean_, X.iloc[:100].mean().to_numpy())


# ---------- 入れ子 CPCV（合成データで小さく回す） ----------
def _synthetic_ds(n=420, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, name="Date")
    ret = rng.normal(0.0005, 0.02, n)
    close = 1000 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    prices = pd.DataFrame({"adj_open": open_, "adj_high": high, "adj_low": low, "adj_close": close,
                           "raw_open": open_, "raw_close": close}, index=idx)
    feats = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=["f1", "f2", "f3", "vx"])
    feats["semi_ewm_vol"] = 0.02
    return {"features": feats, "prices": prices}


TINY = {"model_features": ["f1", "f2", "f3", "vx"], "model_interactions": [["f1", "vx"]],
        "barrier_g_fixed": [0.02], "barrier_g_vol_multiples": [0.5], "huber_alpha_grid": [0.01],
        "signal_k_grid": [0.0, 0.5], "cpcv_n_groups": 6, "cpcv_n_test_groups": 2, "cpcv_embargo_days": 20,
        "huber_epsilon": 1.5, "huber_max_iter": 300,
        "lgbm_params": {"objective": "huber", "alpha": 1.5, "num_leaves": 4, "min_child_samples": 20,
                        "learning_rate": 0.05, "n_estimators": 60, "seed": 0, "verbose": -1},
        "lgbm_early_stopping_rounds": 10, "barrier_max_hold_days": 20, "barrier_vol_scale_days": 5,
        "barrier_fill_within_days": [5, 20], "round_trip_cost": 0.001}


@pytest.fixture(scope="module")
def nested():
    from common_utils.position_simulator import PositionRules
    rules = PositionRules(lot_units=100, max_units=300, add_on_drop=0.04, round_trip_cost=0.001, capital_jpy=1_500_000)
    return run_nested_cpcv(_synthetic_ds(), TINY, rules)


def test_nested_selection(nested):
    """内側の選択（g・α・k・木の本数）に使った行は、外側の学習行（purge・embargo 後）に含まれる。"""
    for s in nested["splits"]:
        used = set(s["inner_rows_used"])
        assert used <= set(s["train_rows"])
        assert not used & set(s["test_rows"])


def test_predictions_cover_test_rows_and_trials_are_logged(nested):
    pred = nested["predictions"]
    assert set(pred["split"]) == set(range(15))
    for s in nested["splits"]:
        got = set(pred.loc[pred["split"] == s["split"], "row"])
        assert got == set(s["test_rows"])
    # 試行ログ: 外側15 × g 2 × α 1 × k 2
    assert len(nested["trials"]) == 15 * 2 * 1 * 2
    assert {"split", "g_kind", "g_value", "alpha", "k", "inner_sharpe"} <= set(nested["trials"].columns)
