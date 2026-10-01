"""v2 の学習（設計書 v2 モデル構成・評価設計、フェーズ3）: Huber 回帰と浅い LightGBM の単純平均を入れ子 CPCV で。

外側: CPCV（cpcv_n_groups・cpcv_n_test_groups、ラベル期間で purge、embargo）。各分割のテスト行を1回だけ予測する。
内側: 外側の学習グループで leave-one-group-out（内側の検証グループの周りも purge・embargo）。
  候補 g（固定率・週次ボラ倍率）× Huber の α × シグナルの k を、検証グループをシミュレータで回した
  コスト込みの時価評価シャープの平均で選ぶ。LightGBM の木の本数は内側の検証グループで early stopping し、
  外側の最終学習では選ばれた g の中央値を使う。
標準化は学習 fold の統計量だけ。重みは学習 fold 内の平均独自性（uniqueness）。特徴量選択はしない。
フェーズ4（2026-10-01）で不合格。保守しないが、research/trial_log/phase3 の結果を再現できるように動く状態で残す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_utils.cpcv import CombinatorialPurgedCV, label_end_positions, uniqueness_weights  # noqa: E402
from common_utils.position_simulator import PositionRules, simulate_positions, summarize  # noqa: E402
from research.v2_model.signal_rule import buy_signal  # noqa: E402
from common_utils.target_utils import v2_barrier_labels  # noqa: E402


def design_matrix(features: pd.DataFrame, cols: list[str], interactions: list[list[str]]) -> pd.DataFrame:
    X = features[list(cols)].copy()
    for a, b in interactions:
        X[f"{a}_x_{b}"] = features[a] * features[b]
    return X.astype("float64")


def fit_huber(X: pd.DataFrame, y: pd.Series, w: np.ndarray, alpha: float, epsilon: float, max_iter: int):
    scaler = StandardScaler().fit(X.to_numpy())                 # 学習 fold だけで標準化
    model = HuberRegressor(alpha=alpha, epsilon=epsilon, max_iter=max_iter)
    model.fit(scaler.transform(X.to_numpy()), y.to_numpy(), sample_weight=w)
    return scaler, model


def fit_lgbm(X: pd.DataFrame, y: pd.Series, w: np.ndarray, params: dict, n_estimators: int | None = None,
             X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None, early_stopping_rounds: int | None = None):
    import lightgbm as lgb
    p = dict(params)
    n_max = int(p.pop("n_estimators", 1000))
    p.setdefault("verbose", -1)
    model = lgb.LGBMRegressor(n_estimators=int(n_estimators or n_max), n_jobs=1, deterministic=True,
                              force_row_wise=True, **p)
    if X_val is not None and early_stopping_rounds and len(X_val) > 0:
        model.fit(X.to_numpy(), y.to_numpy(), sample_weight=w, eval_set=[(X_val.to_numpy(), y_val.to_numpy())],
                  callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)])
        best = int(model.best_iteration_ or n_max)
    else:
        model.fit(X.to_numpy(), y.to_numpy(), sample_weight=w)
        best = int(n_estimators or n_max)
    return model, best


def _g_candidates(cfg: dict) -> list[tuple[str, float]]:
    return ([("fixed", float(g)) for g in cfg["barrier_g_fixed"]]
            + [("vol", float(m)) for m in cfg["barrier_g_vol_multiples"]])


def _g_series(kind: str, value: float, sigma: pd.Series, v: int):
    return value if kind == "fixed" else value * sigma * np.sqrt(v)


def _sharpe_on_block(prices: pd.DataFrame, rows: np.ndarray, signal: pd.Series, g, rules: PositionRules) -> float:
    block = prices.iloc[rows]
    gg = g.iloc[rows] if isinstance(g, pd.Series) else g
    daily, trades = simulate_positions(block, signal.reindex(block.index).fillna(False), gg, rules)
    s = summarize(daily, trades)["sharpe"]
    return 0.0 if not np.isfinite(s) else float(s)


def run_nested_cpcv(ds: dict, cfg: dict, rules: PositionRules, progress=None) -> dict:
    feats, prices = ds["features"], ds["prices"]
    idx = feats.index
    n = len(idx)
    v = int(cfg["barrier_vol_scale_days"])
    sigma = feats["semi_ewm_vol"]
    X_all = design_matrix(feats, cfg["model_features"], cfg["model_interactions"])
    x_ok = (X_all.notna().all(axis=1) & sigma.notna()).to_numpy()
    c = float(cfg["round_trip_cost"])
    cands = _g_candidates(cfg)
    labels = {cand: v2_barrier_labels(prices, sigma, _g_series(*cand, sigma, v), int(cfg["barrier_max_hold_days"]), v,
                                      tuple(cfg["barrier_fill_within_days"])) for cand in cands}
    # purge には全候補の中で最も長いラベル期間を使う（保守的）
    end = np.max([label_end_positions(l["bar_days_to_fill"].to_numpy(), l["bar_censored"].to_numpy(),
                                      int(cfg["barrier_max_hold_days"]), n) for l in labels.values()], axis=0)
    cv = CombinatorialPurgedCV(cfg["cpcv_n_groups"], cfg["cpcv_n_test_groups"], cfg["cpcv_embargo_days"])
    groups = cv.groups(n)
    hp = dict(cfg["lgbm_params"])
    es = int(cfg["lgbm_early_stopping_rounds"])

    def rows_with_label(rows, lab):
        return rows[x_ok[rows] & lab["bar_y"].iloc[rows].notna().to_numpy()]

    pred_rows, trial_rows, split_info = [], [], []
    for sid, train, test, gids in cv.split(end):
        scores: dict = {}
        best_iters: dict = {cand: [] for cand in cands}
        used: set = set()
        for vg in [g for g in range(cfg["cpcv_n_groups"]) if g not in gids]:
            val = np.intersect1d(groups[vg], train)
            if len(val) == 0:
                continue
            inner = cv.purge(np.setdiff1d(train, groups[vg]), [groups[vg]], end)
            va = val[x_ok[val]]
            for cand in cands:
                lab = labels[cand]
                tr = rows_with_label(inner, lab)
                va_l = rows_with_label(va, lab)
                if len(tr) < 50 or len(va) == 0:
                    continue
                w = uniqueness_weights(end, n, tr)
                y = lab["bar_y"]
                lgbm, best = fit_lgbm(X_all.iloc[tr], y.iloc[tr], w, hp, X_val=X_all.iloc[va_l], y_val=y.iloc[va_l],
                                      early_stopping_rounds=es)
                best_iters[cand].append(best)
                p_l = lgbm.predict(X_all.iloc[va].to_numpy())
                g = _g_series(*cand, sigma, v)
                for alpha in cfg["huber_alpha_grid"]:
                    sc, hub = fit_huber(X_all.iloc[tr], y.iloc[tr], w, alpha, cfg["huber_epsilon"], cfg["huber_max_iter"])
                    y_hat = pd.Series((hub.predict(sc.transform(X_all.iloc[va].to_numpy())) + p_l) / 2, index=idx[va])
                    for k in cfg["signal_k_grid"]:
                        sig_ = buy_signal(y_hat, sigma, c, float(k), v)
                        scores.setdefault((cand, float(alpha), float(k)), []).append(_sharpe_on_block(prices, val, sig_, g, rules))
                used |= set(tr.tolist()) | set(va.tolist())

        best_key = max(scores, key=lambda key: (np.mean(scores[key]), key[2], -key[1]))
        for key, vals in scores.items():
            (kind, gv), alpha, k = key
            trial_rows.append({"split": sid, "g_kind": kind, "g_value": gv, "alpha": alpha, "k": k,
                               "inner_sharpe": float(np.mean(vals)), "inner_sharpes": [round(x, 4) for x in vals],
                               "selected": key == best_key})

        (cand, alpha, k) = best_key
        lab = labels[cand]
        tr = rows_with_label(train, lab)
        w = uniqueness_weights(end, n, tr)
        y = lab["bar_y"]
        n_est = int(np.median(best_iters[cand])) if best_iters[cand] else None
        sc, hub = fit_huber(X_all.iloc[tr], y.iloc[tr], w, alpha, cfg["huber_epsilon"], cfg["huber_max_iter"])
        lgbm, _ = fit_lgbm(X_all.iloc[tr], y.iloc[tr], w, hp, n_estimators=n_est)
        te_ok = test[x_ok[test]]
        ph = pd.Series(np.nan, index=idx[test])
        pl = pd.Series(np.nan, index=idx[test])
        if len(te_ok):
            ph.loc[idx[te_ok]] = hub.predict(sc.transform(X_all.iloc[te_ok].to_numpy()))
            pl.loc[idx[te_ok]] = lgbm.predict(X_all.iloc[te_ok].to_numpy())
        y_hat = (ph + pl) / 2
        g = _g_series(*cand, sigma, v)
        g_rows = g.reindex(idx[test]).to_numpy() if isinstance(g, pd.Series) else np.full(len(test), g)
        sig_ = buy_signal(y_hat, sigma, c, float(k), v)
        scale = sigma.reindex(idx[test]) * np.sqrt(v)
        for j, r in enumerate(test):
            d = idx[r]
            pred_rows.append({"split": sid, "row": int(r), "Date": d, "test_group": int(np.searchsorted([gg[-1] for gg in groups], r)),
                              "y_hat_huber": ph[d], "y_hat_lgbm": pl[d], "y_hat": y_hat[d], "sigma": sigma[d],
                              "mu_hat": y_hat[d] * scale[d], "signal": bool(sig_[d]), "g": g_rows[j],
                              "g_kind": cand[0], "g_value": cand[1], "alpha": alpha, "k": k, "n_estimators": n_est,
                              "y_true": lab["bar_y"].iloc[r], "ret_true": lab["bar_ret"].iloc[r]})
        split_info.append({"split": sid, "test_groups": list(gids), "train_rows": train.tolist(), "test_rows": test.tolist(),
                           "inner_rows_used": sorted(used), "selected": {"g_kind": cand[0], "g_value": cand[1],
                                                                          "alpha": alpha, "k": k, "n_estimators": n_est}})
        if progress:
            progress(sid, best_key, np.mean(scores[best_key]))

    return {"predictions": pd.DataFrame(pred_rows), "trials": pd.DataFrame(trial_rows), "splits": split_info,
            "paths": cv.paths(), "groups": [g.tolist() for g in groups]}
