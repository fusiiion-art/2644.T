"""設計書 v2 フェーズ4（短縮版）: 新しい学習はせず、フェーズ3の出力でモデルを判定する。

- モデル: research/trial_log/phase3/ の15分割の予測を5パスにつなぎ、シミュレータで時価評価する
- ベースライン: 毎朝買う（g は config の phase4_baseline_g）、何もしない、SOX 当夜リターンが正の朝だけ買う、
  買う確率をモデルに合わせたランダム売買（帰無分布）。特徴量3個の Ridge は学習が要るので行わない
- DSR: N = フェーズ3までの累計 + この実行で増えた試行（SOX 符号の g 2通り）。V はフェーズ3内側の試行のシャープの分散
- PBO: フェーズ3内側の (検証グループ × 候補63) のシャープ行列で CSCV
- 開発期間（dev_end まで）だけを使う。凍結ホールドアウトは読まない
出力: research/trial_log/phase4/summary.json
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from common_utils.cpcv import CombinatorialPurgedCV  # noqa: E402
from common_utils.position_simulator import PositionRules, simulate_positions, summarize  # noqa: E402
from common_utils.selection_bias import deflated_sharpe, pbo_from_block_scores, probabilistic_sharpe  # noqa: E402
from generate.features_v2 import build_v2_dataset, v2_config  # noqa: E402

P3 = ROOT / "research" / "trial_log" / "phase3"
OUT = ROOT / "research" / "trial_log" / "phase4"


def model_paths(pred: pd.DataFrame, cfg: dict) -> list[pd.DataFrame]:
    cv = CombinatorialPurgedCV(cfg["cpcv_n_groups"], cfg["cpcv_n_test_groups"], cfg["cpcv_embargo_days"])
    out = []
    for path in cv.paths():
        parts = [pred[(pred["split"] == sid) & (pred["test_group"] == g)] for g, sid in sorted(path.items())]
        out.append(pd.concat(parts).set_index("Date").sort_index())
    return out


def run(prices: pd.DataFrame, signal: pd.Series, g, rules: PositionRules, ppy: int) -> tuple[dict, pd.Series]:
    daily, trades = simulate_positions(prices, signal, g, rules)
    s = summarize(daily, trades, ppy)
    s["trades_per_year"] = s["n_trades"] / (len(daily) / ppy)
    return s, daily["ret"]


def rolling_ic_positive_share(seq: pd.DataFrame, window: int) -> float:
    ok = seq.dropna(subset=["y_hat", "y_true"])
    ics = [ok["y_hat"].iloc[i - window:i].corr(ok["y_true"].iloc[i - window:i], method="spearman")
           for i in range(window, len(ok) + 1)]
    ics = np.asarray(ics, dtype=float)
    return float(np.mean(ics[np.isfinite(ics)] > 0)) if len(ics) else np.nan


def main() -> dict:
    cfg = v2_config()
    rules = PositionRules.from_config()
    ppy = int(cfg["trading_days_per_year"])
    end = pd.Timestamp(cfg["dev_end"])
    ds = build_v2_dataset()
    prices = ds["prices"].loc[:end]
    feats = ds["features"].loc[:end]

    pred = pd.read_csv(P3 / "predictions.csv", parse_dates=["Date"])
    trials = pd.read_csv(P3 / "trials.csv")
    assert pred["Date"].max() <= end, "フェーズ3の予測に凍結ホールドアウトが混ざっている"

    # --- モデルの5パス
    paths = model_paths(pred, cfg)
    model = []
    for p, seq in enumerate(paths):
        assert seq.index.equals(prices.index), f"パス {p} が開発期間の全行をちょうど1回ずつ覆っていない"
        s, r = run(prices, seq["signal"].astype(bool), seq["g"], rules, ppy)
        s.update(path=p, buy_rate=float(seq["signal"].mean()),
                 rolling_ic_positive_share=rolling_ic_positive_share(seq, int(cfg["phase4_rolling_ic_window"])))
        model.append((s, r, seq))

    # --- ベースライン（決定的なもの）
    always = pd.Series(True, index=prices.index)
    sox_pos = feats["sox_ret_overnight"] > 0
    base = {}
    for g in cfg["phase4_baseline_g"]:
        base[f"always_buy_g{g}"] = run(prices, always, float(g), rules, ppy)
        base[f"sox_sign_g{g}"] = run(prices, sox_pos, float(g), rules, ppy)
    base["do_nothing"] = ({"sharpe": 0.0, "ann_return_on_capital": 0.0, "max_drawdown_jpy": 0.0,
                           "n_trades": 0, "trades_per_year": 0.0}, pd.Series(0.0, index=prices.index))
    best_base = max(v[0]["sharpe"] for v in base.values())

    # --- ランダム売買（買う確率と g をパスに合わせる）
    rng = np.random.default_rng(int(cfg["phase4_random_seed"]))
    for s, _, seq in model:
        draws = []
        for _ in range(int(cfg["phase4_random_draws"])):
            sig = pd.Series(rng.random(len(seq)) < s["buy_rate"], index=seq.index)
            draws.append(run(prices, sig, seq["g"], rules, ppy)[0]["sharpe"])
        draws = np.asarray(draws, dtype=float)
        s["random_sharpe_median"] = float(np.nanmedian(draws))
        s["random_percentile_of_model"] = float(np.mean(draws < s["sharpe"]))

    # --- DSR
    new_trials = len(cfg["phase4_baseline_g"])          # SOX 符号の g ぶん（毎朝買うの g 比較はフェーズ2で計上済み）
    n_trials = int(cfg["phase4_trials_before"]) + new_trials
    v = float(np.var(trials["inner_sharpe"].to_numpy() / np.sqrt(ppy), ddof=1))
    for s, r, _ in model:
        s.update({f"dsr_{k}": val for k, val in deflated_sharpe(r, n_trials, v).items()})
    mean_path_ret = pd.concat([r for _, r, _ in model], axis=1).mean(axis=1)
    pooled = deflated_sharpe(mean_path_ret, n_trials, v)

    # 参考: 毎朝買うルール自体の PSR（N はこのルールの g の候補数だけ）
    always_rets = [base[f"always_buy_g{g}"][1] for g in cfg["phase4_baseline_g"]]
    v_always = float(np.var([r.mean() / r.std(ddof=1) for r in always_rets], ddof=1))
    always_ref = {f"g{g}": {"psr_vs_0": probabilistic_sharpe(r, 0.0),
                            **deflated_sharpe(r, len(always_rets), v_always)}
                  for g, r in zip(cfg["phase4_baseline_g"], always_rets)}

    # --- PBO（分割ごとに 検証グループ × 候補 の行列）
    mats = []
    for _, grp in trials.groupby("split"):
        rows = [ast.literal_eval(x) for x in grp["inner_sharpes"]]
        if len({len(x) for x in rows}) == 1 and len(rows[0]) % 2 == 0:
            mats.append(np.asarray(rows, dtype=float).T)
    pbo = pbo_from_block_scores(mats)

    # --- 判定
    sh = [s["sharpe"] for s, _, _ in model]
    verdict = {
        "dsr_pass": bool(pooled["dsr"] >= cfg["phase4_dsr_pass"]),
        "pbo_pass": bool(pbo["pbo"] < cfg["phase4_pbo_pass"]),
        "beats_all_baselines_every_path": bool(min(sh) > best_base),
        "rolling_ic_majority_positive_every_path": bool(all(s["rolling_ic_positive_share"] > 0.5 for s, _, _ in model)),
        "trades_per_year_ok_every_path": bool(all(s["trades_per_year"] >= cfg["phase4_min_trades_per_year"] for s, _, _ in model)),
        "interval_coverage": "未評価（分位点を出すモデルが無い）",
    }
    verdict["pass"] = all(v_ for k, v_ in verdict.items() if isinstance(v_, bool))

    summary = {
        "dev_period": [str(prices.index[0].date()), str(end.date())],
        "n_trials_for_dsr": n_trials, "new_trials_this_run": new_trials, "trial_sharpe_var_per_period": v,
        "model_paths": [{k: val for k, val in s.items()} for s, _, _ in model],
        "model_mean_path": pooled,
        "baselines": {k: v_[0] for k, v_ in base.items()},
        "always_buy_reference": always_ref,
        "pbo": pbo,
        "verdict": verdict,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: s[k] for k in ("n_trials_for_dsr", "model_mean_path", "pbo", "verdict")}, ensure_ascii=False, indent=1, default=float))
    for p in s["model_paths"]:
        print(p["path"], round(p["sharpe"], 3), round(p["dsr_dsr"], 4), round(p["trades_per_year"], 1),
              round(p["rolling_ic_positive_share"], 2), round(p["random_percentile_of_model"], 2), round(p["random_sharpe_median"], 3))
    for k, b in s["baselines"].items():
        print(k, round(b["sharpe"], 3), round(b.get("trades_per_year", 0), 1), b.get("max_drawdown_jpy"))
    print(json.dumps(s["always_buy_reference"], ensure_ascii=False, indent=1, default=float))
