"""設計書 v2 フェーズ2: 翌朝判断に上積みがあるかを、少数の変数の IC だけで確かめる。

判定ルール（結果を見る前に固定。設計書 v2 フェーズ2の完了条件）:
- 主判定: 変数 {sox_ret_overnight, semi_mom_5, semi_mom_20} × ホライズン h ∈ {1, 5, 20} の9検定。
  ラベルは fwd_oc_h（行 D の寄付で買い、h 営業日目の大引けで評価。v2 ラベルの打ち切り評価と同じ形）。
- 検定: Spearman IC を順位の回帰係数として Newey–West（lag = max(h-1, 5)）で t 値にし、両側 p 値。
  多重比較は Bonferroni（α = 0.05 / 9）。符号は問わない。
- 参考（判定に使わない。試行ログには数える）: fwd_oo_h（寄付→寄付）、寄付前ギャップ gap との IC、
  現行と同じ t-1 版（sox_ret_overnight_old）。
- 日経先物の夜間変化はデータが無く未測定（キャッシュに無い）。
- 期間は開発期間（〜2026-07-03）のみ。凍結ホールドアウト（2026-07-06〜）は使わない。

注意: 夜間 SOX と「翌始値→翌々始値」の関係（Spearman −0.118）は 2026-09-30 の探索で既に見ており、
この検定は事前登録にはならない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from generate.features_v2 import build_v2_dataset  # noqa: E402

DEV_END = pd.Timestamp("2026-07-03")
PRIMARY_VARS = ["sox_ret_overnight", "semi_mom_5", "semi_mom_20"]
HORIZONS = [1, 5, 20]
ALPHA = 0.05


def hac_spearman(x: np.ndarray, y: np.ndarray, lag: int) -> dict:
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    b = float(np.mean(rx * ry))
    u = rx * (ry - b * rx)
    n = len(u)
    s = float(np.mean(u * u))
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * float(np.mean(u[k:] * u[:-k]))
    t = b / np.sqrt(s / n)
    return {"ic": b, "t_hac": float(t), "p_hac": float(2 * (1 - norm.cdf(abs(t)))), "n": int(n),
            "p_naive": float(spearmanr(x, y).pvalue)}


def measure(features: pd.DataFrame, labels: pd.DataFrame, var: str, label: str, lag: int) -> dict:
    d = features[[var]].join(labels[[label]]).dropna()
    d = d[d.index <= DEV_END]
    res = hac_spearman(d[var].to_numpy(), d[label].to_numpy(), lag)
    half = d.index[len(d) // 2]
    res["ic_first_half"] = float(spearmanr(d.loc[:half, var], d.loc[:half, label]).statistic)
    res["ic_second_half"] = float(spearmanr(d.loc[half:, var], d.loc[half:, label]).statistic)
    res["period"] = [str(d.index[0].date()), str(d.index[-1].date())]
    return res


def main() -> dict:
    ds = build_v2_dataset()
    f, lab = ds["features"], ds["labels"]
    bonf = ALPHA / (len(PRIMARY_VARS) * len(HORIZONS))
    rows = []
    for var in PRIMARY_VARS:
        for h in HORIZONS:
            r = measure(f, lab, var, f"fwd_oc_{h}", max(h - 1, 5))
            rows.append({"role": "primary", "var": var, "label": f"fwd_oc_{h}", **r,
                         "significant_bonferroni": r["p_hac"] < bonf})
    for var in PRIMARY_VARS + ["sox_ret_overnight_old"]:
        for h in HORIZONS:
            r = measure(f, lab, var, f"fwd_oo_{h}", max(h - 1, 5))
            rows.append({"role": "reference", "var": var, "label": f"fwd_oo_{h}", **r})
        r = measure(f, lab, var, "gap", 5)
        rows.append({"role": "reference", "var": var, "label": "gap", **r})
    for h in HORIZONS:
        r = measure(f, lab, "sox_ret_overnight_old", f"fwd_oc_{h}", max(h - 1, 5))
        rows.append({"role": "reference", "var": "sox_ret_overnight_old", "label": f"fwd_oc_{h}", **r})

    table = pd.DataFrame(rows)
    primary = table[table.role == "primary"]
    passed = primary[primary.significant_bonferroni]
    summary = {
        "bonferroni_alpha": bonf,
        "n_tests_total": int(len(table)),
        "n_primary": int(len(primary)),
        "decision": ("フェーズ3へ進む（有意: " + ", ".join(f"{r.var}×{r.label}" for r in passed.itertuples()) + "）")
                    if len(passed) else "有意な IC なし。特徴量の設計から見直す（フェーズ3に進まない）",
        "unmeasured": ["日経225先物の夜間変化（データ未取得）"],
        "table": table.to_dict(orient="records"),
    }
    out = ROOT / "research" / "trial_log" / "2026-09-30_phase2_ic.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(table[["role", "var", "label", "ic", "t_hac", "p_hac", "p_naive", "n", "ic_first_half", "ic_second_half"]]
          .round(4).to_string(index=False))
    print(f"\nBonferroni α = {bonf:.4f}  →  {summary['decision']}")
    return summary


if __name__ == "__main__":
    main()
