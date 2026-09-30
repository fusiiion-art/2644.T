"""audit/results/wf_*.json と予測 CSV から F の表（Markdown）を作る。"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

R = Path(__file__).resolve().parent / "results"
KEYS = [("hit_rate", "方向的中率", "{:.1%}"), ("always_up_hit_rate", "「常に上昇」の的中率", "{:.1%}"),
        ("IC_spearman", "IC (Spearman)", "{:+.3f}"), ("IC_pvalue", "IC の p 値", "{:.2f}"),
        ("sharpe_cost0.00%", "年率シャープ 往復0%", "{:+.2f}"), ("sharpe_cost0.05%", "年率シャープ 往復0.05%", "{:+.2f}"),
        ("sharpe_cost0.10%", "年率シャープ 往復0.10%", "{:+.2f}"), ("round_trips", "取引回数（往復）", "{:.0f}"),
        ("days_in_market", "保有日数", "{:.0f}"), ("buyhold_sharpe", "参考: B&H シャープ(コスト0)", "{:+.2f}")]


def table(variant):
    d = json.loads((R / f"wf_{variant}.json").read_text())
    per = pd.DataFrame(d["per_seed"]).T
    ens = d["seed_mean_prediction"]
    lines = [f"| 指標 | {len(per)}シード中央値 [最小, 最大] | シード平均予測 |", "|---|---|---|"]
    for k, name, fmt in KEYS:
        s = per[k].astype(float)
        lines.append(f"| {name} | {fmt.format(s.median())} [{fmt.format(s.min())}, {fmt.format(s.max())}] | {fmt.format(ens[k])} |")
    meta = {"oos": f'{d["oos_first_date"]} 〜 {d["oos_last_date"]}', "n_eval_days": ens["n_eval_days"],
            "n_with_pred": ens["n_with_pred"], "status": d["status_counts_oos"]}
    # DSR 入力用: シード平均予測・コスト0 の日次戦略リターンの歪度/尖度
    p = pd.read_csv(R / f"wf_{variant}_predictions.csv").dropna(subset=["realized_open_to_open"])
    pos = (p.pred_seed_mean > 0).astype(float)
    ret = pos * p.realized_open_to_open
    meta["dsr_inputs"] = {"T_days": int(len(ret)), "skew": float(skew(ret)), "kurtosis_pearson": float(kurtosis(ret, fisher=False))}
    return "\n".join(lines), meta


if __name__ == "__main__":
    for v in sys.argv[1:] or ["asis", "uslag1", "bfix"]:
        t, m = table(v)
        print(f"### {v}\n{json.dumps(m, ensure_ascii=False)}\n{t}\n")
