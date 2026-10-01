"""設計書 v2 フェーズ3: 入れ子 CPCV で Huber＋浅い LightGBM を学習し、各 fold の予測と試行ログを残す。

- データは開発期間（config の dev_end まで）だけ。凍結ホールドアウトは使わない。
- 出力は research/trial_log/phase3/ 以下（予測・試行ログ・分割ごとの選択・パスの参考評価）。
- パスの時価評価は参考値。DSR・PBO・ベースライン比較・ホールドアウトでの判定はフェーズ4で行う。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from common_utils.position_simulator import PositionRules, simulate_positions, summarize  # noqa: E402
from research.v2_model.trainer_v2 import run_nested_cpcv  # noqa: E402
from generate.features_v2 import build_v2_dataset, v2_config  # noqa: E402

OUT = ROOT / "research" / "trial_log" / "phase3"


def path_preview(res: dict, prices: pd.DataFrame, rules: PositionRules) -> list[dict]:
    pred = res["predictions"]
    out = []
    for p, path in enumerate(res["paths"]):
        parts = [pred[(pred["split"] == sid) & pred["row"].isin(res["groups"][g])] for g, sid in sorted(path.items())]
        seq = pd.concat(parts).set_index("Date").sort_index()
        daily, trades = simulate_positions(prices.loc[seq.index], seq["signal"], seq["g"], rules)
        s = summarize(daily, trades)
        s.update(path=p, n_buy_signals=int(seq["signal"].sum()))
        out.append(s)
    return out


def main() -> None:
    cfg = v2_config()
    rules = PositionRules.from_config()
    ds = build_v2_dataset()
    end = pd.Timestamp(cfg["dev_end"])
    ds_dev = {"features": ds["features"].loc[:end], "prices": ds["prices"].loc[:end]}
    t0 = time.time()
    res = run_nested_cpcv(ds_dev, cfg, rules,
                          progress=lambda sid, key, sc: print(f"split {sid:2d}: g={key[0]} α={key[1]} k={key[2]} 内側シャープ={sc:+.2f}", flush=True))
    OUT.mkdir(parents=True, exist_ok=True)
    res["predictions"].to_csv(OUT / "predictions.csv", index=False)
    res["trials"].to_csv(OUT / "trials.csv", index=False)
    splits = [{k: v for k, v in s.items() if k not in ("train_rows", "test_rows", "inner_rows_used")}
              | {"n_train": len(s["train_rows"]), "n_test": len(s["test_rows"]), "n_inner_used": len(s["inner_rows_used"])}
              for s in res["splits"]]
    preview = path_preview(res, ds_dev["prices"], rules)
    pred = res["predictions"]
    ok = pred.dropna(subset=["y_hat", "y_true"])
    summary = {
        "dev_period": [str(ds_dev["features"].index[0].date()), str(end.date())],
        "n_rows": int(len(ds_dev["features"])), "n_splits": len(res["splits"]), "n_paths": len(res["paths"]),
        "n_inner_trials": int(len(res["trials"])),
        "oos_ic_spearman_pooled": float(ok["y_hat"].corr(ok["y_true"], method="spearman")),
        "selected_per_split": splits,
        "path_preview_not_final": preview,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "selected_per_split"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
