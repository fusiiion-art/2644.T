"""再学習つき拡大窓のウォークフォワード検証（監査 G-9 の修正）。

旧版は全期間で学習済みの本番モデルで後半50%を予測していた（in-sample）うえ、予測 k を target[k] と
比べていた（学習ラベルは target[k+1] なので1日ずれ）。本版は:
- wf_refit_every_rows 行ごとに、その時点より前の行だけでレジーム別にモデルを学習し直す
  （学習手順は ExpertTrainer.fit_in_memory = 本番の _train_full_period と同じ関数）。
- 行 k の予測は、直近 SEQUENCE_LENGTH 行（本番の predicter と同じ連続行）から作る。
- actual は学習ラベルと同じ区間（窓の最終行 k → target[k+1]）。
- 出力 CSV の列（regime, actual, pred_<model>）は従来どおりなので、supervisor は真の OOF だけを読む。
数値パラメータは semi2644/config/config.yaml の v2 節（wf_min_train_rows, wf_refit_every_rows）。
"""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


def walk_forward_oof(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], dict],
    predict_fn: Callable[[str, object, pd.DataFrame], float],
    target_col: str,
    min_train: int,
    refit_every: int,
    regime_col: str = "regime",
) -> pd.DataFrame:
    """拡大窓で再学習しながら、行 min_train 以降の各行を1回だけ予測する。

    fit_fn(train_df) -> {model_name: {regime: fitted or None}}。train_df は予測行より前の行だけ。
    predict_fn(model_name, fitted, history_df) -> 予測値。history_df は予測行までの行（その行を含む）。
    """
    rows = []
    n = len(df)
    target = df[target_col].to_numpy(dtype=float)
    for j in range(min_train, n, refit_every):
        fitted = fit_fn(df.iloc[:j])
        for k in range(j, min(j + refit_every, n)):
            regime = df[regime_col].iloc[k]
            row = {"Date": df.index[k], "regime": regime,
                   "actual": target[k + 1] if k + 1 < n else np.nan,
                   "train_rows": j}
            for name, by_regime in fitted.items():
                model = (by_regime or {}).get(regime)
                row[f"pred_{name}"] = predict_fn(name, model, df.iloc[: k + 1]) if model is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows).set_index("Date")


# ---------- 既存パイプライン（timescale_modules）向けの実装 ----------
def _best_finite_params(cfg, regime: str, model_name: str) -> dict | None:
    """BEST_PARAMS、なければ Optuna DB の「値が有限の完了試行」の最良（監査 G-7: ±inf を最良にしない）。"""
    params = getattr(cfg, "BEST_PARAMS", {}).get(regime, {}).get(model_name)
    if params:
        return params
    import optuna
    study_name = f"{cfg.MODULE_NAME}_{regime}_{model_name}_optimization"
    db = PROJECT_ROOT / "logs" / "optuna_db" / f"{study_name}.db"
    if not db.exists():
        return None
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and np.isfinite(t.value)]
    return min(done, key=lambda t: t.value).params if done else None


def _make_fit_fn(cfg, model_map: dict, models: list[str], params: dict, accelerator: str):
    from common_utils.trainer_model import ExpertTrainer

    def fit_fn(train: pd.DataFrame) -> dict:
        out = {m: {} for m in models}
        for regime in cfg.REGIMES:
            df_regime = train[train["regime"] == regime]
            for m in models:
                p = params.get((regime, m))
                if p is None or m not in cfg.REGIME_EXPERTS.get(regime, []):
                    out[m][regime] = None
                    continue
                try:
                    et = ExpertTrainer(m, model_map[m], df_regime.copy(), cfg, PROJECT_ROOT,
                                       f"{cfg.MODULE_NAME}_{regime}_{m}_wf", PROJECT_ROOT / "logs")
                    out[m][regime] = et.fit_in_memory(p, accelerator=accelerator)
                except Exception as e:  # 行数不足など
                    print(f"  ⚠️ {m}/{regime} (train rows={len(df_regime)}): {e}")
                    out[m][regime] = None
        return out
    return fit_fn


def _predict_fn(name: str, fitted: dict, history: pd.DataFrame) -> float:
    import torch
    seq_len = fitted["seq_len"]
    if len(history) < seq_len:
        return np.nan
    x = np.nan_to_num(history[fitted["features"]].to_numpy(dtype=float)[-seq_len:], nan=0.0, posinf=0.0, neginf=0.0)
    x = fitted["scaler"].transform(x).astype(np.float32)
    model = fitted["model"]
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    with torch.no_grad():
        out = model(torch.from_numpy(x).unsqueeze(0).to(device))
    out = out[0] if isinstance(out, (tuple, list)) else out
    return float(out.detach().cpu().reshape(-1)[0])


def run_walk_forward(timescale: str, models: list[str] | None = None, accelerator: str = "auto") -> pd.DataFrame:
    from common_utils.regime_detector import RegimeDetector
    from data.adjust import load_price_config

    cfg = importlib.import_module(f"timescale_modules.{timescale}.config_{timescale}")
    model_map = importlib.import_module(f"timescale_modules.{timescale}.trainer_{timescale}").MODEL_MAP
    v2 = load_price_config()["v2"]

    df = pd.read_parquet(PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME)
    df.columns = [c.replace(".", "_") for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= cfg.DATA_START_DATE].set_index("Date").sort_index()
    df = RegimeDetector.detect_simple_vix(df, cfg.REGIME_VIX_COLUMN, cfg.REGIME_VIX_THRESHOLD)

    models = models or sorted({m for ms in cfg.REGIME_EXPERTS.values() for m in ms if m in model_map})
    params = {(r, m): _best_finite_params(cfg, r, m) for r in cfg.REGIMES for m in models}
    print(f"models={models}  params found for: {[k for k, v in params.items() if v]}")
    fit_fn = _make_fit_fn(cfg, model_map, models, params, accelerator)
    target_col = f"target_{cfg.PREDICTION_HORIZON}"
    return walk_forward_oof(df, fit_fn, _predict_fn, target_col,
                            int(v2["wf_min_train_rows"]), int(v2["wf_refit_every_rows"]))


def _save(result: pd.DataFrame, timescale: str) -> Path:
    base = PROJECT_ROOT / "logs" / f"{timescale}_walkforward"
    base.mkdir(parents=True, exist_ok=True)
    versions = [int(d.name.split("_")[1]) for d in base.iterdir()
                if d.is_dir() and d.name.startswith("version_") and d.name.split("_")[1].isdigit()]
    out_dir = base / f"version_{max(versions, default=-1) + 1}"
    out_dir.mkdir()
    path = out_dir / f"wf_results_{timescale}.csv"
    result.to_csv(path)
    result.to_csv(base / f"wf_results_{timescale}.csv")      # supervisor が読む最新版
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="再学習つき拡大窓のウォークフォワード（真の OOF 予測）")
    ap.add_argument("--timescale", default="daily_2644t")
    ap.add_argument("--models", default=None, help="カンマ区切り（例: ridge）。省略時は REGIME_EXPERTS のすべて")
    ap.add_argument("--accelerator", default="auto")
    args = ap.parse_args()
    res = run_walk_forward(args.timescale, args.models.split(",") if args.models else None, args.accelerator)
    path = _save(res, args.timescale)
    print(f"保存先: {path}  （{len(res)} 行）")
    for col in [c for c in res.columns if c.startswith("pred_")]:
        v = res[[col, "actual"]].dropna()
        if len(v) > 10:
            print(f"  {col}: Spearman={v[col].corr(v['actual'], method='spearman'):+.4f}  "
                  f"Direction={((v[col] > 0) == (v['actual'] > 0)).mean():.2%}  n={len(v)}")
