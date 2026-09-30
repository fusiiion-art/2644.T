"""監査 F / A-方法3: 現行 Ridge (RidgeLightning) のウォークフォワード OOS 評価。

既存コードは一切変更しない。モデルクラスはリポジトリの models.linear_model.RidgeLightning を
そのまま import し、学習手順は common_utils/trainer_model.py の _train_full_period を
行番号付きで写経して再現する（trainer_model.py は mlflow/lightgbm に依存するため import しない）。

variant:
  asis   : data/features_daily_2644t.parquet をそのまま使う（現行）
  uslag1 : 米国系列由来の列（sox_/nvda_/tsm_/vxn_/SOX_/NVDA_）を追加で 1 行（=1 平日）遅らせる
  bfix   : 参考。2024-10-09 の 1:2 分割で壊れたラベル/ラグ特徴量 3 セルだけを分割調整値に置換

評価リターンは生データ (data/cache/yf_data_daily_2644t.parquet) の 2644.T 始値から、
東京の実営業日のみで open(s+1)->open(s+2) を計算する（2024-10-09 より前は /2 で分割調整）。
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True  # リポジトリ内に __pycache__ を作らない
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import timescale_modules.daily_2644t.config_daily_2644t as cfg
from models.linear_model import RidgeLightning

warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

SEQ_LEN = cfg.SEQUENCE_LENGTH            # 32
TARGET = f"target_{cfg.PREDICTION_HORIZON}"
US_PREFIXES = ("sox_", "nvda_", "tsm_", "vxn_", "SOX_", "NVDA_")
SPLIT_DATE = pd.Timestamp("2024-10-09")
COSTS = [0.0, 0.0005, 0.0010]            # 往復コスト


# ---------- trainer_model.py の写経 ----------
def drop_constant_columns(df):  # trainer_model.py:202-210
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    std_vals = df[numeric_cols].std(skipna=True)
    const = std_vals[(std_vals == 0) | (std_vals.isna())].index.tolist()
    return df.drop(columns=const) if const else df


def feature_candidates(df):  # trainer_model.py:229-268
    patterns = []
    for s in cfg.FEATURE_SETS:
        patterns.extend(cfg.FEATURE_SETS_DEFINITIONS.get(s, []))
    cols = set(df.columns)
    matched = set()
    for p in patterns:
        rx = re.compile(p)
        matched |= {c for c in cols if rx.match(c)}
    meta = {"regime", "Date", "time_idx", "group", cfg.TARGET_COLUMN}
    meta |= {c for c in cols if c.startswith("target_")}
    cleaned = matched - meta - set(cfg.BLACKLIST_FEATURES)
    if not cleaned:
        return sorted(c for c in df.columns if c not in meta and not c.startswith("target_"))
    return sorted(cleaned)


def create_sequences(features, target, seq_len=SEQ_LEN):  # trainer_model.py:217-227
    n = len(features) - seq_len
    if n <= 0:
        return None, None
    idx = np.arange(n)[:, None] + np.arange(seq_len)[None, :]
    X = features[idx]
    y = target[seq_len:seq_len + n]   # 窓の最終行の「次の行」のターゲット
    return torch.from_numpy(X).float(), torch.from_numpy(y).float().view(-1, 1)


def train_regime_model(df_regime, n_feat, hp, seed, root_dir):  # trainer_model.py:549-628
    df_regime = drop_constant_columns(df_regime)
    sel = feature_candidates(df_regime)[:n_feat]
    X_raw = np.nan_to_num(df_regime[sel].values, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(X_raw)
    X_seq, y_seq = create_sequences(scaler.transform(X_raw), df_regime[TARGET].values)
    if X_seq is None or len(X_seq) < 2:
        return None
    pl.seed_everything(seed, workers=True, verbose=False)
    loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    model = RidgeLightning(input_channels=len(sel), context_window=SEQ_LEN, target_window=cfg.PREDICTION_HORIZON,
                           loss_type=cfg.LOSS_TYPE, dropout=hp["dropout"], lr=hp["lr"], weight_decay=hp["weight_decay"])
    trainer = pl.Trainer(max_epochs=cfg.MAX_EPOCHS, accelerator="cpu", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         precision=cfg.PRECISION, gradient_clip_val=cfg.GRADIENT_CLIP_VAL,
                         accumulate_grad_batches=cfg.ACCUMULATE_GRAD_BATCHES, default_root_dir=root_dir)
    trainer.fit(model, loader)
    model.eval()
    return {"model": model, "scaler": scaler, "features": sel, "n_train_seq": int(len(X_seq))}


def predict_at(m, grid, k):  # predicter_model.py:59-83 (直近 SEQ_LEN 行、連続した全レジームの行)
    win = grid[m["features"]].iloc[k - SEQ_LEN + 1:k + 1]
    if win.isna().any().any():
        win = win.ffill().bfill().fillna(0)
    X = m["scaler"].transform(win.values)
    with torch.no_grad():
        mu, _ = m["model"](torch.from_numpy(X.astype(np.float32)).unsqueeze(0))
    return float(mu.item())


# ---------- データ ----------
def load_current_model_hparams():
    out = {}
    for regime, ver in [("risk_on", 3), ("risk_off", 1)]:
        d = ROOT / "logs" / f"daily_2644t_{regime}_ridge_final_training" / f"version_{ver}"
        ck = torch.load(d / "swa.ckpt", map_location="cpu", weights_only=False)
        hp = dict(ck["hyper_parameters"])
        feats = json.loads((d / "features.json").read_text())
        out[regime] = {"dropout": hp["dropout"], "lr": hp["lr"], "weight_decay": hp["weight_decay"],
                       "n_features": len(feats), "features_json": feats, "ckpt": str(d.relative_to(ROOT) / "swa.ckpt")}
    return out


def load_grid(variant):
    g = pd.read_parquet(ROOT / "data" / cfg.OUTPUT_FILENAME).reset_index(drop=True)
    if variant == "uslag1":
        us_cols = [c for c in g.columns if c.startswith(US_PREFIXES)]
        g[us_cols] = g[us_cols].shift(1)
    elif variant == "bfix":
        yf = pd.read_parquet(ROOT / "data" / "cache" / cfg.YF_CACHE_FILENAME).set_index("Date")
        o, ac = yf["2644.T_open"], yf["2644.T_adj_close"]
        d_pre, d_post = pd.Timestamp("2024-10-08"), SPLIT_DATE
        fixed_target = np.log((o[d_post] + 1e-9) / (o[d_pre] / 2.0 + 1e-9))
        fixed_ret = np.log(ac[d_post] / (ac[d_pre] / 2.0))
        for c in ["target_1", "target_1_open_to_open"]:
            g.loc[g.Date == d_pre, c] = fixed_target
        g.loc[g.Date == pd.Timestamp("2024-10-10"), "semi_adj_close_return_lag1"] = fixed_ret
        g.loc[g.Date == pd.Timestamp("2024-10-11"), "semi_adj_close_return_lag2"] = fixed_ret
    # RegimeDetector.detect_simple_vix と同じ判定 (regime_detector.py:30)
    g["regime"] = np.where(g[cfg.REGIME_VIX_COLUMN] > cfg.REGIME_VIX_THRESHOLD, "risk_off", "risk_on")
    return g


def realized_returns(grid):
    """行 k (東京営業日の大引けで予測) -> 翌営業日始値で買い、翌々営業日始値で手仕舞いのリターン。"""
    import exchange_calendars as xc
    cal = xc.get_calendar("XTKS")
    yf = pd.read_parquet(ROOT / "data" / "cache" / cfg.YF_CACHE_FILENAME).set_index("Date")
    o = yf["2644.T_open"].dropna().copy()
    o[o.index < SPLIT_DATE] = o[o.index < SPLIT_DATE] / 2.0   # 評価用の分割調整
    sess = o.index
    pos = {d: i for i, d in enumerate(sess)}
    r = pd.Series(np.nan, index=grid.index)
    status = pd.Series("not_session", index=grid.index, dtype=object)
    for k, d in grid["Date"].items():
        if d not in pos:
            continue
        i = pos[d]
        if i + 2 >= len(sess):
            status[k] = "no_future"
            continue
        n1, n2 = sess[i + 1], sess[i + 2]
        if cal.next_session(d) != n1 or cal.next_session(n1) != n2:   # 2025-10-24 など欠損営業日を跨ぐ
            status[k] = "gap_in_data"
            continue
        r[k] = o[n2] / o[n1] - 1.0
        status[k] = "ok"
    return r, status


# ---------- 指標 ----------
def evaluate(pred, r):
    df = pd.DataFrame({"pred": pred, "r": r}).dropna(subset=["r"])
    out = {"n_eval_days": int(len(df)), "n_with_pred": int(df.pred.notna().sum())}
    v = df.dropna(subset=["pred"])
    nz = v[v.r != 0]
    out["hit_rate"] = float((np.sign(nz.pred) == np.sign(nz.r)).mean())
    out["always_up_hit_rate"] = float((nz.r > 0).mean())
    out["n_hit_days(r!=0)"] = int(len(nz))
    out["pred_up_ratio"] = float((v.pred > 0).mean())
    ic = spearmanr(v.pred, v.r)
    out["IC_spearman"] = float(ic.statistic)
    out["IC_pvalue"] = float(ic.pvalue)
    out["pearson"] = float(np.corrcoef(v.pred, v.r)[0, 1])
    pos = (df.pred > 0).astype(float).values              # long/flat（predicter_model.py:210, 380 と同じ）
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    entries = int(((np.diff(np.concatenate([[0.0], pos]))) > 0).sum())
    for c in COSTS:
        ret = pos * df.r.values - (c / 2.0) * turn
        out[f"sharpe_cost{c*100:.2f}%"] = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))
        out[f"ann_return_cost{c*100:.2f}%"] = float(ret.mean() * 252)
    out["round_trips"] = entries
    out["days_in_market"] = int(pos.sum())
    bh = df.r.values
    out["buyhold_sharpe"] = float(bh.mean() / bh.std(ddof=1) * np.sqrt(252))
    out["buyhold_ann_return"] = float(bh.mean() * 252)
    return out


def run(variant, seeds, min_train, refit_every, root_dir):
    hps = load_current_model_hparams()
    grid = load_grid(variant)
    r, status = realized_returns(grid)
    n = len(grid)
    preds = {s: pd.Series(np.nan, index=grid.index) for s in seeds}
    used_regime = pd.Series(np.nan, index=grid.index, dtype=object)
    refit_log = []
    for j in range(min_train, n, refit_every):
        # 行 j の大引け時点で既知: 行 0..j-1 のターゲット（open(j) まで）
        train = grid.iloc[:j]
        for s in seeds:
            models = {}
            for regime in cfg.REGIMES:
                dfr = train[train["regime"] == regime].copy()
                m = train_regime_model(dfr, hps[regime]["n_features"], hps[regime], s, root_dir)
                models[regime] = m
                if s == seeds[0]:
                    refit_log.append({"refit_row": j, "refit_date": str(grid.Date[j].date()), "regime": regime,
                                      "n_rows": int(len(dfr)), "trained": m is not None,
                                      "features": m["features"] if m else None})
            for k in range(j, min(j + refit_every, n)):
                if status[k] != "ok" or k < SEQ_LEN - 1:
                    continue
                reg = grid.at[k, "regime"]
                used_regime[k] = reg
                m = models.get(reg)
                if m is not None:
                    preds[s][k] = predict_at(m, grid, k)
        print(f"[{variant}] refit at row {j} ({grid.Date[j].date()}) done", flush=True)

    oos = status.index >= min_train
    res = {"variant": variant, "seeds": seeds, "min_train_rows": min_train, "refit_every_rows": refit_every,
           "oos_first_date": str(grid.Date[min_train].date()), "oos_last_date": str(grid.Date[n - 1].date()),
           "status_counts_oos": status[oos].value_counts().to_dict(), "hparams": hps, "per_seed": {}}
    for s in seeds:
        res["per_seed"][s] = evaluate(preds[s][oos], r[oos])
    ens = pd.concat([preds[s] for s in seeds], axis=1).mean(axis=1, skipna=False)
    res["seed_mean_prediction"] = evaluate(ens[oos], r[oos])
    for reg in cfg.REGIMES:
        mask = oos & (used_regime == reg).values
        res[f"seed_mean_prediction_{reg}_only"] = evaluate(ens[mask], r[mask])
    out = pd.DataFrame({"Date": grid.Date, "regime": used_regime, "realized_open_to_open": r, "status": status,
                        **{f"pred_seed{s}": preds[s] for s in seeds}, "pred_seed_mean": ens})
    return res, out[oos], refit_log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["asis", "uslag1", "bfix"], required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--min-train", type=int, default=252)
    ap.add_argument("--refit-every", type=int, default=21)
    ap.add_argument("--root-dir", default=None, help="Lightning の default_root_dir（リポジトリ外を推奨）")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    torch.set_num_threads(1)
    res, preds, refit_log = run(a.variant, seeds, a.min_train, a.refit_every, a.root_dir)
    od = Path(__file__).resolve().parent / "results"
    od.mkdir(exist_ok=True)
    (od / f"wf_{a.variant}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    preds.to_csv(od / f"wf_{a.variant}_predictions.csv", index=False)
    pd.DataFrame(refit_log).to_csv(od / f"wf_{a.variant}_refits.csv", index=False)
    print(json.dumps(res["seed_mean_prediction"], indent=2))
