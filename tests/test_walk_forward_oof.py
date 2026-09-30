"""監査 G-9: 既存パイプラインのウォークフォワードは再学習つき拡大窓で、真の OOF だけを出す。

- 行 k の予測に使うモデルは、行 k より前の行（ラベルは open(k) までで確定）だけで学習する。
- 評価に使う actual は、学習ラベルと同じ区間（窓の最終行 k → target[k+1]）にそろえる。
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis.walk_forward_validator import walk_forward_oof
from common_utils.trainer_model import ExpertTrainer
from models.linear_model import RidgeLightning


def _df(n=80):
    idx = pd.bdate_range("2024-01-01", periods=n, name="Date")
    return pd.DataFrame({"regime": np.where(np.arange(n) % 3 == 0, "risk_off", "risk_on"),
                         "target_1": np.arange(n, dtype=float), "x": np.linspace(0, 1, n)}, index=idx)


def test_predictions_use_only_past_rows():
    df = _df()
    fitted_on = []

    def fit_fn(train):
        fitted_on.append(train.index[-1])
        return {"m": {r: {"n": len(train)} for r in ["risk_on", "risk_off"]}}

    def predict_fn(name, fitted, history):
        return float(fitted["n"])                       # 学習に使った行数を返す

    out = walk_forward_oof(df, fit_fn, predict_fn, "target_1", min_train=30, refit_every=10)
    pos = np.arange(30, len(df))
    assert list(out.index) == list(df.index[30:])
    assert (out["pred_m"].to_numpy() <= pos).all()      # 行 k の予測は行 0..k-1 だけで学習したモデル
    assert (out["train_rows"].to_numpy() <= pos).all()
    assert fitted_on == list(df.index[[29, 39, 49, 59, 69]])
    # actual は学習ラベルと同じ区間: target[k+1]
    assert out["actual"].iloc[:-1].tolist() == df["target_1"].iloc[31:].tolist()
    assert np.isnan(out["actual"].iloc[-1])
    assert (out["regime"] == df["regime"].iloc[30:]).all()


def test_missing_regime_model_gives_nan():
    df = _df(50)

    def fit_fn(train):
        return {"m": {"risk_on": {"n": len(train)}, "risk_off": None}}

    out = walk_forward_oof(df, fit_fn, lambda n, f, h: 1.0, "target_1", min_train=20, refit_every=10)
    assert out.loc[out["regime"] == "risk_off", "pred_m"].isna().all()
    assert out.loc[out["regime"] == "risk_on", "pred_m"].notna().all()


def test_fit_in_memory_trains_real_ridge_on_cpu():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({"f_a": rng.normal(size=n), "f_b": rng.normal(size=n)})
    df["target_1"] = 0.5 * df["f_a"].shift(-1).fillna(0) + rng.normal(scale=0.01, size=n)
    cfg = SimpleNamespace(PREDICTION_HORIZON=1, FEATURE_SETS=["S"], FEATURE_SETS_DEFINITIONS={"S": [r"^f_"]},
                          SEQUENCE_LENGTH=4, MAX_EPOCHS=2, BATCH_SIZE=16, PRECISION=32, GRADIENT_CLIP_VAL=1.0,
                          ACCUMULATE_GRAD_BATCHES=1, LOSS_TYPE="mse", USE_CFI=False, BLACKLIST_FEATURES=[])
    et = ExpertTrainer("ridge", RidgeLightning, df, cfg, ".", "wf_test", ".")
    fitted = et.fit_in_memory({"dropout": 0.0, "lr": 0.01, "weight_decay": 0.001, "n_features_to_select": 2},
                              accelerator="cpu")
    assert fitted["features"] == ["f_a", "f_b"] and fitted["seq_len"] == 4
    assert fitted["scaler"].n_samples_seen_ == n
    x = fitted["scaler"].transform(df[fitted["features"]].to_numpy()[-4:])
    import torch
    mu, _ = fitted["model"](torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
    assert np.isfinite(mu.item())
