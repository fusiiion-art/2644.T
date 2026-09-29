import json
from pathlib import Path
import pandas as pd
import importlib

from common_utils.predicter_model import ExpertPredicter
import timescale_modules.daily_2644t.config_daily_2644t as cfg


def test_prediction_smoke_uses_saved_artifacts():
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / 'logs' / 'daily_2644t_risk_on_ridge_final_training' / 'version_3'
    assert (model_dir / 'best.ckpt').exists() or (model_dir / 'swa.ckpt').exists()

    features_path = model_dir / 'features.json'
    assert features_path.exists()
    feature_cols = json.loads(features_path.read_text())
    assert feature_cols

    df = pd.read_parquet(project_root / 'data' / cfg.OUTPUT_FILENAME)
    df = df[df['Date'] >= cfg.DATA_START_DATE].copy()
    df = df.tail(64).copy()
    assert len(df) >= cfg.SEQUENCE_LENGTH

    assets = {
        'model': None,
        'scaler': None,
        'features': feature_cols,
    }

    from models.linear_model import RidgeLightning
    ckpt_path = model_dir / 'swa.ckpt'
    if not ckpt_path.exists():
        ckpt_path = model_dir / 'best.ckpt'
    model = RidgeLightning.load_from_checkpoint(ckpt_path, map_location='cpu')
    assets['model'] = model

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = df[feature_cols].fillna(0).to_numpy()
    scaler.fit(X)
    assets['scaler'] = scaler

    predicter = ExpertPredicter('ridge', assets, cfg)
    mu, sigma = predicter.predict(df)
    assert isinstance(mu, float)
    assert isinstance(sigma, float)
