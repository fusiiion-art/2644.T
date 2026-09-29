import sys
from pathlib import Path
import warnings
import os
import argparse
import optuna
import torch
import pytorch_lightning as pl
import logging
from typing import Optional, List, Dict, Type

# --- パス設定 ---
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# --- 共通モジュールとモデルのインポート ---
from common_utils.trainer_model import TrainingOrchestrator
from models.linear_model import RidgeLightning, ElasticNetLightning
from models.dlinear_model import DLinearLightning, NLinearLightning
from models.lightgbm_model import LightGBMLightning
from models.tcn_attention_model import TCNAttentionLightning
import timescale_modules.daily_2644t.config_daily_2644t as cfg

# ★ログ設定: loggingの出力を有効化（これがないとlogging.infoが出力されない）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ★T+1予測は複数の軽量モデルで統合する
MODEL_MAP: Dict[str, Type[pl.LightningModule]] = {
    "ridge": RidgeLightning,
    "elasticnet": ElasticNetLightning,
    "dlinear": DLinearLightning,
    "nlinear": NLinearLightning,
    "lightgbm": LightGBMLightning,
    "tcn_attention": TCNAttentionLightning,
}

# Joblib Temp Dir Setup
safe_temp_dir = 'C:\\joblib_temp'
try:
    default_temp = os.environ.get('TEMP', 'C:\\temp')
    is_ascii_path = True; default_temp.encode('ascii')
except UnicodeEncodeError: is_ascii_path = False
except Exception: pass 
if ' ' in default_temp or not is_ascii_path:
    try:
        os.makedirs(safe_temp_dir, exist_ok=True)
        os.environ['JOBLIB_TEMP_FOLDER'] = safe_temp_dir
    except Exception: pass

def main(new_study: bool, production: bool, n_jobs: int, models_to_run_str: Optional[str]):
    orchestrator = TrainingOrchestrator(
        config=cfg,
        project_root=project_root,
        model_map=MODEL_MAP,
        run_name_prefix="daily_2644t",
        output_dir_name="models_daily_2644t"
    )
    if not production and getattr(cfg, 'N_TRIALS', 1) > 1:
        cfg.N_TRIALS = 1
    orchestrator.run(new_study=new_study, production=production, n_jobs=n_jobs, models_to_run_str=models_to_run_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-study", action="store_true")
    parser.add_argument("--production", "-p", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--models", type=str, default=None)
    args = parser.parse_args()
    main(args.new_study, args.production, args.n_jobs, args.models)
