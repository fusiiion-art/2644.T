import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import joblib
import logging
import warnings
import json
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler

# --- プロジェクトルート設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- モジュールインポート ---
# daily_2644t uses lightweight models, not heavy TimesNet/iTransformer
from models.linear_model import RidgeLightning, ElasticNetLightning
from models.dlinear_model import DLinearLightning, NLinearLightning
from models.lightgbm_model import LightGBMLightning
from common_utils.supervisor_model import MetaLearner
from common_utils.limit_engine import LimitEngine
from pytorch_lightning.callbacks import ModelCheckpoint
import timescale_modules.daily_2644t.config_daily_2644t as cfg

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MODEL_CLASS_MAP = {
    "ridge": RidgeLightning,
    "elasticnet": ElasticNetLightning,
    "dlinear": DLinearLightning,
    "nlinear": NLinearLightning,
    "lightgbm": LightGBMLightning,
}

class DailyDecisionController:
    """AI予測とルールベース指値を統合する簡易コントローラ。"""

    def __init__(self):
        self.limit_engine = LimitEngine()

    def generate_daily_decision(self, current_features, ai_log_return):
        """
        日次の意思決定をまとめて行う。current_featuresは特徴量 DataFrame を想定する。
        """
        if current_features is None or current_features.empty:
            return None

        latest = current_features.iloc[-1]
        open_price = latest.get("open")
        atr = latest.get("ATR_14")
        bb_lower = latest.get("BB_lower_2")
        sox_return = latest.get("SOX_return_prev_night")

        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [open_price, atr, bb_lower, sox_return]):
            return None

        order_decision = self.limit_engine.calculate_order(
            open_price=open_price,
            atr=atr,
            bb_lower=bb_lower,
            sox_return=sox_return,
            ai_expected_return=ai_log_return,
        )

        if order_decision["suspend"]:
            logger.info(f"本日のアクション: 見送り ({order_decision['reason']})")
            return None

        return {
            "action": "LIMIT_BUY",
            "price": order_decision["p_buy_limit"],
            "regime": order_decision.get("regime", "unknown"),
            "ai_score": ai_log_return,
        }

    def generate_daily_action(self, current_features, ai_log_return):
        """Backward-compatible wrapper for the old method name."""
        return self.generate_daily_decision(current_features, ai_log_return)


class DailyOOFPredictor:
    """Out-of-Fold予測を生成（取得）するためのクラス"""
    def __init__(self, regime: str, model_name: str, device: str):
        self.regime = regime
        self.model_name = model_name
        self.device = device
        # フォールバック削除に伴い、seq_lenは不要になりましたが互換性のため残します
        self.seq_len = getattr(cfg, "SEQUENCE_LENGTH_S5", 120) if model_name == "s5" else cfg.SEQUENCE_LENGTH

    def load_oof_predictions(self, df_full: pd.DataFrame, n_splits: int = 5) -> pd.Series:
        """
        Walk-Forward ValidationのログファイルからOOF予測をロードする。
        """
        oof_preds = pd.Series(index=df_full.index, data=np.nan)
        
        # WF検証ログのパスを確認
        wf_log_dir = PROJECT_ROOT / "logs" / f"{cfg.MODULE_NAME}_walkforward"
        wf_result_path = wf_log_dir / f"wf_results_{cfg.MODULE_NAME}.csv"
        
        if not wf_result_path.exists():
            error_msg = (
                f"❌ [CRITICAL] OOF prediction log not found for {self.model_name}.\n"
                f"   Path: {wf_result_path}\n"
                f"   Action: Please run 'analysis/walk_forward_validator.py' FIRST.\n"
                f"   Reason: Training Supervisor without proper OOF data causes severe data leakage."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading OOF predictions from: {wf_result_path.name}")
        try:
            wf_df = pd.read_csv(wf_result_path, parse_dates=['Date'], index_col='Date')
            col_name = f"pred_{self.model_name}"
            if col_name in wf_df.columns:
                wf_df = wf_df[~wf_df.index.duplicated(keep='last')]
                target_preds = wf_df[col_name]
                oof_preds = target_preds.reindex(df_full.index)
                valid_count = oof_preds.notna().sum()
                logger.info(f" -> Loaded {valid_count} OOF predictions for {self.model_name}")
                if valid_count == 0:
                    logger.warning(f"⚠️ OOF data for {self.model_name} is all NaN. Check WF validator settings.")
            else:
                logger.warning(f"⚠️ Column '{col_name}' not found in WF logs. Available columns: {wf_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Failed to load OOF logs: {e}")
            raise e
            
        return oof_preds

    def generate_oof_predictions(self, df_full: pd.DataFrame, n_splits: int = 5) -> pd.Series:
        """Backward-compatible wrapper for the old method name."""
        return self.load_oof_predictions(df_full, n_splits=n_splits)


class DailySupervisorTrainer:
    def __init__(self):
        self.module_name = cfg.MODULE_NAME
        self.models_dir = PROJECT_ROOT / f"models_{self.module_name}"
        self.data_path = PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

# Backward-compatible aliases
RuleBasedDecisionController = DailyDecisionController
OOFPredictor = DailyOOFPredictor
SupervisorTrainer = DailySupervisorTrainer
