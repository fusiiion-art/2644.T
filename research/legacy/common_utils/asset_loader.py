import os
import json
import joblib
from pathlib import Path
import logging
from typing import Any, Optional, Dict, List
import torch

# --- モデルクラスの動的インポート (条件付き) ---
MODEL_CLASS_MAP = {}

# Heavy models (daily_1, etc)
try:
    from models.timesnet_model import TimesNetLightning
    MODEL_CLASS_MAP["timesnet"] = TimesNetLightning
except ImportError: pass

try:
    from models.itransformer_model import iTransformerLightning
    MODEL_CLASS_MAP["itransformer"] = iTransformerLightning
except ImportError: pass

try:
    from models.timemixer_model import TimeMixerLightning
    MODEL_CLASS_MAP["timemixer"] = TimeMixerLightning
except ImportError: pass

try:
    from models.s5_model import S5Lightning
    MODEL_CLASS_MAP["s5"] = S5Lightning
except ImportError: pass

try:
    from models.patchtst_model import PatchTSTLightning
    MODEL_CLASS_MAP["patchtst"] = PatchTSTLightning
except ImportError: pass

# Lightweight models (semi_daily_1, etc)
try:
    from models.linear_model import RidgeLightning, ElasticNetLightning
    MODEL_CLASS_MAP["ridge"] = RidgeLightning
    MODEL_CLASS_MAP["elasticnet"] = ElasticNetLightning
except ImportError: pass

try:
    from models.dlinear_model import DLinearLightning, NLinearLightning
    MODEL_CLASS_MAP["dlinear"] = DLinearLightning
    MODEL_CLASS_MAP["nlinear"] = NLinearLightning
except ImportError: pass

try:
    from models.lightgbm_model import LightGBMLightning
    MODEL_CLASS_MAP["lightgbm"] = LightGBMLightning
except ImportError: pass

try:
    from models.tcn_model import TCNLightning
    MODEL_CLASS_MAP["tcn"] = TCNLightning
except ImportError: pass

try:
    from models.tcn_attention_model import TCNAttentionLightning
    MODEL_CLASS_MAP["tcn_attention"] = TCNAttentionLightning
except ImportError: pass

from common_utils.supervisor_model import MetaLearner
MODEL_CLASS_MAP["supervisor"] = MetaLearner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AssetLoader:
    def __init__(self, project_root: Path, module_name: str = "daily_1"):
        self.project_root = project_root
        self.module_name = module_name
        self.logs_dir = project_root / "logs"
        self.models_dir = self.logs_dir / f"models_{module_name}" # 例: logs/models_daily_1
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_class_map = MODEL_CLASS_MAP

    def load_production_model(self, model_name: str, regime: str, use_swa: bool = True) -> Dict[str, Any]:
        """
        models_{module_name}/ フォルダから本番用モデル一式を読み込む。
        
        Args:
            model_name: 'tcn', 'bdh' など
            regime: 'risk_on', 'risk_off'
            use_swa: TrueならSWAモデル、FalseならBestモデルを読み込む
        
        Returns:
            Dict: {'model': LightningModule, 'scaler': scaler, 'features': list, ...}
        """
        prefix = f"{model_name}_{regime}" # 例: tcn_risk_on
        assets = {}
        
        # 1. 特徴量リスト (.json)
        json_path = self.models_dir / f"{prefix}_features.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Features file not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            assets['features'] = json.load(f)
            
        # 2. スケーラー (.joblib)
        scaler_path = self.models_dir / f"{prefix}_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        assets['scaler'] = joblib.load(scaler_path)
        
        # 3. モデル重み (.ckpt)
        suffix = "swa" if use_swa else "best"
        ckpt_path = self.models_dir / f"{prefix}_{suffix}.ckpt"
        if not ckpt_path.exists():
            # SWAがない場合はBestにフォールバック、逆もしかり
            fallback = "best" if use_swa else "swa"
            logger.warning(f"{ckpt_path.name} not found. Trying {fallback}...")
            ckpt_path = self.models_dir / f"{prefix}_{fallback}.ckpt"
            
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
            
        model_class = self.model_class_map.get(model_name)
        if not model_class:
            raise ValueError(f"Unknown model name: {model_name}")
            
        # モデルのロード
        logger.info(f"Loading model from {ckpt_path.name}...")
        try:
            # hparamsがない場合でも読み込めるようにする
            model = model_class.load_from_checkpoint(str(ckpt_path), map_location=lambda storage, loc: storage)
            model.eval()
            model.freeze() # 推論専用にする
            assets['model'] = model
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")
            
        return assets

    def _find_latest_version_path(self, base_log_dir: Path) -> Optional[Path]:
        """指定されたログディレクトリから最新バージョンのパスを見つける"""
        if not base_log_dir.exists():
            return None
        version_paths = sorted(
            [p for p in base_log_dir.iterdir() if p.is_dir() and p.name.startswith('version_')],
            key=lambda p: int(p.name.split('_')[-1]),
            reverse=True
        )
        return version_paths[0] if version_paths else None

    def find_and_load(self, module_name: str, model_name: str, regime: Optional[str] = None) -> Dict[str, Any]:
        """
        モジュール名、モデル名、レジーム名を基に、最新の学習済み資産を自動で捜索し読み込む。
        """
        log_name_pattern = f"{module_name}_{regime}_{model_name}_final_training"
        base_log_dir = self.logs_dir / log_name_pattern
        if (base_log_dir / "best_model.ckpt").exists() or (base_log_dir / "best_model.joblib").exists():
             log_dir = base_log_dir
        else:
             log_dir = self._find_latest_version_path(base_log_dir)

        if not log_dir or not log_dir.exists():
            logger.warning(f"アセットが見つかりませんでした: ログ '{log_name_pattern}' が見つかりません。")
            return {}

        logger.info(f"アセットを '{log_dir.relative_to(self.project_root)}' から読み込んでいます...")
        return self._load_artifacts_from_dir(log_dir, model_name)

    def _load_artifacts_from_dir(self, log_dir: Path, model_name: str) -> Dict[str, Any]:
        """指定されたディレクトリから資産一式を読み込む内部関数"""
        assets = {"log_dir": log_dir}
        model_candidates = ["best_model.ckpt", "best_model.joblib", "checkpoints/best_model.ckpt"]
        model_path = next((log_dir / p for p in model_candidates if (log_dir / p).exists()), None)

        if model_path:
            assets['model_path'] = model_path
            if model_path.suffix == ".joblib":
                assets["model"] = joblib.load(model_path)
            else: # .ckpt
                model_class = self.model_class_map.get(model_name)
                if model_class:
                    assets["model"] = model_class.load_from_checkpoint(str(model_path), hparams_file=None)
                else:
                    logger.error(f"モデルクラスのマッピングが見つかりません: {model_name}")
        else:
            logger.warning("モデルファイルが見つかりませんでした。")

        # スケーラーやパラメータなどの関連資産を読み込み
        for artifact_path in log_dir.glob('*'):
            if artifact_path.is_file() and artifact_path.name not in ["best_model.ckpt", "best_model.joblib"]:
                key_name = artifact_path.stem.replace('best_', '')
                if artifact_path.suffix == ".joblib":
                    assets[key_name] = joblib.load(artifact_path)
                elif artifact_path.suffix == ".json":
                    with open(artifact_path, 'r', encoding='utf-8') as f:
                        assets[key_name] = json.load(f)

        return assets
