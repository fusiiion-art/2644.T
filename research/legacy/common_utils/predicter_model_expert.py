"""学習済みの単一モデル（旧パイプライン）で直近の窓から予測する。

DailyPredicter（supervisor 統合）と可視化シミュレーションは、旧モデル群とともに
research/legacy/common_utils/predicter_model.py へ移した（設計書 v2 フェーズ3）。
"""
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch


class ExpertPredicter:
    """
    単一の学習済み専門家モデル (TCN/BDH/S5) をロードし、予測を実行するクラス。
    """
    def __init__(self, model_name: str, assets: Dict[str, Any], config: Any):
        self.model_name = model_name
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = assets.get("model")
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device).eval()
            if hasattr(self.model, 'freeze'):
                self.model.freeze() 

        self.scaler = assets.get("scaler")
        self.feature_cols = assets.get("features")

    def predict(self, df_latest: pd.DataFrame) -> Tuple[float, float]:
        """
        直近のデータフレームを受け取り、明日のリターン予測(mu)と不確実性(sigma)を返す。
        Returns:
            (mu, sigma) のタプル
        """
        missing = [c for c in self.feature_cols if c not in df_latest.columns]
        if missing:
            raise KeyError(f"[{self.model_name}] Missing features for prediction: {missing[:3]}...")

        try:
            # S5モデル対応のシーケンス長
            if self.model_name == "s5":
                seq_len = getattr(self.config, "SEQUENCE_LENGTH_S5", 120)
            else:
                seq_len = self.config.SEQUENCE_LENGTH
            
            if len(df_latest) < seq_len:
                raise ValueError(f"[{self.model_name}] Not enough data. Need {seq_len}, got {len(df_latest)}")
            
            df_seq = df_latest[self.feature_cols].iloc[-seq_len:].copy()
            if df_seq.isna().any().any():
                df_seq = df_seq.ffill().bfill().fillna(0)
            
            X_seq_raw = df_seq.values

        except Exception as e:
            raise ValueError(f"[{self.model_name}] Data processing failed: {e}")

        if self.scaler is None:
             raise ValueError(f"[{self.model_name}] Scaler not found in assets.")
        
        try:
            X_scaled = self.scaler.transform(X_seq_raw)
        except Exception as e:
             raise ValueError(f"[{self.model_name}] Scaling failed. Check feature count. {e}")

        input_tensor = torch.from_numpy(X_scaled.astype(np.float32)).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # タプル (mu, sigma) か 単一テンソル かを判定
            if isinstance(output, tuple):
                mu = output[0].cpu().item()
                sigma = output[1].cpu().item()
            else:
                mu = output.cpu().item()
                # ★修正: sigmaが返されない場合のデフォルト値を 0.0 -> 0.01 (1%) に変更
                # 0.0 だと PortfolioStrategist が取引を拒否するため
                sigma = 0.01
            
            # NaN対策
            if np.isnan(mu): mu = 0.0
            
            # ★修正: sigma の下限値クリッピング (ゼロ除算防止 & 過剰レバレッジ防止)
            # 0.0001 (0.01%) 未満のsigmaは異常値として補正する
            if np.isnan(sigma) or sigma < 1e-4: 
                sigma = 1e-4
            
        return mu, sigma
