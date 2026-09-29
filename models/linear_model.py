"""
linear_model.py - 線形モデル（Ridge/ElasticNet）

少データ（100-2000行）に最適なシンプルな線形モデル。
正則化により過学習を防ぎ、安定した予測を実現。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from common_utils.base_model import BaseLightningModule


class RidgeWrapper:
    """Ridge回帰のラッパー"""
    def __init__(self, alpha: float = 1.0, **kwargs):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self._is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        mu = self.model.predict(X_scaled)
        sigma = np.ones_like(mu) * 0.01  # 固定σ
        return mu, sigma


class ElasticNetWrapper:
    """ElasticNet回帰のラッパー"""
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
        self.scaler = StandardScaler()
        self._is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        mu = self.model.predict(X_scaled)
        sigma = np.ones_like(mu) * 0.01
        return mu, sigma


class RidgeLightning(BaseLightningModule):
    """
    Ridge回帰のPyTorch Lightning実装
    内部で線形層 + L2正則化で実現
    """
    def __init__(self,
                 input_dim: int = None,
                 seq_len: int = 32,
                 alpha: float = 1.0,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-2,  # L2正則化
                 loss_type: str = 'mse',
                 # ★互換性エイリアス (ExpertTrainerからの呼び出し対応)
                 input_channels: int = None,
                 context_window: int = None,
                 target_window: int = None,  # 未使用だが受け取る
                 **kwargs):
        # パラメータ名の互換性対応
        input_dim = input_dim or input_channels
        seq_len = context_window or seq_len
        if input_dim is None:
            raise ValueError("input_dim or input_channels must be specified")
        # weight_decay = alpha として L2正則化を実現
        super().__init__(lr=lr, weight_decay=weight_decay, loss_type=loss_type)
        self.save_hyperparameters()
        
        # シンプルな線形層
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * input_dim, 1)
        self.fc_sigma = nn.Linear(seq_len * input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, Seq, Features]
        """
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            # Flatten: [Batch, Seq * Features]
            x = self.flatten(x)
            x = self.dropout(x)
            
            # 出力
            mu = self.fc(x).squeeze(-1)
            sigma = F.softplus(self.fc_sigma(x)).squeeze(-1) + 1e-6
            
        return mu, sigma
    
    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """探索空間（拡張版）"""
        return {
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 10.0, log=True),
            "lr": trial.suggest_float("lr", 5e-5, 5e-2, log=True),
        }


class ElasticNetLightning(BaseLightningModule):
    """
    ElasticNet風のPyTorch Lightning実装
    L1 + L2 正則化を実現
    """
    def __init__(self,
                 input_dim: int = None,
                 seq_len: int = 32,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-2,
                 loss_type: str = 'mse',
                 # ★互換性エイリアス
                 input_channels: int = None,
                 context_window: int = None,
                 target_window: int = None,
                 **kwargs):
        # パラメータ名の互換性対応
        input_dim = input_dim or input_channels
        seq_len = context_window or seq_len
        if input_dim is None:
            raise ValueError("input_dim or input_channels must be specified")
        super().__init__(lr=lr, weight_decay=weight_decay, loss_type=loss_type)
        self.save_hyperparameters()
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        
        # シンプルな線形層
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * input_dim, 1)
        self.fc_sigma = nn.Linear(seq_len * input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            x = self.flatten(x)
            x = self.dropout(x)
            
            mu = self.fc(x).squeeze(-1)
            sigma = F.softplus(self.fc_sigma(x)).squeeze(-1) + 1e-6
            
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        """L1正則化を追加"""
        x, y = batch
        out = self(x)
        
        if isinstance(out, tuple):
            mu = out[0]
        else:
            mu = out
            
        # 基本損失
        loss = self.loss_fn(mu.unsqueeze(1), y.unsqueeze(1) if y.dim() == 1 else y)
        
        # L1正則化項を追加
        l1_reg = sum(p.abs().sum() for p in self.fc.parameters())
        loss = loss + self.alpha * self.l1_ratio * l1_reg
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """探索空間（拡張版）- 正則化強度を緩和"""
        return {
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 0.5),  # 0.5上限に縮小
            "alpha": trial.suggest_float("alpha", 1e-6, 0.1, log=True),  # 0.1上限に縮小
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1.0, log=True),
            "lr": trial.suggest_float("lr", 5e-5, 5e-2, log=True),
        }
