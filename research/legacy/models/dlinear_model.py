"""
dlinear_model.py - DLinear（少データ用シンプルモデル）

DLinearは時系列分解（トレンド + 季節成分）を線形層で行うシンプルなモデル。
論文でTransformerより小データで高精度と証明されている。
パラメータ数が極めて少なく、1000行程度のデータに最適。

Reference: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from typing import Dict, Any, Tuple
from common_utils.base_model import BaseLightningModule, MovingAvg


class DLinearBackbone(nn.Module):
    """
    DLinear: 時系列分解 + 線形予測
    トレンドと季節成分を分離して予測
    """
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, 
                 moving_avg: int = 25, individual: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        
        # 移動平均でトレンド/季節分解
        self.decomp = MovingAvg(moving_avg, stride=1)
        
        if individual:
            # 特徴量ごとに独立した線形層
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            # 共有の線形層
            self.linear_trend = nn.Linear(seq_len, pred_len)
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        """
        x: [Batch, Seq, Features]
        returns: [Batch, pred_len, Features]
        """
        # 時系列分解
        trend = self.decomp(x)  # [Batch, Seq, Features]
        seasonal = x - trend
        
        # 転置して時間軸に線形層を適用
        trend = trend.permute(0, 2, 1)  # [Batch, Features, Seq]
        seasonal = seasonal.permute(0, 2, 1)
        
        if self.individual:
            trend_out = torch.stack([
                self.linear_trend[i](trend[:, i, :]) 
                for i in range(self.enc_in)
            ], dim=1)  # [Batch, Features, pred_len]
            seasonal_out = torch.stack([
                self.linear_seasonal[i](seasonal[:, i, :]) 
                for i in range(self.enc_in)
            ], dim=1)
        else:
            trend_out = self.linear_trend(trend)  # [Batch, Features, pred_len]
            seasonal_out = self.linear_seasonal(seasonal)
        
        # 合成して転置
        out = trend_out + seasonal_out  # [Batch, Features, pred_len]
        out = out.permute(0, 2, 1)  # [Batch, pred_len, Features]
        
        return out


class DLinearLightning(BaseLightningModule):
    """
    DLinear Lightning Module
    極小データ（100-2000行）に最適なシンプルモデル
    """
    def __init__(self,
                 input_dim: int = None,
                 seq_len: int = 32,
                 pred_len: int = 1,
                 moving_avg: int = 25,
                 individual: bool = False,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 loss_type: str = 'mse',
                 # ★互換性エイリアス
                 input_channels: int = None,
                 context_window: int = None,
                 target_window: int = None,
                 **kwargs):
        # パラメータ名の互換性対応
        input_dim = input_dim or input_channels
        seq_len = context_window or seq_len
        pred_len = target_window or pred_len
        if input_dim is None:
            raise ValueError("input_dim or input_channels must be specified")
        super().__init__(lr=lr, weight_decay=weight_decay, loss_type=loss_type)
        self.save_hyperparameters()
        
        # DLinearバックボーン
        self.backbone = DLinearBackbone(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=input_dim,
            moving_avg=moving_avg,
            individual=individual
        )
        
        # 出力層
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(input_dim, 1)
        self.fc_sigma = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向き伝播
        x: [Batch, Seq, Features]
        returns: (mu, sigma)
        """
        # FP32強制（数値安定性）
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            # DLinear forward
            out = self.backbone(x)  # [Batch, pred_len, Features]
            
            # 最後のステップを使用
            out = out[:, -1, :]  # [Batch, Features]
            out = self.dropout(out)
            
            # 出力
            mu = self.fc_mu(out).squeeze(-1)  # [Batch]
            sigma = F.softplus(self.fc_sigma(out)).squeeze(-1) + 1e-6  # [Batch]
            
        return mu, sigma
    
    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Optunaのハイパーパラメータ探索空間（拡張版）"""
        return {
            "moving_avg": trial.suggest_categorical("moving_avg", [3, 5, 11, 25, 49]),
            "individual": trial.suggest_categorical("individual", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "lr": trial.suggest_float("lr", 5e-5, 5e-2, log=True),
        }


class NLinearLightning(BaseLightningModule):
    """
    NLinear Lightning Module
    DLinearよりさらにシンプル（分解なし）
    """
    def __init__(self,
                 input_dim: int = None,
                 seq_len: int = 32,
                 pred_len: int = 1,
                 individual: bool = False,
                 dropout: float = 0.1,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 loss_type: str = 'mse',
                 # ★安定化オプション
                 normalize_last: bool = True,
                 output_clip: float = 0.3,
                 # ★互換性エイリアス
                 input_channels: int = None,
                 context_window: int = None,
                 target_window: int = None,
                 **kwargs):
        # パラメータ名の互換性対応
        input_dim = input_dim or input_channels
        seq_len = context_window or seq_len
        pred_len = target_window or pred_len
        if input_dim is None:
            raise ValueError("input_dim or input_channels must be specified")
        super().__init__(lr=lr, weight_decay=weight_decay, loss_type=loss_type)
        self.save_hyperparameters()
        self.individual = individual
        self.input_dim = input_dim
        self.normalize_last = normalize_last
        self.output_clip = output_clip
        
        if individual:
            self.linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(input_dim)
            ])
        else:
            self.linear = nn.Linear(seq_len, pred_len)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(input_dim, 1)
        self.fc_sigma = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, Seq, Features]
        """
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            # 最後の値を引く（正規化）- オプション化
            if self.normalize_last:
                seq_last = x[:, -1:, :].detach()
                x = x - seq_last
            else:
                seq_last = torch.zeros_like(x[:, -1:, :])
            
            # 転置
            x = x.permute(0, 2, 1)  # [Batch, Features, Seq]
            
            if self.individual:
                out = torch.stack([
                    self.linear[i](x[:, i, :]) 
                    for i in range(self.input_dim)
                ], dim=1)  # [Batch, Features, pred_len]
            else:
                out = self.linear(x)  # [Batch, Features, pred_len]
            
            # 転置して最後の値を足す
            out = out.permute(0, 2, 1)  # [Batch, pred_len, Features]
            out = out + seq_last
            
            # 最後のステップ
            out = out[:, -1, :]  # [Batch, Features]
            out = self.dropout(out)
            
            mu = self.fc_mu(out).squeeze(-1)
            # ★出力クリッピング: 極端な予測を防止
            if self.output_clip > 0:
                mu = torch.clamp(mu, -self.output_clip, self.output_clip)
            sigma = F.softplus(self.fc_sigma(out)).squeeze(-1) + 1e-6
            
        return mu, sigma
    
    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """探索空間 - 安定化オプション追加"""
        return {
            "individual": trial.suggest_categorical("individual", [True, False]),
            "normalize_last": trial.suggest_categorical("normalize_last", [True, False]),
            "output_clip": trial.suggest_float("output_clip", 0.1, 0.5),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),  # 強化
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
