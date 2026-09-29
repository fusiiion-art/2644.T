"""
base_model.py - モデル基底クラスと共通コンポーネント

このファイルには以下が統合されています：
- BaseLightningModule: 全モデルの基底クラス
- Layers: RevIN, MovingAvg, SeriesDecomp
- Losses: StudentTNLLLoss, SharpeLoss, GaussianNLLLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from typing import Any, Dict, Optional, Union
import math
import numpy as np


# =============================================================================
# Layers - 共通ニューラルネットワーク層
# =============================================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    時系列データの正規化と逆正規化を行う。
    """
    def __init__(self, num_features: int, eps=1e-4, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability (increased to 1e-4)
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()
        self.stdev = torch.clamp(self.stdev, min=self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            safe_weight = torch.clamp(self.affine_weight.abs(), min=self.eps) * torch.sign(self.affine_weight + self.eps)
            x = x / safe_weight
        x = x * self.stdev
        x = x + self.mean
        return x


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # [B, L, C]
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block - トレンドと季節成分に分解
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# =============================================================================
# Loss Functions - カスタム損失関数
# =============================================================================

class StudentTNLLLoss(nn.Module):
    """
    Student's T分布に基づく負の対数尤度損失。
    2引数(mu, target)のみを受け取り、sigmaは内部で固定値を使用。
    """
    def __init__(self, beta=1.0, sigma=1.0, v=3.0):
        super().__init__()
        self.beta = beta
        self.sigma_value = sigma
        self.v_value = v
        
    def forward(self, mu, target):
        eps = 1e-6
        sigma = self.sigma_value + eps
        v = self.v_value
        
        # 入力サニタイズ: NaN/Infを安全な値に置換
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e5, neginf=-1e5)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # 数値安定性: より保守的な範囲にクランプ
        mu = torch.clamp(mu, min=-1e5, max=1e5)
        target = torch.clamp(target, min=-1e5, max=1e5)
        
        # Student's T NLL
        v_tensor = torch.tensor(v, dtype=mu.dtype, device=mu.device)
        pi_tensor = torch.tensor(np.pi, dtype=mu.dtype, device=mu.device)
        
        nll = -torch.lgamma((v_tensor + 1) / 2) + torch.lgamma(v_tensor / 2)
        nll = nll + 0.5 * torch.log(v_tensor * pi_tensor) + np.log(sigma)
        
        residual = (target - mu) / sigma
        residual = torch.clamp(residual, min=-1e3, max=1e3)
        nll = nll + ((v_tensor + 1) / 2) * torch.log(1 + (1 / v_tensor) * residual ** 2)
        
        # 最終NLL値をサニタイズ
        nll = torch.nan_to_num(nll, nan=1.0, posinf=10.0, neginf=0.0)
        nll = torch.clamp(nll, min=-1e3, max=1e3)
        
        return torch.mean(nll) * self.beta


class SharpeLoss(nn.Module):
    """
    シャープレシオを最大化するための損失関数。
    """
    def __init__(self, risk_free_rate: float = 0.0, epsilon: float = 1e-8):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # 入力サニタイズ: NaN/Infを安全な値に置換
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]
        y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e5, neginf=-1e5)
        y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1e5, neginf=-1e5)
        y_pred = torch.clamp(y_pred, -1e5, 1e5)
        y_true = torch.clamp(y_true, -1e5, 1e5)
            
        positions = torch.sign(y_pred)
        strategy_returns = positions * y_true
        
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        
        sharpe_ratio = (mean_return - self.risk_free_rate) / (std_return + self.epsilon)
        annualized_sharpe = sharpe_ratio * math.sqrt(252)
        
        # 結果をサニタイズ
        result = torch.nan_to_num(-annualized_sharpe, nan=0.0, posinf=1e3, neginf=-1e3)
        return torch.clamp(result, -1e3, 1e3)


class GaussianNLLLoss(nn.Module):
    """
    ガウス分布の負の対数尤度損失。
    モデルが予測値(mu)と不確実性(sigma)の両方を出力する場合に使用。
    """
    def __init__(self, epsilon: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 入力サニタイズ: NaN/Infを安全な値に置換
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e5, neginf=-1e5)
        sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1e5, neginf=self.epsilon)
        y = torch.nan_to_num(y, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # 値を安全な範囲にクランプ
        mu = torch.clamp(mu, -1e5, 1e5)
        y = torch.clamp(y, -1e5, 1e5)
        sigma = sigma.clamp(min=self.epsilon, max=1e5)
        
        var = sigma ** 2
        var = torch.clamp(var, min=self.epsilon)  # 対数計算前に正値保証
        
        residual = torch.clamp(y - mu, -1e5, 1e5)
        nll = 0.5 * (math.log(2 * math.pi) + torch.log(var) + residual ** 2 / var)
        
        # 最終NLL値をサニタイズ
        nll = torch.nan_to_num(nll, nan=1.0, posinf=10.0, neginf=0.0)
        nll = torch.clamp(nll, -1e3, 1e3)

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll


class DirectionalLoss(nn.Module):
    """
    方向精度を重視する損失関数。
    MSE損失に加えて、予測と実績の符号が一致しない場合にペナルティを与える。
    これにより、Hit Rate（方向的中率）の向上を目指す。
    """
    def __init__(self, mse_weight: float = 1.0, direction_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 入力サニタイズ
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=-1e5)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e5, neginf=-1e5)
        pred = torch.clamp(pred, -1e5, 1e5)
        target = torch.clamp(target, -1e5, 1e5)
        
        # MSE成分
        mse = F.mse_loss(pred, target)
        
        # 方向一致ボーナス: pred * target > 0 なら方向一致
        # 方向が一致していれば 1, 不一致なら 0
        direction_match = (pred * target > 0).float()
        # 方向不一致率をペナルティとして使用
        direction_penalty = 1.0 - direction_match.mean()
        
        # 合計損失
        total_loss = self.mse_weight * mse + self.direction_weight * direction_penalty
        
        return total_loss


class HybridLoss(nn.Module):
    """
    Directional + Huber のハイブリッド損失関数。
    - Huber: 外れ値に強い（デルタ以下はMSE、超えはL1）
    - Directional: 方向精度重視（符号一致でボーナス）
    
    金融ノイズの多いデータで安定した学習を実現。
    """
    def __init__(self, huber_weight: float = 0.5, direction_weight: float = 0.5, 
                 huber_delta: float = 1.0):
        super().__init__()
        self.huber_weight = huber_weight
        self.direction_weight = direction_weight
        self.huber_loss = nn.HuberLoss(delta=huber_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 入力サニタイズ
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e5, neginf=-1e5)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e5, neginf=-1e5)
        pred = torch.clamp(pred, -1e5, 1e5)
        target = torch.clamp(target, -1e5, 1e5)
        
        # Huber損失（外れ値に強い）
        huber = self.huber_loss(pred, target)
        
        # 方向一致ペナルティ
        direction_match = (pred * target > 0).float()
        direction_penalty = 1.0 - direction_match.mean()
        
        # 合計損失
        total_loss = self.huber_weight * huber + self.direction_weight * direction_penalty
        
        return total_loss


# =============================================================================
# Base Lightning Module - 全モデルの基底クラス
# =============================================================================

class BaseLightningModule(pl.LightningModule):
    """
    共通の学習ループ、オプティマイザ設定、Loss計算ロジックを持つ基底クラス。
    各モデルはこのクラスを継承し、`forward` と `__init__` (アーキテクチャ定義) に専念する。
    """
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-5, loss_type: str = 'student_t', **kwargs):
        super().__init__()
        
        self.lr = lr
        self.weight_decay = weight_decay
        self._loss_type = loss_type
        
        if loss_type == 'student_t':
            self.loss_fn = StudentTNLLLoss(beta=1.0)
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'sharpe':
            self.loss_fn = SharpeLoss()
        elif loss_type == 'gaussian':
            self.loss_fn = GaussianNLLLoss()
        elif loss_type == 'directional':
            self.loss_fn = DirectionalLoss(mse_weight=1.0, direction_weight=0.8)  # ★強化: 0.5→0.8
        elif loss_type == 'hybrid':
            self.loss_fn = HybridLoss(huber_weight=0.5, direction_weight=0.5)  # ★新規: Huber + Directional
        else:
            self.loss_fn = StudentTNLLLoss(beta=1.0)  # Default

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, sigma = out
            loss = self.loss_fn(mu.unsqueeze(1), y)
        else:
            pred = out
            loss = self.loss_fn(pred, y)
            
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        
        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, sigma = out
            loss = self.loss_fn(mu.unsqueeze(1), y)
        else:
            pred = out
            loss = self.loss_fn(pred, y)
            
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        total_steps = 1000  # Default fallback
        if self.trainer and hasattr(self.trainer, 'estimated_stepping_batches'):
            try:
                est = self.trainer.estimated_stepping_batches
                if est:
                    total_steps = est
            except:
                pass
        
        scheduler = OneCycleLR(optimizer, max_lr=self.lr, total_steps=total_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step"
            }
        }
