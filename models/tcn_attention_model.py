"""
tcn_attention_model.py - TCN + Attention ハイブリッドモデル

金融時系列のノイズに強い設計：
- 浅層TCN（2-3層）で局所パターンを抽出
- MultiheadAttention で重要な時間ステップを学習
- 強めのDropoutで過学習を抑制

参考: Perplexity分析より「TCN-Attention は高ボラティリティ金融データで LSTM/GRU より安定」
期待精度: 54-56% 方向精度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from typing import Dict, Any, Tuple
from common_utils.base_model import BaseLightningModule


class TemporalBlock(nn.Module):
    """TCNの基本ブロック：Dilated Causal Convolution + Residual"""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, 
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        
        self.chomp = padding
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """x: [Batch, Channels, Seq]"""
        out = self.conv1(x)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.dropout1(self.relu(self.bn1(out)))
        
        out = self.conv2(out)
        if self.chomp > 0:
            out = out[:, :, :-self.chomp]
        out = self.dropout2(self.relu(self.bn2(out)))
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNAttentionBackbone(nn.Module):
    """TCN + Self-Attention バックボーン"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 n_layers: int = 2, kernel_size: int = 3, 
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        # TCN Layers (浅層に制限)
        layers = []
        num_channels = [hidden_dim] * n_layers
        
        for i in range(n_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation, padding=padding, dropout=dropout
            ))
        
        self.tcn = nn.Sequential(*layers)
        
        # Self-Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # [Batch, Seq, Hidden]
        )
        
        # Layer Normalization for attention output
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.output_dim = hidden_dim
        
    def forward(self, x):
        """
        x: [Batch, Seq, Features] -> [Batch, hidden_dim]
        """
        # TCN expects [Batch, Channels, Seq]
        x = x.transpose(1, 2)  # [Batch, Features, Seq]
        
        # TCN forward
        tcn_out = self.tcn(x)  # [Batch, hidden_dim, Seq]
        
        # Prepare for attention: [Batch, Seq, hidden_dim]
        tcn_out = tcn_out.transpose(1, 2)
        
        # Self-attention
        attn_out, _ = self.attention(tcn_out, tcn_out, tcn_out)
        
        # Residual + LayerNorm
        attn_out = self.layer_norm(tcn_out + attn_out)
        
        # Global average pooling over time
        out = attn_out.mean(dim=1)  # [Batch, hidden_dim]
        
        return out


class TCNAttentionLightning(BaseLightningModule):
    """
    TCN + Attention Lightning Module
    金融時系列のノイズに強い設計（1000-5000行データ向け）
    """
    def __init__(self,
                 input_dim: int = None,
                 seq_len: int = 32,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 kernel_size: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 loss_type: str = 'mse',
                 # 互換性エイリアス
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
        
        # TCN + Attention バックボーン
        self.backbone = TCNAttentionBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            kernel_size=kernel_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 出力層
        self.fc_mu = nn.Linear(hidden_dim, 1)
        self.fc_sigma = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向き伝播
        x: [Batch, Seq, Features]
        returns: (mu, sigma)
        """
        # FP32強制（数値安定性）
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            
            # Backbone forward
            hidden = self.backbone(x)  # [Batch, hidden_dim]
            
            # 出力
            mu = self.fc_mu(hidden).squeeze(-1)  # [Batch]
            sigma = F.softplus(self.fc_sigma(hidden)).squeeze(-1) + 1e-6  # [Batch]
            
        return mu, sigma
    
    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Optunaのハイパーパラメータ探索空間（過学習抑制重視）"""
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),  # 小さめ
            "n_layers": trial.suggest_int("n_layers", 2, 3),  # 浅層に制限
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),  # 小さめ
            "num_heads": trial.suggest_categorical("num_heads", [2, 4]),
            "dropout": trial.suggest_float("dropout", 0.3, 0.5),  # 強めのドロップアウト
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),  # 強め正則化
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        }
