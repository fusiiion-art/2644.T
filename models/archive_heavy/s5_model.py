import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Tuple, Optional

from common_utils.base_model import BaseLightningModule

# --- 1. Robust Loss Function (Student-t) ---
# common_utils.custom_losses に移動済みのため削除

# --- 2. JIT-Compiled Recurrence (Windows Safe) ---
@torch.jit.script
def binary_operator_diag(q_i: torch.Tensor, q_j: torch.Tensor, a_i: torch.Tensor, a_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parallel Scanのための結合法則演算 (今回はJIT化された直列スキャンを使用)
    """
    return a_j * q_i + q_j, a_j * a_i

@torch.jit.script
def ssm_scan(u: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor) -> torch.Tensor:
    """
    Windows環境での安定性と速度を両立させるJITコンパイル済みSSMスキャン。
    """
    batch_size, seq_len, _ = u.shape
    hidden_dim = A_bar.shape[0]
    
    # 状態変数の初期化 (Batch, H)
    x = torch.zeros(batch_size, hidden_dim, dtype=A_bar.dtype, device=A_bar.device)
    xs = []
    
    # Buの事前計算
    Bu = torch.einsum('bli,hi->blh', u, B_bar)
    
    for t in range(seq_len):
        x = A_bar * x + Bu[:, t]
        xs.append(x)
        
    return torch.stack(xs, dim=1)

class S5SSM_Real(nn.Module):
    def __init__(self, ssm_size, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.ssm_size = ssm_size 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # HiPPO Initialization (Complex)
        self.log_delta = nn.Parameter(torch.log(torch.ones(ssm_size) * 0.001))
        self.A_real = nn.Parameter(-0.5 * torch.ones(ssm_size))
        self.A_imag = nn.Parameter(torch.randn(ssm_size))
        self.B_real = nn.Parameter(torch.randn(ssm_size, input_dim))
        self.B_imag = nn.Parameter(torch.randn(ssm_size, input_dim))
        self.C_real = nn.Parameter(torch.randn(hidden_dim, ssm_size))
        self.C_imag = nn.Parameter(torch.randn(hidden_dim, ssm_size))
        self.D = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # ★重要: Windows/Ampere GPUでのNaNを防ぐため、SSM計算中はFP32を強制
        with torch.cuda.amp.autocast(enabled=False):
            u_f32 = u.float()
            
            # 数値安定化のためのクランプ
            u_f32 = torch.clamp(u_f32, min=-1e4, max=1e4)

            # Discretization
            delta = torch.exp(self.log_delta)
            
            # A_bar calculation
            dt_A_real = self.A_real * delta
            dt_A_imag = self.A_imag * delta
            A_bar = torch.complex(dt_A_real, dt_A_imag).exp()
            
            # B_bar calculation
            B_complex = torch.complex(self.B_real, self.B_imag)
            B_bar = B_complex * delta.unsqueeze(-1)
            
            # Recurrence Scan
            u_complex = torch.complex(u_f32, torch.zeros_like(u_f32))
            x_scan = ssm_scan(u_complex, A_bar, B_bar)
            
            # Output mixing
            C_complex = torch.complex(self.C_real, self.C_imag)
            y = torch.einsum('blh,dh->bld', x_scan, C_complex).real
            
            # Direct connection
            du = torch.einsum('di,bli->bld', self.D, u_f32)
            out = y + du
        
        return self.dropout(self.activation(out))

class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(d_model) * 1e-4)

    def forward(self, x):
        query = x[:, -1:, :] 
        key = value = x
        attn_output, _ = self.attn(query, key, value)
        out = self.norm(query + self.dropout(attn_output) * self.gamma)
        return out.squeeze(1)

class S5Model(nn.Module):
    def __init__(self, input_channels, d_model=128, ssm_size=64, n_layers=4, dropout=0.2, n_heads=4):
        super().__init__()
        self.encoder = nn.Linear(input_channels, d_model)
        self.layers = nn.ModuleList([
            S5SSM_Real(ssm_size=ssm_size, input_dim=d_model, hidden_dim=d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = TemporalCrossAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.head_mu = nn.Linear(d_model, 1)
        self.head_sigma = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm(x)
        context = self.cross_attn(x)
        mu = self.head_mu(context)
        sigma = self.softplus(self.head_sigma(context)) + 1e-6
        return mu, sigma

class S5Lightning(BaseLightningModule):
    def __init__(self, input_channels, d_model=128, ssm_size=64, n_layers=4, dropout=0.2, lr=1e-3, weight_decay=1e-5, n_heads=4, **kwargs):
        super().__init__(lr=lr, weight_decay=weight_decay, **kwargs)
        self.save_hyperparameters()
        self.model = S5Model(input_channels, d_model, ssm_size, n_layers, dropout, n_heads)

    def forward(self, x):
        # ★S5用のNaN/Infガード（他のモデルと同様に実装）
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float() # 強制FP32
            
            # 入力サニタイズ
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, min=-1e5, max=1e5)

            mu, sigma = self.model(x)

        # 出力サニタイズ
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e5, neginf=-1e5)
        sigma = torch.nan_to_num(sigma, nan=1e-4, posinf=1e5, neginf=1e-4)
        sigma = torch.clamp(sigma, min=1e-6, max=1e5)

        return mu, sigma

    @staticmethod
    def define_param_space(trial):
        return {
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
            "ssm_size": trial.suggest_categorical("ssm_size", [32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 2, 8),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        }