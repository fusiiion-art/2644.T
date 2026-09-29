import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils.base_model import BaseLightningModule, RevIN, SeriesDecomp

class iTransformerBackbone(nn.Module):
    def __init__(self, n_vars, seq_len, pred_len, d_model=128, n_heads=4, d_ff=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model 
        
        # Decomposition
        # kernel_size should be odd and smaller than seq_len. 
        # If seq_len=32, kernel=25 is okay (padding handles it), but 17 is safer.
        self.decomp = SeriesDecomp(kernel_size=17)

        # Encoders for Seasonal and Trend components
        # Sharing weights or separate? Separate is usually better for distinct dynamics.
        self.enc_embedding_s = nn.Linear(seq_len, d_model)
        self.enc_embedding_t = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder_s = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_t = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Projectors
        self.projector_s = nn.Linear(d_model, pred_len, bias=True)
        self.projector_t = nn.Linear(d_model, pred_len, bias=True)
        
        # Uncertainty (Shared or separate?) - Single uncertainty from combined features
        self.projector_sigma = nn.Linear(d_model * 2, pred_len, bias=True)

    def forward(self, x):
        # x: [Batch, Seq_Len, N_Vars]
        
        # 1. Decomposition
        seasonal, trend = self.decomp(x)
        
        # 2. Invert: [Batch, N_Vars, Seq_Len]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        
        # 3. Embedding
        enc_out_s = self.enc_embedding_s(seasonal) # [Batch, N_Vars, D_model]
        enc_out_t = self.enc_embedding_t(trend)
        
        # 4. Encoder
        enc_out_s = self.encoder_s(enc_out_s)
        enc_out_t = self.encoder_t(enc_out_t)
        
        # 5. Projection
        mu_s = self.projector_s(enc_out_s)
        mu_t = self.projector_t(enc_out_t)
        mu_out = mu_s + mu_t
        
        # Sigma: Concatenate features
        enc_cat = torch.cat([enc_out_s, enc_out_t], dim=-1) # [Batch, N_Vars, D_model*2]
        sigma_out = self.projector_sigma(enc_cat)
        sigma_out = F.softplus(sigma_out) + 1e-6
        
        # 6. Transpose back: [Batch, Pred_Len, N_Vars]
        mu_out = mu_out.permute(0, 2, 1)
        sigma_out = sigma_out.permute(0, 2, 1)
        
        return mu_out, sigma_out

class iTransformerLightning(BaseLightningModule):
    def __init__(self, input_channels, context_window=20, target_window=1, 
                 d_model=128, n_heads=4, d_ff=256, n_layers=2, dropout=0.1, lr=1e-3, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.save_hyperparameters()
        
        # RevIN Layer
        self.revin = RevIN(input_channels, affine=True)
        
        self.model = iTransformerBackbone(
            n_vars=input_channels,
            seq_len=context_window,
            pred_len=target_window,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
        
        self.channel_mixer = nn.Linear(input_channels, 1)

    def forward(self, x):
        # x: [Batch, Seq, Vars]
        
        # ★AMP保護: 常にfloat32で計算（16-bit混合精度でのオーバーフロー防止）
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()  # 入力を強制的にfloat32へ
            
            # ★数値安定性: 極端な値をクランプ
            x = torch.clamp(x, min=-1e6, max=1e6)
            
            # ★NaN/Infチェック: 入力に異常値があれば0で置換
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # RevIN Norm
            x = self.revin(x, 'norm')
            
            # ★RevIN後のNaN/Infチェック
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            mu_all, sigma_all = self.model(x)
        
        # ★NaN/Infチェック: モデル出力
        mu_all = torch.nan_to_num(mu_all, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_all = torch.nan_to_num(sigma_all, nan=1e-4, posinf=1e6, neginf=1e-4)
        
        # RevIN Denorm
        mu_all = self.revin(mu_all, 'denorm')
        sigma_all = sigma_all * self.revin.stdev # Sigma scaling
        
        # ★最終的なNaN/Infクリーンアップ
        mu_all = torch.nan_to_num(mu_all, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_all = torch.clamp(sigma_all, min=1e-6, max=1e6)
        
        # Flatten: [Batch, Vars]
        mu_flat = mu_all.squeeze(1)
        sigma_flat = sigma_all.squeeze(1)
        
        # Mix
        mu = self.channel_mixer(mu_flat)
        sigma = torch.mean(sigma_flat, dim=1, keepdim=True)
        
        return mu.squeeze(-1), sigma.squeeze(-1)

    @staticmethod
    def define_param_space(trial):
        return {
            "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
            "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512]),
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        }
