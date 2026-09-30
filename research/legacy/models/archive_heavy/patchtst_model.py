import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from common_utils.base_model import BaseLightningModule, RevIN

class PatchTSTBackbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride, 
                 n_layers, d_model, n_heads, d_ff, dropout, head_dropout, 
                 attn_dropout, pooling_type='max'):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = int((context_window - patch_len) / stride + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        self.patch_num += 1
        self.c_in = c_in
        self.d_model = d_model
        
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_num, d_model))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_dropout = nn.Dropout(head_dropout)
        
        # Channel-Independent Prediction Head
        self.head_mu = nn.Linear(d_model, target_window)
        self.head_sigma = nn.Linear(d_model, target_window)

    def forward(self, x):
        B, L, C = x.shape
        
        # NOTE: RevIN is handled in LightningModule

        # Patching & CI
        x = x.permute(0, 2, 1)              # [B, C, L]
        x = self.padding_patch_layer(x)     # <--- パディングを実行！ [B, C, L+stride]
        x = x.reshape(B * C, -1, 1)         # [B*C, L', 1]
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride).squeeze(2)
        
        # ★インデックス境界修正: 実際のパッチ数を取得
        actual_patch_num = x.shape[1]

        # Embedding
        enc_out = self.value_embedding(x)
        
        # ★インデックス境界修正: position_embeddingを実際のパッチ数に合わせてスライス
        if actual_patch_num <= self.position_embedding.shape[1]:
            pos_emb = self.position_embedding[:, :actual_patch_num, :]
        else:
            # パッチ数が多すぎる場合は、position_embeddingを繰り返して拡張
            repeat_times = (actual_patch_num // self.position_embedding.shape[1]) + 1
            pos_emb_repeated = self.position_embedding.repeat(1, repeat_times, 1)
            pos_emb = pos_emb_repeated[:, :actual_patch_num, :]
        
        enc_out = enc_out + pos_emb
        enc_out = self.dropout(enc_out)

        # Encoder
        enc_out = self.encoder(enc_out) # [B*C, actual_patch_num, d_model]

        # Global Average Pooling over Patches
        enc_out = torch.mean(enc_out, dim=1)
        enc_out = self.output_dropout(enc_out)
        
        # Predict per channel
        mu_out = self.head_mu(enc_out)      # [B*C, Target_Len]
        sigma_out = self.head_sigma(enc_out) 
        
        # Reshape back
        mu_out = mu_out.reshape(B, C, -1).permute(0, 2, 1)    # [B, T, C]
        sigma_out = sigma_out.reshape(B, C, -1).permute(0, 2, 1)
        sigma_out = F.softplus(sigma_out) + 1e-6

        return mu_out, sigma_out

class PatchTSTLightning(BaseLightningModule):
    def __init__(self, input_channels, context_window=20, target_window=1, 
                 patch_len=8, stride=None, n_layers=3, d_model=128, n_heads=4, 
                 d_ff=256, dropout=0.2, lr=1e-3, **kwargs):
        super().__init__(lr=lr, **kwargs)
        
        # ★stride自動計算: patch_lenの半分（オーバーラップ50%）
        if stride is None:
            stride = max(1, patch_len // 2)
        
        # ★d_model/n_heads互換性チェック
        if d_model % n_heads != 0:
            # n_headsを調整してd_modelで割り切れるようにする
            valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
            n_heads = valid_heads[-1] if valid_heads else 4
        
        self.save_hyperparameters()
        
        # RevIN Layer
        self.revin = RevIN(input_channels, affine=True)

        self.model = PatchTSTBackbone(
            c_in=input_channels, context_window=context_window, target_window=target_window,
            patch_len=patch_len, stride=stride, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout, head_dropout=dropout, attn_dropout=dropout
        )
        
        self.channel_mixer = nn.Linear(input_channels, 1)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        
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
            
            mu_all, sigma_all = self.model(x) # [Batch, 1, Channels]
        
        # ★NaN/Infチェック: モデル出力
        mu_all = torch.nan_to_num(mu_all, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_all = torch.nan_to_num(sigma_all, nan=1e-4, posinf=1e6, neginf=1e-4)
        
        # RevIN Denorm
        mu_all = self.revin(mu_all, 'denorm')
        sigma_all = sigma_all * self.revin.stdev
        
        # ★最終的なNaN/Infクリーンアップ
        mu_all = torch.nan_to_num(mu_all, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_all = torch.clamp(sigma_all, min=1e-6, max=1e6)
        
        # [Batch, Channels]
        mu_flat = mu_all.squeeze(1)
        sigma_flat = sigma_all.squeeze(1)
        
        # Mix
        mu = self.channel_mixer(mu_flat)
        sigma = torch.mean(sigma_flat, dim=1, keepdim=True)
        
        return mu.squeeze(-1), sigma.squeeze(-1)

    @staticmethod
    def define_param_space(trial):
        # ★d_model/n_heads互換性を保証するパラメータ選択
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        # n_headsはd_modelで割り切れる値のみ選択
        if d_model == 64:
            n_heads = trial.suggest_categorical("n_heads", [4, 8])
        elif d_model == 128:
            n_heads = trial.suggest_categorical("n_heads", [4, 8])
        else:  # d_model == 256
            n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])
        
        return {
            "patch_len": trial.suggest_categorical("patch_len", [4, 8, 16]),
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        }