import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from common_utils.base_model import BaseLightningModule, RevIN
import math
import logging

def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**(x - 1).bit_length()


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size() # [Batch, Seq, Channels] (Channels = d_model after embedding in backbone)
        
        # ★FFT保護: 常にfloat32で計算（16-bit混合精度でのオーバーフロー防止）
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float() # 強制的にfloat32へ
            
            # Period detection via FFT
            x_ft = torch.fft.rfft(x, dim=1)
        
        # ★FFT保護: NaN/Inf チェック
        if torch.isnan(x_ft.real).any() or torch.isinf(x_ft.real).any():
            # FFT失敗時はフォールバック（入力をそのまま返す）
            return x
        
        period_list = x_ft.abs().mean(0).mean(1) # Avg amplitude over Batch and Channels
        # ★FFT保護: ゼロ除算防止
        period_list = period_list + 1e-8
        
        # ★インデックス境界修正: top_kがFFT周波数ビン数を超えないようにする
        n_freq_bins = period_list.shape[0]
        k_safe = min(self.k, n_freq_bins)
        if k_safe < 1:
            return x  # FFTビンがない場合はフォールバック
        
        # Top-k periods (安全な数で取得)
        _, top_list = torch.topk(period_list, k_safe)
        top_list = top_list.detach().cpu()
        
        # ★インデックス境界修正: period_weightアクセス時の境界チェック
        x_ft_abs_mean = x_ft.abs().mean(1)  # [B, n_freq_bins]
        # インデックスをクランプして境界内に収める
        top_list_clamped = torch.clamp(top_list, 0, x_ft_abs_mean.shape[1] - 1)
        period_weight = x_ft_abs_mean[:, top_list_clamped] # [B, k_safe]
        
        # ★FFT保護: オーバーフロー防止
        period_weight = torch.clamp(period_weight, -1e6, 1e6)
        # Softmax for weighting
        period_weight = F.softmax(period_weight.float(), dim=1)

        res = []
        for i in range(k_safe):
            period = int(top_list[i].item())  # Python intに変換
            # ★インデックス境界修正: periodを安全な範囲に制限
            period = max(2, min(period, T))  # 2以上、シーケンス長以下
            
            # 1D -> 2D
            temp = x.permute(0, 2, 1) # [B, N, T]
            # Padding to fit period
            pad_len = (period - (self.seq_len % period)) % period
            if pad_len > 0:
                temp = F.pad(temp, (0, pad_len))
            
            B_new, N_new, L = temp.shape
            length = L // period
            if length < 1:
                # reshapeできない場合はスキップ
                res.append(x.clone())
                continue
                
            temp = temp.reshape(B_new, N_new, length, period) # [B, N, L/P, P]
            
            # 2D Conv
            out = self.conv(temp) # [B, N, L/P, P]
            
            # 2D -> 1D
            out = out.reshape(B_new, N_new, -1)
            out = out[:, :, :self.seq_len] # Crop padding
            out = out.permute(0, 2, 1) # [B, T, N]
            res.append(out)
            
        # Aggregation
        res = torch.stack(res, dim=-1) # [B, T, N, k_safe]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1) # [B, T, N, k_safe]
        res = torch.sum(res * period_weight.to(res.dtype), -1) # Weighted sum over k
        
        # ★FFT保護: 最終出力のNaNチェック
        result = res + x
        if torch.isnan(result).any():
            return x  # フォールバック
        
        return result

class TimesNetBackbone(nn.Module):
    def __init__(self, input_channels, seq_len, pred_len, top_k=5, d_model=32, d_ff=64, num_kernels=6, n_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Enforce Power of 2 for 16-bit stability
        d_model_pow2 = _next_power_of_2(d_model)
        if d_model_pow2 != d_model:
            logging.info(f"[TimesNet] Adjusted d_model from {d_model} to {d_model_pow2} (Power of 2 constraint)")
            d_model = d_model_pow2
            
        d_ff_pow2 = _next_power_of_2(d_ff)
        if d_ff_pow2 != d_ff:
            logging.info(f"[TimesNet] Adjusted d_ff from {d_ff} to {d_ff_pow2} (Power of 2 constraint)")
            d_ff = d_ff_pow2
        
        self.embedding = nn.Linear(input_channels, d_model)
        
        self.layers = nn.ModuleList([
            TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
            for _ in range(n_layers)
        ])
        
        self.enc_embedding = nn.Linear(d_model, input_channels) # Back to original dim
        self.predict_layer = nn.Linear(seq_len, pred_len)
        self.predict_sigma = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Channels]
        
        # NOTE: RevIN is handled in LightningModule
        
        # Embedding [B, T, C] -> [B, T, d_model]
        enc_out = self.embedding(x)
        
        # TimesBlocks
        for layer in self.layers:
            enc_out = layer(enc_out)
            
        # Projection
        # [B, T, d_model] -> [B, T, C]
        dec_out = self.enc_embedding(enc_out)
        
        # Predict: [B, T, C] -> [B, C, T] -> [B, C, Pred]
        dec_out = dec_out.permute(0, 2, 1)
        mu = self.predict_layer(dec_out)
        sigma = F.softplus(self.predict_sigma(dec_out)) + 1e-6
        
        # [B, Pred, C]
        mu = mu.permute(0, 2, 1)
        sigma = sigma.permute(0, 2, 1)
        
        return mu, sigma

class TimesNetLightning(BaseLightningModule):
    def __init__(self, input_channels, context_window=20, target_window=1, 
                 top_k=3, d_model=32, d_ff=64, num_kernels=3, n_layers=2, lr=1e-3, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.save_hyperparameters()
        
        # RevIN Layer
        self.revin = RevIN(input_channels, affine=True)
        
        self.model = TimesNetBackbone(
            input_channels=input_channels,
            seq_len=context_window,
            pred_len=target_window,
            top_k=top_k,
            d_model=d_model,
            d_ff=d_ff,
            num_kernels=num_kernels,
            n_layers=n_layers
        )
        
        self.channel_mixer = nn.Linear(input_channels, 1)

    def forward(self, x):
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
        sigma_all = sigma_all * self.revin.stdev
        
        # ★最終的なNaN/Infクリーンアップ
        mu_all = torch.nan_to_num(mu_all, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma_all = torch.clamp(sigma_all, min=1e-6, max=1e6)
        
        mu_flat = mu_all.squeeze(1)
        sigma_flat = sigma_all.squeeze(1)
        
        mu = self.channel_mixer(mu_flat)
        sigma = torch.mean(sigma_flat, dim=1, keepdim=True)
        
        return mu.squeeze(-1), sigma.squeeze(-1)

    @staticmethod
    def define_param_space(trial):
        return {
            "top_k": trial.suggest_int("top_k", 3, 5),
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128, 256]),
            "d_ff": trial.suggest_categorical("d_ff", [64, 128, 256, 512]),
            "num_kernels": trial.suggest_int("num_kernels", 3, 6),
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        }
