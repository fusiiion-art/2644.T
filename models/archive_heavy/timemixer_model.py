import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils.base_model import BaseLightningModule, RevIN, SeriesDecomp, MovingAvg

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing for seasonality (MLP-based)
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low resolution
        # season_list: [scale_0 (fine), scale_1, ...]
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high)

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing for trend (MLP-based)
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window**i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window**i),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, trend_list):
        # mixing low->high resolution
        # trend_list: [scale_0 (fine), ..., scale_N (coarse)]
        trend_list_reverse = trend_list[::-1]
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, pred_len, down_sampling_window, down_sampling_layers, channel_independence=False, d_model=16, d_ff=32, dropout=0.1):
        super().__init__()
        # Decomp
        self.decomp = SeriesDecomp(kernel_size=25) 
        
        # Cross-Scale Mixing
        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.Dropout(dropout),
        )

        self.channel_independence = channel_independence
        
        self.predict_layers = nn.ModuleList([
             nn.Linear(seq_len // (down_sampling_window**i), pred_len)
             for i in range(down_sampling_layers + 1)
        ])

    def forward(self, x_list):
        preds = []
        for i, x in enumerate(x_list):
            x_t = x.permute(0, 2, 1)
            pred = self.predict_layers[i](x_t)
            pred = pred.permute(0, 2, 1)
            preds.append(pred)
            
        res = torch.sum(torch.stack(preds), dim=0)
        return res

class TimeMixerBackbone(nn.Module):
    def __init__(self, seq_len, pred_len, input_channels, down_sampling_window=2, down_sampling_layers=2, d_model=32, dropout=0.1):
        super().__init__()
        
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        
        self.pdm = PastDecomposableMixing(seq_len, pred_len, down_sampling_window, down_sampling_layers, d_model=d_model)
        
        self.enc_in = input_channels
        self.projection_mu = nn.Linear(input_channels, input_channels) 
        self.projection_sigma = nn.Linear(input_channels, input_channels)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        
        # NOTE: RevIN is handled in LightningModule

        # Downsample to create scales
        x_list = [x]
        for i in range(self.down_sampling_layers):
            x_tmp = x_list[-1].permute(0, 2, 1)
            x_tmp = F.avg_pool1d(x_tmp, kernel_size=self.down_sampling_window, stride=self.down_sampling_window)
            x_tmp = x_tmp.permute(0, 2, 1)
            x_list.append(x_tmp)

        out = self.pdm(x_list) # [Batch, Pred, Channels]

        mu = self.projection_mu(out)
        sigma = F.softplus(self.projection_sigma(out)) + 1e-6
        
        return mu, sigma

class TimeMixerLightning(BaseLightningModule):
    def __init__(self, input_channels, context_window=20, target_window=1, 
                 down_sampling_window=2, down_sampling_layers=2, d_model=32, dropout=0.1, lr=1e-3, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.save_hyperparameters()
        
        # RevIN Layer
        self.revin = RevIN(input_channels, affine=True)

        self.model = TimeMixerBackbone(
            seq_len=context_window,
            pred_len=target_window,
            input_channels=input_channels,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            d_model=d_model,
            dropout=dropout
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
        
        # Flatten [Batch, Pred=1, C]
        mu_flat = mu_all.squeeze(1)
        sigma_flat = sigma_all.squeeze(1)
        
        mu = self.channel_mixer(mu_flat)
        sigma = torch.mean(sigma_flat, dim=1, keepdim=True)
        
        return mu.squeeze(-1), sigma.squeeze(-1)

    @staticmethod
    def define_param_space(trial):
        return {
            "down_sampling_window": trial.suggest_categorical("down_sampling_window", [2]),
            "down_sampling_layers": trial.suggest_int("down_sampling_layers", 1, 4),
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        }
