import torch
import torch.nn as nn
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
import optuna
from typing import Dict, Any
try:
    from common_utils.dynamic_ensemble import AttentionGateEnsemble
except Exception:
    AttentionGateEnsemble = None

class MetaLearner(pl.LightningModule):
    """
    複数の専門家モデルの予測を入力として受け取り、最終的な予測を出力する監督者（メタ学習器）。
    ★強化: Attention機構で各Expertの重みを動的に学習。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2, 
                 dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-5,
                 use_attention: bool = True, n_experts: int = 5):
        """
        Args:
            input_dim: 入力特徴量の次元数（専門家の予測値 + メタ特徴量）
            hidden_dim: 隠れ層のニューロン数
            n_layers: 隠れ層の数
            dropout: ドロップアウト率
            lr: 学習率
            weight_decay: L2正則化の強度
            use_attention: Attention機構を使うか
            n_experts: 専門家モデルの数（Attention用）
        """
        super().__init__()
        self.save_hyperparameters()
        self.use_attention = use_attention
        self.n_experts = n_experts

        # ★Attention機構: 各Expertに対する動的重み付け
        if use_attention and n_experts > 0:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_experts),
                nn.Softmax(dim=-1)  # 重みの合計が1になるように正規化
            )

        # メインのMLP
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)
        
        # ★DirectionalLoss対応
        from common_utils.base_model import DirectionalLoss
        self.loss_fn = DirectionalLoss(mse_weight=1.0, direction_weight=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """モデルの順伝播"""
        return self.model(x)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Attention重みを取得（解釈可能性のため）"""
        if self.use_attention and hasattr(self, 'attention'):
            return self.attention(x)
        return None

    def _common_step(self, batch, batch_idx):
        """訓練ステップと検証ステップで共通の処理"""
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.loss_fn(y_hat, y.squeeze(-1) if y.dim() > 1 else y)
        return loss

    def training_step(self, batch, batch_idx):
        """訓練ステップ"""
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """検証ステップ"""
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """オプティマイザを設定"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """★拡張: より広いハイパーパラメータ探索空間"""
        return {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.2, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        }

class SupervisorEnsemble:
    """
    複数の専門家モデルの予測を統合（アンサンブル）するクラス。
    逆分散加重 (Inverse-Variance Weighting) や MetaLearner による統合を行う。
    """
    def __init__(self, supervisor_assets: Dict[str, Any] = None, use_ivw: bool = True, dynamic_ensemble: Any = None):
        self.supervisor_assets = supervisor_assets
        self.use_ivw = use_ivw
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dynamic_ensemble = dynamic_ensemble
        if self.dynamic_ensemble is None and AttentionGateEnsemble is not None:
            # create a default dynamic ensemble if features provided
            try:
                self.dynamic_ensemble = AttentionGateEnsemble(n_experts=len(self.supervisor_assets.get('features', [])) if self.supervisor_assets else 0, state_dim=4)
                self.dynamic_ensemble = self.dynamic_ensemble.to(self.device).eval()
            except Exception:
                self.dynamic_ensemble = None
        
        if self.supervisor_assets:
            self.model = self.supervisor_assets.get('model')
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device).eval()
            self.scaler = self.supervisor_assets.get('scaler')
            self.features = self.supervisor_assets.get('features', [])

    def aggregate(self, expert_predictions: Dict[str, float], expert_sigmas: Dict[str, float], latest_vix: float = 20.0) -> Dict[str, float]:
        """
        予測の集約を行う。
        Returns:
            {
                "mu": float, # 最終予測リターン
                "sigma": float, # 最終予測不確実性
                "weights": Dict[str, float] # 各モデルの重み
            }
        """
        predictions_list = list(expert_predictions.values())
        sigmas_list = list(expert_sigmas.values())
        
        # 1. 逆分散加重 (IVW)
        # Weight_i = 1 / sigma_i^2
        weights = []
        for sigma in sigmas_list:
            if sigma > 0:
                weights.append(1.0 / (sigma ** 2))
            else:
                weights.append(1.0) # Fallback

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)

        mu_weighted = sum(w * pred for w, pred in zip(normalized_weights, predictions_list))
        
        # 加重Sigma: Sigma_weighted = √(Σ(Weight_i * Sigma_i^2)) ? 
        # 厳密な分散の加重平均ではないが、IVWの文脈では分散 = 1/Sum(1/sigma_i^2) となることが多い
        # ここでは保守的に加重平均をとる
        # sigma_weighted = np.sqrt(sum(w * s**2 for w, s in zip(normalized_weights, sigmas_list)))
        import numpy as np
        sigma_weighted = np.sqrt(1.0 / total_weight) if total_weight > 0 else np.mean(sigmas_list)

        result = {
            "mu_weighted": mu_weighted,
            "sigma_weighted": sigma_weighted,
            "weights": {k: w for k, w in zip(expert_predictions.keys(), normalized_weights)}
        }
        
        # 2. MetaLearner (Supervisor Model) Integration
        # もし学習済みのMetaLearnerがあれば、それを使って補正・予測する
        if self.dynamic_ensemble is not None:
            try:
                # build state vector (simple example: [latest_vix, mu_weighted, sigma_weighted, 0])
                import numpy as _np
                state = _np.array([[latest_vix, mu_weighted, sigma_weighted, 0.0]], dtype=float)
                expert_vals = _np.array([[v for v in expert_predictions.values()]], dtype=float)
                import torch as _torch
                with _torch.no_grad():
                    state_t = _torch.from_numpy(state).float().to(self.dynamic_ensemble.attn[0].weight.device)
                    expert_t = _torch.from_numpy(expert_vals).float().to(state_t.device)
                    ens_pred, ens_weights = self.dynamic_ensemble(expert_t, state_t)
                    ens_pred = float(ens_pred.cpu().numpy().ravel()[0])
                    weights_out = ens_weights.cpu().numpy().ravel().tolist()

                result["mu_dynamic"] = ens_pred
                result["weights_dynamic"] = {k: w for k, w in zip(expert_predictions.keys(), weights_out)}
                # prefer dynamic ensemble prediction when available
                result["mu"] = ens_pred
                result["sigma"] = sigma_weighted
                return result
            except Exception as e:
                logging = __import__('logging')
                logging.warning(f"Dynamic ensemble inference failed: {e}")

        if self.supervisor_assets and self.model:
            try:
                # 入力データ作成 (Experts + Meta Features)
                input_data = expert_predictions.copy()
                input_data['meta_std'] = sigma_weighted
                input_data['meta_mean'] = mu_weighted
                
                # DataFrame化してスケーリング
                import pandas as pd
                input_df = pd.DataFrame([input_data])
                final_input = pd.DataFrame(0.0, index=[0], columns=self.features)
                
                for col in self.features:
                    if col in input_df.columns: 
                        final_input[col] = input_df[col]
                    elif 'vix' in col.lower() and latest_vix is not None:
                        final_input[col] = latest_vix
                
                X_sup = self.scaler.transform(final_input.values)
                
                with torch.no_grad():
                    input_tensor = torch.from_numpy(X_sup).float().to(self.device)
                    sup_pred = self.model(input_tensor).item()
                    
                result["mu_supervised"] = sup_pred
                # MetaLearnerがSigmaを出さない場合はWeighted Sigmaを継承
                result["sigma_supervised"] = sigma_weighted 
                
                # 最終決定: Supervisorがいるならそれを使う、いなければIVW
                result["mu"] = sup_pred
                result["sigma"] = sigma_weighted 
                
            except Exception as e:
                print(f"  ⚠️ Supervisor inference failed, falling back to IVW: {e}")
                result["mu"] = mu_weighted
                result["sigma"] = sigma_weighted
        else:
            # IVWのみ
            result["mu"] = mu_weighted
            result["sigma"] = sigma_weighted

        return result
