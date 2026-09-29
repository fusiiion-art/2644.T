"""
lightgbm_model.py - LightGBM Regressor for semi_daily_1

小データ（1,000行）で強力な非線形モデル。
線形モデルでは捉えられない閾値ベースのパターンを捕捉。
"""
import numpy as np
import torch
import pytorch_lightning as pl
import optuna
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from common_utils.base_model import BaseLightningModule


class LightGBMLightning(BaseLightningModule):
    """
    LightGBM Regressor wrapped in PyTorch Lightning format.
    内部でLightGBMを使用するが、I/Oは他のモデルと互換性を持つ。
    
    Note: 実際の学習はLightGBM側で行うが、
    PyTorch Lightningのチェックポイント形式で保存するための互換ラッパー。
    """
    
    def __init__(self,
                 input_dim: int = None,
                 seq_len: int = 32,
                 # LightGBM params
                 num_leaves: int = 31,
                 learning_rate: float = 0.05,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 feature_fraction: float = 0.8,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 0.1,
                 min_child_samples: int = 20,  # ★新規: 過学習防止
                 # Common params
                 dropout: float = 0.0,  # LightGBMでは未使用
                 lr: float = 0.05,  # learning_rateのエイリアス
                 weight_decay: float = 0.1,  # reg_lambdaのエイリアス
                 loss_type: str = 'mse',
                 # Compatibility aliases
                 input_channels: int = None,
                 context_window: int = None,
                 target_window: int = None,
                 **kwargs):
        
        # パラメータ名の互換性対応
        input_dim = input_dim or input_channels
        seq_len = context_window or seq_len
        if input_dim is None:
            raise ValueError("input_dim or input_channels must be specified")
        
        # lrとlearning_rateの整合（Optuna用）
        learning_rate = lr if lr != 0.05 else learning_rate
        reg_lambda = weight_decay if weight_decay != 0.1 else reg_lambda
        
        super().__init__(lr=learning_rate, weight_decay=reg_lambda, loss_type=loss_type)
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.flattened_dim = seq_len * input_dim
        
        # LightGBMパラメータ保存
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'feature_fraction': feature_fraction,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_samples': min_child_samples,  # ★新規
            'verbose': -1,
            'n_jobs': -1,
        }
        self.n_estimators = n_estimators
        
        # LightGBMモデル（後でfitで初期化）
        self.lgb_model: Optional[lgb.Booster] = None
        
        # ダミーのPyTorchパラメータ（PLが要求するため）
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, Seq, Features]
        
        Note: LightGBMはバッチ予測が得意なので、一括で処理。
        """
        batch_size = x.shape[0]
        
        # Flatten: [Batch, Seq * Features]
        # Use reshape instead of view to handle non-contiguous tensors
        x_flat = x.reshape(batch_size, -1).cpu().numpy()
        
        if self.lgb_model is not None:
            # 学習済みモデルで予測
            preds = self.lgb_model.predict(x_flat)
            mu = torch.tensor(preds, dtype=torch.float32, device=x.device)
        else:
            # 未学習の場合はゼロ予測
            mu = torch.zeros(batch_size, device=x.device)
        
        sigma = torch.ones_like(mu) * 0.01
        return mu, sigma
    
    def training_step(self, batch, batch_idx):
        """
        LightGBMは内部で学習するため、このメソッドはダミー。
        実際の学習はfit_lgbmで行う。
        """
        x, y = batch
        mu, _ = self(x)
        loss = torch.nn.functional.mse_loss(mu, y)
        
        # ★重要: 勾配を持たせるためにダミーパラメータを加算
        # これがないと "element 0 of tensors does not require grad" エラーになる
        loss = loss + 0.0 * self.dummy_param.sum()
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def fit_lgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        LightGBMモデルを学習する。
        trainer_model.pyから直接呼び出すことを想定。
        """
        # 3D入力 (Batch, Seq, Features) の場合、2DにFlattenする
        if X_train.ndim == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        if X_val is not None and X_val.ndim == 3:
            X_val = X_val.reshape(X_val.shape[0], -1)
            
        train_data = lgb.Dataset(X_train, label=y_train)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),  # ログ出力なし
        ]
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        self.lgb_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        
        
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """LightGBMモデルを文字列として保存"""
        if self.lgb_model is not None:
            checkpoint["lgb_model_str"] = self.lgb_model.model_to_string()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """LightGBMモデルを復元"""
        if "lgb_model_str" in checkpoint:
            self.lgb_model = lgb.Booster(model_str=checkpoint["lgb_model_str"])

    def configure_optimizers(self):
        """ダミー（LightGBMは内部で最適化）"""
        return torch.optim.Adam([self.dummy_param], lr=1e-4)
    
    @staticmethod
    def define_param_space(trial: optuna.Trial) -> Dict[str, Any]:
        """Optunaのハイパーパラメータ探索空間（拡張版）"""
        return {
            # ★拡張: 探索範囲を広げつつ過学習対策を追加
            "num_leaves": trial.suggest_int("num_leaves", 8, 96),  # 64→96
            "max_depth": trial.suggest_int("max_depth", 3, 10),    # 8→10
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),  # ★新規
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),  # ★新規（過学習防止）
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            # 互換性エイリアス
            "lr": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1.0, log=True),
            "dropout": 0.0,  # 未使用
        }
