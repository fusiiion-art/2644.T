import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
import optuna
from pathlib import Path
import joblib
import json
import ast
import warnings
import logging
import lightgbm as lgb
import mlflow # Phase 3: MLflow Integration

logging.getLogger("joblib").setLevel(logging.WARNING)
from typing import Any, Dict, Tuple, List, Type, Optional
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
# from optuna.integration import PyTorchLightningPruningCallback # 直接実装するため不要化
try:
    from common_utils.purged_cv import PurgedGroupTimeSeriesSplit
except Exception:
    from sklearn.model_selection import TimeSeriesSplit as _TSS

    class PurgedGroupTimeSeriesSplit(_TSS):
        def __init__(self, n_splits=5, purge_gap=0, embargo_gap=0, group_gap=0):
            super().__init__(n_splits=n_splits)
import os
import tempfile
import re
from sklearn.metrics import f1_score

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# --- 共通の評価指標をインポート ---
from common_utils.metrics import directional_accuracy
try:
    from generate.featureregistry import FEATURE_SETS_DEFINITIONS
except ImportError:
    FEATURE_SETS_DEFINITIONS = {}
# CFI
try:
    from common_utils.clustered_importance import ClusteredFeatureImportance
except Exception:
    ClusteredFeatureImportance = None

# 一時ディレクトリ設定
safe_temp_dir = Path(tempfile.gettempdir()) / 'joblib_temp'
try:
    safe_temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ['JOBLIB_TEMP_FOLDER'] = str(safe_temp_dir)
except Exception: pass

class PruningCallbackWrapper(pl.Callback):
    """
    Optunaのプルーニングを行うためのカスタムコールバック。
    PyTorch Lightningのバージョン互換性とTensor/Float変換の問題を回避するため、
    trainer.callback_metricsを書き換えずに直接Optuna APIを叩く実装に変更。
    
    【改善】CV時のプルーニング精度向上:
    - step_offsetを導入し、Fold間でステップカウントを連続させる
    - Fold 1→Fold 2→Fold 3を通した一連の流れとして正しくプルーニング判定が行われる
    """
    def __init__(self, trial: optuna.Trial, monitor: str, step_offset: int = 0):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.step_offset = step_offset  # CV時のFold IDに基づくステップオフセット

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        try:
            # 監視対象のメトリクスを取得
            current_score = trainer.callback_metrics.get(self.monitor)
            
            if current_score is None:
                return

            # Tensorならfloatに変換 (GPU上のTensorや勾配付きTensorへの対応)
            if isinstance(current_score, torch.Tensor):
                current_score = current_score.item() if current_score.numel() == 1 else current_score.mean().item()
            
            # 値が有限でない場合はスキップ
            if not np.isfinite(current_score):
                return

            # 現在のエポックを取得し、step_offsetを加算して統一的なステップ番号を算出
            epoch = trainer.current_epoch
            global_step = epoch + self.step_offset

            # Optunaに報告（グローバルなステップ番号を使用）
            self.trial.report(current_score, step=global_step)

            # プルーニング判定
            if self.trial.should_prune():
                message = f"Trial was pruned at global_step {global_step} (epoch {epoch}, fold_offset {self.step_offset})."
                raise optuna.exceptions.TrialPruned(message)
                
        except optuna.exceptions.TrialPruned:
            # プルーニング例外はそのまま投げる（OptunaがキャッチしてTrialを終了させるため）
            raise
        except Exception as e:
            # その他のエラーはログに出して続行（トレーニングを止めない）
            logging.warning(f"[PruningCallback] Error during pruning check: {e}")

# RFE implementation removed in Phase 2: we now prefer CFI when available and
# fall back to a simple deterministic selection (first-N columns). This keeps
# the codebase lightweight and avoids maintaining a heavy cached LGBM-based
# RFE pipeline.

def select_features(config: Any, X_train: pd.DataFrame, y_train: pd.Series, n_features_to_select: int, run_name: str) -> List[str]:
    """
    Select features honoring the removal of the old RFE logic.
    - If CFI is enabled and available, use it.
    - Otherwise, return the first N columns deterministically and emit a DeprecationWarning.
    """
    use_cfi = getattr(config, 'USE_CFI', False) and ClusteredFeatureImportance is not None
    if use_cfi:
        try:
            cfi_cv = PurgedGroupTimeSeriesSplit(
                n_splits=3,
                purge_gap=getattr(config, 'CFI_CV_PURGE_GAP', getattr(config, 'PREDICTION_HORIZON', 1) + 1),
                embargo_gap=getattr(config, 'CFI_CV_EMBARGO_GAP', 5)
            )
            cfi = ClusteredFeatureImportance(linkage_method=getattr(config, 'CFI_LINKAGE_METHOD', 'ward'), max_clusters=getattr(config, 'CFI_MAX_CLUSTERS', 20), cv_splitter=cfi_cv)
            cfi.fit(X_train, y_train)
            import lightgbm as _lgb
            model_for_cfi = _lgb.LGBMRegressor(n_estimators=100, random_state=42)
            importance_df = cfi.compute_importance(model_for_cfi, X_train, y_train)
            selected = cfi.select_features(importance_df, getattr(config, 'CFI_IMPORTANCE_THRESHOLD', 0.0))
            return selected[:n_features_to_select]
        except Exception as e:
            logging.warning(f"CFI selection failed in select_features: {e}. Falling back to deterministic selection.")

    warnings.warn(
        "RFE implementation has been removed. Falling back to deterministic first-N feature selection.",
        DeprecationWarning
    )
    return list(X_train.columns[:n_features_to_select])




class ExpertTrainer:
    def __init__(
        self, 
        model_name: str, 
        model_class: Type[pl.LightningModule], 
        df_regime: pd.DataFrame, 
        config: Any, 
        project_root: Path,
        run_name: str,       
        log_base_dir: Path   
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.df_regime = self._drop_constant_columns(df_regime)
        self.config = config
        self.project_root = project_root
        self.run_name = run_name          
        self.log_base_dir = log_base_dir   
        
        self.seq_len = self._determine_sequence_length()
        
        try:
            self.all_feature_candidates = self._get_feature_columns_from_config()
        except ValueError as e:
            raise ValueError(f"[{self.run_name}] Failed to initialize RFE candidate pool: {e}")

        # インスタンス生成時に一度だけ重要度計算を実行しておく（推奨）
        # これにより、Objectiveループ内での重い処理を回避できる
        self._precompute_feature_importance()

    def _precompute_feature_importance(self):
        """Optunaの試行前に一度だけ特徴量重要度を計算してキャッシュする"""
        target_col_name = f"target_{self.config.PREDICTION_HORIZON}"
        split_rfe = int(len(self.df_regime) * 0.8)
        X_rfe = self.df_regime[self.all_feature_candidates].iloc[:split_rfe]
        y_rfe = self.df_regime[target_col_name].iloc[:split_rfe]
        # Since the LGBM-based RFE pipeline has been removed, attempt to
        # precompute CFI clusters if CFI is enabled; otherwise, this is a noop.
        use_cfi = getattr(self.config, 'USE_CFI', False) and ClusteredFeatureImportance is not None
        if use_cfi:
            try:
                cfi_cv = PurgedGroupTimeSeriesSplit(
                    n_splits=3,
                    purge_gap=getattr(self.config, 'CFI_CV_PURGE_GAP', getattr(self.config, 'PREDICTION_HORIZON', 1) + 1),
                    embargo_gap=getattr(self.config, 'CFI_CV_EMBARGO_GAP', 5)
                )
                cfi = ClusteredFeatureImportance(linkage_method=getattr(self.config, 'CFI_LINKAGE_METHOD', 'ward'), max_clusters=getattr(self.config, 'CFI_MAX_CLUSTERS', 20), cv_splitter=cfi_cv)
                cfi.fit(X_rfe, y_rfe)
                logging.info(f"[CFI] Precomputed clusters for {self.run_name}")
            except Exception as e:
                logging.warning(f"[CFI] Precompute failed for {self.run_name}: {e}")
        else:
            logging.info(f"[RFE] No precompute necessary for {self.run_name} (RFE removed)")

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # NaNを無視してstd計算、NaNのみの列はNaNを返す
        std_vals = df[numeric_cols].std(skipna=True)
        # 分散が0またはNaN（全てNaN）の列を除外
        constant_or_nan_cols = std_vals[(std_vals == 0) | (std_vals.isna())].index.tolist()
        if constant_or_nan_cols:
            return df.drop(columns=constant_or_nan_cols)
        return df

    def _determine_sequence_length(self) -> int:
        if self.model_name == "s5":
            return getattr(self.config, "SEQUENCE_LENGTH_S5", 120)
        return self.config.SEQUENCE_LENGTH

    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = self.seq_len
        num_samples = len(features) - seq_len
        if num_samples <= 0:
            return torch.empty(0, seq_len, features.shape[1], dtype=torch.float32), torch.empty(0, 1, dtype=torch.float32)

        indices = np.arange(num_samples)[:, None] + np.arange(seq_len)[None, :]
        X = features[indices] 
        y = target[seq_len:seq_len+num_samples] 
        
        return torch.from_numpy(X).float(), torch.from_numpy(y).float().view(-1, 1)

    def _get_feature_columns_from_config(self) -> List[str]:
        if not hasattr(self.config, 'FEATURE_SETS') or not self.config.FEATURE_SETS:
            return [col for col in self.df_regime.columns if 'target' not in col and 'regime' not in col]

        patterns_to_match = []
        feature_sets_definitions = getattr(self.config, 'FEATURE_SETS_DEFINITIONS', FEATURE_SETS_DEFINITIONS)
        for set_name in self.config.FEATURE_SETS:
            if set_name in feature_sets_definitions:
                patterns_to_match.extend(feature_sets_definitions[set_name])

        all_df_columns = set(self.df_regime.columns)
        matched_features = set()
        for pattern_str in patterns_to_match:
            try:
                pattern = re.compile(pattern_str)
                matches = {col for col in all_df_columns if pattern.match(col)}
                matched_features.update(matches)
            except re.error:
                pass

        metadata_cols_to_exclude = {'regime', 'Date', 'time_idx', 'group'}
        if hasattr(self.config, 'TARGET_COLUMN'):
            metadata_cols_to_exclude.add(self.config.TARGET_COLUMN)
        target_cols = {col for col in all_df_columns if col.startswith('target_')}
        metadata_cols_to_exclude.update(target_cols)

        cleaned_features = matched_features - metadata_cols_to_exclude

        if hasattr(self.config, 'BLACKLIST_FEATURES'):
            blacklist = set(self.config.BLACKLIST_FEATURES)
            cleaned_features = cleaned_features - blacklist

        if not cleaned_features:
            fallback_features = [
                col for col in self.df_regime.columns
                if col not in metadata_cols_to_exclude and not col.startswith('target_')
            ]
            return sorted(fallback_features)

        return sorted(list(cleaned_features))

    @staticmethod
    def _parse_params(params: Dict[str, Any]) -> Dict[str, Any]:
        parsed = params.copy()
        if 'hidden_dims' in parsed and isinstance(parsed['hidden_dims'], str):
            try:
                obj = ast.literal_eval(parsed['hidden_dims'])
                parsed['hidden_dims'] = tuple(int(x) for x in obj) if isinstance(obj, list) else tuple(obj)
            except (ValueError, SyntaxError): pass

        int_keys = ['n_layer', 'n_layers', 'kernel_size', 'n_embd', 'n_head', 
                    'mlp_internal_dim_multiplier', 'max_epochs', 'n_features_to_select',
                    'attn_dim', 'attn_heads', 'ssm_size', 'd_model']
        for key in int_keys:
            if key in parsed:
                parsed[key] = int(parsed[key])
        return parsed

    def objective(self, trial: optuna.Trial) -> float:
        try:
            params = self.model_class.define_param_space(trial)
            params = self._parse_params(params)
        
        except Exception as e:
            logging.error(f"[{self.run_name}] Error defining params: {e}")
            return -np.inf

        try:
            # 特徴量選択: キャッシュから高速に取得
            target_col_name = f"target_{self.config.PREDICTION_HORIZON}"
            max_features = max(1, min(len(self.all_feature_candidates), 100))
            n_features_to_select = trial.suggest_int("n_features_to_select", 1, max_features)
            
            # 特徴量選択用のデータ分割（学習データの80%を使用）
            # select_features() 内部でCFIまたはdeterministic selectionが実行される
            # ここでは self.df_regime から必要なデータを作成して渡す
            split_rfe = int(len(self.df_regime) * 0.8)
            X_rfe = self.df_regime[self.all_feature_candidates].iloc[:split_rfe]
            y_rfe = self.df_regime[target_col_name].iloc[:split_rfe]

            # Feature selection: use CFI when enabled, else fallback to RFE cache
            use_cfi = getattr(self.config, 'USE_CFI', False) and ClusteredFeatureImportance is not None
            if use_cfi:
                try:
                    cfi_cv = PurgedGroupTimeSeriesSplit(
                        n_splits=3,
                        purge_gap=getattr(self.config, 'CFI_CV_PURGE_GAP', self.config.PREDICTION_HORIZON + 1),
                        embargo_gap=getattr(self.config, 'CFI_CV_EMBARGO_GAP', 5)
                    )
                    cfi = ClusteredFeatureImportance(linkage_method=getattr(self.config, 'CFI_LINKAGE_METHOD', 'ward'), max_clusters=getattr(self.config, 'CFI_MAX_CLUSTERS', 20), cv_splitter=cfi_cv)
                    cfi.fit(X_rfe, y_rfe)
                    import lightgbm as _lgb
                    model_for_cfi = _lgb.LGBMRegressor(n_estimators=100, random_state=42)
                    importance_df = cfi.compute_importance(model_for_cfi, X_rfe, y_rfe)
                    selected_features = cfi.select_features(importance_df, getattr(self.config, 'CFI_IMPORTANCE_THRESHOLD', 0.0))
                except Exception as e:
                    logging.warning(f"CFI selection failed, falling back to deterministic selection: {e}")
                    selected_features = select_features(self.config, X_rfe, y_rfe, n_features_to_select, self.run_name)
            else:
                selected_features = select_features(self.config, X_rfe, y_rfe, n_features_to_select, self.run_name)
            trial.set_user_attr("n_features_selected", len(selected_features))

            # CV: 5-fold PurgedGroupTimeSeriesSplit (purge + embargo)
            n_splits = 5
            tscv = PurgedGroupTimeSeriesSplit(
                n_splits=n_splits,
                purge_gap=getattr(self.config, 'PREDICTION_HORIZON', 1) + 1,
                embargo_gap=getattr(self.config, 'CV_EMBARGO_GAP', 5)
            )
            
            X_all = self.df_regime[selected_features].values
            y_all = self.df_regime[target_col_name].values
            
            fold_scores = []
            epochs = params.pop('max_epochs', self.config.MAX_EPOCHS)

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_all)):
                # skip folds with too-small train/val splits
                if len(train_idx) <= self.seq_len or len(val_idx) <= self.seq_len:
                    continue
                # Sequence length must be smaller than the available samples after the split.
                if len(train_idx) - self.seq_len <= 0 or len(val_idx) - self.seq_len <= 0:
                    continue
                # NaN/Inf をサニタイズしてからScalerに渡す
                X_train_raw = np.nan_to_num(X_all[train_idx], nan=0.0, posinf=0.0, neginf=0.0)
                X_val_raw = np.nan_to_num(X_all[val_idx], nan=0.0, posinf=0.0, neginf=0.0)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_raw)
                X_val_scaled = scaler.transform(X_val_raw)
                
                X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_all[train_idx])
                X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_all[val_idx])
                
                if len(X_train_seq) < 10 or len(X_val_seq) < 10: continue
                
                train_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)
                val_loader = DataLoader(TensorDataset(X_val_seq, y_val_seq), batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)
                
                model = self.model_class(
                    input_channels=len(selected_features),
                    context_window=self.seq_len,
                    target_window=self.config.PREDICTION_HORIZON,
                    loss_type=getattr(self.config, 'LOSS_TYPE', 'mse'),  # ★Configから損失関数タイプを取得
                    **params
                )
                
                # ★追加: LightGBM等のための手動学習トリガー
                if hasattr(model, "fit_lgbm"):
                    model.fit_lgbm(
                        X_train_seq.numpy(), 
                        y_train_seq.numpy().ravel(), 
                        X_val_seq.numpy(), 
                        y_val_seq.numpy().ravel()
                    )
                
                # 【改善】step_offsetを計算: Fold間でステップカウントを連続させる
                # Fold 0: offset = 0 (0～epochs-1)
                # Fold 1: offset = epochs (epochs～2*epochs-1)
                # Fold 2: offset = 2*epochs (2*epochs～3*epochs-1)
                step_offset = fold_idx * epochs
                # If model provides a manual fit interface (e.g., fit_lgbm, sklearn wrapper),
                # prefer to use it and evaluate predictions directly without invoking PL Trainer.
                is_lightning = isinstance(model, pl.LightningModule)

                if hasattr(model, "fit_lgbm") or (not is_lightning and hasattr(model, 'predict')):
                    # assume model.fit_lgbm has already been called above when present
                    try:
                        X_val_np = X_val_seq.numpy()
                        # sklearn-like predict expects 2D input
                        if X_val_np.ndim == 3:
                            nrows = X_val_np.shape[0]
                            X_flat = X_val_np.reshape(nrows, -1)
                        else:
                            X_flat = X_val_np

                        preds = None
                        if hasattr(model, 'predict'):
                            preds = model.predict(X_flat)
                        else:
                            # fallback: try calling fit_lgbm then predict attribute
                            try:
                                preds = model.predict(X_flat)
                            except Exception:
                                preds = np.full(len(X_flat), np.nan)

                        y_true = y_val_seq.numpy().ravel()
                        # compute metrics
                        valid_mask = ~np.isnan(preds)
                        if valid_mask.sum() == 0:
                            score = float(np.inf)
                        else:
                            preds_valid = preds[valid_mask]
                            y_valid = y_true[valid_mask]
                            mse = float(((y_valid - preds_valid) ** 2).mean())
                            # sharpe approximation: mean(pred * actual) / std(pred * actual)
                            prod = preds_valid * y_valid
                            sharpe = float(np.nanmean(prod) / (np.nanstd(prod) + 1e-9))
                            if getattr(self.config, 'OBJECTIVE_METRIC', 'val_loss') == 'sharpe':
                                score = -sharpe
                            elif getattr(self.config, 'TASK_TYPE', 'regression') == 'classification':
                                # compute F1 on sign threshold
                                from sklearn.metrics import f1_score as _f1
                                y_pred_label = (preds_valid > 0).astype(int)
                                y_true_label = (y_valid > 0).astype(int)
                                score = -float(_f1(y_true_label, y_pred_label))
                            else:
                                score = mse

                    except Exception as e:
                        logging.warning(f"[Objective] Sklearn-model eval failed: {e}")
                        score = float(np.inf)

                    fold_scores.append(score)

                else:
                    # Use PyTorch Lightning trainer for LightningModule models
                    trainer = pl.Trainer(
                        max_epochs=epochs,accelerator="gpu",
                        devices=1,logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False,
                        precision=getattr(self.config, 'PRECISION', 32),
                        gradient_clip_val=getattr(self.config, 'GRADIENT_CLIP_VAL', 0.5),
                        callbacks=[PruningCallbackWrapper(trial, "val_loss", step_offset=step_offset)]
                    )

                    trainer.fit(model, train_loader, val_loader)
                    # after training, run model on validation loader to get predictions
                    try:
                        model.eval()
                        preds_list = []
                        y_list = []
                        for xb, yb in val_loader:
                            with torch.no_grad():
                                xb_device = xb.to(next(model.parameters()).device if len(list(model.parameters()))>0 else torch.device('cpu'))
                                out = model(xb_device)
                                if isinstance(out, tuple) or isinstance(out, list):
                                    out = out[0]
                                preds_list.append(out.detach().cpu().numpy().ravel())
                                y_list.append(yb.numpy().ravel())

                        if preds_list:
                            preds = np.concatenate(preds_list)
                            y_true = np.concatenate(y_list)
                            mse = float(((y_true - preds) ** 2).mean())
                            prod = preds * y_true
                            sharpe = float(np.nanmean(prod) / (np.nanstd(prod) + 1e-9))
                            if getattr(self.config, 'OBJECTIVE_METRIC', 'val_loss') == 'sharpe':
                                score = -sharpe
                            elif getattr(self.config, 'TASK_TYPE', 'regression') == 'classification':
                                score = -float(f1_score((y_true>0).astype(int), (preds>0).astype(int)))
                            else:
                                score = mse
                        else:
                            score = float(np.inf)

                    except Exception as e:
                        logging.warning(f"[Objective] Lightning eval failed: {e}")
                        score = float(np.inf)

                    fold_scores.append(score)
            
            if not fold_scores: return -np.inf
            mean_score = np.mean(fold_scores)
            
            # ★追加: トライアル結果をCSVに保存
            save_trial_summary(
                log_dir=self.log_base_dir,
                run_name=self.run_name,
                trial=trial,
                result_metrics={"best_val_acc": mean_score, "status": "SUCCESS"}
            )
            
            return float(mean_score)

        except optuna.exceptions.TrialPruned as e:
            # ★追加: PrunedトライアルもCSVに記録
            save_trial_summary(
                log_dir=self.log_base_dir,
                run_name=self.run_name,
                trial=trial,
                result_metrics={"best_val_acc": float('nan'), "status": "PRUNED"}
            )
            raise e
        except RuntimeError as e:
            # ★CUDA Error Recovery: GPU状態をクリーンアップ
            error_msg = str(e).lower()
            if 'cuda' in error_msg or 'device' in error_msg or 'assert' in error_msg:
                logging.error(f"[{self.run_name}] CUDA Error detected: {e}")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as cleanup_error:
                    logging.warning(f"[{self.run_name}] CUDA cleanup failed: {cleanup_error}")
            logging.error(f"[{self.run_name}] Trial failed: {e}")
            # ★追加: 失敗トライアルもCSVに記録
            save_trial_summary(
                log_dir=self.log_base_dir,
                run_name=self.run_name,
                trial=trial,
                result_metrics={"best_val_acc": float('nan'), "status": "FAILED_CUDA"}
            )
            return float('inf')
        except Exception as e:
            logging.error(f"[{self.run_name}] Trial failed: {e}")
            # ★追加: 失敗トライアルもCSVに記録
            save_trial_summary(
                log_dir=self.log_base_dir,
                run_name=self.run_name,
                trial=trial,
                result_metrics={"best_val_acc": float('nan'), "status": "FAILED"}
            )
            return float('inf')

    def train_final_model(self, best_params: Dict[str, Any], log_dir: Path):
        self._train_full_period(best_params, log_dir, is_production=False)

    def train_production(self, best_params: Dict[str, Any], output_dir: Path, file_prefix: str):
        self._train_full_period(best_params, output_dir, is_production=True, file_prefix=file_prefix)

    def fit_in_memory(self, params: Dict[str, Any], accelerator: str = "auto") -> Optional[Dict[str, Any]]:
        """ファイルに保存せず、self.df_regime 全体で最終モデルを学習して返す。

        再学習つきウォークフォワード（analysis/walk_forward_validator.py、監査 G-9）用。
        特徴量選択・scaler・シーケンス・モデル生成は _train_full_period と同じ関数を通る。
        """
        train_params, selected_features, scaler, X_seq, y_seq = self._prepare_final_fit(params)
        if len(X_seq) < 2:
            return None
        model = self._make_final_model(train_params, selected_features, X_seq, y_seq)
        loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)
        trainer = pl.Trainer(
            max_epochs=train_params.get('max_epochs', self.config.MAX_EPOCHS),
            accelerator=accelerator, devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            precision=getattr(self.config, 'PRECISION', 32),
            gradient_clip_val=getattr(self.config, 'GRADIENT_CLIP_VAL', 0.5),
            accumulate_grad_batches=getattr(self.config, 'ACCUMULATE_GRAD_BATCHES', 4)
        )
        trainer.fit(model, loader)
        model.eval()
        return {"model": model, "scaler": scaler, "features": selected_features, "seq_len": self.seq_len}

    def _prepare_final_fit(self, params: Dict[str, Any]):
        """最終学習用の特徴量選択・scaler・シーケンスを作る（_train_full_period と fit_in_memory で共有）。"""
        train_params = self._parse_params(params)
        target_col = f"target_{self.config.PREDICTION_HORIZON}"
        n_features = train_params.get("n_features_to_select", len(self.all_feature_candidates))
        # objective と同じデータセット（先頭80%）で選ぶ（詳細は _train_full_period のコメント）
        split_rfe = int(len(self.df_regime) * 0.8)
        X_rfe = self.df_regime[self.all_feature_candidates].iloc[:split_rfe]
        y_rfe = self.df_regime[target_col].iloc[:split_rfe]
        selected_features = select_features(self.config, X_rfe, y_rfe, n_features, self.run_name)
        # NaN/Inf をサニタイズしてからScalerに渡す
        X_raw = np.nan_to_num(self.df_regime[selected_features].values, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        X_seq, y_seq = self._create_sequences(X_scaled, self.df_regime[target_col].values)
        return train_params, selected_features, scaler, X_seq, y_seq

    def _make_final_model(self, train_params: Dict[str, Any], selected_features: List[str], X_seq, y_seq):
        model_params = train_params.copy()
        model_params.pop('n_features_to_select', None)
        model_params.pop('max_epochs', None)
        model = self.model_class(
            input_channels=len(selected_features),
            context_window=self.seq_len,
            target_window=self.config.PREDICTION_HORIZON,
            loss_type=getattr(self.config, 'LOSS_TYPE', 'mse'),  # ★Configから損失関数タイプを取得
            **model_params
        )
        # ★追加: LightGBM等のための手動学習トリガー
        if hasattr(model, "fit_lgbm"):
            model.fit_lgbm(X_seq.numpy(), y_seq.numpy().ravel())
        return model

    def _train_full_period(self, params: Dict[str, Any], output_dir: Path, is_production: bool, file_prefix: str = ""):
        output_dir.mkdir(parents=True, exist_ok=True)

        # 最終学習用の特徴量選択は objective と同じデータセット（先頭80%）での選択をそのまま使う。
        # ★注意: 最終モデル作成時に特徴量を変えると、学習時と条件が変わるため、
        # objectiveで選ばれた特徴量をそのまま使うのが鉄則。
        train_params, selected_features, scaler, X_seq, y_seq = self._prepare_final_fit(params)

        prefix = f"{file_prefix}_" if file_prefix else ""
        with open(output_dir / f"{prefix}features.json", 'w') as f:
            json.dump(selected_features, f, indent=4)
        joblib.dump(scaler, output_dir / f"{prefix}scaler.joblib")

        if len(X_seq) < 2:
            logging.warning(f"[{self.run_name}] Not enough sequence samples for training. Skipping final model training.")
            return
        loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)
        model = self._make_final_model(train_params, selected_features, X_seq, y_seq)
        
        # 【改善】StochasticWeightAveraging (SWA) を除外
        # 理由: weight_norm が適用されたレイヤーの deepcopy でエラー発生
        # 具体的には TCN の causal_conv1d や PyTorchバージョン間の互換性問題で失敗
        # 安全のため本番学習では SWA を使用しない
        callbacks = [
            ModelCheckpoint(monitor="train_loss", save_top_k=1, mode="min", dirpath=output_dir, filename=f"{prefix}best")
        ]
        
        trainer = pl.Trainer(
            max_epochs=train_params.get('max_epochs', self.config.MAX_EPOCHS),
            accelerator="gpu",devices="1", logger=False, 
            callbacks=callbacks, enable_progress_bar=True,
            enable_model_summary=True,
            precision=getattr(self.config, 'PRECISION', 32),
            gradient_clip_val=getattr(self.config, 'GRADIENT_CLIP_VAL', 0.5),
            accumulate_grad_batches=getattr(self.config, 'ACCUMULATE_GRAD_BATCHES', 4)
        )
        
        trainer.fit(model, loader)
        trainer.save_checkpoint(output_dir / f"{prefix}swa.ckpt")
        logging.info(f"Saved final models to {output_dir}")

        # Phase 3: MLflow Logging
        if mlflow.active_run():
            mlflow.end_run()
            
        try:
            experiment_name = f"Semi_Daily_{self.run_name}"
            try:
                if not mlflow.get_experiment_by_name(experiment_name):
                    mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            except Exception:
                # If set_experiment fails (e.g. concurrent creation), just use default
                pass

            with mlflow.start_run(run_name=f"{prefix}train_{'prod' if is_production else 'final'}"):
                # Log Params
                mlflow.log_params(train_params)
                mlflow.log_param("n_features", len(selected_features))
                mlflow.log_param("is_production", is_production)
                mlflow.log_param("model_type", self.model_name)
                
                # Log Artifacts
                mlflow.log_artifact(str(output_dir / f"{prefix}features.json"))
                mlflow.log_artifact(str(output_dir / f"{prefix}scaler.joblib"))
                
                # Log Metrics (Final Loss)
                final_loss = trainer.callback_metrics.get("train_loss")
                if final_loss:
                    mlflow.log_metric("final_train_loss", float(final_loss))
                    
        except Exception as e:
            logging.warning(f"[MLflow] Logging failed: {e}")

def save_trial_summary(log_dir: Path, run_name: str, trial: optuna.trial.Trial, result_metrics: Dict[str, Any]):
    try:
        summary_csv_path = log_dir / "all_trials_summary.csv"
        summary_data = {
            'Run_Name': run_name,
            'Trial_Number': trial.number,
            'Val_Acc_Best': result_metrics.get("best_val_acc", 0.0),
            'Status': result_metrics.get("status", "UNKNOWN"),
            **trial.params
        }
        pd.DataFrame([summary_data]).to_csv(summary_csv_path, mode='a', header=not summary_csv_path.exists(), index=False)
    except: pass


# =============================================================================
# Training Orchestrator - 学習パイプライン全体の管理
# =============================================================================

# RegimeDetector のインポート (循環参照を避けるためここで行う)
from common_utils.regime_detector import RegimeDetector

# Optuna可視化のインポート
try:
    import matplotlib
    matplotlib.use('Agg')
    from optuna.visualization import plot_optimization_history
except ImportError:
    plot_optimization_history = None


class TrainingOrchestrator:
    """
    Encapsulates the training workflow (Regime Optimization & Production Training)
    to enforce DRY principles across different timescale modules.
    
    学習パイプライン全体を管理するオーケストレーター。
    - レジームごとのOptuna最適化
    - 本番モデルの学習
    """
    def __init__(
        self,
        config: Any,
        project_root: Path,
        model_map: Dict[str, Type],
        run_name_prefix: str,
        output_dir_name: str
    ):
        self.config = config
        self.project_root = project_root
        self.model_map = model_map
        self.run_name_prefix = run_name_prefix
        self.output_dir_name = output_dir_name
        
        # Ensure log directories exist
        self.optuna_db_dir = self.project_root / "logs" / "optuna_db"
        self.optuna_db_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.project_root / "logs" / self.output_dir_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def run_regime_optimization(self, regime: str, df_regime: pd.DataFrame, new_study: bool, models_to_run: List[str]):
        """Runs Optuna optimization for a specific regime."""
        worker_info = f"MainProcess {os.getpid()}" 
        logging.info(f"{worker_info} (Regime: {regime}) started.")
        
        expert_model_names = self.config.REGIME_EXPERTS.get(regime, [])
        
        for model_name in expert_model_names:
            if model_name not in models_to_run: continue
                
            model_class = self.model_map.get(model_name)
            if model_class is None: continue
    
            logging.info(f"--- Starting training for expert '{model_name}' (Regime: {regime}) ---")
            run_name = f"{self.run_name_prefix}_{regime}_{model_name}" 
            study_name = f"{run_name}_optimization"
            db_path = self.optuna_db_dir / f"{study_name}.db"
            
            if new_study and db_path.exists():
                try: db_path.unlink()
                except OSError: pass
                
            study = optuna.create_study(
                study_name=study_name, 
                storage=f"sqlite:///{db_path}", 
                load_if_exists=not new_study, 
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), 
                direction="minimize"
            )
    
            log_dir_name = f"{run_name}_final_training" 
            version_base_dir = self.project_root / "logs" / log_dir_name
            version_base_dir.mkdir(parents=True, exist_ok=True) 
            
            existing_versions = [int(d.name.split('_')[-1]) for d in version_base_dir.iterdir() if d.is_dir() and d.name.startswith('version_')]
            next_version = max(existing_versions, default=-1) + 1
            log_dir = version_base_dir / f"version_{next_version}"
            log_dir.mkdir(exist_ok=True)
            
            trainer_instance = ExpertTrainer(
                model_name=model_name, 
                model_class=model_class, 
                df_regime=df_regime, 
                config=self.config, 
                project_root=self.project_root,
                run_name=run_name,
                log_base_dir=log_dir
            )
            
            try:
                study.optimize(trainer_instance.objective, n_trials=self.config.N_TRIALS, n_jobs=1)
            except Exception as e: 
                logging.error(f"[{regime} / {model_name}] Optuna optimization failed: {e}")
                continue
                
            try:
                best_trial = study.best_trial
                logging.info(f"Best trial: {best_trial.value:.4f}")
                trainer_instance.train_final_model(best_trial.params, log_dir)
            except Exception as e: logging.error(f"[{regime} / {model_name}] Failed saving: {e}")

    def _load_best_params_from_optuna(self, regime: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Optuna DBから最適パラメータを自動読み込み。
        BEST_PARAMSが設定されていない場合のフォールバック。
        """
        run_name = f"{self.run_name_prefix}_{regime}_{model_name}"
        study_name = f"{run_name}_optimization"
        db_path = self.optuna_db_dir / f"{study_name}.db"
        
        if not db_path.exists():
            logging.warning(f"[AutoLoad] Optuna DB not found: {db_path}")
            return None
        
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=f"sqlite:///{db_path}"
            )
            
            if len(study.trials) == 0:
                logging.warning(f"[AutoLoad] No trials found in study: {study_name}")
                return None
            
            best_trial = study.best_trial
            logging.info(f"[AutoLoad] Loaded best params from Optuna DB for {regime}/{model_name}")
            logging.info(f"[AutoLoad] Best value: {best_trial.value:.6f}, Trial #{best_trial.number}")
            
            return best_trial.params
            
        except Exception as e:
            logging.error(f"[AutoLoad] Failed to load from Optuna DB: {e}")
            return None

    def run_production_training(self, regime: str, df_regime: pd.DataFrame, models_to_run: List[str]):
        """Runs production training using BEST_PARAMS or auto-loaded from Optuna DB."""
        expert_model_names = self.config.REGIME_EXPERTS.get(regime, [])
        
        for model_name in expert_model_names:
            if model_name not in models_to_run: continue
            model_class = self.model_map.get(model_name)
            if model_class is None: continue
    
            # まずBEST_PARAMSから取得を試みる
            params = None
            if hasattr(self.config, 'BEST_PARAMS'):
                params = self.config.BEST_PARAMS.get(regime, {}).get(model_name)
            
            # BEST_PARAMSがない場合、Optuna DBから自動読み込み
            if not params:
                logging.info(f"[AutoLoad] No BEST_PARAMS for {regime}/{model_name}, trying Optuna DB...")
                params = self._load_best_params_from_optuna(regime, model_name)
            
            if not params:
                logging.warning(f"No parameters available for {regime}/{model_name} (neither BEST_PARAMS nor Optuna DB)")
                continue
    
            logging.info(f"--- [Production] Training: {model_name} ({regime}) ---")
            
            # Specific hack for S5 if needed
            if model_name == "s5" and "initialization" not in params:
                 params["initialization"] = "hippo"
    
            run_name = f"{self.run_name_prefix}_{regime}_{model_name}"
            trainer_instance = ExpertTrainer(
                model_name=model_name, 
                model_class=model_class, 
                df_regime=df_regime, 
                config=self.config, 
                project_root=self.project_root,
                run_name=run_name,
                log_base_dir=self.models_dir
            )
    
            file_prefix = f"{model_name}_{regime}"
            try:
                trainer_instance.train_production(
                    best_params=params,
                    output_dir=self.models_dir,
                    file_prefix=file_prefix)
                logging.info(f"SUCCESS: {file_prefix}")
            except Exception as e:
                logging.error(f"Failed production training for {file_prefix}: {e}")
                import traceback; traceback.print_exc()

    def run(self, new_study: bool, production: bool, n_jobs: int, models_to_run_str: Optional[str]):
        """Main execution entry point."""
        logging.info(f"--- Starting Training Pipeline ({self.run_name_prefix}) ---")
        
        if models_to_run_str: 
            models_to_run = [m.strip().lower() for m in models_to_run_str.split(',') if m.strip()]
        else: 
            models_to_run = list(self.model_map.keys())

        # Load Data
        try:
            data_path = self.project_root / "data" / self.config.OUTPUT_FILENAME
            df = pd.read_parquet(data_path)
            start_date = getattr(self.config, 'DATA_START_DATE', '2020-01-01')
            df = df[df['Date'] >= start_date].copy()
        except Exception as e: 
            logging.error(f"Failed to load data from {data_path}: {e}")
            return

        # Regime Detection
        try:
            detector = RegimeDetector()
            vix_col = getattr(self.config, 'REGIME_VIX_COLUMN', 'vix_close')
            vix_th = getattr(self.config, 'REGIME_VIX_THRESHOLD', 20.0)
            df = detector.detect_simple_vix(df, vix_col, vix_th)
        except Exception as e:
             logging.error(f"Regime detection failed: {e}")
             return

        actual_regimes = list(df['regime'].unique())
        logging.info(f"Target Regimes: {actual_regimes}")

        for r in actual_regimes:
            if hasattr(self.config, 'REGIMES') and r not in self.config.REGIMES: continue
            
            if production:
                self.run_production_training(r, df[df['regime']==r].copy(), models_to_run)
            else:
                self.run_regime_optimization(r, df[df['regime']==r].copy(), new_study, models_to_run)
