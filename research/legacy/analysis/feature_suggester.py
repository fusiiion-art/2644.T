import sys
import os
import torch
import pandas as pd
import numpy as np
import shap
import argparse
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
import japanize_matplotlib
import warnings
import importlib
import re
import joblib
import json
from sklearn.preprocessing import StandardScaler

# --- プロジェクトのパス設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- 必要なモジュールを動的にインポート ---
from models.timesnet_model import TimesNetLightning
from models.itransformer_model import iTransformerLightning
from models.timemixer_model import TimeMixerLightning
from models.s5_model import S5Lightning
from models.patchtst_model import PatchTSTLightning

# featureregistry から定義をインポート
try:
    from generate.featureregistry import FEATURE_SETS_DEFINITIONS
except ImportError:
    print("[ERROR] Failed to import FEATURE_SETS_DEFINITIONS from generate.featureregistry.")
    FEATURE_SETS_DEFINITIONS = {}

# RegimeDetector のインポート
try:
    from common_utils.regime_detector import RegimeDetector
except ImportError:
    print("[ERROR] common_utils.regime_detector が見つかりません。")
    sys.exit(1)

# --- モデル名とクラスの対応表 ---
MODEL_CLASS_MAP = {
    "timesnet": TimesNetLightning,
    "itransformer": iTransformerLightning,
    "timemixer": TimeMixerLightning,
    "s5": S5Lightning,
    "patchtst": PatchTSTLightning,
}

# --- SHAP分析用のラッパークラス ---
class ModelWrapperForSHAP(torch.nn.Module):
    """シーケンスモデル (B, T, C) を受け取り、(B, 1) を返すラッパー"""
    def __init__(self, model_to_wrap):
        super().__init__()
        self.model = model_to_wrap
    
    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        # モデルが (B,) を返す場合 (TCN, BDH, S5)
        output = self.model(x_tensor)
        if output.ndim == 1:
            return output.unsqueeze(-1) # (B, 1) に変換
        return output

# --- ユーティリティ関数 ---

def _create_sequences(features: np.ndarray, seq_len: int) -> torch.Tensor:
    """シーケンスデータセット (Xのみ) を作成する (SHAPの背景データ用)"""
    X = []
    if len(features) <= seq_len:
        # print(f"[WARNING] Data length ({len(features)}) is shorter than SEQUENCE_LENGTH ({seq_len}).")
        return torch.empty(0, seq_len, features.shape[1], dtype=torch.float32)

    for i in range(len(features) - seq_len + 1): # +1 して、最後のシーケンスも含める
        X.append(features[i:i + seq_len])
        
    return torch.tensor(np.array(X), dtype=torch.float32)

# --- メイン関数 ---

def suggest_features(timescale: str, model_name: str, regime: str, top_n=10, shap_samples=100):
    print(f"\n--- {timescale} / {model_name.upper()} / {regime.upper()} のSHAP分析を開始 ---")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f" -> 使用デバイス: {device_str.upper()}")
    
    # 1. config 読み込み
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    # 2. モデルクラスの確認
    if model_name not in MODEL_CLASS_MAP:
        raise ValueError(f"モデル名 '{model_name}' が MODEL_CLASS_MAP に定義されていません。")
    model_class = MODEL_CLASS_MAP[model_name]

    # 3. 学習済みモデル (.ckpt) とスケーラー (.joblib) をロード
    log_dir_name = f"{timescale}_{regime}_{model_name}_final_training"
    base_log_dir = PROJECT_ROOT / "logs" / log_dir_name
    
    # フォルダ存在チェック (フォールバック対応)
    if not base_log_dir.exists():
        fallback_dir_name = f"daily_1_{regime}_{model_name}_final_training"
        fallback_log_dir = PROJECT_ROOT / "logs" / fallback_dir_name
        if fallback_log_dir.exists():
            print(f" -> [Info] '{log_dir_name}' が見つからないため、'{fallback_dir_name}' を使用します。")
            base_log_dir = fallback_log_dir
        else:
            raise FileNotFoundError(f"学習済みモデルのログディレクトリが見つかりません: {base_log_dir}")
    
    versions = sorted(
        [d for d in base_log_dir.glob("version_*") if d.is_dir()], 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    if not versions:
        raise FileNotFoundError(f"バージョンフォルダが見つかりません: {base_log_dir}")
    
    log_dir = versions[0]
    ckpt_path = log_dir / "best_model.ckpt"
    scaler_path = log_dir / "feature_scaler.joblib"
    features_path = log_dir / "features.json"

    if not all([ckpt_path.exists(), scaler_path.exists(), features_path.exists()]):
        # SWAモデルのチェック (best_modelがない場合)
        swa_path = log_dir / "swa_model.ckpt"
        if swa_path.exists():
            print(f" -> best_model.ckpt の代わりに swa_model.ckpt を使用します。")
            ckpt_path = swa_path
        else:
            raise FileNotFoundError(f"モデルファイル、スケーラー、特徴量リストのいずれかが見つかりません: {log_dir}")

    print(f"--- 分析に使用するアセット (in {log_dir.name}) ---")
    print(f"  Model: {ckpt_path.name}")
    print(f"  Scaler: {scaler_path.name}")
    print(f"  Features: {features_path.name}")

    # モデルをロード
    model = model_class.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.to(device) 
    model.eval()
    
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        available_features = json.load(f)

    # 4. データ準備
    print("分析用のデータを準備中...")
    data_path = PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME
    df_features = pd.read_parquet(data_path)
    
    detector = RegimeDetector()
    df_features = detector.detect_simple_vix(df_features, cfg.REGIME_VIX_COLUMN, cfg.REGIME_VIX_THRESHOLD)
    df_regime = df_features[df_features['regime'] == regime].copy()
    
    # S5モデル用のシーケンス長対応
    if model_name == "s5":
        current_seq_len = getattr(cfg, "SEQUENCE_LENGTH_S5", 120)
    else:
        current_seq_len = cfg.SEQUENCE_LENGTH

    # 必要なデータ数をチェック
    min_required = current_seq_len + shap_samples + 10
    if len(df_regime) < min_required:
         raise ValueError(f"レジーム '{regime}' のデータが不足しています（必要: {min_required}, 実際: {len(df_regime)}）。")

    print(f" -> レジーム '{regime}' のデータ {len(df_regime)}件 を使用します (SeqLen: {current_seq_len})。")
    
    # 欠損値処理
    # 特徴量リストにある列だけを抽出して埋める
    # (available_featuresに含まれる列がdfにない場合はエラーになるので注意)
    df_subset = df_regime[available_features].copy()
    df_subset = df_subset.ffill().bfill().fillna(0)
    
    scaled_features = scaler.transform(df_subset)
    all_sequences = _create_sequences(scaled_features, current_seq_len)
    
    # データ数が多すぎる場合はサンプリング
    n_sequences = len(all_sequences)
    
    # 背景データ
    bg_size = min(100, n_sequences)
    bg_indices = np.random.choice(n_sequences, bg_size, replace=False)
    background_data = all_sequences[bg_indices].to(device)
    
    # 分析対象データ
    test_size = min(shap_samples, n_sequences)
    test_indices = np.random.choice(n_sequences, test_size, replace=False)
    test_data = all_sequences[test_indices].to(device)

    print(f" -> 背景データ: {background_data.shape}, 分析対象データ: {test_data.shape}")

    # 5. SHAP値の計算
    print("SHAP値の計算中... (時間がかかる場合があります)")
    wrapped_model = ModelWrapperForSHAP(model)
    
    try:
        explainer = shap.GradientExplainer(wrapped_model, background_data)
        shap_values_raw = explainer.shap_values(test_data)
    except Exception as e:
        print(f"[エラー] SHAP計算中にエラー: {e}")
        return []
    
    # 6. 特徴量の重要度を集計
    shap_values = shap_values_raw[0] if isinstance(shap_values_raw, (list, tuple)) else shap_values_raw
    
    expected_shape = test_data.shape 
    
    if shap_values.ndim == len(expected_shape) + 1:
        shap_values = shap_values.squeeze(-1)
    
    # 平均絶対SHAP値を計算
    shap_values_tensor = torch.from_numpy(shap_values).to(device)
    mean_abs_shap_tensor = torch.abs(shap_values_tensor).mean(dim=(0, 1)) # 時間方向とサンプル方向で平均
    mean_abs_shap = mean_abs_shap_tensor.cpu().numpy()

    if len(available_features) != len(mean_abs_shap):
        print(f"[警告] 特徴量数({len(available_features)})とSHAP次元({len(mean_abs_shap)})が不一致です。")
        return []

    importance_df = pd.DataFrame({'feature': available_features, 'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)

    # 7. 結果の可視化と保存
    print("\n分析結果をプロットしています...")
    output_dir = PROJECT_ROOT / "reports" / timescale / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"importance_plot_{timescale}_{regime}_{model_name}.png"
    
    plt.figure(figsize=(10, 8))
    plot_df = importance_df.head(20).sort_values(by='importance', ascending=True)
    plt.barh(plot_df['feature'], plot_df['importance'])
    plt.title(f"Feature Importance: {timescale} / {model_name.upper()} / {regime.upper()} (Top 20)")
    plt.xlabel("mean(|SHAP value|)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"特徴量重要度チャートを保存しました: {output_path}")

    # JSON保存
    json_output_path = output_dir / f"importance_full_list_{timescale}_{regime}_{model_name}.json"
    importance_df.to_json(json_output_path, orient='records', indent=2, force_ascii=False)
    print(f"全特徴量の重要度リストを保存しました: {json_output_path}")

    # 8. コンソール出力
    top_features = importance_df.head(top_n)['feature'].tolist()
    print(f"\n--- トップ{top_n}の重要特徴量 ({model_name.upper()}) ---")
    for i, feature in enumerate(top_features):
        print(f"{i+1}. {feature}  (Score: {importance_df.iloc[i]['importance']:.4f})")

    return top_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP分析に基づき、特徴量の重要度を算出します。")
    parser.add_argument("--timescale", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["timesnet", "itransformer", "timemixer", "s5", "patchtst"])
    parser.add_argument("--regime", type=str, required=True, choices=["risk_on", "risk_off"])
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--shap_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    suggest_features(args.timescale, args.model, args.regime, args.top_n, args.shap_samples)