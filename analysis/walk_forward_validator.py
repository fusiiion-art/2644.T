"""
ウォークフォワード検証スクリプト (リファクタリング版)
- predicter_model.py と同じデータフロー（ExpertPredicter/AssetLoader）を使用
- これにより訓練時と検証時の特徴量処理が一致
"""
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import warnings
import sys
import argparse
import importlib

# --- プロジェクトルート設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common_utils.regime_detector import RegimeDetector
from common_utils.asset_loader import AssetLoader
from common_utils.predicter_model import ExpertPredicter

warnings.filterwarnings("ignore")

def run_walk_forward(timescale: str):
    """ウォークフォワード検証を実行（predicterと同じデータフローを使用）"""
    print(f"\n{'='*20} ウォークフォワード検証 ({timescale}) を開始 {'='*20}")

    # Config読み込み
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
        print(f" -> 設定ファイル '{config_module_path}' を読み込みました。")
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    # AssetLoader初期化
    loader = AssetLoader(PROJECT_ROOT, timescale)
    
    # データ読み込み
    print("\n--- データの準備 ---")
    data_path = PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME
    if not data_path.exists():
        print(f"[エラー] データファイルが見つかりません: {data_path}")
        return
    
    df_all = pd.read_parquet(data_path)
    df_all.columns = [col.replace('.', '_') for col in df_all.columns]
    if 'Date' in df_all.columns:
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.set_index('Date')
    df_all = df_all.sort_index()
    print(f" -> データ形状: {df_all.shape}")

    # レジーム判定
    detector = RegimeDetector()
    df_all = detector.detect_simple_vix(df_all.copy(), cfg.REGIME_VIX_COLUMN, cfg.REGIME_VIX_THRESHOLD)

    # 結果格納用DataFrame
    target_col = f"target_{cfg.PREDICTION_HORIZON}"
    if target_col not in df_all.columns:
        print(f"[エラー] ターゲット列 '{target_col}' がありません。")
        return
    
    oof_predictions = pd.DataFrame(index=df_all.index)
    oof_predictions['regime'] = df_all['regime']
    oof_predictions['actual'] = df_all[target_col]

    # モデルのロードとキャッシュ
    models_cache = {}
    for regime in cfg.REGIMES:
        models_cache[regime] = {}
        experts = cfg.REGIME_EXPERTS.get(regime, [])
        for model_name in experts:
            try:
                assets = loader.load_production_model(model_name, regime, use_swa=True)
                predicter = ExpertPredicter(model_name, assets, cfg)
                models_cache[regime][model_name] = predicter
                print(f" -> ロード: {model_name}/{regime}")
            except Exception as e:
                print(f" ⚠️ {model_name}/{regime}: {e}")

    # ウォークフォワード予測
    # 直近N日に対して、その時点で利用可能なデータで予測
    print("\n--- 予測実行 ---")
    seq_len = cfg.SEQUENCE_LENGTH
    
    # 予測対象期間（データセットの後半50%）
    n_samples = len(df_all)
    start_idx = max(seq_len + 20, n_samples // 2)  # 少なくとも半分のデータで訓練されていると仮定
    
    for i in range(start_idx, n_samples):
        if i % 50 == 0:
            print(f"  Progress: {i}/{n_samples}")
        
        # その時点までのデータスライス
        current_slice = df_all.iloc[:i+1]
        current_regime = current_slice['regime'].iloc[-1]
        current_date = df_all.index[i]
        
        regime_models = models_cache.get(current_regime, {})
        
        for model_name, predicter in regime_models.items():
            try:
                mu, sigma = predicter.predict(current_slice)
                
                col_name = f"pred_{model_name}"
                if col_name not in oof_predictions.columns:
                    oof_predictions[col_name] = np.nan
                oof_predictions.loc[current_date, col_name] = mu
                
            except Exception as e:
                # 通常は静かに失敗（シーケンス長不足など）
                continue

    # 結果保存
    print("\n--- 結果保存 ---")
    base_output_dir = PROJECT_ROOT / "logs" / f"{timescale}_walkforward"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # バージョン管理
    existing_versions = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
    if existing_versions:
        version_nums = [int(d.name.split("_")[1]) for d in existing_versions if d.name.split("_")[-1].isdigit()]
        next_version = max(version_nums, default=0) + 1
    else:
        next_version = 0
    
    output_dir = base_output_dir / f"version_{next_version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_cols = [c for c in oof_predictions.columns if c.startswith('pred_')]
    if pred_cols:
        # 有効なサンプルのみ保存
        result_df = oof_predictions.dropna(subset=['actual'], how='any')
        result_df = result_df[result_df[pred_cols].notna().any(axis=1)]
        
        csv_path = output_dir / f"wf_results_{timescale}.csv"
        result_df.to_csv(csv_path)
        print(f"保存先: {csv_path}")
        print(f"有効サンプル数: {len(result_df)}")
        
        # 最新版へのコピー
        latest_csv = base_output_dir / f"wf_results_{timescale}.csv"
        result_df.to_csv(latest_csv)
        
        # 精度レポート
        print("\n--- 精度レポート ---")
        for col in pred_cols:
            valid = result_df[[col, 'actual']].dropna()
            if len(valid) > 10:
                corr = valid[col].corr(valid['actual'])
                direction_match = ((valid[col] > 0) == (valid['actual'] > 0)).mean()
                print(f"  {col}: Corr={corr:.4f}, Direction={direction_match:.2%}")
    else:
        print("有効な予測がありませんでした。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timescale", default="daily_2644t", help="Target timescale module")
    args = parser.parse_args()
    run_walk_forward(args.timescale)
