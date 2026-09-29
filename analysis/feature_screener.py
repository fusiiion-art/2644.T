import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.style as style
import warnings
import importlib
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# --- プロジェクトのパス設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from common_utils.metrics import directional_accuracy
import torch

# Matplotlibのスタイルと警告設定
style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def screen_features(timescale: str):
    """
    特徴量ファイルに含まれる特徴量を一つずつテストし、ターゲットに対する
    単独での予測性能を評価（スクリーニング）する。
    """
    print(f"\n--- '{timescale}'の新特徴量スクリーニングを開始 ---")

    # --- 1. 設定とデータの読み込み ---
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    feature_file = PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME
    if not feature_file.exists():
        print(f"\n[エラー] 特徴量ファイルが見つかりません: {feature_file}")
        sys.exit(1)
        
    df = pd.read_parquet(feature_file)
    df.columns = [c.replace('.', '_') for c in df.columns]
    df.dropna(subset=[cfg.TARGET_COLUMN], inplace=True)
    print(f" -> 特徴量ファイルを読み込みました (特徴量数: {len(df.columns)})")

    # --- 2. 既存の特徴量と候補特徴量の特定 ---
    # ★修正: EXPERTS_CONFIGの形式がList[Dict]に変更されているため対応
    existing_features = set()
    experts_cfg = getattr(cfg, 'EXPERTS_CONFIG', {})
    if isinstance(experts_cfg, dict):
        for model_cfg in experts_cfg.values():
            if isinstance(model_cfg, dict):
                existing_features.update([f.replace('.', '_') for f in model_cfg.get("feature_columns", [])])
    
    # メタデータ列を除外
    metadata_cols = {'Date', 'regime', 'time_idx', 'group', cfg.TARGET_COLUMN}
    target_cols = {col for col in df.columns if col.startswith('target_')}
    metadata_cols.update(target_cols)
    
    candidate_features = [col for col in df.columns 
                          if col not in existing_features 
                          and col not in metadata_cols
                          and not col.startswith('target_')]
    
    if not candidate_features:
        print("\n[情報] 評価対象となる新しい特徴量候補が見つかりませんでした。")
        return

    print(f" -> {len(candidate_features)}個の新しい特徴量候補を評価します。")

    # --- 3. 各候補特徴量の性能評価 ---
    results = []
    y = df[cfg.TARGET_COLUMN]

    for feature in candidate_features:
        X = df[[feature]].copy()
        
        # 欠損値を平均で補完 (シンプルな方法)
        X.fillna(X.mean(), inplace=True)
        if X.isnull().values.any(): # それでもNaNが残る場合
            continue

        # データを訓練用とテスト用に分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # シンプルなLightGBMモデルで学習
        model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # テストデータで予測
        predictions = model.predict(X_test)
        
        # 方向性精度で性能を評価（numpy配列で計算）
        try:
            y_test_arr = np.asarray(y_test).flatten()
            pred_arr = np.asarray(predictions).flatten()
            accuracy = directional_accuracy(pred_arr, y_test_arr)
        except Exception:
            continue
        
        results.append({"feature": feature, "directional_accuracy": accuracy})

    if not results:
        print("\n[警告] どの特徴量も評価できませんでした。")
        return

    # --- 4. 結果の集計と報告 ---
    summary_df = pd.DataFrame(results).sort_values(by="directional_accuracy", ascending=False)
    
    print("\n--- 新特徴量 予測性能ランキング ---")
    print(summary_df.head(20).to_string(formatters={'directional_accuracy': '{:,.2%}'.format}))
    
    report_dir = PROJECT_ROOT / "reports" / timescale / "screening"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = report_dir / "feature_screening_report.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n -> 詳細レポートをCSVとして保存しました: {csv_path}")

    # --- 5. 上位特徴量の可視化 ---
    plt.figure(figsize=(12, 10))
    plot_df = summary_df.head(20).sort_values(by="directional_accuracy", ascending=True)
    plt.barh(plot_df['feature'], plot_df['directional_accuracy'])
    plt.title(f'Top 20 Candidate Features by Directional Accuracy ({timescale})')
    plt.xlabel('Directional Accuracy')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Random Guess (50%)')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    plt.tight_layout()
    
    img_path = report_dir / "feature_screening_chart.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> 上位特徴量のチャートを保存しました: {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新しい特徴量候補の予測性能を高速に評価（スクリーニング）します。")
    parser.add_argument("--timescale", required=True, help="評価対象のタイムスケール (例: 'daily_1')")
    args = parser.parse_args()
    
    screen_features(args.timescale)
