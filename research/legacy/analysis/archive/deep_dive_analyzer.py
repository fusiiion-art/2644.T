import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import warnings
import importlib

# --- プロジェクトのパス設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlibのスタイルと警告設定
style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def analyze_worst_predictions(df: pd.DataFrame, cfg, output_dir: Path, n_cases=5):
    """
    モデルが最も大きく予測を外した事例を特定し、その前後の状況を可視化する。
    """
    print(f"\n--- ワースト予測 トップ{n_cases}事例の分析を開始 ---")
    
    expert_cols = [c for c in df.columns if c.startswith('pred_')]
    if not expert_cols:
        print(" -> 予測カラムが見つからないため、スキップします。")
        return

    for pred_col in expert_cols:
        model_name = pred_col.replace('pred_', '')
        # 誤差の絶対値を計算
        df['error'] = (df[pred_col] - df['actual']).abs()
        worst_cases = df.nlargest(n_cases, 'error')

        if worst_cases.empty:
            continue

        print(f"\n -> モデル '{model_name.upper()}' のワーストケース:")
        
        fig, axes = plt.subplots(n_cases, 1, figsize=(15, 4 * n_cases), sharex=True)
        if n_cases == 1: axes = [axes]
        
        fig.suptitle(f'Worst Prediction Cases for {model_name.upper()}', fontsize=16)

        for i, (idx, case) in enumerate(worst_cases.iterrows()):
            # 失敗した日の前後60日間のデータをプロット
            start_idx = df.index.get_loc(idx) - 60
            end_idx = df.index.get_loc(idx) + 20 # 未来も少し見せる
            
            subset = df.iloc[max(0, start_idx):min(len(df), end_idx)]
            
            ax = axes[i]
            # 実測値（ターゲット）の推移
            ax.plot(subset.index, subset['actual'].cumsum(), label='Actual Return (Cumulative)', color='grey')
            
            # 予測したポイントを強調
            ax.axvline(idx, color='red', linestyle='--', label=f"Error Date: {idx.date()}")
            
            # 詳細情報をタイトルに
            ax.set_title(f"Case {i+1}: {idx.date()} | Regime: {case.get('regime', 'Unknown')} | Pred: {case[pred_col]:.4f} vs Actual: {case['actual']:.4f}")
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        output_path = output_dir / f"worst_cases_{model_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   -> ワーストケースのチャートを保存しました: {output_path}")

def analyze_prediction_correlation(df: pd.DataFrame, output_dir: Path):
    """モデル間の予測値の相関を分析し、ヒートマップで可視化する"""
    print("\n--- モデル間 予測相関分析を開始 ---")
    
    expert_cols = [c for c in df.columns if c.startswith('pred_')]
    target_col = 'actual'
    
    if len(expert_cols) < 2:
        print(" -> 比較対象のモデルが2つ未満のため、スキップします。")
        return
        
    # 実測値も含めて相関を見る
    cols_to_analyze = expert_cols + [target_col]
    correlation_matrix = df[cols_to_analyze].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", vmin=-1, vmax=1)
    plt.title('Prediction Correlation Matrix (Model vs Actual)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = output_dir / "prediction_correlation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f" -> 予測値の相関ヒートマップを保存しました: {output_path}")

def analyze_prediction_distribution(df: pd.DataFrame, output_dir: Path):
    """各モデルの予測値の分布を可視化する"""
    print("\n--- 予測値 分布分析を開始 ---")
    
    expert_cols = [c for c in df.columns if c.startswith('pred_')]
    if not expert_cols:
        print(" -> 予測カラムが見つからないため、スキップします。")
        return
        
    n_models = len(expert_cols)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models), sharex=True)
    if n_models == 1: axes = [axes]
    
    fig.suptitle('Prediction Distribution Analysis', fontsize=16)
    
    for i, pred_col in enumerate(expert_cols):
        model_name = pred_col.replace('pred_', '')
        ax = axes[i]
        
        # ヒストグラム
        sns.histplot(df[pred_col], ax=ax, kde=True, bins=50, color='blue', alpha=0.3, label='Prediction')
        # 実測値の分布も重ねて比較（スケールが合う場合）
        sns.histplot(df['actual'], ax=ax, kde=True, bins=50, color='grey', alpha=0.3, label='Actual')
        
        ax.set_title(f"Distribution for {model_name.upper()}")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / "prediction_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> 予測値の分布グラフを保存しました: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="モデルの挙動を深掘り分析する診断ツール。")
    parser.add_argument("--timescale", default="ni_daily_1", help="分析対象のタイムスケール")
    args = parser.parse_args()
    
    timescale = args.timescale
    print(f"\n{'='*20} 深層挙動分析を開始: {timescale} {'='*20}")

    # --- 1. 設定の読み込み ---
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    # --- 2. ログファイルの読み込み (パス修正済み) ---
    # walk_forward_validator.py が保存した場所を指定
    log_dir = PROJECT_ROOT / "logs" / f"{timescale}_walkforward"
    log_file = log_dir / f"wf_results_{timescale}.csv"
    
    if not log_file.exists():
        print(f"\n[エラー] ウォークフォワード検証ログが見つかりません: {log_file}")
        print(f"       先に 'python analysis/walk_forward_validator.py --timescale {timescale}' を実行してください。")
        sys.exit(1)
    
    print(f" -> ログファイルを読み込み中: {log_file.name}")
    df_wf = pd.read_csv(log_file, parse_dates=['Date'], index_col='Date')
    df_wf.sort_index(inplace=True)

    # --- 3. 出力ディレクトリの準備 ---
    output_dir = PROJECT_ROOT / "reports" / timescale / "deep_dive"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 4. 各分析の実行 ---
    analyze_prediction_correlation(df_wf.copy(), output_dir)
    analyze_prediction_distribution(df_wf.copy(), output_dir)
    analyze_worst_predictions(df_wf.copy(), cfg, output_dir)
    
    print(f"\n{'='*20} 深層挙動分析が完了しました {'='*20}")
    print(f"レポートは {output_dir} に保存されています。")

if __name__ == "__main__":
    main()