import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.style as style
import warnings
import importlib
# ▼▼▼ 修正: 必要な型ヒントを追加 ▼▼▼
from typing import Dict
# ▲▲▲ 修正箇所 ▲▲▲

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

# ▼▼▼ 修正: performance_analyzer.py から指標計算関数を移植 ▼▼▼
def calculate_metrics(returns: pd.Series, annualization_factor=252) -> Dict:
    """リターン系列からパフォーマンス指標を計算するヘルパー関数"""
    if returns.empty or returns.isnull().all():
        return {
            "Total Return": 0.0, "Annualized Return": 0.0,
            "Annualized Volatility": 0.0, "Sharpe Ratio": 0.0, "Max Drawdown": 0.0
        }
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    days = len(returns)
    annual_return = (1 + total_return)**(annualization_factor / days) - 1 if days > 0 else 0
    annual_vol = returns.std() * np.sqrt(annualization_factor)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()
    return {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }
# ▲▲▲ 修正箇所 ▲▲▲


def analyze_regime_performance(timescale: str):
    """
    ウォークフォワード検証の結果を読み込み、市場レジーム（環境）別に
    パフォーマンスを分析・可視化する。
    """
    print(f"\n--- '{timescale}'の市場環境別パフォーマンス分析を開始 ---")

    # --- 1. 設定とデータの読み込み ---
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    # ウォークフォワード検証結果のログを読み込み
    # ▼▼▼ 修正: ファイルパスを walk_forward_validator.py の出力先に合わせる ▼▼▼
    log_file = PROJECT_ROOT / "logs" / f"{timescale}_walkforward" / f"wf_results_{timescale}.csv"
    # ▲▲▲ 修正箇所 ▲▲▲
    if not log_file.exists():
        print(f"\n[エラー] ウォークフォワード検証ログが見つかりません: {log_file}")
        print("先に walk_forward_validator.py を実行してください。")
        sys.exit(1)

    # ▼▼▼ 修正: CSVの読み込み方を performance_analyzer.py と統一 ▼▼▼
    try:
        df = pd.read_csv(log_file, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True)
        pred_cols = [col for col in df.columns if col.startswith('pred_')]
        df.dropna(subset=['actual'] + pred_cols, inplace=True)
        print(f" -> ウォークフォワード検証ログを読み込みました ({len(df)}件)")
    except Exception as e:
        print(f"\n[エラー] ログファイルの読み込み中にエラーが発生しました: {e}")
        return
    # ▲▲▲ 修正箇所 ▲▲▲

    # --- 2. 市場レジームの定義 ---
    # このスクリプトでは、VIXの長期MAに基づいたレジームを「再定義」して分析する
    # (walk_forward_validator.py が使ったレジームとは異なる可能性がある)
    
    # ▼▼▼ 修正: 特徴量データをロードし、VIXカラム名を config から取得 ▼▼▼
    feature_file = PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME
    if not feature_file.exists():
        print(f"[エラー] レジーム分析用の特徴量ファイルが見つかりません: {feature_file}")
        return
        
    df_features = pd.read_parquet(feature_file)
    if 'Date' in df_features.columns:
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        df_features = df_features.set_index('Date')
    
    # VIXカラム名を config から取得
    vix_col = cfg.REGIME_VIX_COLUMN
    if vix_col not in df_features.columns:
        print(f"[警告] VIXカラム '{vix_col}' が特徴量データにないため、レジーム分析をスキップします。")
        return

    # WFVの結果(df)に、特徴量データ(df_features)からVIXカラムを結合
    df_analysis = df.join(df_features[[vix_col]], how='inner')

    df_analysis['vix_ma_200'] = df_analysis[vix_col].rolling(window=200, min_periods=50).mean()
    df_analysis['regime_custom'] = np.where(df_analysis[vix_col] > df_analysis['vix_ma_200'], 'Risk-Off (高ボラ)', 'Risk-On (低ボラ)')
    df_analysis.dropna(subset=['regime_custom'], inplace=True)
    
    print(" -> 市場を2つのカスタムレジームに分類しました:")
    print(df_analysis['regime_custom'].value_counts())
    # ▲▲▲ 修正箇所 ▲▲▲

    # --- 3. レジーム別パフォーマンスの集計 ---
    expert_cols = [c for c in df_analysis.columns if c.startswith('pred_')]
    results = []
    
    # 各モデルの戦略リターンを計算 (performance_analyzer.py と同じロジック)
    df_analysis['Benchmark_return'] = df_analysis['actual']
    for pred_col in expert_cols:
        model_name = pred_col.replace('pred_', '').upper()
        position = np.sign(df_analysis[pred_col])
        df_analysis[f'{model_name}_return'] = position.shift(1) * df_analysis['actual']
    
    # 最初のNaNを0で埋める
    df_analysis.fillna(0.0, inplace=True)

    # モデルリストに 'Benchmark' を追加
    model_return_cols = ['Benchmark_return'] + [f'{col.replace("pred_", "").upper()}_return' for col in expert_cols]

    for model_ret_col in model_return_cols:
        model_name = model_ret_col.replace('_return', '')
        
        for regime_name, regime_df in df_analysis.groupby('regime_custom'):
            if regime_df.empty: continue
            
            # パフォーマンス指標を計算
            metrics = calculate_metrics(regime_df[model_ret_col])
            
            # 方向性精度 (Accuracy)
            if model_name == "Benchmark":
                # ベンチマークは常にBuyなので、実績がプラスだった割合
                accuracy = (regime_df['actual'] > 0).mean()
            else:
                # モデルのポジションと実績の符号が一致した割合
                # (position は shift(1) する前のものを使う)
                pred_col = f"pred_{model_name.lower()}"
                position = np.sign(regime_df[pred_col])
                correct_direction = (position * regime_df['actual']) > 0
                accuracy = correct_direction.mean()
            
            metrics["Model"] = model_name
            metrics["Regime"] = regime_name
            metrics["Accuracy"] = accuracy
            metrics["Trades"] = len(regime_df)
            results.append(metrics)

    summary_df = pd.DataFrame(results).pivot(index="Model", columns="Regime", values=["Accuracy", "Annualized Return", "Sharpe Ratio", "Max Drawdown"])
    
    print("\n--- 市場レジーム別 パフォーマンスサマリー (VIX 200MA基準) ---")
    
    # 表示フォーマットを動的に作成
    formatters = {}
    if 'Risk-On (低ボラ)' in summary_df.columns.get_level_values(1):
        formatters[('Accuracy', 'Risk-On (低ボラ)')] = '{:,.2%}'.format
        formatters[('Annualized Return', 'Risk-On (低ボラ)')] = '{:,.2%}'.format
        formatters[('Sharpe Ratio', 'Risk-On (低ボラ)')] = '{:,.2f}'.format
        formatters[('Max Drawdown', 'Risk-On (低ボラ)')] = '{:,.2%}'.format
    if 'Risk-Off (高ボラ)' in summary_df.columns.get_level_values(1):
        formatters[('Accuracy', 'Risk-Off (高ボラ)')] = '{:,.2%}'.format
        formatters[('Annualized Return', 'Risk-Off (高ボラ)')] = '{:,.2%}'.format
        formatters[('Sharpe Ratio', 'Risk-Off (高ボラ)')] = '{:,.2f}'.format
        formatters[('Max Drawdown', 'Risk-Off (高ボラ)')] = '{:,.2%}'.format
        
    print(summary_df.to_string(formatters=formatters))

    # --- 4. グラフの作成と保存 ---
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 資産曲線を計算
    for col in model_return_cols:
        model_name = col.replace('_return', '')
        equity_col = f'{model_name}_equity'
        df_analysis[equity_col] = (1 + df_analysis[col]).cumprod()
        
        linestyle = '--' if model_name == "Benchmark" else '-'
        linewidth = 1.5 if model_name == "Benchmark" else 2
        color = 'grey' if model_name == "Benchmark" else None
        
        ax.plot(df_analysis.index, df_analysis[equity_col], label=model_name, linewidth=linewidth, linestyle=linestyle, zorder=2, color=color)

    # レジームに応じて背景色を変更
    risk_off_periods = df_analysis[df_analysis['regime_custom'] == 'Risk-Off (高ボラ)']
    # 連続した期間のみ描画（簡略化）
    for start_date in risk_off_periods.index.to_series().diff().dt.days.gt(1).cumsum().unique():
         period_data = risk_off_periods[risk_off_periods.index.to_series().diff().dt.days.gt(1).cumsum() == start_date]
         if not period_data.empty:
             ax.axvspan(period_data.index.min(), period_data.index.max(), color='red', alpha=0.1, zorder=0)

    ax.set_yscale('log')
    ax.set_title(f'Equity Curves Across Market Regimes ({timescale} - VIX 200MA Crossover)', fontsize=16)
    ax.set_ylabel('Cumulative Return (Log Scale)')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.text(0.01, 0.02, 'Shaded Area = Risk-Off (High Volatility) Regime', transform=ax.transAxes, fontsize=10, color='red', alpha=0.7)

    plt.tight_layout()
    
    report_dir = PROJECT_ROOT / "reports" / timescale
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "regime_performance_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n--- 分析完了 ---")
    print(f"レジーム分析レポートを保存しました: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ウォークフォワード検証結果を市場レジーム別に分析します。")
    parser.add_argument("--timescale", required=True, help="分析対象のタイムスケール (例: 'daily_1')")
    args = parser.parse_args()
    
    analyze_regime_performance(args.timescale)
