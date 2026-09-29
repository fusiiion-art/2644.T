import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import importlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import warnings
from typing import Dict

# --- Project Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlibのスタイルと警告設定
style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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

    # ドローダウン計算
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


def analyze_performance(timescale: str):
    """
    指定されたタイムスケールのウォークフォワード検証結果を分析し、
    パフォーマンス指標の計算と結果の可視化を行う。
    """
    print(f"\n--- '{timescale}'のパフォーマンス分析を開始 ---")

    # --- 1. 設定ファイルの動的読み込み ---
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
        print(f" -> 設定ファイル '{config_module_path}' を読み込みました。")
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    # --- 2. ウォークフォワード検証結果の読み込み ---
    # ▼▼▼ 修正: ファイルパスを walk_forward_validator.py の出力先に合わせる ▼▼▼
    log_file = PROJECT_ROOT / "logs" / f"{timescale}_walkforward" / f"wf_results_{timescale}.csv"
    # ▲▲▲ 修正箇所 ▲▲▲
    if not log_file.exists():
        print(f"\n[エラー] ウォークフォワード検証結果ファイルが見つかりません: {log_file}")
        print(f"       先に analysis/walk_forward_validator.py を実行してください。")
        sys.exit(1)

    try:
        # ▼▼▼ 修正: index_col を 'Date' に変更 (CSVに合わせて) ▼▼▼
        df = pd.read_csv(log_file, parse_dates=['Date'], index_col='Date')
        # ▲▲▲ 修正箇所 ▲▲▲
        df.sort_index(inplace=True)
        # 予測値('pred_*')または実績('actual')のいずれかがNaNの行は分析に不要なため削除
        pred_cols = [col for col in df.columns if col.startswith('pred_')]
        df.dropna(subset=['actual'] + pred_cols, inplace=True) # inplace=True を追加

        if df.empty:
            print("\n[エラー] ログファイルが空か、有効なデータがありません。")
            return

        print(f" -> ウォークフォワード検証結果を読み込みました ({len(df)}件の有効な記録)")

    except Exception as e:
        print(f"\n[エラー] ログファイルの読み込み中にエラーが発生しました: {e}")
        return


    # --- 3. パフォーマンス指標の計算 ---
    results = {}
    equity_curves = pd.DataFrame(index=df.index)

    # ベンチマーク (Buy & Hold)
    benchmark_returns = df['actual']
    results["Benchmark"] = calculate_metrics(benchmark_returns)
    equity_curves["Benchmark"] = (1 + benchmark_returns).cumprod()

    # 各モデル戦略
    for pred_col in pred_cols:
        model_name = pred_col.replace('pred_', '').upper()
        # ポジションサイズを決定 (予測値 > 0 なら +1, <= 0 なら -1 と仮定)
        position = np.sign(df[pred_col])
        # 前日のポジションを当日のリターンに乗じる (取引は前日終値で行うと仮定)
        strategy_returns = position.shift(1) * df['actual']
        # 最初のNaNを0で埋める
        strategy_returns = strategy_returns.fillna(0.0)

        results[model_name] = calculate_metrics(strategy_returns)
        equity_curves[model_name] = (1 + strategy_returns).cumprod()

    # --- 4. 結果のサマリー表示 ---
    summary_df = pd.DataFrame(results)
    # 表示用にフォーマット調整
    summary_formatted = summary_df.copy()
    for col in summary_formatted.columns:
        summary_formatted[col] = summary_formatted[col].apply(lambda x: f"{x:.2%}" if isinstance(x, (float, np.float64)) and col != "Sharpe Ratio" else (f"{x:.2f}" if isinstance(x, (float, np.float64)) else x))


    print("\n--- パフォーマンスサマリー ---")
    print(summary_formatted)

    # --- 5. グラフの作成と保存 ---
    n_plots = len(equity_curves.columns)
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 8)) # ドローダウンは別途表示も検討

    # 資産曲線グラフ
    for col in equity_curves.columns:
        linestyle = '--' if col == "Benchmark" else '-'
        linewidth = 1.5 if col == "Benchmark" else 2
        color = 'grey' if col == "Benchmark" else None
        ax1.plot(equity_curves.index, equity_curves[col], label=col, linestyle=linestyle, linewidth=linewidth, color=color)

    ax1.set_yscale('log')
    ax1.set_title(f'Walk-Forward Performance ({timescale})', fontsize=16)
    ax1.set_ylabel('Cumulative Return (Log Scale)')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    report_dir = PROJECT_ROOT / "reports" / timescale
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "performance_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # メモリを解放

    print(f"\n--- 分析完了 ---")
    print(f"パフォーマンスレポートを保存しました: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ウォークフォワード検証結果を分析し、パフォーマンスレポートを生成します。")
    parser.add_argument("--timescale", required=True, help="分析対象のタイムスケール (例: 'daily_1')")
    args = parser.parse_args()

    analyze_performance(args.timescale)
