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
import optuna
import re # <-- 正規表現のために追加

# --- プロジェクトのパス設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ▼▼▼ 修正: featureregistry から定義をインポート ▼▼▼
try:
    from generate.featureregistry import FEATURE_SETS_DEFINITIONS
except ImportError:
    print("[ERROR] Failed to import FEATURE_SETS_DEFINITIONS from generate.featureregistry.")
    print("        Please ensure 'generate' directory is in sys.path and featureregistry.py exists.")
    FEATURE_SETS_DEFINITIONS = {}
# ▲▲▲ 修正箇所 ▲▲▲

# Matplotlibのスタイルと警告設定
style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")


def analyze_parameter_importance(storage_path: str, experts: list, module_name: str, output_dir: Path):
    """Optunaの学習履歴から、ハイパーパラメータの重要度を可視化する"""
    print("\n--- ハイパーパラメータ重要度分析を開始 ---")
    
    # ▼▼▼ 修正: cfg.EXPERTS ではなく、config から取得した expert リストを直接使用 ▼▼▼
    #    (この関数は main から呼び出される前提)
    for expert_name in experts:
        # ▼▼▼ 修正: Optuna Study 名を config_daily_1.py の体系に合わせる ▼▼▼
        # study_name = f"{module_name}-{expert_name}-study" # 旧
        # (config_daily_1.py の REGIMES と REGIME_EXPERTS を知らないと study 名を復元できない)
        # (Optuna DB を直接ロードするアプローチに変更)
        pass # Optuna DB のロードロジックを main 側で実施するように変更
    
    # (ロジックを main に移動するため、ここでは何もしない)
    # ▲▲▲ 修正箇所 ▲▲▲
    pass


# ▼▼▼ 修正: config.FEATURE_SETS に基づいて特徴量カラムを動的に取得する関数 (trainer_model.py から移植) ▼▼▼
def _get_feature_columns_from_config(cfg, all_df_columns: set) -> list:
    """
    config.py の FEATURE_SETS リストに基づき、
    featureregistry.py の定義 (Regex) を使って特徴量カラムを動的に取得する。
    """

    if not hasattr(cfg, 'FEATURE_SETS') or not cfg.FEATURE_SETS:
        raise ValueError("config.FEATURE_SETS is not defined or empty.")

    print(f"[INFO] Building feature list based on config.FEATURE_SETS: {cfg.FEATURE_SETS}")

    patterns_to_match = []
    missing_definitions = []
    for set_name in cfg.FEATURE_SETS:
        if set_name in FEATURE_SETS_DEFINITIONS:
            patterns_to_match.extend(FEATURE_SETS_DEFINITIONS[set_name])
        else:
            missing_definitions.append(set_name)

    if missing_definitions:
        print(f"[WARNING] Definitions for {missing_definitions} not found in featureregistry. Skipping.")

    if not patterns_to_match:
        raise ValueError("No feature patterns found. FEATURE_SETS in config might be empty or all definitions are missing.")

    matched_features = set()

    for pattern_str in patterns_to_match:
        try:
            pattern = re.compile(pattern_str)
            matches = {col for col in all_df_columns if pattern.match(col)}
            matched_features.update(matches)
        except re.error as e:
            print(f"[WARNING] Invalid regex pattern '{pattern_str}': {e}")

    final_feature_list = sorted(list(matched_features))

    if not final_feature_list:
        raise ValueError("No features matched the patterns defined in config.FEATURE_SETS.")

    print(f"[INFO] Selected {len(final_feature_list)} features for correlation analysis.")
    return final_feature_list
# ▲▲▲ 修正箇所 ▲▲▲


def analyze_feature_correlation(cfg, output_dir: Path):
    """特徴量データ内の相関を計算し、ヒートマップとして可視化する"""
    print("\n--- 特徴量 相関分析を開始 ---")
    
    feature_file = PROJECT_ROOT / "data" / cfg.OUTPUT_FILENAME
    if not feature_file.exists():
        print(f"[エラー] 特徴量ファイルが見つかりません: {feature_file}")
        return
        
    df_features = pd.read_parquet(feature_file)
    # (カラム名の . を _ に置換する処理は不要。parquet 時点で対応済みのはず)
    
    # ▼▼▼ 修正: TCN_FEATURE_COLUMNS を cfg.FEATURE_SETS から動的に取得 ▼▼▼
    try:
        all_df_columns = set(df_features.columns)
        target_features = _get_feature_columns_from_config(cfg, all_df_columns)
    except ValueError as e:
        print(f"[エラー] 特徴量リストの取得に失敗: {e}")
        return
    # ▲▲▲ 修正箇所 ▲▲▲

    if not target_features:
        print("[警告] 相関分析の対象となる特徴量が見つかりません。")
        return

    print(f" -> {len(target_features)}個の特徴量で相関行列を計算中...")
    correlation_matrix = df_features[target_features].corr()

    # 相関行列が大きすぎる (例: 100x100以上) 場合はヒートマップを簡略化
    if len(target_features) > 100:
        print(f" -> 特徴量 ({len(target_features)}個) が多すぎるため、ヒートマップのテキスト注釈 (annot) を無効にします。")
        show_annotations = False
        fig_size = (20, 16)
    else:
        show_annotations = False # デフォルトでは無効 (多すぎると見づらいため)
        fig_size = (16, 12)

    plt.figure(figsize=fig_size)
    # ▼▼▼ 修正: annot=show_annotations に変更 ▼▼▼
    sns.heatmap(correlation_matrix, cmap='viridis', annot=show_annotations, fmt=".1f") 
    plt.title(f'Feature Correlation Heatmap ({cfg.MODULE_NAME})', fontsize=16)
    
    # ▼▼▼ 修正: 特徴量が多すぎる場合は軸ラベルを非表示にする ▼▼▼
    if len(target_features) > 50:
        plt.xticks([])
        plt.yticks([])
        print(" -> 特徴量が50個を超えたため、軸ラベルを非表示にします。")
    else:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    # ▲▲▲ 修正箇所 ▲▲▲

    plt.tight_layout()

    output_path = output_dir / f"feature_correlation_{cfg.MODULE_NAME}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" -> 特徴量の相関ヒートマップを保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="プロジェクトの総合診断ツールを実行します。")
    parser.add_argument("--timescale", required=True, help="分析対象のタイムスケール (例: 'daily_1')")
    args = parser.parse_args()
    
    timescale = args.timescale
    print(f"\n{'='*20} 総合診断を開始: {timescale} {'='*20}")

    # --- 1. 設定ファイルの動的読み込み ---
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
    except ImportError:
        print(f"\n[エラー] タイムスケール '{timescale}' の設定ファイルが見つかりませんでした。")
        sys.exit(1)

    # --- 2. 出力ディレクトリの準備 ---
    output_dir = PROJECT_ROOT / "reports" / timescale / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 3. 各分析の実行 ---
    
    # ▼▼▼ 修正: Optuna DB のロードと分析ロジックを修正 ▼▼▼
    print("\n--- ハイパーパラメータ重要度分析を開始 ---")
    
    # config_daily_1.py に基づき、実行すべきモデルとレジームの組み合わせを取得
    if not hasattr(cfg, 'REGIME_EXPERTS') or not hasattr(cfg, 'REGIMES'):
        print("[警告] config に REGIME_EXPERTS または REGIMES が定義されていません。Optuna 分析をスキップします。")
    else:
        # config に定義されているすべての study 名をリストアップ
        study_names_in_config = []
        for regime, models in cfg.REGIME_EXPERTS.items():
            for model in models:
                study_names_in_config.append(f"daily_1_{regime}_{model}_optimization")
        
        # Optuna DB からロードを試みる
        for study_name in study_names_in_config:
            # trainer_daily_1.py が保存する .db ファイルのパス
            db_path = PROJECT_ROOT / "logs" / "optuna_db" / f"{study_name}.db"
            
            if not db_path.exists():
                print(f"[情報] Optuna データベース '{db_path.name}' が見つかりません。スキップします。")
                continue

            storage_path = f"sqlite:///{db_path}"
            
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_path)
                
                if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                    print(f" -> '{study_name}' には完了した試行がないため、スキップします。")
                    continue

                # plot_param_importances は plotly が必要
                try:
                    fig = optuna.visualization.plot_param_importances(study)
                    output_path = output_dir / f"{study_name}_param_importances.html"
                    fig.write_html(str(output_path))
                    print(f" -> '{study_name}' の重要度グラフを保存しました: {output_path}")
                except ImportError:
                    print(f"[警告] 'plotly' がインストールされていないため、{study_name} のグラフを生成できません。")
                except Exception as e:
                     print(f" -> '{study_name}' のグラフ生成中にエラーが発生しました: {e}")

            except KeyError:
                 print(f" -> '{study_name}' の学習履歴が見つからないか、分析できるデータがありません。")
            except Exception as e:
                print(f" -> '{study_name}' の分析中にエラーが発生しました: {e}")
    # ▲▲▲ 修正箇所 ▲▲▲

    analyze_feature_correlation(cfg, output_dir)
    
    print(f"\n{'='*20} 総合診断が完了しました {'='*20}")
    print(f"レポートは {output_dir} に保存されています。")

if __name__ == "__main__":
    main()