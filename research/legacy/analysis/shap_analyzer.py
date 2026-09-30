import sys
import pandas as pd
import numpy as np
import shap
import torch
import joblib
from pathlib import Path
import argparse
import importlib
import json
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のため

# --- プロジェクトのパス設定 ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- モデル定義のインポート ---
# これにより、保存されたモデルを正しくロードできる
from common_utils.supervisor_model import MetaLearner

def analyze_shap(timescale: str, regime: str):
    """
    指定されたタイムスケールとレジームの監督者モデルについて、
    SHAP分析を行い、判断根拠を可視化する。
    """
    print(f"\n--- SHAP分析開始: タイムスケール[{timescale}], レジーム[{regime.upper()}] ---")

    # --- 1. 設定とアセットの読み込み ---
    try:
        config_module_path = f"timescale_modules.{timescale}.config_{timescale}"
        cfg = importlib.import_module(config_module_path)
    except ImportError:
        print(f"[エラー] 設定ファイルが見つかりません: {config_module_path}")
        return

    try:
        sup_config = cfg.REGIME_SUPERVISOR_CONFIG[regime]
        supervisor_name = sup_config['name']
    except (AttributeError, KeyError):
        print(f"[エラー] configファイルに '{regime}' 用の 'REGIME_SUPERVISOR_CONFIG' が正しく設定されていません。")
        return

    # --- 2. 最新の学習済みモデルとスケーラーのパスを取得 ---
    base_log_dir = PROJECT_ROOT / "logs" / f"{cfg.MODULE_NAME}_{supervisor_name}_final"
    
    print(f" -> ログディレクトリを検索中: {base_log_dir}")
    
    # ▼▼▼ 修正: .exists() から .is_dir() に変更 ▼▼▼
    # パスが存在し、かつそれが「ディレクトリ」であることを確認
    if not base_log_dir.is_dir():
        print(f"[エラー] 監督者モデルのログディレクトリが見つかりません。(またはディレクトリではありません)")
        print(f"       パス: {base_log_dir}")
        print(f"       (supervisor_{timescale}.py を先に実行する必要があります)")
        return
    # ▲▲▲ 修正箇所 ▲▲▲

    # base_log_dir がディレクトリであることを確認してから iterdir() を実行
    versions = sorted([d for d in base_log_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not versions:
        print(f"[エラー] ログディレクトリに有効なバージョンが見つかりません: {base_log_dir}")
        return

    latest_version_dir = versions[0]
    print(f" -> 最新バージョン (version_{latest_version_dir.name}) を使用します。")
    
    model_path = latest_version_dir / "supervisor_model.joblib"
    scaler_path = latest_version_dir / "supervisor_scaler.joblib"
    features_path = latest_version_dir / "supervisor_features.json"

    if not all([model_path.exists(), scaler_path.exists(), features_path.exists()]):
        print(f"[エラー] 必要なアセット (model, scaler, features) が見つかりません。")
        print(f"       {model_path}")
        print(f"       {scaler_path}")
        print(f"       {features_path}")
        return

    # --- 3. アセットのロード ---
    print(" -> モデル、スケーラー、特徴量リストをロード中...")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            features = json.load(f)
    except Exception as e:
        print(f"[エラー] アセットのロード中にエラーが発生しました: {e}")
        return

    # --- 4. SHAP分析の実行 ---
    print(" -> SHAP Explainer を準備中...")
    
    # SHAP分析用のデータセット (背景データ) が必要
    # ここでは、最新の予測に使われたデータセットの親 (X_train) を探す
    # (注意: 本来は supervisor_trainer.py から X_train を保存すべき)
    #
    # [仮の実装]
    # もし X_train が保存されていればそれを使う。
    # なければ、最新の予測ログから「単一のインスタンス」で説明を試みる。
    
    explainer = shap.KernelExplainer(model.predict_proba, scaler.transform(np.zeros((1, len(features)))))
    print(f" -> KernelExplainer を初期化しました (ダミーデータ使用)。")


    # 最新の予測インスタンスをロードして説明する
    last_prediction_log = PROJECT_ROOT / "logs" / "predictions" / f"{timescale}_last_prediction_context.json"
    if not last_prediction_log.exists():
        print(f"[エラー] 最新の予測ログ '{last_prediction_log.name}' が見つかりません。")
        print(f"       (predict_daily_1.py を一度実行してください。)")
        return
        
    last_context = json.load(open(last_prediction_log))
    if last_context['regime'] != regime:
        print(f"[情報] 最新の予測はレジーム'{last_context['regime']}'で行われました。分析対象の'{regime}'とは異なります。")
        return

    instance_to_explain = pd.DataFrame([last_context['supervisor_input']])[features]
    instance_scaled = scaler.transform(instance_to_explain)
    
    shap_values = explainer.shap_values(instance_scaled)

    # --- 5. 結果の可視化と保存 ---
    print(" -> 分析結果を可視化中...")
    output_dir = PROJECT_ROOT / "reports" / timescale / "xai"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    # predict_proba の出力 (クラス0, クラス1) のうち、クラス1 (上昇予測) のSHAP値を使用
    shap.summary_plot(shap_values[1], features=features, plot_type="bar", show=False)
    plt.title(f"最終予測の判断根拠 (SHAP値) - レジーム: {regime.upper()}")
    plt.xlabel("最終予測への貢献度 (SHAP Value for 'UP')")
    plt.tight_layout()
    
    save_path = output_dir / f"shap_summary_{regime}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"\n--- SHAP分析完了 ---")
    print(f"結果は {save_path} に保存されました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="監督者モデルのSHAP分析を実行します。")
    parser.add_argument("--timescale", type=str, required=True, help="タイムスケール (例: daily_1)")
    parser.add_argument("--regime", type=str, required=True, help="レジーム (例: risk_on)")
    
    args = parser.parse_args()
    
    analyze_shap(timescale=args.timescale, regime=args.regime.lower()) # レジーム名を小文字に正規化
