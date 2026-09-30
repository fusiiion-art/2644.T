import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import datetime
import subprocess

from common_utils.regime_detector import RegimeDetector
from common_utils.asset_loader import AssetLoader
from common_utils.supervisor_model import SupervisorEnsemble
from common_utils.risk_manager import RiskManager
from common_utils.drift_detector import DriftDetector

plt.rcParams['font.family'] = 'sans-serif'

class ExpertPredicter:
    """
    単一の学習済み専門家モデル (TCN/BDH/S5) をロードし、予測を実行するクラス。
    """
    def __init__(self, model_name: str, assets: Dict[str, Any], config: Any):
        self.model_name = model_name
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = assets.get("model")
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device).eval()
            if hasattr(self.model, 'freeze'):
                self.model.freeze() 

        self.scaler = assets.get("scaler")
        self.feature_cols = assets.get("features")

    def predict(self, df_latest: pd.DataFrame) -> Tuple[float, float]:
        """
        直近のデータフレームを受け取り、明日のリターン予測(mu)と不確実性(sigma)を返す。
        Returns:
            (mu, sigma) のタプル
        """
        missing = [c for c in self.feature_cols if c not in df_latest.columns]
        if missing:
            raise KeyError(f"[{self.model_name}] Missing features for prediction: {missing[:3]}...")

        try:
            # S5モデル対応のシーケンス長
            if self.model_name == "s5":
                seq_len = getattr(self.config, "SEQUENCE_LENGTH_S5", 120)
            else:
                seq_len = self.config.SEQUENCE_LENGTH
            
            if len(df_latest) < seq_len:
                raise ValueError(f"[{self.model_name}] Not enough data. Need {seq_len}, got {len(df_latest)}")
            
            df_seq = df_latest[self.feature_cols].iloc[-seq_len:].copy()
            if df_seq.isna().any().any():
                df_seq = df_seq.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            X_seq_raw = df_seq.values

        except Exception as e:
            raise ValueError(f"[{self.model_name}] Data processing failed: {e}")

        if self.scaler is None:
             raise ValueError(f"[{self.model_name}] Scaler not found in assets.")
        
        try:
            X_scaled = self.scaler.transform(X_seq_raw)
        except Exception as e:
             raise ValueError(f"[{self.model_name}] Scaling failed. Check feature count. {e}")

        input_tensor = torch.from_numpy(X_scaled.astype(np.float32)).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # タプル (mu, sigma) か 単一テンソル かを判定
            if isinstance(output, tuple):
                mu = output[0].cpu().item()
                sigma = output[1].cpu().item()
            else:
                mu = output.cpu().item()
                # ★修正: sigmaが返されない場合のデフォルト値を 0.0 -> 0.01 (1%) に変更
                # 0.0 だと PortfolioStrategist が取引を拒否するため
                sigma = 0.01
            
            # NaN対策
            if np.isnan(mu): mu = 0.0
            
            # ★修正: sigma の下限値クリッピング (ゼロ除算防止 & 過剰レバレッジ防止)
            # 0.0001 (0.01%) 未満のsigmaは異常値として補正する
            if np.isnan(sigma) or sigma < 1e-4: 
                sigma = 1e-4
            
        return mu, sigma

class DailyPredicter:
    """
    日次推論パイプラインのオーケストレーター。
    1. データロード
    2. レジーム検知
    3. 全エキスパート推論
    4. Supervisorによる統合
    5. トレードプラン作成・マニフェスト保存
    """
    def __init__(self, project_root: Path, config: Any):
        self.project_root = project_root
        self.config = config
        self.loader = AssetLoader(project_root, getattr(config, 'MODULE_NAME', 'daily_2644t'))
        
    def run_pipeline(self, df: pd.DataFrame, df_prices: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        print(f"🚀 Starting Daily Prediction Pipeline for {self.config.MODULE_NAME}")
        
        # 1. レジーム検知
        detector = RegimeDetector()
        vix_col = next((c for c in df.columns if 'vix' in c.lower() and 'return' not in c.lower()), None)
        latest_vix = df[vix_col].iloc[-1] if vix_col else 15.0
        current_regime = "risk_off" if latest_vix > self.config.REGIME_VIX_THRESHOLD else "risk_on"
        print(f"  🛡️ Current Regime: {current_regime.upper()} (VIX: {latest_vix:.2f})")
        
        # 2. Expert Predictions
        experts = self.config.REGIME_EXPERTS.get(current_regime, [])
        expert_preds = {}
        expert_sigmas = {}
        
        print("\n🤖 Expert Predictions:")
        for expert_name in experts:
            try:
                # Productionモデルを優先ロード (SWAなど)
                assets = self.loader.load_production_model(expert_name, current_regime, use_swa=True)
                predicter = ExpertPredicter(expert_name, assets, self.config)
                mu, sigma = predicter.predict(df)
                expert_preds[expert_name] = mu
                expert_sigmas[expert_name] = sigma
                print(f"  - {expert_name}: {mu:.6f} (Sigma: {sigma:.4f})")
            except Exception as e:
                print(f"  ❌ {expert_name} failed: {e}")
        
        if not expert_preds:
            print("  ⚠️ No expert predictions available.")
            return {}

        # 3. Supervisor Aggregation
        try:
            sup_assets = self.loader.load_production_model("supervisor", current_regime, use_swa=False)
        except:
            sup_assets = None # SupervisorなしでもIVWは動作可能
            
        ensemble = SupervisorEnsemble(sup_assets, use_ivw=True)
        agg_result = ensemble.aggregate(expert_preds, expert_sigmas, latest_vix)
        
        final_mu = agg_result["mu"]
        final_sigma = agg_result["sigma"]
        print(f"\n  📈 Ensemble Result: Mu={final_mu:.6f}, Sigma={final_sigma:.6f}")
        
        # 4. トレードロジック & ドリフト検知 & マニフェスト
        # ここは戦略依存だが、標準的なものを実装
        
        # ... (TBD: 呼び出し元でやるか、ここに入れるか。User要望は「Predicterの機能をまとめろ」なので入れるのが正解)
        # 簡易実装: RiskManager呼び出しなどはConfig依存が強いため、主要な数値計算までをここで返す形にするか、
        # あるいは汎用的なRiskManagerを呼ぶか。
        
        return {
            "date": df['Date'].iloc[-1] if 'Date' in df.columns else datetime.date.today(),
            "regime": current_regime,
            "latest_vix": latest_vix,
            "expert_predictions": expert_preds,
            "expert_sigmas": expert_sigmas,
            "final_mu": final_mu,
            "final_sigma": final_sigma,
            "aggregation_details": agg_result
        }
    
    def generate_manifest_and_print_advice(self, result: Dict[str, Any], latest_price: float, df_original: pd.DataFrame):
        """
        結果を受け取り、ユーザー向けアドバイス表示とマニフェスト保存を行う
        """
        mu = result["final_mu"]
        sigma = result["final_sigma"]
        
        # --- Drift Detection ---
        detector = DriftDetector(psi_threshold=0.25)
        drift_report = {}
        try:
             # 直近90日を参照
            reference_df = df_original.iloc[-90:-1]
            current_df = df_original.iloc[[-1]]
            drift_cols = [c for c in ['vix', 'rsi_14', 'macd', 'atr'] if c in df_original.columns]
            if not drift_cols and 'vix' in df_original.columns: drift_cols = ['vix']
            
            if drift_cols:
                drift_report = detector.detect_drift(reference_df, current_df, drift_cols)
                if drift_report['drift_detected']:
                    print(f"  ⚠️ DRIFT WARNING: {drift_report['summary']}")
        except Exception as e:
            print(f" Drift check warning: {e}")

        # --- Advice ---
        # 簡易的なロジック（ハーフケリー）
        # Configから読み込むべきだが、デフォルト値で実装
        print("\n" + "="*40)
        print("💰 AI FUSION STRATEGY DECISION")
        print("="*40)
        
        target_pos = 0.0
        if mu > 0:
            raw_kelly = mu / (sigma**2 + 1e-6)
            target_pos = raw_kelly * 0.5 # Half Kelly
            target_pos = np.clip(target_pos, 0.0, 1.0)
            print(f"  🤖 BUY SIGNAL: Size {target_pos*100:.1f}%")
        else:
            print(f"  🐻 SELL/WAIT: Mu is negative ({mu:.4f})")
            
        # --- Manifest Save ---
        manifest_dir = self.project_root / "logs" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        
        manifest = {
            "timestamp": datetime.datetime.now().isoformat(),
            "regime": result["regime"],
            "latest_price": latest_price,
            "prediction": {
                "mu": mu,
                "sigma": sigma,
                "target_position": target_pos
            },
            "drift_report": drift_report,
            "experts": result["expert_predictions"]
        }
        
        with open(manifest_dir / f"manifest_{today_str}.json", 'w') as f:
            json.dump(manifest, f, indent=4)
        print(f"  📜 Manifest saved to: {manifest_dir / f'manifest_{today_str}.json'}")

# ==========================================
#  以下、可視化・シミュレーション用関数
# ==========================================

def load_all_models_for_simulation(loader: Any, config: Any) -> Dict[str, Any]:
    models_cache = {}
    for regime in ["risk_on", "risk_off"]:
        models_cache[regime] = {}
        experts = config.REGIME_EXPERTS.get(regime, [])
        for exp_name in experts:
            try:
                assets = loader.load_production_model(exp_name, regime, use_swa=True)
                models_cache[regime][exp_name] = ExpertPredicter(exp_name, assets, config)
            except Exception: pass
        try:
            sup_assets = loader.load_production_model("supervisor", regime, use_swa=False)
            models_cache[regime]["supervisor"] = sup_assets
        except Exception: pass
    return models_cache

def run_simulation_and_visualize(df: pd.DataFrame, loader: Any, config: Any, project_root: Path):
    print("\n" + "="*30)
    print("📊 RUNNING VISUALIZATION SIMULATION (Probabilistic)")
    print("="*30)
    
    TEST_DAYS = 100
    if len(df) < TEST_DAYS + 30:
        print("⚠️ Not enough data for visualization.")
        return

    test_df = df.iloc[-(TEST_DAYS + 20):].reset_index(drop=True)
    models_cache = load_all_models_for_simulation(loader, config)
    results = []

    date_col = 'Date' if 'Date' in df.columns else 'date'
    
    for i in tqdm(range(20, len(test_df))):
        current_slice = test_df.iloc[i-20 : i+1]
        target_date = test_df[date_col].iloc[i]
        
        vix_col = config.REGIME_VIX_COLUMN
        if vix_col not in current_slice.columns: continue
        latest_vix = current_slice[vix_col].iloc[-1]
        current_regime = "risk_off" if latest_vix > config.REGIME_VIX_THRESHOLD else "risk_on"

        row = {
            "Date": target_date,
            "Regime": current_regime,
            "Actual_Return": test_df[config.TARGET_COLUMN].iloc[i] if i < len(test_df) else 0.0
        }

        regime_models = models_cache.get(current_regime, {})
        expert_preds = {}
        expert_sigmas = {}
        
        for name, predictor in regime_models.items():
            if name == "supervisor": continue
            try:
                mu, sigma = predictor.predict(current_slice)
                expert_preds[name] = mu
                expert_sigmas[name] = sigma
                row[f"{name}_mu"] = mu
                row[f"{name}_sigma"] = sigma
            except:
                row[f"{name}_mu"] = np.nan
        
        if "supervisor" in regime_models and expert_preds:
            try:
                sup = regime_models["supervisor"]
                vals = [v for v in expert_preds.values() if not np.isnan(v)]
                sigmas = [s for s in expert_sigmas.values() if not np.isnan(s)]
                
                input_data = expert_preds.copy()
                input_data['meta_std'] = np.std(vals) if len(vals) > 1 else 0.0
                # 平均的なモデルの自信度も入力に入れたい場合（将来的な拡張）
                # input_data['avg_model_sigma'] = np.mean(sigmas) if sigmas else 0.0
                
                input_df = pd.DataFrame([input_data])
                final_input = pd.DataFrame(0.0, index=[0], columns=sup['features'])
                for c in sup['features']:
                    if c in input_df.columns: final_input[c] = input_df[c]
                    elif c == config.REGIME_VIX_COLUMN: final_input[c] = latest_vix
                
                X_sup = sup['scaler'].transform(final_input.values)
                with torch.no_grad():
                    # Supervisorはまだ確率出力ではない(MetaLearner)ためスカラー
                    row["Supervisor_pred"] = sup['model'](torch.from_numpy(X_sup).float()).item()
            except:
                row["Supervisor_pred"] = np.nan
        else:
            row["Supervisor_pred"] = np.nan
            
        results.append(row)

    if not results: return

    res_df = pd.DataFrame(results).set_index("Date")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 上段: Expert Predictions (Confidence Interval)
    ax1 = axes[0]
    ax1.set_title("Expert Models: Prediction vs Actual (with Uncertainty)", fontsize=12, fontweight='bold')
    ax1.plot(res_df.index, res_df['Actual_Return'], label='Actual', color='gray', alpha=0.5, linestyle='--')
    
    colors = {'tcn': 'blue', 'bdh': 'green', 's5': 'purple'}
    for m in ['tcn', 'bdh', 's5']:
        col_mu = f"{m}_mu"
        col_sigma = f"{m}_sigma"
        if col_mu in res_df.columns:
            # 予測線
            ax1.plot(res_df.index, res_df[col_mu], label=m.upper(), color=colors.get(m, 'black'), alpha=0.8)
            # 信頼区間 (±1σ) の描画
            if col_sigma in res_df.columns:
                lower = res_df[col_mu] - res_df[col_sigma]
                upper = res_df[col_mu] + res_df[col_sigma]
                ax1.fill_between(res_df.index, lower, upper, color=colors.get(m, 'black'), alpha=0.1)

    ax1.legend(loc='upper left')
    ax1.set_ylabel("Return")

    # 中段: Supervisor
    ax2 = axes[1]
    ax2.set_title("AI Supervisor Decision", fontsize=12, fontweight='bold')
    if 'Supervisor_pred' in res_df.columns:
        ax2.plot(res_df.index, res_df['Supervisor_pred'], label='AI Prediction', color='red', linewidth=2)
    ax2.axhline(0, color='black', alpha=0.5)
    ylim = ax2.get_ylim()
    ax2.fill_between(res_df.index, ylim[0], ylim[1], where=(res_df['Regime']=='risk_off'), color='red', alpha=0.1, label='Risk Off')
    ax2.legend(loc='upper left')

    # 下段: 累積リターン
    ax3 = axes[2]
    ax3.set_title("Performance Simulation", fontsize=12, fontweight='bold')
    actual_cum = (1 + res_df['Actual_Return']).cumprod()
    ax3.plot(res_df.index, actual_cum, label='Buy & Hold', color='gray', linestyle='--')
    
    if 'Supervisor_pred' in res_df.columns:
        signals = np.sign(res_df['Supervisor_pred'].fillna(0))
        signals = np.where(signals > 0, 1, 0)
        strat_ret = signals * res_df['Actual_Return']
        ai_cum = (1 + strat_ret).cumprod()
        ax3.plot(res_df.index, ai_cum, label='AI Strategy', color='gold', linewidth=2)

    ax3.legend(loc='upper left')
    
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "prediction_chart_probabilistic.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  ✅ Chart saved to: {out_path}")