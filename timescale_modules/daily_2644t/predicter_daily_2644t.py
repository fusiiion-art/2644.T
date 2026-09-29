import pandas as pd
import torch
from pathlib import Path
import sys
import numpy as np
import warnings

# --- Project Setup ---
try:
    project_root = Path(__file__).resolve().parents[2]
except NameError:
    project_root = Path('.').resolve()

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import timescale_modules.daily_2644t.config_daily_2644t as cfg
from common_utils.predicter_model import DailyPredicter

warnings.filterwarnings('ignore', category=UserWarning)

def get_latest_data_and_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    最新の特徴量データと、価格データ（YFキャッシュ）の両方を読み込む。
    """
    data_path = project_root / "data" / cfg.OUTPUT_FILENAME
    yf_cache_path = project_root / "data" / cfg.YF_CACHE_FILENAME
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
        
    df = pd.read_parquet(data_path)
    date_col = 'Date' if 'Date' in df.columns else 'date'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
    df_prices = None
    if yf_cache_path.exists():
        try:
            df_raw = pd.read_parquet(yf_cache_path)
            raw_date_col = 'Date' if 'Date' in df_raw.columns else 'date'
            if raw_date_col in df_raw.columns:
                df_raw[raw_date_col] = pd.to_datetime(df_raw[raw_date_col])
                df_raw = df_raw.sort_values(raw_date_col).reset_index(drop=True)
                df_prices = df_raw
        except Exception:
            pass
    return df, df_prices

def main():
    print("🚀 Starting Daily Prediction Pipeline (Refactored)")
    
    try:
        df, df_prices = get_latest_data_and_prices()
        
        # Initialize Predicter
        predicter = DailyPredicter(project_root, cfg)
        
        # Run Pipeline
        result = predicter.run_pipeline(df, df_prices)
        
        if not result:
            print("⚠️ Pipeline returned no result.")
            return

        # Latest Price determination (Simple logic here or inside Orchestrator)
        # For advice printing, we need a price.
        latest_price = 0.0
        if df_prices is not None and "Close" in df_prices.columns:
            latest_price = df_prices["Close"].iloc[-1]
        elif "close" in df.columns:
            latest_price = df["close"].iloc[-1]
            
        # Generate Manifest & Advice
        predicter.generate_manifest_and_print_advice(result, latest_price, df)
        
    except Exception as e:
        print(f"🚨 Pipeline Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
