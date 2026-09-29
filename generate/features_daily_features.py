import logging
import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import argparse
import importlib

# --- Project Path Setup ---
try:
    # .parent (generate/) -> .parents[1] (Project Root)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except (NameError, IndexError):
    PROJECT_ROOT = Path('.').resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Import Common Logic ---
from common_utils.data_fetcher import (
    get_fred_data, get_yfinance_data, get_tiingo_data, get_stooq_data,
    apply_column_mapping, strict_validate, write_atomic
)
from common_utils.target_utils import (
    create_targets, create_open_targets, create_multi_targets, infer_open_column
)
from generate.feature_engine_core import FeatureEngine

# [TEMP DISABLED 2026-01-07] alternative_dataはエラーの温床のため一時的に無効化
AlternativeDataScraper = None

from common_utils.feature_registry import FeatureRegistry

# --- モジュール固有の特徴量エンジンクマッピング ---
MODULE_ENGINE_MAP = {
    "semi": ("generate.features_2644t", "FeatureEngineDaily2644T"),
    "2644t": ("generate.features_2644t", "FeatureEngineDaily2644T"),
    "daily": ("generate.features_2644t", "FeatureEngineDaily2644T"),
}

def get_feature_engine_class(module_name: str):
    """
    モジュール名に基づいて適切な特徴量エンジンクラスを返す。
    例: 'daily_2644t' -> FeatureEngineDaily2644T
    見つからない場合は汎用 FeatureEngine を返す。
    """
    # モジュール名のプレフィックスを抽出 (例: 'daily_2644t' -> 'daily')
    module_prefix = module_name.split('_')[0]
    
    if module_prefix in MODULE_ENGINE_MAP:
        module_path, class_name = MODULE_ENGINE_MAP[module_prefix]
        try:
            engine_module = importlib.import_module(module_path)
            engine_class = getattr(engine_module, class_name)
            logging.info(f"✅ Using module-specific engine: {class_name}")
            return engine_class
        except (ImportError, AttributeError) as e:
            logging.warning(f"⚠️ Failed to load {class_name}: {e}. Falling back to generic FeatureEngine.")
    
    return FeatureEngine

def report_missing_values(df: pd.DataFrame, stage: str = ""):
    """Calculates and logs the missing value statistics in a clean format."""
    logging.info(f"--- Missing Value Report ({stage}) ---")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]
    if missing_stats.empty:
        logging.info("No missing values found. Excellent!")
        return

    total_rows = len(df)
    report_df = pd.DataFrame({
        'Missing Count': missing_stats,
        'Percentage': (missing_stats / total_rows * 100).round(2)
    }).sort_values(by='Percentage', ascending=False)

    logging.warning(f"Missing values detected in {stage}:") 
    for col, row in report_df.iterrows():
        logging.warning(f"  - Column '{col}': {row['Missing Count']} missing values ({row['Percentage']}%)")
    logging.info("--- End of Report ---")

def load_config(module_name: str):
    """
    指定されたモジュール名に基づいて設定ファイルを動的にインポートする
    例: module_name='daily_5' -> timescale_modules.daily_5.config_daily_5
    """
    # 2644.T専用 naming alias support (unified -> daily_2644t)
    module_alias_map = {
        'daily_2644t': 'daily_2644t',
    }
    module_lookup = module_alias_map.get(module_name, module_name)

    try:
        config_path = f"timescale_modules.{module_lookup}.config_{module_lookup}"
        config = importlib.import_module(config_path)
        logging.info(f"✅ Successfully loaded config for: {module_name} (resolved to {module_lookup})")
        return config
    except ImportError as e:
        logging.error(f"❌ Failed to import config module '{module_lookup}': {e}")
        logging.error(f"   Ensure that 'timescale_modules/{module_lookup}/config_{module_lookup}.py' exists.")
        return None

def run_pipeline_for_module(module_name: str):
    """Executes the feature generation pipeline for a single module."""
    logging.info(f"\n{'='*10} Processing Module: {module_name} {'='*10}")
    
    # --- 1. Load Dynamic Config ---
    config = load_config(module_name)
    if config is None:
        return

    # --- 2. Load Config Variables ---
    start_date = getattr(config, "DATA_START_DATE", "2020-01-01")
    
    yf_symbols = getattr(config, "YF_SYMBOLS", {})
    fred_symbols = getattr(config, "FRED_SYMBOLS", {}) 
    tiingo_symbols = getattr(config, "TIINGO_SYMBOLS", {})
    stooq_symbols = getattr(config, "STOOQ_SYMBOLS", {})
    
    yf_cache_name = getattr(config, "YF_CACHE_FILENAME", f"yf_data_{module_name}.parquet")
    fred_cache_name = getattr(config, "FRED_CACHE_FILENAME", f"fred_data_{module_name}.parquet")
    data_interval = getattr(config, "DATA_INTERVAL", "1d")
    
    # --- 3. Fetch Standard Data ---
    logging.info(f"[{module_name}] Fetching Standard Data from {start_date}...")
    df_yf = get_yfinance_data(yf_symbols, start_date, data_interval, yf_cache_name)
    df_fred = get_fred_data(fred_symbols, start_date, fred_cache_name) 
    df_tiingo = get_tiingo_data(tiingo_symbols, start_date, use_cache=True)
    df_stooq = get_stooq_data(stooq_symbols, start_date, use_cache=True)
    
    # --- 4. Fetch Alternative Data (NEW) ---
    df_alt = pd.DataFrame()
    if AlternativeDataScraper is not None:
        logging.info(f"[{module_name}] Fetching Alternative Data (Scraping)...")
        try:
            scraper = AlternativeDataScraper()
            df_alt = scraper.get_all_alternative_data()
            if not df_alt.empty:
                logging.info(f"✅ Alternative data fetched: {len(df_alt)} rows (TSMC Revenue, etc.)")
            else:
                logging.warning("⚠️ Alternative data is empty.")
        except Exception as e:
            logging.error(f"Failed to fetch alternative data: {e}")

    # --- 5. Merge All Data ---
    logging.info(f"[{module_name}] Merging and Preprocessing...")
    
    # df_alt をマージリストに追加
    data_frames = [df for df in [df_yf, df_fred, df_tiingo, df_stooq, df_alt] if not df.empty]
    
    if not data_frames:
        logging.error(f"[{module_name}] No data was fetched. Skipping.")
        return

    base_df = data_frames[0]
    for df_next in data_frames[1:]:
        # オルタナティブデータは日付が飛び飛び(月次)の可能性があるため、how='outer'で結合
        base_df = pd.merge(base_df, df_next, on='Date', how='outer')

    base_df['Date'] = pd.to_datetime(base_df['Date'])
    base_df = base_df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    base_df = apply_column_mapping(base_df, getattr(config, "COLUMN_MAP", {}))
    base_df = base_df.loc[:, ~base_df.columns.duplicated(keep='first')]
    
    # 営業日ベースの再インデックス
    date_range = pd.bdate_range(start=start_date, end=date.today().strftime("%Y-%m-%d"))
    base_df = base_df.set_index('Date').reindex(date_range).reset_index().rename(columns={'index':'Date'})
    
    # ★重要: オルタナティブデータ（月次）を日次に展開
    # ffill() することで、発表された最新の値を次の発表日まで維持する
    logging.info(f"[{module_name}] Applying ffill() to propagate monthly/sparse data...")
    base_df.ffill(inplace=True)
    
    # 先頭の欠損削除 (Look-ahead bias防止のため bfill はしない)
    original_len = len(base_df)
    base_df.dropna(inplace=True)
    dropped_len = original_len - len(base_df)
    
    if dropped_len > 0:
        logging.info(f"[{module_name}] Dropped {dropped_len} rows from the beginning (initial missing values).")
    
    if base_df.empty:
        logging.error(f"[{module_name}] All data was dropped after cleaning. Check if data sources cover the start date.")
        return

    report_missing_values(base_df, stage="Post-Merge & Clean")
    
    # --- 6. Feature Engineering ---
    logging.info(f"[{module_name}] Generating Features...")
    try:
        feature_params = getattr(config, "FEATURE_PARAMS", {})
        required_cols = feature_params.get("REQUIRED_COLUMNS", set())
        # オルタナティブデータは必須カラムではない場合が多いので、strict_validateには含めないか、
        # configで調整する。ここでは基本カラムのみチェック。
        strict_validate(base_df, required_cols)
    except RuntimeError as e:
        logging.error(f"[{module_name}] Validation failure: {e}")
        return

    EngineClass = get_feature_engine_class(module_name)
    feature_engine = EngineClass(config, base_df)
    df_features = feature_engine.run_all()
    
    df_features.dropna(inplace=True)
    logging.info(f"[{module_name}] Dropped NaN rows generated by feature engineering.")

    # --- 7. Target Creation & Selection ---
    horizon = getattr(config, "PREDICTION_HORIZON", 1)
    
    # ターゲット生成: 複数ターゲットをまとめて作る
    open_col = getattr(config, "OPEN_COL", None)
    high_col = getattr(config, "HIGH_COL", None)
    low_col = getattr(config, "LOW_COL", None)
    close_col = getattr(config, "CLOSE_COL", None)
    target_type = getattr(config, "TARGET_TYPE", "close_to_close")
    open_col = infer_open_column(df_features, open_col)

    if open_col and target_type in ["open_to_open", "open_to_close", "high_low", "close_to_close", "multi_target"]:
        logging.info(f"[{module_name}] Creating targets with target_type={target_type} using {open_col}")
        df_with_targets = create_multi_targets(
            df_features,
            [horizon],
            open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
        )
    else:
        # 従来の Close ベースターゲット（後方互換性）
        target_col_source = getattr(config, "TARGET_COLUMN", "")
        df_with_targets = create_targets(df_features, [horizon], target_col_source)
    
    registry = FeatureRegistry(
        getattr(config, "FEATURE_SETS", []), 
        getattr(config, "FEATURE_SETS_DEFINITIONS", {})
    )
    final_cols = registry.select_features(df_with_targets.columns)
    target_cols = [c for c in df_with_targets.columns if 'target_' in c]
    required_cols = ['Date'] + target_cols
    
    if hasattr(config, 'REGIME_VIX_COLUMN'):
        regime_col = config.REGIME_VIX_COLUMN
        if regime_col in df_with_targets.columns:
            required_cols.append(regime_col)
    
    output_cols = required_cols + [c for c in final_cols if c not in required_cols]
    output_cols = list(dict.fromkeys(output_cols))
    
    df_final = df_with_targets[output_cols].copy()
    prediction_target = f'target_{horizon}'
    
    if prediction_target not in df_final.columns:
        logging.error(f"[{module_name}] Target column '{prediction_target}' missing. Skipping save.")
        return
    
    df_final.dropna(subset=[prediction_target], inplace=True)

    if not df_final.empty:
        output_filename = getattr(config, "OUTPUT_FILENAME", f"features_{module_name}.parquet")
        output_path = PROJECT_ROOT / "data" / output_filename
        write_atomic(df_final, output_path)
        logging.info(f"✅ [{module_name}] SUCCESS: Saved to {output_path.name} ({len(df_final)} rows)")
    else:
        logging.warning(f"[{module_name}] No valid data generated.")

def main():
    """Daily data pipeline wrapper that supports multiple horizons/modules."""
    
    parser = argparse.ArgumentParser(description="Generate features for daily modules.")
    parser.add_argument(
        '--modules', 
        type=str, 
        default='daily_2644t',
        help="Comma-separated list of modules to run. Defaults to 'daily_2644t' (2644.T専用)。"
    )
    args = parser.parse_args()
    module_list = [m.strip() for m in args.modules.split(',') if m.strip()]

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    logging.info(f"====== Batch Feature Generation Started for: {module_list} ======")

    for module_name in module_list:
        try:
            run_pipeline_for_module(module_name)
        except Exception as e:
            logging.error(f"🚨 CRITICAL ERROR in [{module_name}]: {e}")
            import traceback
            traceback.print_exc()
            continue

    logging.info("====== All Requested Pipelines Completed ======")

if __name__ == "__main__":
    main()