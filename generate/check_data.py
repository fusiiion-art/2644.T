import pandas as pd
import sys
import argparse
import importlib
from pathlib import Path

def main():
    """
    指定されたモジュールのParquetファイルを読み込み、
    データの概要、Dateカラムの状態、オルタナティブデータの有無を表示する。
    """
    parser = argparse.ArgumentParser(description="Check features parquet file.")
    parser.add_argument(
        '--module',
        type=str,
        default='daily_2644t',
        help="Module name (default: daily_2644t)")
    args = parser.parse_args()
    
    module_name = args.module

    # --- 1. プロジェクトルートとファイルパスの設定 ---
    try:
        # PROJECT_ROOT/generate/check_data.py -> PROJECT_ROOT
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except (NameError, IndexError):
        PROJECT_ROOT = Path('.').resolve()
    
    sys.path.append(str(PROJECT_ROOT)) # モジュールインポート用

    # Config読み込み (OUTPUT_FILENAMEを取得するため)
    try:
        config_module = importlib.import_module(f"timescale_modules.{module_name}.config_{module_name}")
        output_filename = getattr(config_module, "OUTPUT_FILENAME", f"features_{module_name}.parquet")
        print(f"[INFO] Loaded config for {module_name}. Target file: {output_filename}")
    except ImportError as e:
        print(f"[ERROR] Could not load config for module '{module_name}': {e}")
        output_filename = f"features_{module_name}.parquet"
        print(f"[WARN] Falling back to default filename: {output_filename}")

    DATA_FILE_PATH = PROJECT_ROOT / "data" / output_filename

    # --- 2. Pandas 表示オプション設定 ---
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(f"\n====== データファイル [ {DATA_FILE_PATH.name} ] の内容チェック ======\n")

    # --- 3. ファイルの読み込み ---
    if not DATA_FILE_PATH.exists():
        print(f"[エラー] ファイルが見つかりません: {DATA_FILE_PATH}")
        return

    try:
        df = pd.read_parquet(DATA_FILE_PATH)
    except Exception as e:
        print(f"[エラー] ファイルの読み込みに失敗しました: {e}")
        return

    # --- 4. 必須項目チェック ---
    print("--- 0. 必須項目 (Date, Targets) チェック ---")
    
    # Dateチェック
    if 'Date' in df.columns:
        print("✅ 'Date' column exists.")
    else:
        print("❌ 'Date' column MISSING! (It might be in index?)")
        if isinstance(df.index, pd.DatetimeIndex):
             print("   -> Date appears to be in the INDEX.")
    
    # Targetチェック
    target_cols = [c for c in df.columns if 'target_' in c]
    if target_cols:
        print(f"✅ Found {len(target_cols)} target columns: {target_cols}")
    else:
        print("❌ NO target columns found!")

    # Alternative Dataチェック
    print("\n--- 0.1 オルタナティブデータ (TSMC) チェック ---")
    tsmc_cols = [c for c in df.columns if 'tsmc_' in c]
    if tsmc_cols:
         print(f"✅ Found {len(tsmc_cols)} TSMC columns: {tsmc_cols}")
         # 欠損チェック
         tsmc_nulls = df[tsmc_cols].isnull().sum().sum()
         if tsmc_nulls == 0:
             print("   -> No missing values in TSMC data.")
         else:
             print(f"   -> ⚠️ Found {tsmc_nulls} missing values in TSMC data.")
    else:
        print("❌ NO TSMC columns found (Alternative Data missing or failed to merge).")


    # --- 5. データ概要の表示 ---
    print("\n--- 1. データフレームの基本情報 (df.info()) ---")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    print("\n--- 2. データの先頭 5行 (df.tail()) ---")
    print(df.tail())

    # --- 6. 欠損値 (NaN) レポート ---
    print("\n--- 3. 欠損値 (NaN) レポート (df.isnull().sum()) ---")
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]

    if missing_stats.empty:
        print("\n[結果] 素晴らしい！ 最終データに欠損値は検出されませんでした。")
    else:
        total_rows = len(df)
        report_df = pd.DataFrame({
            'Missing Count (NaN)': missing_stats,
            'Percentage (%)': (missing_stats / total_rows * 100).round(2)
        }).sort_values(by='Percentage (%)', ascending=False)
        
        print("\n[結果] 警告: 以下のカラムで欠損値が検出されました:")
        print(report_df)
    
    print("\n====== チェック完了 ======")

if __name__ == "__main__":
    main()
