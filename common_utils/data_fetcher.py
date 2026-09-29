"""
共通データ取得関数
YahooFinance, FRED, Tiingo, Stooq からのデータ取得を提供
"""
import os
import pandas as pd
from pathlib import Path
import logging
import time
import ssl
import warnings
import sys
from datetime import date, datetime, timedelta
from functools import wraps
import requests
import re

# --- Library Imports ---
try:
    import yfinance as yf
    from fredapi import Fred
except ImportError as e:
    logging.error(f"Required libraries are not installed: {e}")
    sys.exit(1)

# --- API Key Configuration ---
FRED_API_KEY = "8a3d39fae2aa6cadb793291d22458e0d"
TIINGO_API_KEY = "YOUR_TIINGO_API_KEY_HERE"

# --- Project Path Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except (NameError, IndexError):
    PROJECT_ROOT = Path('.').resolve()

CACHE_DATA_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

warnings.simplefilter(action='ignore', category=FutureWarning)
ssl._create_default_https_context = ssl._create_unverified_context


def retry(tries=3, delay=5, backoff=2):
    """リトライデコレータ"""
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Connection error in '{f.__name__}'. Retrying...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
                except Exception as e:
                    logging.error(f"Unexpected error in '{f.__name__}': {e}", exc_info=True)
                    raise
            return f(*args, **kwargs)
        return f_retry
    return deco_retry


def write_atomic(df: pd.DataFrame, path: Path):
    """アトミック書き込み"""
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
        logging.info(f"Successfully wrote file: {path}")
    except Exception as e:
        if tmp_path.exists():
            os.remove(tmp_path)
        raise


@retry()
def get_yfinance_data(symbols: dict, start_date: str, interval: str, cache_filename: str) -> pd.DataFrame:
    """Yahoo Finance からデータ取得"""
    if not symbols:
        return pd.DataFrame()
    cache_file = CACHE_DATA_DIR / cache_filename
    
    df_existing = pd.DataFrame()
    fetch_start_date = pd.to_datetime(start_date).normalize()

    if cache_file.exists():
        try:
            df_existing = pd.read_parquet(cache_file)
            if 'Date' in df_existing.columns and not df_existing.empty:
                df_existing['Date'] = pd.to_datetime(df_existing['Date']).dt.tz_localize(None)
                last_cached_date = df_existing['Date'].max()
                fetch_start_date = (pd.to_datetime(last_cached_date).normalize() + timedelta(days=1))
            else:
                raise ValueError("Cache file is invalid.")
        except Exception as e:
            logging.warning(f"Cache file {cache_file} is corrupted. Refetching all data. Error: {e}")
            df_existing = pd.DataFrame()
            if cache_file.exists():
                cache_file.unlink()

    today_utc = pd.Timestamp.utcnow().tz_convert(None).normalize()

    if isinstance(fetch_start_date, pd.Timestamp) and fetch_start_date.tz is not None:
        fetch_start_date = fetch_start_date.tz_convert(None)
    fetch_start_date = pd.to_datetime(fetch_start_date).normalize()

    if fetch_start_date >= today_utc:
        adjusted = today_utc - timedelta(days=1)
        logging.info(f"Adjusted fetch_start_date from {fetch_start_date.date()} to {adjusted.date()} to avoid start > end.")
        fetch_start_date = adjusted

    if not df_existing.empty:
        last_cached_date = pd.to_datetime(df_existing['Date'].max()).normalize()
        if fetch_start_date <= last_cached_date:
            logging.info("No new yfinance data to fetch after cache check.")
            # キャッシュクリーンアップ: 現在のシンボルに関連しない古い列を削除
            sanitized_tickers = [re.escape(ticker) for ticker in symbols.values()]
            pattern = re.compile(f"^(Date|{'|'.join([f'{t}_' for t in sanitized_tickers])})", re.IGNORECASE)
            cols_to_keep = [col for col in df_existing.columns if pattern.match(col)]
            cols_dropped = [col for col in df_existing.columns if not pattern.match(col)]
            if cols_dropped:
                logging.info(f"YF Cache Cleanup: Dropping {len(cols_dropped)} old columns. Example: {cols_dropped[:3]}")
                df_existing = df_existing[cols_to_keep]
                write_atomic(df_existing, cache_file)
            return df_existing

    logging.info(f"Fetching YFinance data for {len(symbols)} symbols from {fetch_start_date.strftime('%Y-%m-%d')}...")

    start_str = fetch_start_date.strftime('%Y-%m-%d')
    try:
        df_new = yf.download(list(symbols.values()), start=start_str, interval=interval, auto_adjust=False, progress=False)
    except Exception as e:
        logging.error(f"Failed to download from yfinance: {e}")
        return df_existing

    if df_new.empty:
        logging.info("No new data was returned from YFinance.")
        return df_existing

    if isinstance(df_new.columns, pd.MultiIndex):
        df_new.columns = [f"{ticker}_{col.lower().replace(' ', '_')}" for col, ticker in df_new.columns]
    else:
        if len(symbols) == 1:
            ticker = list(symbols.values())[0]
            df_new.columns = [f"{ticker}_{col.lower().replace(' ', '_')}" for col in df_new.columns]
        else:
            df_new.columns = [str(col).lower().replace(' ', '_') for col in df_new.columns]

    df_new.reset_index(inplace=True)
    df_new['Date'] = pd.to_datetime(df_new['Date']).dt.tz_localize(None)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    
    # キャッシュクリーンアップ: 現在のシンボルに関連しない古い列を削除
    sanitized_tickers = [re.escape(ticker) for ticker in symbols.values()]
    pattern = re.compile(f"^(Date|{'|'.join([f'{t}_' for t in sanitized_tickers])})", re.IGNORECASE)
    cols_to_keep = [col for col in df_combined.columns if pattern.match(col)]
    cols_dropped = [col for col in df_combined.columns if not pattern.match(col)]
    if cols_dropped:
        logging.info(f"YF Cache Cleanup: Dropping {len(cols_dropped)} old columns. Example: {cols_dropped[:3]}")
        df_combined = df_combined[cols_to_keep]
    
    write_atomic(df_combined, cache_file)
    return df_combined


@retry()
def get_fred_data(series: dict, start_date: str, cache_filename: str) -> pd.DataFrame:
    """FRED からデータ取得"""
    if not series:
        return pd.DataFrame()
    cache_file = CACHE_DATA_DIR / cache_filename

    if cache_file.exists():
        try:
            df_cache = pd.read_parquet(cache_file)
            if 'Date' in df_cache.columns and not df_cache.empty:
                df_cache['Date'] = pd.to_datetime(df_cache['Date'])
                if df_cache['Date'].max().date() >= date.today() - timedelta(days=2):
                    logging.info(f"FRED cache is recent. Using data from {cache_file}")
                    return df_cache
        except Exception as e:
            logging.warning(f"Cache file {cache_file} is corrupted. Refetching. Error: {e}")
            if cache_file.exists():
                cache_file.unlink()
            
    logging.info("Fetching fresh data from FRED (full history)...")
    fred = Fred(api_key=FRED_API_KEY)
    df_list = [fred.get_series(sid, observation_start=start_date).rename(name) for name, sid in series.items()]
    if not df_list:
        return pd.DataFrame()
    
    df_fred = pd.concat(df_list, axis=1)
    df_fred = df_fred.reset_index().rename(columns={'index': 'Date'})
    df_fred['Date'] = pd.to_datetime(df_fred['Date']).dt.tz_localize(None)
    write_atomic(df_fred, cache_file)
    return df_fred


@retry()
def get_tiingo_data(symbols: dict, start_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Tiingo からデータ取得"""
    try:
        from tiingo import TiingoClient
    except ImportError:
        logging.warning("tiingo library not installed")
        return pd.DataFrame()
    
    if not symbols or not TIINGO_API_KEY or "YOUR_TIINGO" in TIINGO_API_KEY:
        return pd.DataFrame()
    logging.info(f"Fetching {len(symbols)} symbols from Tiingo...")
    client = TiingoClient({'api_key': TIINGO_API_KEY, 'session': True})
    df_list = []
    for name, ticker in symbols.items():
        try:
            df_ticker = client.get_dataframe(ticker, startDate=start_date, frequency='daily')
            df_ticker.index = df_ticker.index.tz_localize(None)
            df_ticker.columns = [re.sub(r'adj(Close|Open|High|Low|Volume)$', r'adj_\1', col).lower() for col in df_ticker.columns]
            df_list.append(df_ticker.add_prefix(f"{name}_"))
        except Exception as e:
            logging.error(f"Failed to fetch {ticker} from Tiingo: {e}")
    if not df_list:
        return pd.DataFrame()
    
    df_concat = pd.concat(df_list, axis=1).reset_index()
    return df_concat.rename(columns={'date': 'Date'})


@retry()
def get_stooq_data(symbols: dict, start_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Stooq からデータ取得"""
    if not symbols:
        return pd.DataFrame()
    logging.info(f"Fetching {len(symbols)} symbols from Stooq...")
    base_url = "https://stooq.com/q/d/l/"
    start_date_fmt = pd.to_datetime(start_date).strftime('%Y%m%d')
    df_list = []
    for name, ticker in symbols.items():
        url = f"{base_url}?s={ticker.replace('^', '.')}&d1={start_date_fmt}&i=d"
        try:
            df_ticker = pd.read_csv(url)
            if 'Date' not in df_ticker.columns:
                continue
            df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
            df_ticker = df_ticker.add_prefix(f"{name}_")
            df_list.append(df_ticker)
        except Exception as e:
            logging.error(f"Failed to fetch {ticker} from Stooq: {e}")

    if not df_list:
        return pd.DataFrame()
    
    base_df = None
    for df_item in df_list:
        date_col_original = df_item.filter(like='_Date').columns[0]
        df_item = df_item.rename(columns={date_col_original: 'Date'})
        if base_df is None:
            base_df = df_item
        else:
            base_df = pd.merge(base_df, df_item, on='Date', how='outer')
            
    return base_df


def strict_validate(df: pd.DataFrame, required_columns: set):
    """データ検証"""
    # 1. Column Existence
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"Validation Failed: Required columns missing: {list(missing_cols)}")
    
    # 2. Duplicate Dates
    if 'Date' in df.columns:
        if df['Date'].duplicated().any():
            raise RuntimeError("Validation Failed: Duplicate dates detected in 'Date' column.")
        
        # 3. Monotonicity
        if not df['Date'].is_monotonic_increasing:
            raise RuntimeError("Validation Failed: Dates are not strictly increasing.")
             
        # 4. Timezone check
        if pd.api.types.is_datetime64_any_dtype(df['Date']) and df['Date'].dt.tz is not None:
            raise RuntimeError("Validation Failed: 'Date' column must be timezone-naive.")

    # 5. NaN Check
    nan_rows = df[list(required_columns)].isnull().any(axis=1).sum()
    if nan_rows > 0:
        logging.warning(f"Found {nan_rows} rows with NaN in required columns post-fill.")
    
    logging.info("Data validation successful.")


def apply_column_mapping(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """カラム名マッピング"""
    rename_dict = {}
    sorted_map = sorted(column_map.items(), key=lambda item: len(item[0]), reverse=True)

    for original_col in df.columns:
        if original_col == 'Date':
            continue
        original_col_lower = str(original_col).lower()
        original_col_lower = re.sub(r'adjclose$', '_adj_close', original_col_lower)
        original_col_lower = re.sub(r'(?<!_)(open|high|low|close|volume)$', r'_\1', original_col_lower)

        for keyword, new_name in sorted_map:
            if keyword.lower() in original_col_lower:
                if any(s in new_name for s in ['_nav', '_open', '_high', '_low', '_close']):
                    rename_dict[original_col] = new_name
                else: 
                    suffix = original_col_lower.replace(keyword.lower(), '').replace(' ', '_')
                    rename_dict[original_col] = f"{new_name.lower()}{suffix}"
                break 
                
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df
