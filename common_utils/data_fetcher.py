"""
共通データ取得関数
YahooFinance, FRED, Tiingo, Stooq からのデータ取得を提供
"""
import os
import pandas as pd
from pathlib import Path
import logging
import time
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

# --- Project Path Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except (NameError, IndexError):
    PROJECT_ROOT = Path('.').resolve()

CACHE_DATA_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- API Key Configuration ---
# 秘密情報はソースに書かない（監査 G-11）。環境変数、または git 管理外の semi2644/config/secrets.env（KEY=VALUE 形式）から読む。
SECRETS_ENV_PATH = PROJECT_ROOT / "semi2644" / "config" / "secrets.env"


def _read_secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if SECRETS_ENV_PATH.exists():
        for line in SECRETS_ENV_PATH.read_text(encoding="utf-8").splitlines():
            key, sep, val = line.partition("=")
            if sep and key.strip() == name:
                return val.strip().strip('"').strip("'") or None
    return None


FRED_API_KEY = _read_secret("FRED_API_KEY")
TIINGO_API_KEY = _read_secret("TIINGO_API_KEY")

warnings.simplefilter(action='ignore', category=FutureWarning)


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


def _naive_dates(values) -> pd.Series:
    d = pd.to_datetime(pd.Series(values))
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize()


def _read_yf_cache(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(cache_file)
        if 'Date' not in df.columns or df.empty:
            raise ValueError("Cache file is invalid.")
        df['Date'] = _naive_dates(df['Date']).to_numpy()
        return df.sort_values('Date').reset_index(drop=True)
    except Exception as e:
        logging.warning(f"Cache file {cache_file} is corrupted. Refetching all data. Error: {e}")
        cache_file.unlink(missing_ok=True)
        return pd.DataFrame()


def _download_yf(symbols: dict, start: pd.Timestamp, interval: str) -> pd.DataFrame:
    """yfinance から取得し、列名を '<ticker>_<field>' に揃える。失敗・空なら空の DataFrame。"""
    logging.info(f"Fetching YFinance data for {len(symbols)} symbols from {start.strftime('%Y-%m-%d')}...")
    try:
        df_new = yf.download(list(symbols.values()), start=start.strftime('%Y-%m-%d'), interval=interval,
                             auto_adjust=False, progress=False)
    except Exception as e:
        logging.error(f"Failed to download from yfinance: {e}")
        return pd.DataFrame()
    if df_new is None or df_new.empty:
        return pd.DataFrame()
    if isinstance(df_new.columns, pd.MultiIndex):
        df_new.columns = [f"{ticker}_{col.lower().replace(' ', '_')}" for col, ticker in df_new.columns]
    elif len(symbols) == 1:
        ticker = list(symbols.values())[0]
        df_new.columns = [f"{ticker}_{col.lower().replace(' ', '_')}" for col in df_new.columns]
    else:
        df_new.columns = [str(col).lower().replace(' ', '_') for col in df_new.columns]
    df_new = df_new.reset_index()
    df_new['Date'] = _naive_dates(df_new['Date']).to_numpy()
    return df_new


def _revised_columns(existing: pd.DataFrame, new: pd.DataFrame, exclude_date, rtol: float) -> list:
    """重なり期間（最終キャッシュ日を除く）で close / adj_close が変わった列。分割・配当の遡及調整の検知用。"""
    common = sorted((set(existing['Date']) & set(new['Date'])) - {exclude_date})
    cols = [c for c in new.columns if c.endswith(('_close', '_adj_close')) and c in existing.columns]
    if not common or not cols:
        return []
    old = existing.set_index('Date').loc[common, cols].astype(float)
    cur = new.set_index('Date').loc[common, cols].astype(float)
    rel = ((cur - old).abs() / old.abs()).where(old.notna() & cur.notna())
    return [c for c in cols if (rel[c] > rtol).any()]


def _keep_symbol_columns(df: pd.DataFrame, symbols: dict) -> pd.DataFrame:
    """キャッシュクリーンアップ: 現在のシンボルに関連しない古い列を削除"""
    sanitized_tickers = [re.escape(ticker) for ticker in symbols.values()]
    pattern = re.compile(f"^(Date|{'|'.join([f'{t}_' for t in sanitized_tickers])})", re.IGNORECASE)
    cols_dropped = [col for col in df.columns if not pattern.match(col)]
    if cols_dropped:
        logging.info(f"YF Cache Cleanup: Dropping {len(cols_dropped)} old columns. Example: {cols_dropped[:3]}")
    return df[[col for col in df.columns if pattern.match(col)]]


@retry()
def get_yfinance_data(symbols: dict, start_date: str, interval: str, cache_filename: str,
                      overlap_days: int | None = None, revision_rtol: float | None = None) -> pd.DataFrame:
    """Yahoo Finance からデータ取得（差分キャッシュつき）。

    - 毎回、最終キャッシュ日から overlap_days 日さかのぼって取り直し、重なる行は新しい値で置き換える
      （場中に取った途中足が残らない）。
    - 重なり期間の過去値が revision_rtol を超えて変わっていたら、取得元が分割・配当を遡及調整したとみなし、
      全期間を取り直して新旧を混ぜない（監査 B）。
    既定値は semi2644/config/config.yaml の data 節。
    """
    if not symbols:
        return pd.DataFrame()
    if overlap_days is None or revision_rtol is None:
        from data.adjust import load_price_config
        data_cfg = load_price_config()["data"]
        overlap_days = data_cfg["overlap_days"] if overlap_days is None else overlap_days
        revision_rtol = data_cfg["revision_rtol"] if revision_rtol is None else revision_rtol
    cache_file = CACHE_DATA_DIR / cache_filename
    start = pd.Timestamp(start_date).normalize()
    existing = _read_yf_cache(cache_file)

    if existing.empty:
        combined = _download_yf(symbols, start, interval)
    else:
        last_cached = existing['Date'].max()
        new = _download_yf(symbols, max(start, last_cached - pd.Timedelta(days=int(overlap_days))), interval)
        if new.empty:
            logging.info("No new data was returned from YFinance.")
            combined = existing
        else:
            revised = _revised_columns(existing, new, last_cached, float(revision_rtol))
            if revised:
                logging.warning(f"Cached history was revised upstream (e.g. split/dividend adjustment): {revised[:5]}. "
                                f"Refetching the full history.")
                full = _download_yf(symbols, start, interval)
                if full.empty:
                    raise RuntimeError("過去値の改定を検知したが全期間の取り直しに失敗。キャッシュは更新しない。")
                combined = full
            else:
                combined = pd.concat([existing, new], ignore_index=True)
                combined = combined.drop_duplicates(subset=['Date'], keep='last')

    if combined.empty:
        return combined
    combined = combined.sort_values('Date').reset_index(drop=True)
    combined = _keep_symbol_columns(combined, symbols)
    write_atomic(combined, cache_file)
    return combined


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
            
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY が未設定です。環境変数か semi2644/config/secrets.env に設定してください。")
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
    
    if not symbols or not TIINGO_API_KEY:
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
