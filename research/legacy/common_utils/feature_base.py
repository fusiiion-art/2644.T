"""
共通特徴量エンジン基底クラス
モジュール固有の特徴量は継承して追加
"""
import pandas as pd
import numpy as np
import logging
import re
import joblib
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# pandas_ta をインポート
try:
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
    import pandas_ta as ta
    import jpholiday
except ImportError as e:
    logging.error(f"Required libraries not installed: {e}")


class FeatureEngineBase:
    """
    共通特徴量エンジン基底クラス
    モジュール固有の特徴量は継承して _add_module_specific_features() をオーバーライド
    """
    
    def __init__(self, config=None, df: Optional[pd.DataFrame] = None):
        self.df = df.copy() if df is not None else None
        self.config = config
        self.params: Dict = getattr(config, 'FEATURE_PARAMS', {})
        self.horizon: int = getattr(config, 'PREDICTION_HORIZON', 1)
        self.main_asset_price_col: str = getattr(config, 'MAIN_ASSET', "")
        self.main_asset_return_col: str = getattr(config, 'TARGET_COLUMN', "")
        self.main_asset_prefix: str = re.sub(r'(_nav|_close|_adj_close)$', '', self.main_asset_price_col)
        self.price_cols_map: Dict[str, str] = {}
        self.params_store: Dict[str, Any] = {}

    def save_params(self, path: Path):
        """正規化パラメータを保存"""
        joblib.dump(self.params_store, path)

    def load_params(self, path: Path):
        """正規化パラメータを読み込み"""
        if not path.exists():
            raise FileNotFoundError(f"Parameter file not found: {path}")
        self.params_store = joblib.load(path)
        logging.info(f"Loaded feature engine params from {path}")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習モード: パラメータをフィットして変換"""
        self.df = df.copy()
        return self.run_all(mode='train')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """推論モード: 保存済みパラメータで変換"""
        self.df = df.copy()
        if not self.params_store and not getattr(self.config, 'SKIP_NORMALIZATION_CHECK', False):
            raise RuntimeError("params_store is empty. Call load_params() first.")
        return self.run_all(mode='inference')

    def run_all(self, mode: str = 'train') -> pd.DataFrame:
        """全特徴量を計算"""
        logging.info(f"Starting feature engineering for: {self.main_asset_price_col} (Mode: {mode})")
        
        # 1. タイムゾーン補正
        self._apply_timezone_lag()
        
        # 2. 基本特徴量
        self._prepare_base_returns()
        self._add_exogenous_features()
        self._add_main_asset_technicals()
        self._add_statistical_features()    
        self._add_intermarket_features()   
        self._add_smoothing_features()      
        self._add_volatility_features()     
        self._add_structural_features()     
        self._add_lag_features()
        self._add_calendar_features()
        self._add_suggested_features()
        
        # 3. モジュール固有特徴量（継承先でオーバーライド）
        self._add_module_specific_features()
        
        # 4. 正規化
        self._normalize_features(mode=mode)
        
        # 5. 欠損値処理
        self.df.bfill(inplace=True)
        self.df.ffill(inplace=True)
        self._add_time_index()
        
        logging.info("Feature engineering complete.")
        return self.df

    def _add_module_specific_features(self):
        """モジュール固有の特徴量（継承先でオーバーライド）"""
        pass

    def _apply_timezone_lag(self):
        """米国/欧州データのタイムゾーン補正（1日ラグ）"""
        lag_targets = [
            'sp500', 'nasdaq', 'sox', 'nvda', 'vix', 'vxn', 'us_10y', 'us_2y',
            'vt', 'crude_oil', 'gold', 'eur_usd', 'dollar_index', 'apple', 'google',
            'aapl', 'msft', 'tsm', 'avgo', 'asml', 'amd', 'txn', 'qcom', 'intc', 'mu', 'amat'
        ]
        
        cols_to_shift = []
        for col in self.df.columns:
            if any(t in col for t in lag_targets) and 'target' not in col and 'Date' not in col:
                cols_to_shift.append(col)
        
        if cols_to_shift:
            logging.info(f"Applying Timezone Shift (Lag-1) to {len(cols_to_shift)} columns.")
            self.df[cols_to_shift] = self.df[cols_to_shift].shift(1)

    def _prepare_base_returns(self):
        """基本リターン計算"""
        price_cols_map = {}
        prefix_map = {col: re.match(r'(.+?)(_nav|_close|_adj_close)$', col) for col in self.df.columns}
        valid_price_cols = {col: match.group(1) for col, match in prefix_map.items() if match}

        new_cols = {}
        for col, prefix in valid_price_cols.items():
            if f"{prefix}_adj_close" in price_cols_map.values():
                continue
            if col.endswith("_adj_close"):
                price_cols_map[prefix] = col
            elif col.endswith("_close") and price_cols_map.get(prefix) is None:
                price_cols_map[prefix] = col
            elif col.endswith("_nav") and price_cols_map.get(prefix) is None:
                price_cols_map[prefix] = col

        self.price_cols_map = price_cols_map
        
        for price_col in self.price_cols_map.values():
            return_col = f"{price_col}_return"
            price = self.df[price_col]
            prev_price = self.df[price_col].shift(1)
            conditions = (price > 0) & (prev_price > 0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                new_cols[return_col] = np.where(conditions, np.log(price / prev_price), np.nan)

        if new_cols:
            new_df = pd.DataFrame(new_cols, index=self.df.index)
            self.df = pd.concat([self.df, new_df], axis=1)

    def _add_exogenous_features(self):
        """外生変数特徴量"""
        if 'us_10y' in self.df.columns and 'us_2y' in self.df.columns:
            self.df['us_yield_spread'] = self.df['us_10y'] - self.df['us_2y']
        if 'vix_close' in self.df.columns: self.df['vix_close_diff'] = self.df['vix_close'].diff()
        if 'dollar_index' in self.df.columns: self.df['dollar_index_diff'] = self.df['dollar_index'].diff()
        if 'credit_spread' in self.df.columns: self.df['credit_spread_diff'] = self.df['credit_spread'].diff()
        if 'expected_inflation' in self.df.columns: self.df['expected_inflation_diff'] = self.df['expected_inflation'].diff()
        if 'vvix_close' in self.df.columns: self.df['vvix_close_diff'] = self.df['vvix_close'].diff()
        if 'pc_ratio_close' in self.df.columns: self.df['pc_ratio_close_diff'] = self.df['pc_ratio_close'].diff()

    def _add_main_asset_technicals(self):
        """メインアセットのテクニカル指標"""
        if self.main_asset_price_col not in self.df.columns:
            return
        price = self.df[self.main_asset_price_col]
        prefix = self.main_asset_prefix
        
        # 25日移動平均乖離率
        if len(price.dropna()) >= self.params.get("ma_window", 25):
            self.df[f'{prefix}_ma_25'] = self.df.ta.sma(close=price, length=self.params.get("ma_window", 25))
            self.df[f'{prefix}_ma_25_dev'] = np.where(
                self.df[f'{prefix}_ma_25'] > 0,
                price / (self.df[f'{prefix}_ma_25'] + 1e-9) - 1,
                np.nan
            )

        # ATR / Close 比率
        if f"{prefix}_high" in self.df.columns and f"{prefix}_low" in self.df.columns:
            self.df['ATR_14'] = self.df.ta.atr(
                high=self.df[f'{prefix}_high'],
                low=self.df[f'{prefix}_low'],
                close=price,
                length=self.params.get("atr_window", 14)
            )
            self.df['ATR_Ratio'] = self.df['ATR_14'] / (price + 1e-9)

        # RSI
        self.df.ta.rsi(close=price, length=self.params.get("rsi_window", 14), 
                       append=True, col_names=('RSI_14',))
        
        # MACD
        self.df.ta.macd(close=price, fast=self.params.get("macd_fast", 12), slow=self.params.get("macd_slow", 26), signal=self.params.get("macd_signal", 9), append=True, col_names=(f'{prefix}_macd', f'{prefix}_macd_hist', f'{prefix}_macd_signal'))
        
        # Bollinger Bands
        self.df.ta.bbands(close=price, length=self.params.get("bb_window", 20), std=self.params.get("bb_std", 2), append=True, col_names=(f'{prefix}_bb_lower', f'{prefix}_bb_mid', f'{prefix}_bb_upper', f'{prefix}_bb_bw', f'{prefix}_bb_pct'))

        high_col, low_col, close_col = f"{prefix}_high", f"{prefix}_low", f"{prefix}_close"
        if all(c in self.df.columns for c in [high_col, low_col, close_col]):
            self.df.ta.adx(high=self.df[high_col], low=self.df[low_col], close=self.df[close_col], length=self.params.get("adx_window", 14), append=True, col_names=(f'{prefix}_adx', f'{prefix}_adx_dmp', f'{prefix}_adx_dmn', f'{prefix}_adx_unused'))
            
            k_len, d_len, smooth_k = 14, 3, 3
            self.df.ta.stoch(high=self.df[high_col], low=self.df[low_col], close=self.df[close_col], k=k_len, d=d_len, smooth_k=smooth_k, append=True, col_names=(f'STOCHk_{k_len}_{d_len}_{smooth_k}', f'STOCHd_{k_len}_{d_len}_{smooth_k}', f'STOCHh_{k_len}_{d_len}_{smooth_k}'))

    def _add_statistical_features(self):
        """統計的特徴量"""
        if self.main_asset_return_col not in self.df.columns: return
        ret = self.df[self.main_asset_return_col]
        for w in self.params.get('statistical_windows', []):
            self.df[f'main_ret_skew_{w}d'] = ret.rolling(window=w, min_periods=w//2).skew()
            self.df[f'main_ret_kurt_{w}d'] = ret.rolling(window=w, min_periods=w//2).kurt()

    def get_ret_col_by_prefix(self, prefix):
        return self.price_cols_map.get(prefix, f"{prefix}_adj_close") + "_return"

    def _add_intermarket_features(self):
        """市場間特徴量"""
        pairs = self.params.get('intermarket_pairs', [])
        corr_windows = self.params.get('intermarket_corr_windows', [])
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            p1, p2 = pair[0], pair[1]
            c1, c2 = self.get_ret_col_by_prefix(p1), self.get_ret_col_by_prefix(p2)
            if c1 in self.df.columns and c2 in self.df.columns:
                for w in corr_windows:
                    self.df[f'corr_{p1}_{p2}_{w}d'] = self.df[c1].rolling(w).corr(self.df[c2])
                    
    def _add_smoothing_features(self):
        """平滑化特徴量"""
        if self.main_asset_return_col in self.df.columns:
            ret = self.df[self.main_asset_return_col]
            for k in self.params.get('rolling_ret_windows', []): 
                self.df[f'rolling_mean_ret_{k}'] = ret.rolling(window=k).mean()
            for span in self.params.get('ema_spans', []): 
                self.df[f'ema_return_{span}'] = ret.ewm(span=span, adjust=False).mean()

    def _add_volatility_features(self):
        """ボラティリティ特徴量"""
        if self.main_asset_return_col in self.df.columns:
            ret = self.df[self.main_asset_return_col]
            for n in self.params.get('vol_windows', [20]):
                self.df[f'realized_vol_{n}'] = ret.rolling(window=n).std() * np.sqrt(252)
                
        if 'vix_close' in self.df.columns:
            vix_ret = self.df['vix_close'].pct_change()
            self.df['vix_volatility'] = vix_ret.rolling(20).std()

    def _add_structural_features(self):    
        """構造的特徴量"""
        if 'effr' in self.df.columns: self.df['DFF'] = self.df['effr']
        if self.main_asset_price_col in self.df.columns:
            price = self.df[self.main_asset_price_col]
            ema_trend = price.ewm(span=self.params.get('trend_window', 30), adjust=False).mean()
            self.df['trend_component'] = price - ema_trend
            
        vix_col = self.params.get('REGIME_VIX_COLUMN', 'vix_close') 
        if hasattr(self.config, 'REGIME_VIX_THRESHOLD') and vix_col in self.df.columns:
             threshold = getattr(self.config, 'REGIME_VIX_THRESHOLD')
             self.df['regime'] = (self.df[vix_col] > threshold).astype(int)

    def _add_lag_features(self):
        """ラグ特徴量"""
        return_cols = self.df.filter(like='_return').columns
        lag_periods = self.params.get('lag_periods', [1, 2])
        for col in return_cols:
            for p in lag_periods:
                self.df[f'{col}_lag{p}'] = self.df[col].shift(p)

    def _add_calendar_features(self):
        """カレンダー特徴量"""
        idx = pd.to_datetime(self.df['Date'])
        self.df['day_of_week'] = idx.dt.dayofweek
        self.df['month'] = idx.dt.month
        self.df['is_holiday_jp'] = idx.dt.date.astype('O').apply(jpholiday.is_holiday).astype(int)
        
    def _add_suggested_features(self):
        """複合的/提案特徴量"""
        epsilon = 1e-10
        if 'STOCHk_14_3_3' in self.df.columns and 'STOCHd_14_3_3' in self.df.columns:
            self.df['STOCHk_14_3_3_div_STOCHd_14_3_3'] = (self.df['STOCHk_14_3_3'] / (self.df['STOCHd_14_3_3'] + epsilon))

        if 'corr_VT_CL_F' in self.df.columns and 'vix_volatility' in self.df.columns:
            self.df['corr_VT_CL_F_div_vix_volatility'] = (self.df['corr_VT_CL_F'] / (self.df['vix_volatility'] + epsilon))

        if 'DFF' in self.df.columns and 'RSI_14' in self.df.columns:
            self.df['DFF_div_RSI_14'] = (self.df['DFF'] / (self.df['RSI_14'] + epsilon))

        if 'vix_volatility' in self.df.columns and 'RSI_14' in self.df.columns:
            self.df['vix_volatility_div_RSI_14'] = (self.df['vix_volatility'] / (self.df['RSI_14'] + epsilon))
        
        if 'credit_spread_diff' in self.df.columns and 'vix_close_diff' in self.df.columns:
            self.df['credit_spread_div_vix_diff'] = (self.df['credit_spread_diff'] / (self.df['vix_close_diff'] + epsilon))

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _add_time_index(self):
        """時間インデックス"""
        self.df['time_idx'] = (self.df['Date'] - self.df['Date'].min()).dt.days

    def _normalize_features(self, mode: str = 'train'):
        """特徴量の正規化（RobustScaler）"""
        exclude = ['Date', self.main_asset_price_col, self.main_asset_return_col, 'time_idx'] + \
                  [c for c in self.df.columns if 'target' in c or 'regime' in c] + \
                  ['day_of_week', 'month', 'is_holiday_jp']
        
        price_patterns = ['_open', '_close', '_high', '_low', '_adj_close', '_nav', '_volume']
        for col in self.df.columns:
            if any(pat in col for pat in price_patterns) and col not in exclude:
                exclude.append(col)
        
        cols_to_norm = [c for c in self.df.select_dtypes(include=np.number).columns if c not in exclude]
        
        if mode == 'train':
            stats = {}
            for c in cols_to_norm:
                series = self.df[c].dropna()
                if series.empty:
                    median, iqr = 0.0, 1.0
                else:
                    median = series.median()
                    q75, q25 = series.quantile(0.75), series.quantile(0.25)
                    iqr = q75 - q25
                    if iqr == 0:
                        iqr = 1.0
                
                stats[c] = {'median': float(median), 'iqr': float(iqr)}
                self.df[c] = (self.df[c] - median) / (iqr + 1e-9)
                self.df[c] = self.df[c].clip(-10, 10)
            
            self.params_store['normalization'] = stats
            logging.info(f"Fitted normalization stats for {len(stats)} columns.")
        else:
            if 'normalization' not in self.params_store:
                logging.warning("Normalization params missing! Skipping.")
                return
            
            stats = self.params_store['normalization']
            for c in cols_to_norm:
                if c in stats:
                    median, iqr = stats[c]['median'], stats[c]['iqr']
                    self.df[c] = (self.df[c] - median) / (iqr + 1e-9)
                    self.df[c] = self.df[c].clip(-10, 10)
