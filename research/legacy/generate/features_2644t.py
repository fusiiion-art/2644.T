"""
半導体ETF (2644.T) 用の特徴量エンジン
FeatureEngineBase を継承し、半導体固有の特徴量を追加
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# プロジェクトルート設定
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except (NameError, IndexError):
    PROJECT_ROOT = Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common_utils.feature_base import FeatureEngineBase


class FeatureEngineDaily2644T(FeatureEngineBase):
    """
    2644.T専用特徴量エンジン
    SOX, NVDA, TSM など半導体関連固有の特徴量を追加
    """
    
    def _add_module_specific_features(self):
        """半導体固有の特徴量"""
        logging.info("Calculating 2644.T specific semiconductor features...")
        epsilon = 1e-9

        # --- T-01: SOX Overnight Gap ---
        if 'sox_open' in self.df.columns and 'sox_close' in self.df.columns:
            self.df['SOX_Overnight_Gap'] = (
                self.df['sox_open'] / (self.df['sox_close'].shift(1) + epsilon) - 1
            )
        
        # --- T-02: SOX Intraday Force ---
        if 'sox_close' in self.df.columns and 'sox_open' in self.df.columns:
            self.df['SOX_Intraday_Force'] = (
                self.df['sox_close'] / (self.df['sox_open'] + epsilon) - 1
            )

        # --- R-01: Tech Risk Premium (VXN - VIX) ---
        if 'vxn_close' in self.df.columns and 'vix_close' in self.df.columns:
            self.df['Tech_Risk_Premium'] = self.df['vxn_close'] - self.df['vix_close']
        
        # --- M-02: Sector Relative Strength (vs Nikkei 225) ---
        target_prefix = getattr(self, 'main_asset_prefix', 'semi')
        target_return_col = f"{target_prefix}_adj_close_return"
        semi_ret = self.df.get(target_return_col, pd.Series(0, index=self.df.index))
        nikkei_ret = self.df.get('nikkei_225_adj_close_return', pd.Series(0, index=self.df.index))
        
        self.df['Sector_Relative_Strength_5d'] = (
            semi_ret.rolling(5).sum() - nikkei_ret.rolling(5).sum()
        )

        # --- F-03: NVDA Impact Factor ---
        if 'nvda_adj_close_return' in self.df.columns:
            corr = semi_ret.rolling(20).corr(self.df['nvda_adj_close_return']).fillna(0)
            self.df['NVDA_Impact_Factor'] = self.df['nvda_adj_close_return'] * corr

        # --- M-01: Gap Fill Probability Proxy ---
        open_col = f"{target_prefix}_open"
        close_col = f"{target_prefix}_close"
        if open_col in self.df.columns and close_col in self.df.columns:
            prev_close = self.df[close_col].shift(1)
            gap = self.df[open_col] - prev_close
            intraday = self.df[close_col] - self.df[open_col]
            self.df['JP_Gap_Fill_Force'] = np.sign(gap) * np.sign(intraday)

        logging.info("Finished Semi-conductor specific features.")


# Backward-compatible alias
FeatureEngine2644T = FeatureEngineDaily2644T
