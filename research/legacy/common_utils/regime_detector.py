import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    市場レジーム（環境）を判定する専門家。
    シンプルなVIX閾値ベースと、複雑なテクニカル指標ベースの両方に対応。
    """
    @staticmethod
    def detect_simple_vix(df: pd.DataFrame, vix_column: str, threshold: float) -> pd.DataFrame:
        """
        [静的メソッド] VIXの閾値に基づき、'risk_on'/'risk_off'レジームを判定する。
        インスタンス化不要で呼び出せる。
        
        Args:
            df (pd.DataFrame): VIXカラムを含むデータフレーム
            vix_column (str): VIXデータのカラム名
            threshold (float): risk_on/offを分ける閾値
            
        Returns:
            pd.DataFrame: 'regime'カラムが追加されたデータフレーム
        """
        if vix_column not in df.columns:
            raise ValueError(f"VIXカラム '{vix_column}' がデータフレームに存在しません。")
        
        df_copy = df.copy()
        df_copy['regime'] = np.where(df_copy[vix_column] > threshold, 'risk_off', 'risk_on')
        logger.info(f"VIX閾値({threshold})に基づき、シンプルなレジームを判定しました。")
        return df_copy

    @classmethod
    def detect_complex(cls, df: pd.DataFrame, short_ma: int = 20, long_ma: int = 60, atr_period: int = 14, **kwargs) -> pd.DataFrame:
        """
        [クラスメソッド] 移動平均とATRに基づき、トレンドとボラティリティの複合レジームを判定する。
        """
        df_copy = df.copy()
        close_col = kwargs.get('close_col', 'close') # デフォルトのカラム名を指定
        high_col = kwargs.get('high_col', 'high')
        low_col = kwargs.get('low_col', 'low')

        # トレンド判定
        df_copy['short_ma'] = df_copy[close_col].rolling(window=short_ma).mean()
        df_copy['long_ma'] = df_copy[close_col].rolling(window=long_ma).mean()
        df_copy['trend'] = np.where(df_copy['short_ma'] > df_copy['long_ma'], 'Bull', 'Bear')

        # ボラティリティ判定 (ATR)
        tr = pd.concat([
            df_copy[high_col] - df_copy[low_col],
            abs(df_copy[high_col] - df_copy[close_col].shift()),
            abs(df_copy[low_col] - df_copy[close_col].shift())
        ], axis=1).max(axis=1)
        df_copy['atr'] = tr.ewm(span=atr_period, adjust=False).mean()
        # 例えば、ATRが過去N日のXパーセンタイルを超えたら高ボラなど
        vol_threshold = df_copy['atr'].rolling(252).quantile(0.75)
        df_copy['volatility'] = np.where(df_copy['atr'] > vol_threshold, 'High', 'Low')

        # 複合レジーム
        df_copy['complex_regime'] = df_copy['trend'] + '_' + df_copy['volatility']
        
        logger.info("複合的なテクニカル指標に基づき、レジームを判定しました。")
        return df_copy.drop(columns=['short_ma', 'long_ma', 'atr'])
