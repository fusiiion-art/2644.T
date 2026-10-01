import pandas as pd
import numpy as np
from typing import Dict, Optional

class RiskManager:
    """
    市場リスクとAIの自己評価（自信）を統合的に評価し、
    具体的な損切り・利食い価格を動的に定義する。
    
    ★統合: PortfolioStrategist のケリー基準ポジションサイジングも含む。
    
    機能:
    - ATRベースのストップロス・テイクプロフィット計算
    - ケリー基準に基づく最適ポジションサイズ計算
    """
    def __init__(self, 
                 atr_period: int = 14, 
                 stop_loss_multiplier: float = 2.0, 
                 take_profit_multiplier: float = 3.0,
                 sigma_sensitivity: float = 0.5,
                 annualization_factor: int = 252,
                 # ★統合: PortfolioStrategist のパラメータ
                 risk_aversion: float = 1.0,
                 max_leverage: float = 2.0):
        # ATR関連
        self.atr_period = atr_period
        self.base_stop_loss_multiplier = stop_loss_multiplier
        self.base_take_profit_multiplier = take_profit_multiplier
        self.sigma_sensitivity = sigma_sensitivity
        self.annualization_factor = annualization_factor
        # ★統合: ケリー基準関連
        self.risk_aversion = risk_aversion
        self.max_leverage = max_leverage

    def _find_column(self, df: pd.DataFrame, keywords: list) -> Optional[str]:
        """指定されたキーワードリストに一致するカラム名を探索する（大文字小文字無視）"""
        cols_lower = {c.lower(): c for c in df.columns}
        for k in keywords:
            if k.lower() in cols_lower:
                return cols_lower[k.lower()]
        return None

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """最新のATR（Average True Range）を安全に計算する。"""
        high_col = self._find_column(df, ['High', 'high', 'Highest'])
        low_col = self._find_column(df, ['Low', 'low', 'Lowest'])
        close_col = self._find_column(df, ['Close', 'close', 'Adj Close', 'adj_close'])

        if not (high_col and low_col and close_col):
            raise ValueError(f"OHLC columns missing in historical data. Found: {df.columns.tolist()}")

        high = df[high_col]
        low = df[low_col]
        close = df[close_col]

        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        
        last_atr = atr.iloc[-1]
        
        if pd.isna(last_atr) or last_atr <= 0:
            fallback_atr = close.iloc[-1] * 0.01
            print(f"⚠️ ATR calculation failed or zero. Using fallback (1% of price): {fallback_atr:.2f}")
            return fallback_atr
            
        return last_atr

    # ★統合: PortfolioStrategist のメソッド
    def calculate_kelly_fraction(self, mu: float, sigma: float) -> float:
        """
        ケリー基準を用いて、最適なポジションサイズを計算する。

        Args:
            mu (float): 期待リターン（AIのμ予測）。
            sigma (float): リターンの標準偏差（AIのσ予測）。

        Returns:
            float: 最適なポジションサイズ（-max_leverageから+max_leverageの範囲）。
                   例: 0.5は資産の50%を買い、-0.2は20%を売る。
        """
        if sigma <= 1e-6:
            return 0.0

        # ケリー基準の基本式: f = μ / σ^2
        # リスク許容度を考慮: f = (1 / risk_aversion) * (μ / σ^2)
        optimal_fraction = (1 / self.risk_aversion) * (mu / (sigma**2))
        
        # 最大レバレッジの制約を適用
        position_size = np.clip(optimal_fraction, -self.max_leverage, self.max_leverage)
        
        return position_size

    def define_trade_plan(self, historical_data: pd.DataFrame, entry_price: float, 
                          predicted_direction: int, predicted_sigma: float) -> Dict[str, float]:
        """トレードプランの策定"""
        try:
            atr_value = self._calculate_atr(historical_data)
        except ValueError as e:
            print(f"⚠️ RiskManager Error: {e}")
            return None

        annualized_sigma = predicted_sigma * np.sqrt(self.annualization_factor)
        sigma_adjustment_factor = np.exp(-self.sigma_sensitivity * max(0, annualized_sigma - 0.20))
        
        stop_loss_multiplier = self.base_stop_loss_multiplier * sigma_adjustment_factor
        take_profit_multiplier = self.base_take_profit_multiplier

        print("\n--- 🧠 RiskManager's Dynamic Adjustment ---")
        print(f"Time Scale (Annualization Factor)   : {self.annualization_factor}")
        print(f"AI's Uncertainty (annualized sigma) : {annualized_sigma:.2%}")
        print(f"Sigma Adjustment Factor             : {sigma_adjustment_factor:.2f}")
        print(f"Adjusted Stop-Loss Multiplier       : {stop_loss_multiplier:.2f} (Base: {self.base_stop_loss_multiplier})")
        
        if predicted_direction >= 0:  # Buy
            stop_loss = entry_price - (atr_value * stop_loss_multiplier)
            take_profit = entry_price + (atr_value * take_profit_multiplier)
        else:  # Sell
            stop_loss = entry_price + (atr_value * stop_loss_multiplier)
            take_profit = entry_price - (atr_value * take_profit_multiplier)
        
        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr_value,
            "adjusted_sl_multiplier": stop_loss_multiplier
        }


# ★後方互換性: PortfolioStrategist のエイリアス
class PortfolioStrategist(RiskManager):
    """
    後方互換性のためのエイリアス。
    新規コードでは RiskManager を直接使用してください。
    """
    def __init__(self, risk_aversion: float = 1.0, max_leverage: float = 2.0):
        super().__init__(risk_aversion=risk_aversion, max_leverage=max_leverage)