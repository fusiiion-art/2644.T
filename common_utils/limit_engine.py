import numpy as np


class LimitEngine:
    def __init__(
        self,
        atr_k_bull=0.6,
        atr_k_bear=1.5,
        sox_suspend_thresh=-0.05,
        sox_discount_thresh=-0.03,
    ):
        self.atr_k_bull = atr_k_bull
        self.atr_k_bear = atr_k_bear
        self.sox_suspend_thresh = sox_suspend_thresh
        self.sox_discount_thresh = sox_discount_thresh

    def calculate_order(self, open_price, atr, bb_lower, sox_return, ai_expected_return):
        """
        AIの予測リターンと、SOX指数のルールベースを統合して指値を決定する。
        """
        if sox_return <= self.sox_suspend_thresh:
            return {
                "suspend": True,
                "p_buy_limit": None,
                "reason": "SOX急落による強制サスペンド",
            }

        if ai_expected_return > 0.002:
            base_limit = open_price - (self.atr_k_bull * atr)
            regime = "Bullish_Shallow"
        else:
            base_limit = min(open_price - (self.atr_k_bear * atr), bb_lower)
            regime = "Bearish_Deep"

        if sox_return <= self.sox_discount_thresh:
            base_limit *= 0.985

        p_buy_limit = np.floor(base_limit)

        return {
            "suspend": False,
            "p_buy_limit": p_buy_limit,
            "regime": regime,
        }
