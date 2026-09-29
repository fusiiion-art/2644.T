import pandas as pd
import numpy as np

class FeatureEngineering:
    """
    ETF固有の特徴量（理論価格乖離など）および
    フラクショナル差分（Fractional Differentiation）などを計算するクラス。
    """
    def __init__(self):
        pass

    def calculate_fair_value_gap(self, etf_price: pd.Series, constituent_data: pd.DataFrame) -> pd.Series:
        """
        構成銘柄から推定されるフェアバリュー（理論価格）と実際のETF価格の乖離を計算。
        アービトラージ的な特徴量として利用。
        """
        # TODO: フェアバリュー推定ロジックの実装
        pass

    def get_weights_ffd(self, d: float, thres: float = 1e-5) -> np.ndarray:
        """
        フラクショナル差分のための重みを計算。
        d: 差分の階数 (0 < d < 1)
        """
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(self, series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
        """
        固定幅ウィンドウ (FFD: Fixed-Width Window Fractional Differentiation)
        を用いてフラクショナル差分を計算する。
        情報（メモリ）を保持しつつ、時系列を定常化するために用いる。
        """
        # TODO: フラクショナル差分適用ロジックの実装
        pass
