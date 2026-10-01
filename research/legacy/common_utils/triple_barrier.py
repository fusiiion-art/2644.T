import pandas as pd
import numpy as np
from typing import List


class TripleBarrierLabeler:
    def __init__(self,
                 upper_multiplier: float = 1.5,
                 lower_multiplier: float = 1.5,
                 max_holding_period: int = 5,
                 vol_window: int = 20):
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.max_holding_period = max_holding_period
        self.vol_window = vol_window

    def _realized_vol(self, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change().fillna(0)
        return returns.rolling(window=self.vol_window, min_periods=1).std()

    def label(self, prices: pd.Series, events: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        rv = self._realized_vol(prices)
        for ev in events:
            if ev not in prices.index:
                continue
            start_idx = prices.index.get_loc(ev)
            start_price = prices.iloc[start_idx]
            vol = rv.iloc[start_idx] if start_idx < len(rv) else rv.iloc[-1]
            up = start_price * (1 + self.upper_multiplier * vol)
            down = start_price * (1 - self.lower_multiplier * vol)

            touched = 'time'
            label = 0
            holding = self.max_holding_period
            ret = prices.iloc[min(start_idx + self.max_holding_period, len(prices)-1)] / start_price - 1

            # iterate forward until a barrier touched or max holding
            for offset in range(1, self.max_holding_period+1):
                idx = start_idx + offset
                if idx >= len(prices):
                    break
                p = prices.iloc[idx]
                if p >= up:
                    touched = 'upper'
                    label = 1
                    holding = offset
                    ret = p / start_price - 1
                    break
                if p <= down:
                    touched = 'lower'
                    label = -1
                    holding = offset
                    ret = p / start_price - 1
                    break

            rows.append({'event_date': prices.index[start_idx],
                         'barrier_touched': touched,
                         'label': label,
                         'holding_period': holding,
                         'return': ret})

        return pd.DataFrame(rows).set_index('event_date')
