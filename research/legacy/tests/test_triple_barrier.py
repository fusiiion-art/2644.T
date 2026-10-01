import pandas as pd
import numpy as np
from common_utils.triple_barrier import TripleBarrierLabeler


def test_triple_barrier_labels_basic():
    dates = pd.date_range('2020-01-01', periods=30)
    # create increasing price with a jump
    prices = pd.Series(np.linspace(100, 110, 30), index=dates)
    # events at day 5 and day 15
    events = pd.DatetimeIndex([dates[5], dates[15]])
    labeler = TripleBarrierLabeler(upper_multiplier=0.01, lower_multiplier=0.01, max_holding_period=5, vol_window=5)
    df = labeler.label(prices, events)
    assert 'label' in df.columns
    assert len(df) == 2
