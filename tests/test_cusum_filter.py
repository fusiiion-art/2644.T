import numpy as np
from common_utils.cusum_filter import symmetric_cusum_filter


def test_symmetric_cusum_filter_detects_events():
    # create returns with spikes
    r = np.zeros(100)
    r[10] = 0.05
    r[50] = -0.06
    events = symmetric_cusum_filter(r, threshold=0.01)
    assert isinstance(events, np.ndarray)
    assert len(events) >= 2
