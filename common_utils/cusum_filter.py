import numpy as np
try:
    from numba import njit
except Exception:
    def njit(func=None, **kwargs):
        return func


@njit
def symmetric_cusum_filter(returns: np.ndarray, threshold: float = -1.0, vol_window: int = 20, multiplier: float = 1.0):
    """
    Symmetric CUSUM filter with optional dynamic threshold based on realized volatility.

    Args:
        returns: array of returns
        threshold: if > 0, treated as a fixed scalar threshold. If <= 0, compute dynamic
                   threshold = rolling_std(returns, vol_window) * multiplier.
        vol_window: window size for realized vol when dynamic thresholding is used
        multiplier: scalar multiplier applied to realized vol

    Returns:
        numpy array of event indices where cusum exceeded threshold
    """
    n = len(returns)
    pos_sum = 0.0
    neg_sum = 0.0
    events = []

    # precompute dynamic thresholds if requested
    use_dynamic = (threshold <= 0.0)
    if use_dynamic:
        # rolling std (simple, not optimized) - compute for each i the std of prior vol_window returns
        realized = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - vol_window + 1
            if start < 0:
                start = 0
            window = returns[start:i+1]
            # compute std
            if len(window) <= 1:
                realized[i] = 0.0
            else:
                # population std
                mean = 0.0
                for v in window:
                    mean += v
                mean = mean / len(window)
                s = 0.0
                for v in window:
                    s += (v - mean) * (v - mean)
                realized[i] = (s / (len(window) - 1)) ** 0.5

    for i in range(n):
        r = returns[i]
        pos_sum = max(0.0, pos_sum + r)
        neg_sum = min(0.0, neg_sum + r)

        if use_dynamic:
            th = realized[i] * multiplier
        else:
            th = threshold

        if th <= 0:
            # avoid triggering on zero threshold
            continue

        if pos_sum > th or -neg_sum > th:
            events.append(i)
            pos_sum = 0.0
            neg_sum = 0.0

    return np.array(events, dtype=np.int64)
