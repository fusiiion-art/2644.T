import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils.limit_engine import LimitEngine


def test_suspend_on_strong_sox_drop():
    engine = LimitEngine()
    result = engine.calculate_order(
        open_price=5000,
        atr=100,
        bb_lower=4800,
        sox_return=-0.06,
        ai_expected_return=0.004,
    )
    assert result["suspend"] is True


def test_bullish_regime_uses_shallow_limit():
    engine = LimitEngine()
    result = engine.calculate_order(
        open_price=5000,
        atr=100,
        bb_lower=4800,
        sox_return=-0.01,
        ai_expected_return=0.004,
    )
    assert result["suspend"] is False
    assert result["regime"] == "Bullish_Shallow"
    assert result["p_buy_limit"] < 5000
