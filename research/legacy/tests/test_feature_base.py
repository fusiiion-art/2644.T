import pandas as pd
from types import SimpleNamespace

from common_utils.feature_base import FeatureEngineBase


def test_intermarket_features_support_triplets_in_pairs_config():
    config = SimpleNamespace(
        FEATURE_PARAMS={
            "intermarket_pairs": [("semi_adj_close", "sox_adj_close", "return_vs_return")],
            "intermarket_corr_windows": [20],
        },
        PREDICTION_HORIZON=1,
        MAIN_ASSET="",
        TARGET_COLUMN="",
    )
    engine = FeatureEngineBase(config=config)
    engine.df = pd.DataFrame(
        {
            "semi_adj_close_return": [0.01, 0.02, 0.03],
            "sox_adj_close_return": [0.02, 0.03, 0.04],
        }
    )
    engine.price_cols_map = {
        "semi_adj_close": "semi_adj_close",
        "sox_adj_close": "sox_adj_close",
    }

    engine._add_intermarket_features()

    assert "corr_semi_adj_close_sox_adj_close_20d" in engine.df.columns
