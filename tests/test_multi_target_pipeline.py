import pandas as pd

from common_utils.target_utils import create_multi_targets


def test_create_multi_targets_adds_primary_and_secondary_targets():
    df = pd.DataFrame({
        '2644t_open': [100.0, 102.0, 104.0],
        '2644t_high': [105.0, 107.0, 109.0],
        '2644t_low': [98.0, 100.0, 102.0],
        '2644t_adj_close': [101.0, 103.0, 105.0],
    })

    out = create_multi_targets(df, horizons=[1], open_col='2644t_open', high_col='2644t_high', low_col='2644t_low', close_col='2644t_adj_close')

    assert 'target_1' in out.columns
    assert 'target_1_open_to_open' in out.columns
    assert 'target_1_high_low' in out.columns
    assert 'target_1_close_to_close' in out.columns
    assert out['target_1'].notna().all()
