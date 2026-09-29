import pandas as pd
import pytest

from common_utils.trainer_model import select_features


class DummyConfig:
    pass


def test_select_features_deprecation():
    cfg = DummyConfig()
    cfg.USE_RFE = False
    X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6], 'f3': [7, 8, 9]})
    y = pd.Series([0, 1, 0])

    with pytest.warns(DeprecationWarning):
        selected = select_features(cfg, X, y, 2, 'run_test')

    assert selected == ['f1', 'f2']
