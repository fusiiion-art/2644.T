import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from common_utils.clustered_importance import ClusteredFeatureImportance


def test_clustered_importance_basic():
    # create correlated features
    rng = np.random.RandomState(0)
    n = 200
    a = rng.normal(size=n)
    X = pd.DataFrame({
        'f1': a + rng.normal(scale=0.01, size=n),
        'f2': a * 1.0 + rng.normal(scale=0.01, size=n),
        'f3': rng.normal(size=n),
    })
    y = pd.Series(0.5 * X['f1'] + 0.2 * X['f3'] + rng.normal(scale=0.01, size=n))

    cfi = ClusteredFeatureImportance(linkage_method='ward', max_clusters=3)
    cfi.fit(X, y)
    model = LinearRegression()
    imp_df = cfi.compute_importance(model, X, y)
    assert not imp_df.empty
    assert 'importance_score' in imp_df.columns
