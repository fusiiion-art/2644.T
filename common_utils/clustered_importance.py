import numpy as np
import pandas as pd
from typing import List, Optional, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


class ClusteredFeatureImportance:
    def __init__(self, linkage_method: str = 'ward', max_clusters: int = 20, cv_splitter=None):
        self.linkage_method = linkage_method
        self.max_clusters = max_clusters
        self.cv_splitter = cv_splitter
        self.cluster_assignments = None
        self.dendro_Z = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ClusteredFeatureImportance':
        # compute correlation distance
        corr = X.corr().fillna(0).values
        # transform to distance
        dist = np.sqrt(np.maximum(0.0, 0.5 * (1 - corr)))
        # convert to condensed distance for linkage
        try:
            condensed = squareform(dist, checks=False)
        except Exception:
            # fallback: zero-diagonal
            condensed = squareform(dist)

        Z = linkage(condensed, method=self.linkage_method)
        self.dendro_Z = Z

        best_k = 1
        best_score = -1.0
        best_assign = None
        for k in range(2, min(self.max_clusters, X.shape[1]) + 1):
            labels = fcluster(Z, k, criterion='maxclust')
            try:
                score = silhouette_score(dist, labels, metric='precomputed')
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_k = k
                best_assign = labels

        if best_assign is None:
            best_assign = np.arange(1, X.shape[1]+1)

        self.cluster_assignments = pd.Series(best_assign, index=X.columns)
        return self

    def compute_importance(self, model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compute clustered MDA: for each cluster, shuffle all features in that cluster together
        across rows and measure OOS performance drop using cv_splitter if provided.
        """
        baseline_score = self._score_cv(model, X, y)
        results = []
        clusters = sorted(set(self.cluster_assignments.values))
        for c in clusters:
            cols = self.cluster_assignments[self.cluster_assignments == c].index.tolist()
            X_shuffled = X.copy()
            # permute cluster jointly
            perm_idx = np.random.permutation(len(X))
            X_shuffled[cols] = X_shuffled[cols].iloc[perm_idx].reset_index(drop=True)
            score = self._score_cv(model, X_shuffled, y)
            importance = baseline_score - score
            results.append({'cluster_id': int(c), 'features_in_cluster': cols, 'importance_score': float(importance)})

        return pd.DataFrame(results).sort_values('importance_score', ascending=False)

    def select_features(self, importance_df: pd.DataFrame, importance_threshold: float = 0.0) -> List[str]:
        selected = []
        for _, row in importance_df.iterrows():
            if row['importance_score'] >= importance_threshold:
                selected.extend(row['features_in_cluster'])
        return selected

    def _score_cv(self, model: Any, X: pd.DataFrame, y: pd.Series) -> float:
        if self.cv_splitter is None:
            # single fit
            try:
                model.fit(X, y)
                preds = model.predict(X)
                # default: negative MSE-like (higher better)
                return -float(((y - preds) ** 2).mean())
            except Exception:
                return 0.0

        scores = []
        for train_idx, test_idx in self.cv_splitter.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            try:
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                scores.append(-float(((y_te - preds) ** 2).mean()))
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores))
