from sklearn.model_selection import BaseCrossValidator
import numpy as np


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purging + Embargo 付き時系列交差検証。

    - purge_gap: テストセット開始前にパージする期間（ラベル評価期間分のリーク防止）
    - embargo_gap: テストセット終了後にエンバーゴする期間（テスト直後のデータのリーク防止）

    学習データ = [0, purge_start) ∪ [embargo_end, n_samples)  （テスト区間 + purge + embargo を除外）
    """
    def __init__(self, n_splits=5, purge_gap=5, embargo_gap=5, group_gap=0):
        self.n_splits = int(n_splits)
        self.purge_gap = int(purge_gap)
        self.embargo_gap = int(embargo_gap)
        self.group_gap = int(group_gap)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        # create contiguous test windows
        for i in range(self.n_splits):
            test_start = int(i * n_samples / self.n_splits)
            test_end = int((i + 1) * n_samples / self.n_splits)

            # purge: exclude purge_gap samples before test_start from training
            purge_start = max(0, test_start - self.purge_gap)

            # embargo: exclude embargo_gap samples after test_end from training
            embargo_end = min(n_samples, test_end + self.embargo_gap)

            # training = before purge zone + after embargo zone
            train_before = indices[indices < purge_start]
            train_after = indices[indices >= embargo_end]
            train_indices = np.concatenate([train_before, train_after])

            test_indices = indices[test_start:test_end]

            if len(train_indices) == 0:
                continue

            yield train_indices, test_indices

