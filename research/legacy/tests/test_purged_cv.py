import numpy as np
from common_utils.purged_cv import PurgedGroupTimeSeriesSplit


def test_purged_cv_splits_count():
    X = np.arange(100)
    cv = PurgedGroupTimeSeriesSplit(n_splits=4, purge_gap=2, embargo_gap=2)
    splits = list(cv.split(X))
    assert len(splits) == 4


def test_purge_gap_excludes_before_test():
    """purge_gap 分だけテスト開始前のデータが学習セットから除外される"""
    X = np.arange(100)
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=5, embargo_gap=0)
    for train_idx, test_idx in cv.split(X):
        if len(train_idx) == 0:
            continue
        test_start = test_idx[0]
        purge_start = max(0, test_start - 5)
        # 学習データに purge 区間のインデックスが含まれないこと
        for idx in range(purge_start, test_start):
            assert idx not in train_idx, f"index {idx} in purge zone found in train"


def test_embargo_gap_excludes_after_test():
    """embargo_gap 分だけテスト終了後のデータが学習セットから除外される"""
    X = np.arange(100)
    embargo = 5
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=0, embargo_gap=embargo)
    for train_idx, test_idx in cv.split(X):
        if len(test_idx) == 0:
            continue
        test_end = test_idx[-1] + 1
        embargo_end = min(100, test_end + embargo)
        # 学習データに embargo 区間のインデックスが含まれないこと
        for idx in range(test_end, embargo_end):
            assert idx not in train_idx, f"index {idx} in embargo zone found in train"


def test_embargo_allows_post_embargo_data():
    """embargo 区間の後ろのデータは学習セットに含まれる"""
    X = np.arange(200)
    embargo = 5
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=3, embargo_gap=embargo)
    splits = list(cv.split(X))
    # 中間フォールド（最終フォールド以外）ではembargo後のデータが学習に含まれるはず
    for train_idx, test_idx in splits[:-1]:  # 最終フォールドは後ろにデータがない可能性あり
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        test_end = test_idx[-1] + 1
        embargo_end = min(200, test_end + embargo)
        if embargo_end < 200:
            # embargo 直後のインデックスが学習に含まれること
            assert embargo_end in train_idx, f"index {embargo_end} (post-embargo) should be in train"


def test_train_test_no_overlap():
    """学習データとテストデータに重複がないこと"""
    X = np.arange(100)
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=5, embargo_gap=5)
    for train_idx, test_idx in cv.split(X):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"overlap between train and test: {overlap}"

