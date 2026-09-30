"""Combinatorial Purged Cross-Validation（設計書 v2 評価設計、López de Prado AFML 12章）。

- 行を n_groups 個の連続グループに分け、n_test_groups 個をテストにする全組み合わせ（6・2 なら15分割）。
- 学習行のラベル期間 [i, end_i] がテストのラベル期間 [テスト先頭, テストのラベル終端] と重なれば除外（purge）。
- テストのラベル終端の後ろ embargo 行も学習から除外（embargo）。
- 各グループは C(n_groups-1, n_test_groups-1) 回テストになり、それをつないで同数のバックテスト・パスを作る。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def label_end_positions(days_to_fill, censored, max_hold: int, n: int) -> np.ndarray:
    """各行のラベルが最後に使う行位置。約定済みは約定日、打ち切り・不明は max_hold 日目（データ終端で切る）。"""
    days = np.asarray(days_to_fill, dtype=float)
    span = np.where(np.isfinite(days), days, float(max_hold))
    return np.minimum(np.arange(n) + span - 1, n - 1).astype(int)


def uniqueness_weights(end: np.ndarray, n: int, rows: np.ndarray | None = None) -> np.ndarray:
    """平均独自性（AFML 4章）。rows を渡すとその行どうしの重なりだけで数える（学習 fold 内の重み用）。"""
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=int)
    conc = np.zeros(n + 1)
    np.add.at(conc, rows, 1.0)
    np.add.at(conc, end[rows] + 1, -1.0)
    conc = np.cumsum(conc)[:n]
    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    return (csum[end[rows] + 1] - csum[rows]) / (end[rows] - rows + 1)


class CombinatorialPurgedCV:
    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 20):
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.embargo = int(embargo)

    def groups(self, n: int) -> list[np.ndarray]:
        return [g.astype(int) for g in np.array_split(np.arange(n), self.n_groups)]

    def purge(self, candidates: np.ndarray, blocks: list[np.ndarray], end: np.ndarray) -> np.ndarray:
        """candidates（学習候補の行位置）から、各テストブロックと重なる行と embargo 行を除く。"""
        keep = np.ones(len(candidates), dtype=bool)
        for block in blocks:
            if len(block) == 0:
                continue
            a, e = int(block[0]), int(end[block].max())
            keep &= ~((candidates <= e) & (end[candidates] >= a))
            keep &= ~((candidates > e) & (candidates <= e + self.embargo))
        return candidates[keep]

    def split(self, end: np.ndarray):
        """(split_id, 学習行, テスト行, テストのグループ番号) を返す。"""
        n = len(end)
        groups = self.groups(n)
        for sid, gids in enumerate(combinations(range(self.n_groups), self.n_test_groups)):
            test = np.concatenate([groups[g] for g in gids])
            rest = np.setdiff1d(np.arange(n), test)
            train = self.purge(rest, [groups[g] for g in gids], end)
            yield sid, train, test, gids

    def paths(self) -> list[dict]:
        """バックテスト・パス。各パスは {グループ番号: そのグループの予測を取る split_id}。"""
        splits = list(combinations(range(self.n_groups), self.n_test_groups))
        per_group = {g: [sid for sid, gids in enumerate(splits) if g in gids] for g in range(self.n_groups)}
        n_paths = len(per_group[0])
        return [{g: per_group[g][p] for g in range(self.n_groups)} for p in range(n_paths)]
