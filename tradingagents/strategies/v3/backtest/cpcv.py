"""Combinatorial Purged Cross-Validation (López de Prado 2018).

Splits the index into ``n_groups`` contiguous groups, picks ``test_groups`` per
split (yielding ``C(n_groups, test_groups)`` splits), purges samples whose label
horizon overlaps the test range, and embargoes ``embargo`` bars on each side of
the test groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargo_gaps: list[tuple[int, int]]


def cpcv_splits(
    n_samples: int,
    n_groups: int = 8,
    test_groups: int = 2,
    embargo: int = 14,
    min_train: int = 252,
) -> Iterator[CPCVSplit]:
    if test_groups >= n_groups:
        raise ValueError("test_groups must be < n_groups")
    if n_samples < n_groups * 2 * embargo:
        raise ValueError(
            f"n_samples={n_samples} too small for n_groups={n_groups}, embargo={embargo}"
        )

    group_size = n_samples // n_groups
    boundaries = [(i * group_size, (i + 1) * group_size) for i in range(n_groups)]
    boundaries[-1] = (boundaries[-1][0], n_samples)

    for test_combo in combinations(range(n_groups), test_groups):
        test_idx_list: list[int] = []
        embargo_gaps: list[tuple[int, int]] = []
        for g in test_combo:
            lo, hi = boundaries[g]
            test_idx_list.extend(range(lo, hi))
            embargo_lo = max(0, lo - embargo)
            embargo_hi = min(n_samples, hi + embargo)
            embargo_gaps.append((embargo_lo, embargo_hi))

        test_idx = np.array(sorted(test_idx_list))
        embargo_set: set[int] = set()
        for elo, ehi in embargo_gaps:
            embargo_set.update(range(elo, ehi))

        train_idx = np.array(
            [i for i in range(n_samples) if i not in embargo_set]
        )
        if len(train_idx) < min_train:
            continue
        yield CPCVSplit(train_idx=train_idx, test_idx=test_idx, embargo_gaps=embargo_gaps)
