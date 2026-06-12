from __future__ import annotations

import numpy as np
import pytest

from tradingagents.strategies.v3.backtest.cpcv import (
    CPCVSplit,
    cpcv_splits,
)


def test_cpcv_split_count():
    splits = list(cpcv_splits(n_samples=720, n_groups=8, test_groups=2, embargo=14))
    assert len(splits) == 28  # C(8,2)


def test_cpcv_no_train_test_overlap():
    for split in cpcv_splits(n_samples=720, n_groups=8, test_groups=2, embargo=14):
        train_set = set(split.train_idx.tolist())
        test_set = set(split.test_idx.tolist())
        assert train_set.isdisjoint(test_set)


def test_cpcv_embargo_respected():
    embargo = 14
    for split in cpcv_splits(n_samples=720, n_groups=8, test_groups=2, embargo=embargo):
        if len(split.test_idx) == 0 or len(split.train_idx) == 0:
            continue
        for gap_start, gap_end in split.embargo_gaps:
            for tr in split.train_idx:
                assert not (gap_start <= tr < gap_end), (
                    f"train idx {tr} fell inside embargo gap [{gap_start},{gap_end})"
                )


def test_cpcv_min_train_size():
    n = 720
    for split in cpcv_splits(n_samples=n, n_groups=8, test_groups=2, embargo=14):
        assert len(split.train_idx) >= 252, "train must be at least 252 bars"


def test_cpcv_invalid_inputs():
    with pytest.raises(ValueError):
        list(cpcv_splits(n_samples=100, n_groups=8, test_groups=2, embargo=14))
    with pytest.raises(ValueError):
        list(cpcv_splits(n_samples=720, n_groups=8, test_groups=10, embargo=14))
