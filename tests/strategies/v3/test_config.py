from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.strategies.v3.config import V3Config


def test_v3_config_defaults():
    cfg = V3Config()
    assert cfg.target_annual_vol == 0.15
    assert cfg.max_leverage == 2.0
    assert cfg.horizons == (3, 7, 14, 21)
    assert cfg.cpcv_n_groups == 8
    assert cfg.cpcv_test_groups == 2
    assert cfg.embargo_bars == 14
    assert isinstance(cfg.microstructure_parquet_dir, Path)


def test_v3_config_override():
    cfg = V3Config(target_annual_vol=0.10, max_leverage=3.0)
    assert cfg.target_annual_vol == 0.10
    assert cfg.max_leverage == 3.0
    assert cfg.horizons == (3, 7, 14, 21)


def test_v3_config_horizon_weights_default():
    cfg = V3Config()
    weights = cfg.horizon_weights("trending")
    assert sum(weights.values()) == 1.0
    assert weights[14] + weights[21] > weights[3] + weights[7]
    weights_mr = cfg.horizon_weights("mean_reverting")
    assert weights_mr[3] + weights_mr[7] > weights_mr[14] + weights_mr[21]
    weights_neutral = cfg.horizon_weights("uncertain")
    assert all(abs(w - 0.25) < 1e-9 for w in weights_neutral.values())


def test_v3_config_horizon_weights_rejects_nonstandard_horizons():
    cfg = V3Config(horizons=(1, 3, 7, 14, 21))
    with pytest.raises(ValueError, match="default horizons"):
        cfg.horizon_weights("trending")


def test_v3_config_horizon_weights_sum_to_one_all_modes():
    cfg = V3Config()
    for mode in ("trending", "mean_reverting", "uncertain"):
        weights = cfg.horizon_weights(mode)
        assert sum(weights.values()) == pytest.approx(1.0), (
            f"{mode} sums to {sum(weights.values())}, not 1.0"
        )
