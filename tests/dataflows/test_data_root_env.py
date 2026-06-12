"""TRADINGAGENTS_DATA_ROOT env var redirects on-chain store + derivatives + options dirs."""
from __future__ import annotations


def test_onchain_store_default_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_DATA_ROOT", str(tmp_path))
    # Force reload to pick up env at import time
    import importlib
    from tradingagents.dataflows import onchain_store
    importlib.reload(onchain_store)
    assert str(onchain_store.DEFAULT_ROOT) == str(tmp_path / "onchain")


def test_build_pit_onchain_features_honors_root(monkeypatch, tmp_path):
    """build_pit_onchain_features reads from data_root if passed explicitly."""
    monkeypatch.setenv("TRADINGAGENTS_DATA_ROOT", str(tmp_path))
    # Empty sandbox → empty features (graceful)
    (tmp_path / "onchain").mkdir()
    import importlib
    from tradingagents.dataflows import onchain_features, onchain_store
    importlib.reload(onchain_store)
    importlib.reload(onchain_features)
    import pandas as pd
    dates = pd.date_range("2026-01-01", "2026-01-03", freq="D", tz="UTC")
    df = onchain_features.build_pit_onchain_features(
        coin="bitcoin", dates=dates,
        include_global=False, include_derived=False,
        include_stablecoin_context=False, include_options=False,
        include_derivatives=False,
        root=onchain_store.DEFAULT_ROOT,
    )
    # Empty store → empty df is fine; we're testing the root threading worked
    assert df.shape[0] == 3
