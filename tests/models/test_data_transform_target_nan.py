"""Regression test for the ffill-masks-target-NaN bug surfaced by the
2026-05-25 V5 parity check.

`data_transform()` originally did `.ffill().fillna(0)` over ALL columns
including the `prices_h{h}` target columns. This silently forward-filled the
last-h rows whose targets must be NaN (price h days in the future is unknown
on the most recent bars). Downstream `walk_forward_pooled` and
`fit_pooled_full` then `.dropna(subset=[target_col])` no-op'd on those rows
and the model produced bogus predictions for dates with no real label —
manifested as a SOL walk-forward `-10.04` price on 2026-05-25.

Fix: exclude `prices_h*` from the ffill/fillna step so the dropna in the
training/eval path actually fires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.models import model_utils as mu


def _fake_pretransform_frame(n: int = 60) -> pd.DataFrame:
    """Minimal frame matching what build_pooled_dataset produces per coin."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    prices = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100.0
    return pd.DataFrame(
        {
            "prices": prices,
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "total_volumes": np.ones(n) * 1_000.0,
            "id": "crypto",
        },
        index=pd.Index(dates, name="Date"),
    )


def test_data_transform_preserves_nan_in_target_for_last_h_rows():
    """Last h rows of `prices_h{h}` must be NaN, not ffill'd to last value."""
    df = _fake_pretransform_frame(n=60)
    first_day_future = df.index[-1] + pd.Timedelta(days=1)

    _, df_final = mu.data_transform(
        df, first_day_future=first_day_future,
        include_future_row=True, horizons=(7, 14),
    )

    assert "prices_h7" in df_final.columns
    assert "prices_h14" in df_final.columns

    # h=7: with include_future_row=True and the internal shift, the last 7 rows
    # must have NaN target. Before the fix, ffill silently propagated the last
    # known value into these rows and dropna() then never fired.
    last7 = df_final["prices_h7"].iloc[-7:]
    assert last7.isna().all(), (
        f"prices_h7 last 7 rows should all be NaN (ffill must skip target "
        f"columns); got values:\n{last7}"
    )

    last14 = df_final["prices_h14"].iloc[-14:]
    assert last14.isna().all(), (
        f"prices_h14 last 14 rows should all be NaN; got values:\n{last14}"
    )


def test_data_transform_non_target_columns_still_ffilled():
    """Non-target columns must still be ffilled — fix must be scoped to prices_h*."""
    df = _fake_pretransform_frame(n=60)
    # Inject a NaN in a non-target rolling feature to confirm ffill still works.
    df.loc[df.index[5], "open"] = np.nan
    first_day_future = df.index[-1] + pd.Timedelta(days=1)

    _, df_final = mu.data_transform(
        df, first_day_future=first_day_future,
        include_future_row=True, horizons=(7,),
    )

    # 'open' is not a target column; ffill should have eliminated all NaNs.
    assert df_final["open"].notna().all(), (
        "open column should be ffilled — fix must be scoped to prices_h*"
    )
