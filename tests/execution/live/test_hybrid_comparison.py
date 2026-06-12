"""Tests for compare_quant_hybrid — quant vs hybrid equity-curve comparison.

TDD: tests are written first; they fail until compare_quant_hybrid is
implemented in rebacktest.py.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _seed_snapshots(db_path: Path, rows: list[tuple[str, str, float]]) -> None:
    """Seed portfolio_snapshots into a fresh Journal DB.

    Args:
        db_path: path to create the SQLite file.
        rows: list of (cycle_id, ts, total_value) tuples.  cycle_id must be
              YYYY-MM-DD so compute_live_metrics (and our reader) count them.
    """
    from tradingagents.execution.live.journal import Journal

    j = Journal(str(db_path))
    for cycle_id, ts, total_value in rows:
        j.log_portfolio_snapshot(
            cycle_id=cycle_id,
            total_value=total_value,
            usdt_balance=total_value,
            position_qty_per_coin={},
            unrealized_pnl=0.0,
        )
    j.close()


# ── fixture helpers ──────────────────────────────────────────────────────────

# 6 daily cycles, same dates for both books so the overlap is the full window.
_DATES = [
    "2026-05-01",
    "2026-05-02",
    "2026-05-03",
    "2026-05-04",
    "2026-05-05",
    "2026-05-06",
]

# Quant: rises 10000 → 10500 over 6 steps (+0.5% each step, ~+5% total)
_QUANT_VALUES = [10000.0, 10050.0, 10100.0, 10200.0, 10350.0, 10500.0]

# Hybrid: rises 10000 → 11000 over the same 6 steps (~+10% total, outperforms)
_HYBRID_VALUES = [10000.0, 10100.0, 10200.0, 10400.0, 10700.0, 11000.0]


def _make_rows(values: list[float]) -> list[tuple[str, str, float]]:
    return [
        (d, f"{d}T00:05:00+00:00", v)
        for d, v in zip(_DATES, values)
    ]


# ── tests ────────────────────────────────────────────────────────────────────


def test_compare_quant_hybrid_keys(tmp_path):
    """compare_quant_hybrid returns the three required top-level keys."""
    from tradingagents.execution.live.rebacktest import compare_quant_hybrid

    qdb = tmp_path / "quant.db"
    hdb = tmp_path / "hybrid.db"
    _seed_snapshots(qdb, _make_rows(_QUANT_VALUES))
    _seed_snapshots(hdb, _make_rows(_HYBRID_VALUES))

    result = compare_quant_hybrid(qdb, hdb, coins=["bitcoin", "ethereum"])

    assert set(result.keys()) == {"quant", "hybrid", "delta", "window"}


def test_compare_quant_hybrid_metric_subkeys(tmp_path):
    """Each block (quant/hybrid/delta) has sharpe, ret, maxdd."""
    from tradingagents.execution.live.rebacktest import compare_quant_hybrid

    qdb = tmp_path / "quant.db"
    hdb = tmp_path / "hybrid.db"
    _seed_snapshots(qdb, _make_rows(_QUANT_VALUES))
    _seed_snapshots(hdb, _make_rows(_HYBRID_VALUES))

    result = compare_quant_hybrid(qdb, hdb, coins=[])

    for block in ("quant", "hybrid", "delta"):
        assert "sharpe" in result[block], f"missing sharpe in {block}"
        assert "ret" in result[block], f"missing ret in {block}"
        assert "maxdd" in result[block], f"missing maxdd in {block}"


def test_compare_quant_hybrid_delta_ret_sign(tmp_path):
    """delta.ret > 0 when hybrid outperforms quant (hybrid +10% > quant +5%)."""
    from tradingagents.execution.live.rebacktest import compare_quant_hybrid

    qdb = tmp_path / "quant.db"
    hdb = tmp_path / "hybrid.db"
    _seed_snapshots(qdb, _make_rows(_QUANT_VALUES))
    _seed_snapshots(hdb, _make_rows(_HYBRID_VALUES))

    result = compare_quant_hybrid(qdb, hdb, coins=["bitcoin"])

    assert result["delta"]["ret"] > 0, (
        f"Expected hybrid to outperform quant; delta.ret={result['delta']['ret']}"
    )
    # Sanity: hybrid return ≈ 10%, quant ≈ 5%
    assert result["hybrid"]["ret"] == pytest.approx(0.10, abs=0.01)
    assert result["quant"]["ret"] == pytest.approx(0.05, abs=0.01)


def test_compare_quant_hybrid_window_n(tmp_path):
    """window.n matches the number of points in the overlap (6 here)."""
    from tradingagents.execution.live.rebacktest import compare_quant_hybrid

    qdb = tmp_path / "quant.db"
    hdb = tmp_path / "hybrid.db"
    _seed_snapshots(qdb, _make_rows(_QUANT_VALUES))
    _seed_snapshots(hdb, _make_rows(_HYBRID_VALUES))

    result = compare_quant_hybrid(qdb, hdb, coins=[])

    assert result["window"]["n"] == len(_DATES)
    assert result["window"]["start"] == _DATES[0]
    assert result["window"]["end"] == _DATES[-1]


def test_compare_quant_hybrid_overlap_clipping(tmp_path):
    """When the two journals start at different dates the overlap is respected.

    Quant has 6 rows (2026-05-01 → 05-06).
    Hybrid starts 2 days later (2026-05-03 → 05-06) → overlap = 4 rows.
    """
    from tradingagents.execution.live.rebacktest import compare_quant_hybrid

    dates_h = _DATES[2:]          # 2026-05-03 … 05-06 (4)

    qdb = tmp_path / "quant.db"
    hdb = tmp_path / "hybrid.db"
    # Quant: all 6 dates as before.
    _seed_snapshots(qdb, _make_rows(_QUANT_VALUES))
    # Hybrid: only the last 4 dates — must explicitly pair dates_h with values.
    hybrid_late_rows = [
        (d, f"{d}T00:05:00+00:00", v)
        for d, v in zip(dates_h, _HYBRID_VALUES[2:])
    ]
    _seed_snapshots(hdb, hybrid_late_rows)

    result = compare_quant_hybrid(qdb, hdb, coins=["bitcoin"])

    assert result["window"]["n"] == 4
    assert result["window"]["start"] == "2026-05-03"
    assert result["window"]["end"] == "2026-05-06"


def test_compare_quant_hybrid_insufficient_data(tmp_path):
    """Fewer than 2 overlapping daily snapshots → safe NaN/zero defaults,
    no exception raised."""
    from tradingagents.execution.live.rebacktest import compare_quant_hybrid

    qdb = tmp_path / "quant.db"
    hdb = tmp_path / "hybrid.db"
    # Each journal has only 1 row — no overlap long enough for metrics.
    _seed_snapshots(qdb, [("2026-05-01", "2026-05-01T00:05:00+00:00", 10000.0)])
    _seed_snapshots(hdb, [("2026-05-01", "2026-05-01T00:05:00+00:00", 10000.0)])

    result = compare_quant_hybrid(qdb, hdb, coins=[])

    assert set(result.keys()) == {"quant", "hybrid", "delta", "window"}
    assert result["window"]["n"] <= 1
