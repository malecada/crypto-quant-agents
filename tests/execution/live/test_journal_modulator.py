"""modulator_outputs journaling (hybrid runner write-path)."""
from __future__ import annotations

import sqlite3

from tradingagents.execution.live.journal import Journal


def test_log_modulator_roundtrip(tmp_path):
    db = str(tmp_path / "j.db")
    j = Journal(db)
    j.log_modulator(cycle_id="c1", coin="ethereum", multiplier=1.2,
                    effective_weight=0.35, llm_confidence=0.7,
                    regime="trend_up", fallback=False)
    j.close()
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT cycle_id, coin, multiplier, effective_weight, llm_confidence, "
        "regime, fallback FROM modulator_outputs").fetchone()
    conn.close()
    assert row == ("c1", "ethereum", 1.2, 0.35, 0.7, "trend_up", 0)


def test_log_modulator_fallback_row(tmp_path):
    db = str(tmp_path / "j.db")
    j = Journal(db)
    j.log_modulator(cycle_id="c1", coin="bitcoin", multiplier=1.0,
                    effective_weight=0.0, llm_confidence=None,
                    regime=None, fallback=True)
    j.close()
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT llm_confidence, regime, fallback FROM modulator_outputs"
    ).fetchone()
    conn.close()
    assert row == (None, None, 1)


def test_log_modulator_upsert_replaces(tmp_path):
    db = str(tmp_path / "j.db")
    j = Journal(db)
    j.log_modulator(cycle_id="c1", coin="bitcoin", multiplier=1.0,
                    effective_weight=0.0, fallback=True)
    j.log_modulator(cycle_id="c1", coin="bitcoin", multiplier=1.3,
                    effective_weight=0.5, fallback=False)
    j.close()
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT multiplier FROM modulator_outputs").fetchall()
    conn.close()
    assert rows == [(1.3,)]


def test_fallback_detection_matches_extract_semantics():
    # Mirrors hybrid_runner's is_fallback expression against
    # extract_modulator_outputs degrade conditions.
    from tradingagents.execution.live.hybrid_compose import extract_modulator_outputs

    for mp in (None, {}, {"llm_multiplier": None, "effective_weight": 0.4},
               {"llm_multiplier": 1.2, "effective_weight": None}):
        is_fallback = (not mp or mp.get("llm_multiplier") is None
                       or mp.get("effective_weight") is None)
        assert is_fallback is True
        assert extract_modulator_outputs(mp) == (1.0, 0.0)

    mp = {"llm_multiplier": 1.2, "effective_weight": 0.4}
    assert (not mp or mp.get("llm_multiplier") is None
            or mp.get("effective_weight") is None) is False
    assert extract_modulator_outputs(mp) == (1.2, 0.4)
