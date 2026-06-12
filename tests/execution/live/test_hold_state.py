"""P1: hold_state journal table roundtrip."""


def test_hold_state_roundtrip(tmp_path):
    from tradingagents.execution.live.journal import Journal
    j = Journal(str(tmp_path / "j.db"))
    assert j.get_hold_state("bitcoin") is None
    j.upsert_hold_state(coin="bitcoin", current_dir=1, bars_held=3,
                        entry_price=50000.0, entry_base=0.42,
                        entry_cycle="2026-06-01")
    st = j.get_hold_state("bitcoin")
    assert st == {"coin": "bitcoin", "current_dir": 1, "bars_held": 3,
                  "entry_price": 50000.0, "entry_base": 0.42,
                  "entry_cycle": "2026-06-01"}
    # Replace (flat the position)
    j.upsert_hold_state(coin="bitcoin", current_dir=0, bars_held=0,
                        entry_price=0.0, entry_base=0.0, entry_cycle="2026-06-02")
    st2 = j.get_hold_state("bitcoin")
    assert st2["current_dir"] == 0 and st2["bars_held"] == 0
    # Independent coins
    j.upsert_hold_state(coin="ethereum", current_dir=-1, bars_held=9,
                        entry_price=3000.0, entry_base=-0.31,
                        entry_cycle="2026-06-02")
    assert j.get_hold_state("ethereum")["current_dir"] == -1
    assert j.get_hold_state("bitcoin")["current_dir"] == 0
    j.close()
