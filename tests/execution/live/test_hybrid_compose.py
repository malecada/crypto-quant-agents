# tests/execution/live/test_hybrid_compose.py
import math
from tradingagents.execution.live.hybrid_compose import compose_final

def test_multiplier_one_is_identity():
    assert compose_final(base=0.8, multiplier=1.0, effective_weight=0.7) == 0.8

def test_effective_weight_zero_is_identity():
    assert compose_final(base=0.8, multiplier=1.5, effective_weight=0.0) == 0.8

def test_full_formula():
    # base * (1 + eff_w*(mult-1)) = 0.8*(1+0.5*(1.4-1)) = 0.8*1.2 = 0.96
    assert math.isclose(compose_final(base=0.8, multiplier=1.4, effective_weight=0.5), 0.96)

def test_negative_base_preserves_sign():
    # short base, mult>1 levers the short further
    assert math.isclose(compose_final(base=-0.5, multiplier=1.2, effective_weight=1.0), -0.6)


from tradingagents.execution.live.hybrid_compose import extract_modulator_outputs

def test_extract_none_is_pure_quant():
    mult, eff_w = extract_modulator_outputs(None)
    assert (mult, eff_w) == (1.0, 0.0)

def test_extract_missing_keys_is_pure_quant():
    mult, eff_w = extract_modulator_outputs({"coin": "bitcoin"})
    assert (mult, eff_w) == (1.0, 0.0)

def test_extract_reads_fields():
    mp = {"llm_multiplier": 1.3, "effective_weight": 0.6, "position": 999.0}
    assert extract_modulator_outputs(mp) == (1.3, 0.6)

def test_extract_clamps_multiplier_to_contract_bounds():
    # ModulatedPosition bounds llm_multiplier to [0, 1.5]
    assert extract_modulator_outputs({"llm_multiplier": 9.0, "effective_weight": 0.5})[0] == 1.5
    assert extract_modulator_outputs({"llm_multiplier": -9.0, "effective_weight": 0.5})[0] == 0.0

def test_extract_clamps_effective_weight_to_unit_interval():
    # effective_weight is a weight, bounded to [0, 1]
    assert extract_modulator_outputs({"llm_multiplier": 1.0, "effective_weight": 9.0})[1] == 1.0
    assert extract_modulator_outputs({"llm_multiplier": 1.0, "effective_weight": -1.0})[1] == 0.0
