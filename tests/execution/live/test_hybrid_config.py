from tradingagents.execution.live.hybrid_compose import build_hybrid_config, HYBRID_ANALYSTS


def test_pins_gpt_4o_mini_both_slots():
    cfg = build_hybrid_config(quant_pred_dir="/tmp/x")
    assert cfg["deep_think_llm"] == "gpt-4o-mini"
    assert cfg["quick_think_llm"] == "gpt-4o-mini"
    assert cfg["llm_provider"] == "openai"


def test_points_quant_pred_dir():
    cfg = build_hybrid_config(quant_pred_dir="/tmp/cycle/preds")
    assert cfg["quant_pred_dir"] == "/tmp/cycle/preds"


def test_replay_cache_off_for_live():
    assert build_hybrid_config(quant_pred_dir="/tmp/x")["replay_cache"] is False


def test_analyst_set_drops_sentiment():
    assert HYBRID_ANALYSTS == ["market", "onchain", "prediction"]
    assert "crypto_sentiment" not in HYBRID_ANALYSTS
