from unittest.mock import MagicMock
from tradingagents.graph.signal_processing import SignalProcessor


def test_extract_confidence_high():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "HIGH"
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("FINAL TRANSACTION PROPOSAL: **BUY**\nConfidence: HIGH") == "HIGH"


def test_extract_confidence_unknown_when_not_mentioned():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "UNKNOWN"
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("Some text without confidence") == "UNKNOWN"


def test_extract_confidence_normalizes_whitespace_case():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "  medium  "
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("Some text ... Confidence: Medium ...") == "MEDIUM"


def test_extract_confidence_defaults_unknown_on_bad_llm_output():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "definitely not a valid label"
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("...") == "UNKNOWN"


def test_extract_confidence_strips_trailing_punctuation():
    """The inference rubric prompt allows an LLM to emit 'HIGH.' or 'HIGH confidence' —
    the parser must still map that to HIGH, not UNKNOWN."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "HIGH."
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("strong buy signal") == "HIGH"


def test_extract_confidence_takes_first_word_when_explanatory():
    """If the LLM adds 'HIGH confidence' we still get HIGH."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "LOW confidence — hedged HOLD"
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("monitor closely, conflicting signals") == "LOW"


def test_extract_confidence_handles_markdown_wrap():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "**MEDIUM**"
    sp = SignalProcessor(mock_llm)
    assert sp.extract_confidence("clear lean but acknowledges risks") == "MEDIUM"
