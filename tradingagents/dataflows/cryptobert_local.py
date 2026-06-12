"""Local CryptoBERT sentiment baseline (Tier B6).

ElKulako/cryptobert via HuggingFace transformers — runs on CPU and
gives a deterministic sentiment label per text. Used by the Subjective
agent as a numerical anchor that does not depend on the LLM's training-
corpus prior on specific tickers.

Model is loaded lazily on first call and cached in module state. If
``transformers`` is not installed the helper returns ``None`` rather
than raising, so the rest of the pipeline degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_pipeline = None
_load_failed = False


def _ensure_pipeline():
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    try:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="ElKulako/cryptobert",
            top_k=None,  # return all label scores
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"CryptoBERT load failed: {exc}; sentiment will be None")
        _load_failed = True
        _pipeline = None
    return _pipeline


def sentiment(text: str) -> Optional[dict]:
    """Return ``{label: str, score: float, polarity: float ∈ [-1, 1]}`` or ``None``.

    Polarity = score(bullish) - score(bearish) so it can be averaged
    cross-corpus. CryptoBERT labels: ``Bearish | Neutral | Bullish``.
    """
    pipe = _ensure_pipeline()
    if pipe is None or not text:
        return None
    try:
        # Truncate to model max length (BERT-base = 512 tokens)
        out = pipe(text[:1500], truncation=True)
        if not out:
            return None
        scores = out[0] if isinstance(out[0], list) else out
        score_map = {item["label"].lower(): float(item["score"]) for item in scores}
        bull = score_map.get("bullish", 0.0)
        bear = score_map.get("bearish", 0.0)
        polarity = bull - bear
        top = max(scores, key=lambda x: x["score"])
        return {
            "label": top["label"],
            "score": float(top["score"]),
            "polarity": float(polarity),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"CryptoBERT inference failed: {exc}")
        return None
