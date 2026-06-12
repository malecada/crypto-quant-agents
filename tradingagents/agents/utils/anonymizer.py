"""Asset-name anonymization for LLM prompts (Glasserman & Lin 2309.17322).

The literal token "Bitcoin" is a bearish trigger in pretrained LLMs because
the training corpus is dominated by 2018/2022 crashes, regulatory FUD,
Mt. Gox, Luna, FTX. Replacing it with an anonymous alias ("Asset_X") and
re-attaching the identity only at the Portfolio Manager (Layer 3) cuts
this prior. Choi et al. (2510.07517) shows the same effect for debate
aggregators — persona-anonymized debates reduce sycophancy bias toward
whichever side aligns with the base model's prior.

This module is the *single* source of truth for the alias mapping so
masking and un-masking remain reversible. The alias is deterministic
per ``(coin, propagate_id)`` so multi-coin runs use stable labels
within one analysis pass.
"""

from __future__ import annotations

import hashlib
from threading import Lock

# Stable 26-letter alphabet for alias suffixes (skip ambiguous I/O)
_ALIAS_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ"

# Module-level mapping persisted across the lifetime of one ``propagate()``.
# Each ``configure()`` call resets the mapping for the new run.
_lock = Lock()
_mapping: dict[str, str] = {}
_reverse: dict[str, str] = {}
_enabled: bool = False


def configure(enabled: bool) -> None:
    """Reset the per-run mapping. Call at the start of each ``propagate()``."""
    global _enabled
    with _lock:
        _mapping.clear()
        _reverse.clear()
        _enabled = bool(enabled)


def is_enabled() -> bool:
    return _enabled


def mask(coin: str) -> str:
    """Return a stable alias for ``coin`` (creates one on first call).

    No-op if anonymization is disabled — returns ``coin`` unchanged.
    """
    if not _enabled:
        return coin
    coin_norm = coin.lower()
    with _lock:
        if coin_norm in _mapping:
            return _mapping[coin_norm]
        # Deterministic alias suffix from a hash so repeated runs of the
        # same coin always get the same alias, but multi-coin runs still
        # get distinct labels.
        h = hashlib.sha256(coin_norm.encode()).digest()
        idx = h[0] % len(_ALIAS_ALPHA)
        # Disambiguate if the slot is already taken by a different coin
        # (rare, but possible when coins hash to the same first byte).
        while True:
            alias = f"Asset_{_ALIAS_ALPHA[idx]}"
            if alias not in _reverse:
                break
            idx = (idx + 1) % len(_ALIAS_ALPHA)
        _mapping[coin_norm] = alias
        _reverse[alias] = coin_norm
        return alias


def unmask(text: str, coin: str) -> str:
    """Re-attach the real coin name to a previously-masked text."""
    if not _enabled:
        return text
    alias = _mapping.get(coin.lower())
    if alias is None:
        return text
    return text.replace(alias, coin)


def alias_to_coin() -> dict[str, str]:
    """Return a copy of the active alias→coin map (for PM un-mask)."""
    with _lock:
        return dict(_reverse)
