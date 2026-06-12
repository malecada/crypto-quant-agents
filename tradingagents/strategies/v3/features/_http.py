"""Shared HTTP helpers for V3 feature fetchers."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitError(RuntimeError):
    """Raised when an upstream API returns 429."""


def with_backoff(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_backoff: float = 1.0,
    max_backoff: float = 60.0,
) -> T:
    """Run ``fn``; on RateLimitError, retry with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            wait = min(base_backoff * (2**attempt), max_backoff)
            logger.warning("429, sleeping %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
    raise RateLimitError(f"Failed after {max_retries} retries")
