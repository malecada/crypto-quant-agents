"""Self-MoA wrapper: replay-cached N-sample inference (Tier B5).

Self-MoA (Li, Lin, Xia, Jin 2025, arXiv 2502.00674) shows that
aggregating ``N`` samples from a single top model beats mixing different
models by +6.6% on AlpacaEval. We use it to compute calibration
uncertainty for the modulator's multiplier:

    samples = MultiSampleCachedChatModel(model, n=5).invoke(messages)
    multipliers = [parse_multiplier(s.content) for s in samples]
    mean = numpy.mean(multipliers)
    uncertainty = numpy.std(multipliers)

This wrapper extends the existing ``CachedChatModel`` so backtests are
deterministic: each ``sample_idx`` gets its own cache entry keyed on
``(prompt_hash, sample_idx, temperature)``. First run materialises all
N samples; reruns are SQLite reads (no API spend).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from typing import Any, Iterable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load.dump import dumpd
from langchain_core.messages import AIMessage, BaseMessage

from tradingagents.llm_clients.replay_cache import (
    _DEFAULT_DB_PATH,
    _hash_input,
)

logger = logging.getLogger(__name__)


class MultiSampleCachedChatModel:
    """N-sample wrapper for Self-MoA with replay-cache support.

    NOT a ``BaseChatModel`` itself — exposes a single ``sample_n()``
    method that returns a list of ``AIMessage`` so callers can compute
    aggregate statistics. Use this only for the modulator path; the rest
    of the graph keeps the single-call ``CachedChatModel``.
    """

    def __init__(
        self,
        delegate: BaseChatModel,
        n: int = 5,
        temperature: float = 0.5,
        db_path: str = _DEFAULT_DB_PATH,
        enabled: bool = True,
    ):
        self.delegate = delegate
        self.n = int(n)
        self.temperature = float(temperature)
        self.db_path = db_path
        self.enabled = enabled
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            import os
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache_multi (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                sample_idx INTEGER,
                response_json TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        conn.commit()

    def _model_name(self) -> str:
        for attr in ("model", "model_name", "deployment_name"):
            v = getattr(self.delegate, attr, None)
            if v:
                return str(v)
        return self.delegate.__class__.__name__

    def _key(
        self, messages: Any, sample_idx: int, temperature: float
    ) -> str:
        base = _hash_input(messages, None, self._model_name())
        suffix = f"|s{sample_idx}|t{temperature:.3f}"
        return hashlib.sha256((base + suffix).encode()).hexdigest()

    def _try_cache(self, key: str) -> Optional[AIMessage]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT response_json FROM llm_cache_multi WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
            return AIMessage(content=payload.get("content", ""))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"multi_sample cache decode failed: {exc}")
            return None

    def _store(self, key: str, sample_idx: int, msg: AIMessage):
        conn = self._get_conn()
        try:
            content = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content)
            )
        except Exception:
            content = str(msg.content)
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache_multi "
            "(cache_key, model, sample_idx, response_json) "
            "VALUES (?, ?, ?, ?)",
            (key, self._model_name(), sample_idx, json.dumps({"content": content})),
        )
        conn.commit()

    def _set_temperature(self, t: float):
        """Best-effort temperature override on the wrapped model."""
        for attr in ("temperature",):
            if hasattr(self.delegate, attr):
                try:
                    setattr(self.delegate, attr, t)
                except Exception:
                    pass

    def sample_n(self, messages: Iterable[BaseMessage]) -> list[AIMessage]:
        """Return ``n`` samples for the same prompt, replay-cached."""
        msg_list = list(messages)
        out: list[AIMessage] = []
        if self.enabled:
            self._set_temperature(self.temperature)
        for i in range(self.n):
            key = self._key(msg_list, i, self.temperature)
            cached = self._try_cache(key) if self.enabled else None
            if cached is not None:
                out.append(cached)
                continue
            try:
                resp = self.delegate.invoke(msg_list)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"sample {i} failed: {exc}")
                continue
            if not isinstance(resp, AIMessage):
                resp = AIMessage(content=str(resp))
            if self.enabled:
                self._store(key, i, resp)
            out.append(resp)
        return out
