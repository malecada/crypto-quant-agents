"""LLM replay cache for deterministic and cost-free backtest reruns.

Wraps a LangChain chat model so that identical prompts return cached
responses from SQLite.  This makes warm backtest reruns deterministic
(same prompt -> same response) and free (no API call on cache hit).

Usage:
    from tradingagents.llm_clients.replay_cache import CachedChatModel

    llm = ChatOpenAI(model="gpt-4o")
    cached_llm = CachedChatModel(llm, db_path="data/llm_cache.db")

    # First call hits API and stores response
    result = cached_llm.invoke(messages)

    # Second identical call returns cached response (no API call)
    result = cached_llm.invoke(messages)
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.load import dumpd, load

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.path.join("data", "llm_replay_cache.db")


def _hash_input(messages: Any, bound_tools: Any, model_name: str) -> str:
    """Produce a deterministic SHA-256 hash of the LLM input."""
    # Serialize messages via LangChain's dumpd for consistency
    try:
        if isinstance(messages, (list, tuple)):
            serialized_msgs = [dumpd(m) if isinstance(m, BaseMessage) else str(m) for m in messages]
        else:
            serialized_msgs = dumpd(messages) if isinstance(messages, BaseMessage) else str(messages)
    except Exception:
        serialized_msgs = str(messages)

    payload = json.dumps(
        {"messages": serialized_msgs, "tools": str(bound_tools), "model": model_name},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedChatModel(BaseChatModel):
    """Wrapper that caches LLM responses in SQLite for replay.

    Delegates all calls to the underlying model but intercepts invoke()
    to check the cache first.  Cache key = SHA-256(messages + tools + model).
    """

    delegate: Any  # The wrapped LangChain chat model
    db_path: str = _DEFAULT_DB_PATH
    enabled: bool = True
    _local: Any = None  # threading.local for per-thread connections

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, delegate: BaseChatModel, db_path: str = _DEFAULT_DB_PATH, enabled: bool = True, **kwargs):
        super().__init__(delegate=delegate, db_path=db_path, enabled=enabled, **kwargs)
        # Bypass Pydantic v2 which skips private (underscore) attribute assignment.
        object.__setattr__(self, "_local", threading.local())
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        local = object.__getattribute__(self, "_local") if hasattr(self, "__dict__") and "_local" in self.__dict__ else None
        if local is None:
            local = threading.local()
            object.__setattr__(self, "_local", local)
        if not hasattr(local, "conn") or local.conn is None:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            local.conn = sqlite3.connect(self.db_path)
        return local.conn

    def _init_db(self):
        conn = self._get_conn()
        # Enable WAL so multiple processes can read/write concurrently
        # without "database is locked" errors during parallel signal gen.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")  # 10s before raising lock error
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT,
                response_json TEXT,
                created_at REAL
            )
        """)
        conn.commit()

    def _lookup(self, key: str) -> Optional[BaseMessage]:
        """Return cached response or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            return load(json.loads(row[0]))
        except Exception as e:
            logger.warning(f"Cache deserialization failed for {key[:12]}...: {e}")
            return None

    def _store(self, key: str, response: BaseMessage):
        """Store a response in the cache."""
        try:
            response_json = json.dumps(dumpd(response), default=str)
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, model, response_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, self._get_model_name(), response_json, time.time()),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Cache store failed for {key[:12]}...: {e}")

    def _get_model_name(self) -> str:
        return getattr(self.delegate, "model_name", getattr(self.delegate, "model", "unknown"))

    # ── BaseChatModel interface ──────────────────────────────────────

    @property
    def _llm_type(self) -> str:
        return f"cached-{getattr(self.delegate, '_llm_type', 'chat')}"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Required by BaseChatModel. Delegates to the wrapped model."""
        return self.delegate._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        if not self.enabled:
            return self.delegate.invoke(input, config, **kwargs)

        bound_tools = getattr(self.delegate, "bound_tools", None) or getattr(self.delegate, "tools", None)
        key = _hash_input(input, bound_tools, self._get_model_name())

        cached = self._lookup(key)
        if cached is not None:
            logger.debug(f"LLM cache HIT: {key[:12]}...")
            return cached

        logger.debug(f"LLM cache MISS: {key[:12]}...")
        response = self.delegate.invoke(input, config, **kwargs)
        self._store(key, response)
        return response

    def bind_tools(self, tools, **kwargs):
        """Return a new CachedChatModel wrapping the tool-bound delegate."""
        bound_delegate = self.delegate.bind_tools(tools, **kwargs)
        return CachedChatModel(bound_delegate, db_path=self.db_path, enabled=self.enabled)

    # Forward attribute access to the delegate for LangChain compatibility
    def __getattr__(self, name):
        if name in ("delegate", "db_path", "enabled", "_local"):
            raise AttributeError(name)
        return getattr(self.delegate, name)

    # ── Cache management ─────────────────────────────────────────────

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM llm_cache").fetchone()
        return {
            "entries": row[0],
            "oldest": row[1],
            "newest": row[2],
        }

    def clear_cache(self):
        """Delete all cached entries."""
        conn = self._get_conn()
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
        logger.info("LLM replay cache cleared")
