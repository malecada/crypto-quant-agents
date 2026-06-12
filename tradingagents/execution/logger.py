"""Persistent SQLite trade journal for the TradingAgents crypto framework.

Records every trade decision, full analyst reports from each propagate() run,
portfolio snapshots, and daily summaries for audit and thesis documentation.

Ported from Krypto-v0/src_live/logger.py, adapted for the TradingAgents
multi-agent architecture (5-level signal scale, analyst report storage,
crypto-native fields).

Self-contained: no dependencies beyond Python stdlib + sqlite3.
"""

import csv
import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    symbol            TEXT    NOT NULL,
    side              TEXT    NOT NULL,          -- BUY / SELL / HOLD
    quantity          REAL    NOT NULL DEFAULT 0,
    price             REAL    NOT NULL DEFAULT 0,
    stop_loss         REAL,
    signal            TEXT,                      -- BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
    confidence        TEXT,
    rf_prediction     REAL,
    arima_prediction  REAL,
    agent_output      TEXT,                      -- abbreviated final decision text
    pipeline_mode     TEXT,
    risk_check_reason TEXT,
    order_id          TEXT,
    status            TEXT    NOT NULL           -- EXECUTED / REJECTED / HOLD / DRY_RUN
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    total_value   REAL    NOT NULL,
    usdt_balance  REAL,
    positions_json TEXT                          -- JSON-encoded position map
);

CREATE TABLE IF NOT EXISTS daily_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          TEXT    NOT NULL UNIQUE,
    total_pnl     REAL,
    trade_count   INTEGER DEFAULT 0,
    wins          INTEGER DEFAULT 0,
    losses        INTEGER DEFAULT 0,
    rejected      INTEGER DEFAULT 0,
    holds         INTEGER DEFAULT 0,
    pipeline_mode TEXT
);

CREATE TABLE IF NOT EXISTS analyst_reports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id          INTEGER NOT NULL,
    market_report     TEXT,
    sentiment_report  TEXT,
    onchain_report    TEXT,
    prediction_report TEXT,
    debate_summary    TEXT,
    final_decision    TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(id)
);
"""

# ---------------------------------------------------------------------------
# Migrations -- each entry is a single ALTER TABLE statement.
# They are executed idempotently: if the column already exists the
# OperationalError is silently caught.
# ---------------------------------------------------------------------------

_MIGRATIONS: List[str] = [
    # --- trades ---
    "ALTER TABLE trades ADD COLUMN usdt_value REAL DEFAULT 0",
    # --- daily_summary ---
    "ALTER TABLE daily_summary ADD COLUMN starting_portfolio REAL DEFAULT 0",
    "ALTER TABLE daily_summary ADD COLUMN ending_portfolio REAL DEFAULT 0",
    "ALTER TABLE daily_summary ADD COLUMN dry_runs INTEGER DEFAULT 0",
    # --- analyst_reports ---
    "ALTER TABLE analyst_reports ADD COLUMN news_report TEXT",
    "ALTER TABLE analyst_reports ADD COLUMN fundamentals_report TEXT",
    "ALTER TABLE analyst_reports ADD COLUMN investment_plan TEXT",
    "ALTER TABLE analyst_reports ADD COLUMN risk_debate_summary TEXT",
]


def _get_default_db_path() -> Path:
    """Resolve the default DB path from TradingAgents config or fallback."""
    try:
        from tradingagents.dataflows.config import get_config

        config = get_config()
        # Allow config to specify a journal path explicitly.
        journal_path = config.get("trade_journal_db")
        if journal_path:
            return Path(journal_path)
        # Use the project_dir from config as base.
        project_dir = config.get("project_dir", ".")
        return Path(project_dir) / "data" / "trade_journal.db"
    except Exception:
        return Path("data/trade_journal.db")


class TradeJournal:
    """SQLite-backed trade journal for TradingAgents.

    Usage::

        journal = TradeJournal()                   # uses config-derived path
        journal = TradeJournal("my_journal.db")    # explicit path

        trade_id = journal.log_trade(
            symbol="bitcoin",
            side="BUY",
            quantity=0.05,
            price=62_000.0,
            signal="BUY",
            confidence="high",
        )

        journal.log_analyst_reports(trade_id, final_state)
        journal.log_portfolio_snapshot(100_000.0, 38_000.0, {"bitcoin": 0.05})
        journal.close()
    """

    def __init__(self, db_path: Optional[str] = None):
        """Create or open the trade journal database.

        Args:
            db_path: Path to the SQLite file.  When *None* the path is
                     resolved via ``tradingagents.dataflows.config.get_config``
                     with a fallback to ``data/trade_journal.db``.
        """
        resolved = Path(db_path) if db_path else _get_default_db_path()
        self.db_path = resolved
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._run_migrations()
        logger.info("TradeJournal opened at %s", self.db_path)

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _run_migrations(self) -> None:
        """Apply idempotent schema migrations (add-column only)."""
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    def _table_exists(self, name: str) -> bool:
        """Check whether a table exists in the database."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row[0] > 0

    def _column_exists(self, table: str, column: str) -> bool:
        """Check whether a column exists in *table*."""
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cursor.fetchall())

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float = 0.0,
        price: float = 0.0,
        signal: Optional[str] = None,
        confidence: Optional[str] = None,
        rf_pred: Optional[float] = None,
        arima_pred: Optional[float] = None,
        stop_loss: Optional[float] = None,
        agent_output: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        risk_check_reason: Optional[str] = None,
        order_id: Optional[str] = None,
        status: str = "EXECUTED",
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert a trade record and return its row id.

        Args:
            symbol: CoinGecko ID or ticker (e.g. ``"bitcoin"``).
            side: ``BUY``, ``SELL``, or ``HOLD``.
            quantity: Amount of the base asset traded.
            price: Execution / reference price at time of trade.
            signal: The 5-level TradingAgents signal
                    (``BUY`` / ``OVERWEIGHT`` / ``HOLD`` / ``UNDERWEIGHT`` / ``SELL``).
            confidence: Free-text confidence level (e.g. ``"high"``).
            rf_pred: Random Forest predicted price.
            arima_pred: ARIMA predicted price.
            stop_loss: Stop-loss price, if applicable.
            agent_output: Abbreviated portfolio-manager output text.
            pipeline_mode: ``"models_only"`` or ``"agent"`` etc.
            risk_check_reason: Why a trade was rejected, if applicable.
            order_id: Exchange order ID, if an order was placed.
            status: ``EXECUTED``, ``REJECTED``, ``HOLD``, or ``DRY_RUN``.
            timestamp: ISO-format timestamp; defaults to *now*.

        Returns:
            The integer primary-key ``trade_id`` of the inserted row.
        """
        ts = timestamp or datetime.utcnow().isoformat()
        # Truncate agent_output to keep the DB manageable.
        abbreviated = (
            (agent_output[:2000] + "...")
            if agent_output and len(agent_output) > 2000
            else agent_output
        )

        cur = self._conn.execute(
            """INSERT INTO trades
               (timestamp, symbol, side, quantity, price, stop_loss,
                signal, confidence, rf_prediction, arima_prediction,
                agent_output, pipeline_mode, risk_check_reason,
                order_id, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                symbol,
                side,
                quantity,
                price,
                stop_loss,
                signal,
                confidence,
                rf_pred,
                arima_pred,
                abbreviated,
                pipeline_mode,
                risk_check_reason,
                order_id,
                status,
            ),
        )
        self._conn.commit()
        trade_id = cur.lastrowid
        logger.info(
            "Logged trade #%d: %s %s %.6f %s @ %.2f [%s]",
            trade_id,
            side,
            symbol,
            quantity,
            signal or "-",
            price,
            status,
        )
        return trade_id

    # ------------------------------------------------------------------
    # Analyst report storage
    # ------------------------------------------------------------------

    def log_analyst_reports(self, trade_id: int, state: Dict[str, Any]) -> None:
        """Store the full agent state from a ``propagate()`` call.

        Extracts the analyst reports, debate summaries, and final decision
        from the TradingAgents ``AgentState`` dictionary and persists them
        alongside the trade record.

        Args:
            trade_id: The ``trade_id`` returned by :meth:`log_trade`.
            state: The ``final_state`` dictionary returned by
                   ``TradingAgentsGraph.propagate()``.  Keys used:

                   - ``market_report``
                   - ``sentiment_report``
                   - ``news_report``
                   - ``fundamentals_report``
                   - ``onchain_report``
                   - ``prediction_report``
                   - ``investment_debate_state`` (converted to JSON)
                   - ``risk_debate_state`` (converted to JSON)
                   - ``investment_plan``
                   - ``final_trade_decision``
        """

        def _safe(key: str, max_len: int = 50_000) -> Optional[str]:
            """Extract a string value; truncate if enormous."""
            val = state.get(key)
            if val is None:
                return None
            if isinstance(val, dict):
                val = json.dumps(val, default=str)
            val = str(val)
            if len(val) > max_len:
                val = val[:max_len] + "\n... [truncated]"
            return val

        # Build a combined debate summary from investment + risk debates.
        debate_parts: List[str] = []
        inv_debate = state.get("investment_debate_state")
        if inv_debate:
            summary = (
                inv_debate
                if isinstance(inv_debate, str)
                else json.dumps(inv_debate, default=str)
            )
            debate_parts.append(f"=== Investment Debate ===\n{summary}")
        risk_debate = state.get("risk_debate_state")
        if risk_debate:
            summary = (
                risk_debate
                if isinstance(risk_debate, str)
                else json.dumps(risk_debate, default=str)
            )
            debate_parts.append(f"=== Risk Debate ===\n{summary}")
        debate_summary = "\n\n".join(debate_parts) if debate_parts else None

        # Truncate debate_summary if needed.
        if debate_summary and len(debate_summary) > 100_000:
            debate_summary = debate_summary[:100_000] + "\n... [truncated]"

        self._conn.execute(
            """INSERT INTO analyst_reports
               (trade_id, market_report, sentiment_report, onchain_report,
                prediction_report, debate_summary, final_decision,
                news_report, fundamentals_report, investment_plan,
                risk_debate_summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                _safe("market_report"),
                _safe("sentiment_report"),
                _safe("onchain_report"),
                _safe("prediction_report"),
                debate_summary,
                _safe("final_trade_decision"),
                _safe("news_report"),
                _safe("fundamentals_report"),
                _safe("investment_plan"),
                # Separate risk_debate_summary for easy querying.
                _safe("risk_debate_state") if risk_debate else None,
            ),
        )
        self._conn.commit()
        logger.info("Stored analyst reports for trade #%d", trade_id)

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def log_portfolio_snapshot(
        self,
        total_value: float,
        usdt_balance: float = 0.0,
        positions: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a point-in-time portfolio snapshot.

        Args:
            total_value: Total portfolio value in USDT.
            usdt_balance: Free USDT balance.
            positions: Mapping of symbol to quantity/value, stored as JSON.
            timestamp: ISO-format; defaults to *now*.
        """
        ts = timestamp or datetime.utcnow().isoformat()
        positions_json = json.dumps(positions, default=str) if positions else None

        self._conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, total_value, usdt_balance, positions_json)
               VALUES (?, ?, ?, ?)""",
            (ts, total_value, usdt_balance, positions_json),
        )
        self._conn.commit()
        logger.debug("Portfolio snapshot: $%.2f", total_value)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def update_daily_summary(
        self,
        target_date: Optional[str] = None,
        total_pnl: Optional[float] = None,
        trade_count: Optional[int] = None,
        wins: Optional[int] = None,
        losses: Optional[int] = None,
        rejected: Optional[int] = None,
        holds: Optional[int] = None,
        pipeline_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute (or override) a daily summary row.

        If explicit counts are provided they are used directly.  Otherwise
        the method computes them from the trades table for *target_date*.

        Args:
            target_date: ISO date string (``YYYY-MM-DD``); defaults to today.
            total_pnl: Override the P&L value; computed from trades if *None*.
            trade_count: Override; computed if *None*.
            wins: Override; computed if *None*.
            losses: Override; computed if *None*.
            rejected: Override; computed if *None*.
            holds: Override; computed if *None*.
            pipeline_mode: Label for the pipeline used that day.

        Returns:
            A dict mirroring the ``daily_summary`` row.
        """
        target_date = target_date or date.today().isoformat()

        # Auto-compute from the trades table when callers omit values.
        all_today = self._conn.execute(
            "SELECT * FROM trades WHERE date(timestamp) = ?",
            (target_date,),
        ).fetchall()

        executed = [t for t in all_today if t["status"] == "EXECUTED"]

        if trade_count is None:
            trade_count = len(executed)
        if rejected is None:
            rejected = sum(1 for t in all_today if t["status"] == "REJECTED")
        if holds is None:
            holds = sum(1 for t in all_today if t["status"] == "HOLD")

        # Simple P&L approximation: SELL inflow minus BUY outflow.
        if total_pnl is None:
            pnl_row = self._conn.execute(
                """SELECT COALESCE(SUM(
                       CASE WHEN side='SELL' THEN quantity * price
                            WHEN side='BUY'  THEN -(quantity * price)
                            ELSE 0 END
                   ), 0) AS pnl
                   FROM trades
                   WHERE date(timestamp) = ? AND status = 'EXECUTED'""",
                (target_date,),
            ).fetchone()
            total_pnl = float(pnl_row["pnl"])

        if wins is None:
            wins = sum(1 for t in executed if t["side"] == "SELL")
        if losses is None:
            losses = trade_count - wins

        summary: Dict[str, Any] = {
            "date": target_date,
            "total_pnl": total_pnl,
            "trade_count": trade_count,
            "wins": wins,
            "losses": losses,
            "rejected": rejected,
            "holds": holds,
            "pipeline_mode": pipeline_mode or "",
        }

        self._conn.execute(
            """INSERT OR REPLACE INTO daily_summary
               (date, total_pnl, trade_count, wins, losses,
                rejected, holds, pipeline_mode)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                target_date,
                total_pnl,
                trade_count,
                wins,
                losses,
                rejected,
                holds,
                pipeline_mode or "",
            ),
        )
        self._conn.commit()
        logger.info(
            "Daily summary for %s: PnL=%.2f, trades=%d",
            target_date,
            total_pnl,
            trade_count,
        )
        return summary

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve trade records with optional filters.

        Args:
            symbol: Filter by asset symbol (exact match).
            start_date: Inclusive lower bound (ISO date ``YYYY-MM-DD``).
            end_date: Inclusive upper bound (ISO date ``YYYY-MM-DD``).

        Returns:
            List of dicts, each representing a trade row.
        """
        conditions: List[str] = ["1=1"]
        params: List[Any] = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_date:
            conditions.append("date(timestamp) >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date(timestamp) <= ?")
            params.append(end_date)

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM trades WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve daily summary rows within a date range.

        Args:
            start_date: Inclusive lower bound (ISO date).
            end_date: Inclusive upper bound (ISO date).

        Returns:
            List of dicts, each representing a daily_summary row.
        """
        conditions: List[str] = ["1=1"]
        params: List[Any] = []

        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM daily_summary WHERE {where} ORDER BY date",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_analyst_reports(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve the analyst reports associated with a trade.

        Args:
            trade_id: The trade's primary key.

        Returns:
            A dict with all report columns, or *None* if no reports stored.
        """
        row = self._conn.execute(
            "SELECT * FROM analyst_reports WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_portfolio_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve portfolio snapshots within a date range.

        Args:
            start_date: Inclusive lower bound (ISO date).
            end_date: Inclusive upper bound (ISO date).

        Returns:
            List of dicts.
        """
        conditions: List[str] = ["1=1"]
        params: List[Any] = []

        if start_date:
            conditions.append("date(timestamp) >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date(timestamp) <= ?")
            params.append(end_date)

        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT * FROM portfolio_snapshots WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str) -> int:
        """Export the trades table to a CSV file.

        Args:
            output_path: Destination file path.

        Returns:
            Number of rows written.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        rows = self._conn.execute(
            "SELECT * FROM trades ORDER BY timestamp"
        ).fetchall()
        if not rows:
            logger.warning("No trades to export.")
            return 0

        columns = rows[0].keys()
        with open(output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        count = len(rows)
        logger.info("Exported %d trades to %s", count, output)
        return count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("TradeJournal closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return f"TradeJournal(db_path={self.db_path!r})"
