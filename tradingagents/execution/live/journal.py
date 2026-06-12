"""SQLite forensic journal — one writer per pipeline step.

All schema in schema.sql. Designed for post-hoc reconstruction of any cycle.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Journal:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        # J1: WAL + busy_timeout so the runner's second raw connection (frequency
        # guard), the always-on monitor service, and ta-rebacktest can read/write
        # concurrently without "database is locked" aborting a cycle mid-loop.
        # Min-hold adds a hold_state write per coin, raising contention further.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout=10000;")
        self._conn.execute("PRAGMA foreign_keys = ON;")
        with open(_SCHEMA_PATH) as f:
            self._conn.executescript(f.read())
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def migrate(self) -> None:
        """Apply additive V5 schema columns to an existing v1 DB. Idempotent."""
        migrations = [
            ("predictions", "bundle_route", "TEXT"),
            ("retrains", "routes", "TEXT"),
            ("cycles", "n_trades", "INTEGER"),
            ("cycles", "notes", "TEXT"),
            ("cycles", "critical_data_fail_sources", "TEXT"),
            ("cycles", "supplementary_stale_sources", "TEXT"),
        ]
        conn = sqlite3.connect(self.db_path)
        try:
            for table, col, dtype in migrations:
                existing = {r[1] for r in conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()}
                if not existing:
                    # Table doesn't exist on this DB — skip (additive, non-failing).
                    continue
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")
            conn.commit()
        finally:
            conn.close()

    def log_cycle_start(self, cycle_id: str, *, git_sha: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO cycles (cycle_id, start_ts, git_commit_sha) "
            "VALUES (?, ?, ?)",
            (cycle_id, _utcnow_iso(), git_sha),
        )
        self._conn.commit()

    def log_cycle_end(self, cycle_id: str, *, status: str, error_msg: str = "") -> None:
        self._conn.execute(
            "UPDATE cycles SET end_ts = ?, status = ?, error_msg = ? WHERE cycle_id = ?",
            (_utcnow_iso(), status, error_msg, cycle_id),
        )
        self._conn.commit()

    def log_prediction(self, *, cycle_id, coin, horizon, model_path_sha,
                        pred_value, ref_price, signal_h7, signal_h14, consensus_signal,
                        pred_quantile_low=None, pred_quantile_high=None) -> None:
        self._conn.execute(
            "INSERT INTO predictions (cycle_id, coin, horizon, model_path_sha, "
            "pred_value, pred_quantile_low, pred_quantile_high, ref_price, "
            "signal_h7, signal_h14, consensus_signal) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, coin, horizon, model_path_sha, pred_value,
             pred_quantile_low, pred_quantile_high, ref_price,
             signal_h7, signal_h14, consensus_signal),
        )
        self._conn.commit()

    def log_sizing(self, *, cycle_id, coin, realized_vol, target_vol, kelly,
                    confidence, base_size, leverage, sma30_multiplier,
                    final_size_notional) -> None:
        self._conn.execute(
            "INSERT INTO sizing (cycle_id, coin, realized_vol, target_vol, kelly, "
            "confidence, base_size, leverage, sma30_multiplier, final_size_notional) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, coin, realized_vol, target_vol, kelly, confidence,
             base_size, leverage, sma30_multiplier, final_size_notional),
        )
        self._conn.commit()

    def log_risk_check(self, cycle_id, coin, check_name, passed: bool,
                        value, threshold, reason: str) -> None:
        self._conn.execute(
            "INSERT INTO risk_checks (cycle_id, coin, check_name, passed, value, threshold, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, coin, check_name, 1 if passed else 0, value, threshold, reason),
        )
        self._conn.commit()

    def log_trade(self, *, cycle_id, coin, side, qty, entry_price, exit_price,
                   pnl, fees, slippage, order_id, stop_loss_id, status) -> int:
        """Insert a trade row and return its autoincrement id.

        The id lets callers later backfill realized PnL + commission via
        :meth:`update_trade_fills` once Binance has reported fills for the
        order (the placement response does not include them).
        """
        cur = self._conn.execute(
            "INSERT INTO trades (cycle_id, coin, side, qty, entry_price, exit_price, "
            "pnl, fees, slippage, order_id, stop_loss_id, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, coin, side, qty, entry_price, exit_price, pnl, fees,
             slippage, order_id, stop_loss_id, status),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def update_trade_fills(self, trade_id: int, *, fees: float,
                            realized_pnl: float) -> None:
        """Backfill `fees` and `pnl` for an existing trade row by id.

        The live runner records trades at entry time with `fees=None,
        pnl=None`; Binance Futures only exposes the per-fill `commission`
        and `realizedPnl` via `/fapi/v1/userTrades?orderId=...` after the
        order has filled. After a successful `place_market_order` the
        runner sums those fields across the fills and calls this method.

        No-op when `trade_id` does not exist (e.g. journal rotated between
        insert and update).
        """
        self._conn.execute(
            "UPDATE trades SET fees=?, pnl=? WHERE id=?",
            (fees, realized_pnl, int(trade_id)),
        )
        self._conn.commit()

    def log_portfolio_snapshot(self, cycle_id, total_value, usdt_balance,
                                position_qty_per_coin: dict, unrealized_pnl) -> None:
        self._conn.execute(
            "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, usdt_balance, "
            "position_qty_per_coin, unrealized_pnl) VALUES (?, ?, ?, ?, ?, ?)",
            (cycle_id, _utcnow_iso(), total_value, usdt_balance,
             json.dumps(position_qty_per_coin), unrealized_pnl),
        )
        self._conn.commit()

    def peak_total_value(self) -> float:
        row = self._conn.execute(
            "SELECT MAX(total_value) FROM portfolio_snapshots"
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def log_feature_snapshot(self, cycle_id, coin, feature_name, value, source) -> None:
        self._conn.execute(
            "INSERT INTO feature_snapshots (cycle_id, coin, feature_name, value, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (cycle_id, coin, feature_name, value, source),
        )
        self._conn.commit()

    def log_model_artifact(self, *, retrain_id, model_path, train_window_start,
                            train_window_end, train_rows, train_dir_acc_h7,
                            train_dir_acc_h14, sha256) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO model_artifacts (retrain_id, ts, model_path, "
            "train_window_start, train_window_end, train_rows, "
            "train_dir_acc_h7, train_dir_acc_h14, sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (retrain_id, _utcnow_iso(), model_path, train_window_start,
             train_window_end, train_rows, train_dir_acc_h7,
             train_dir_acc_h14, sha256),
        )
        self._conn.commit()

    # ── V5 record_* API (Task 12) ──────────────────────────────────────
    # These complement the legacy log_* methods. The runner uses them for
    # terminal cycle records (with status + n_trades + notes + V5 columns),
    # retrain artifact summary (with routes), and per-prediction rows
    # (with bundle_route). Backward compatible: every new column is optional.

    def record_cycle(self, *, cycle_id: str, start_ts: str, end_ts: str,
                      status: str, n_trades: int = 0, notes: str | None = None,
                      critical_data_fail_sources: str | None = None,
                      supplementary_stale_sources: str | None = None) -> None:
        """Upsert a terminal cycle record. Uses INSERT OR REPLACE keyed on
        cycle_id so calling record_cycle on a row already created by
        log_cycle_start updates it in place.
        """
        # Preserve git_commit_sha if a prior log_cycle_start row exists.
        existing = self._conn.execute(
            "SELECT git_commit_sha FROM cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        git_sha = existing[0] if existing else None
        self._conn.execute(
            "INSERT OR REPLACE INTO cycles (cycle_id, start_ts, end_ts, status, "
            "error_msg, git_commit_sha, n_trades, notes, "
            "critical_data_fail_sources, supplementary_stale_sources) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, start_ts, end_ts, status, None, git_sha,
             n_trades, notes,
             critical_data_fail_sources, supplementary_stale_sources),
        )
        self._conn.commit()

    def record_retrain(self, *, retrain_id: str, cycle_id: str,
                        checkpoint_path: str, checkpoint_sha: str,
                        n_train_rows: int, train_window_start: str,
                        train_dir_acc: float, status: str,
                        routes: str | None = None) -> None:
        """Insert (or replace) a row in retrains capturing the V5 composite."""
        self._conn.execute(
            "INSERT OR REPLACE INTO retrains (retrain_id, cycle_id, "
            "checkpoint_path, checkpoint_sha, n_train_rows, "
            "train_window_start, train_dir_acc, status, routes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (retrain_id, cycle_id, checkpoint_path, checkpoint_sha,
             n_train_rows, train_window_start, train_dir_acc, status, routes),
        )
        self._conn.commit()

    def record_predictions(self, *, cycle_id: str, preds_df) -> None:
        """Insert one row per (coin, horizon) from a V5 predict.run_predict
        DataFrame. Expects columns: coin, horizon, prediction, ref_price,
        bundle_route. Silently no-ops on an empty frame.
        """
        if preds_df is None or len(preds_df) == 0:
            return
        rows = [
            (cycle_id, str(r["coin"]), int(r["horizon"]),
             float(r["prediction"]), float(r["ref_price"]),
             str(r["bundle_route"]) if "bundle_route" in r else None)
            for _, r in preds_df.iterrows()
        ]
        # `predictions` columns: cycle_id, coin, horizon, model_path_sha,
        # pred_value, pred_quantile_low, pred_quantile_high, ref_price,
        # signal_h7, signal_h14, consensus_signal, bundle_route.
        # V5 record path leaves quant/signal columns NULL — they are
        # populated by log_prediction() during the per-coin sizing loop.
        self._conn.executemany(
            "INSERT INTO predictions (cycle_id, coin, horizon, pred_value, "
            "ref_price, bundle_route) VALUES (?, ?, ?, ?, ?, ?)",
            [(cid, coin, h, pred, ref, route)
             for (cid, coin, h, pred, ref, route) in rows],
        )
        self._conn.commit()

    # ── P1 stateful min-hold ───────────────────────────────────────────
    def get_hold_state(self, coin: str) -> dict | None:
        """Return the per-coin hold state, or None if the coin has no row yet."""
        row = self._conn.execute(
            "SELECT coin, current_dir, bars_held, entry_price, entry_base, "
            "entry_cycle FROM hold_state WHERE coin = ?", (coin,),
        ).fetchone()
        if row is None:
            return None
        return {
            "coin": row[0], "current_dir": int(row[1]), "bars_held": int(row[2]),
            "entry_price": float(row[3]), "entry_base": float(row[4]),
            "entry_cycle": row[5],
        }

    def upsert_hold_state(self, *, coin, current_dir, bars_held,
                          entry_price, entry_base, entry_cycle) -> None:
        """Insert or replace the per-coin hold state after a cycle's sizing."""
        self._conn.execute(
            "INSERT OR REPLACE INTO hold_state (coin, current_dir, bars_held, "
            "entry_price, entry_base, entry_cycle, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (coin, int(current_dir), int(bars_held), float(entry_price),
             float(entry_base), entry_cycle, _utcnow_iso()),
        )
        self._conn.commit()

    def log_modulator(self, *, cycle_id: str, coin: str, multiplier: float,
                      effective_weight: float, llm_confidence: float | None = None,
                      regime: str | None = None, fallback: bool = False) -> None:
        """Persist the hybrid modulator outputs for one coin/cycle.

        Written even when the modulator degraded to pure quant (1.0, 0.0)
        with fallback=True, so the UI can label the row honestly.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO modulator_outputs "
            "(cycle_id, coin, multiplier, effective_weight, llm_confidence, "
            "regime, fallback) VALUES (?,?,?,?,?,?,?)",
            (cycle_id, coin, multiplier, effective_weight, llm_confidence,
             regime, 1 if fallback else 0),
        )
        self._conn.commit()

    def log_shadow_decision(self, *, cycle_id, coin, live_signal, backtest_signal,
                             live_size, backtest_size) -> None:
        agree = 1 if live_signal == backtest_signal else 0
        if abs(backtest_size) > 1e-9:
            size_delta_pct = abs(live_size - backtest_size) / abs(backtest_size)
        else:
            size_delta_pct = 0.0 if abs(live_size) < 1e-9 else float("inf")
        self._conn.execute(
            "INSERT INTO shadow_decisions (cycle_id, coin, live_signal, backtest_signal, "
            "agree, live_size, backtest_size, size_delta_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (cycle_id, coin, live_signal, backtest_signal, agree,
             live_size, backtest_size, size_delta_pct),
        )
        self._conn.commit()


if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="V5 journal migration")
    parser.add_argument("--migrate", action="store_true", required=True,
                        help="Apply additive V5 schema columns to the configured journal DB")
    parser.add_argument("--db", default=os.environ.get(
        "JOURNAL_DB", "/opt/tradingagents/data/trade_journal.db"))
    args = parser.parse_args()
    if not os.path.exists(args.db):
        print(f"ERROR: DB does not exist at {args.db}", file=sys.stderr)
        print("If this is a fresh deployment, ensure the live runner has been started "
              "at least once (which creates the DB via schema.sql) before running --migrate.",
              file=sys.stderr)
        sys.exit(2)
    Journal(args.db).migrate()
    print(f"V5 migration applied to {args.db}")
