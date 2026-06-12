"""Weekly V5 drift check — live journal metrics + parity refetch-and-replay.

`compute_live_metrics` summarises the live `portfolio_snapshots` table.
`run_weekly_parity` shells `scripts/parity_refetch_and_replay.py`, which
refetches every data source fresh into a sandbox, replays V5 MIX over the
live cycle window, and diffs against the live journal — the V5-correct
successor to the retired V1 `baseline_strategy_v2` re-backtest.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# repo root: tradingagents/execution/live/rebacktest.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


def compute_live_metrics(live_start_date, live_end_date) -> dict:
    """Compute Sharpe / Return / MaxDD / win-rate from `portfolio_snapshots`.

    Reads `$DATA_DIR/trade_journal.db` (default ``data/``) and computes
    metrics over the inclusive date range [live_start_date, live_end_date].
    Returns NaN/zero defaults if the DB is missing or has fewer than two
    snapshots in the window — callers must tolerate that.
    """
    import re
    import sqlite3
    import numpy as np

    db = Path(os.environ.get("DATA_DIR", "data")) / "trade_journal.db"
    if not db.exists():
        return {
            "sharpe": float("nan"),
            "return_pct": 0.0,
            "max_dd": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
        }
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT cycle_id, ts, total_value FROM portfolio_snapshots "
        "WHERE date(ts) >= ? AND date(ts) <= ? ORDER BY ts",
        (live_start_date, live_end_date),
    ).fetchall()
    conn.close()
    # Keep only scheduled daily cycles (cycle_id == YYYY-MM-DD) and collapse to
    # one equity point per trading day (the last snapshot of the day). This
    # drops manual/deploy/dryrun cycles and the deploy-day balance reset: its
    # pre/post-reset double-snapshot (same daily cycle_id) would otherwise
    # inject a phantom return — e.g. a faucet top-up read as +7.8% profit on
    # week 1. Rows are ts-ordered, so the last write per cycle_id wins.
    _daily = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    by_day: dict[str, float] = {}
    for cycle_id, _ts, val in rows:
        if cycle_id is not None and _daily.match(cycle_id) and val is not None:
            by_day[cycle_id] = val
    if len(by_day) < 2:
        return {
            "sharpe": float("nan"),
            "return_pct": 0.0,
            "max_dd": 0.0,
            "n_trades": len(by_day),
            "win_rate": 0.0,
        }
    values = np.array([by_day[d] for d in sorted(by_day)], dtype=float)
    rets = np.diff(values) / values[:-1]
    if len(rets) > 1 and np.std(rets, ddof=1) > 0:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.max((peak - cum) / peak)) if len(cum) else 0.0
    return {
        "sharpe": sharpe,
        "return_pct": float((values[-1] - values[0]) / values[0]),
        "max_dd": dd,
        "n_trades": len(values),
        "win_rate": float(np.mean(rets > 0)) if len(rets) else 0.0,
    }


def _journal_db() -> Path:
    """The live trade journal path ($DATA_DIR/trade_journal.db)."""
    return Path(os.environ.get("DATA_DIR", "data")) / "trade_journal.db"


def compute_daily_pnl_pct(current_value: float, asof_date: str) -> float:
    """Today's PnL fraction vs the most recent snapshot *before* asof_date.

    L1 fix: the old kill-switch fed `compute_live_metrics(today, today)`, which
    only saw today's single snapshot and always returned 0.0. Here we compare
    the live current equity (`current_value`, which the runner already reads as
    `portfolio_before`) against the prior-day close so a real intraday loss is
    visible to `check_daily_loss`. Returns 0.0 when there is no prior snapshot
    or the values are non-positive (safe: no false kill on day one).
    """
    import sqlite3

    db = _journal_db()
    if not db.exists() or current_value <= 0:
        return 0.0
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT total_value FROM portfolio_snapshots WHERE date(ts) < ? "
        "ORDER BY ts DESC LIMIT 1",
        (asof_date,),
    ).fetchone()
    conn.close()
    if not row or row[0] is None or row[0] <= 0:
        return 0.0
    prior = float(row[0])
    return (current_value - prior) / prior


def compute_drawdown_from_peak(current_value: float, asof_date: str) -> float:
    """Drawdown fraction of current equity from the running peak.

    Peak is the max snapshot value on or before asof_date, combined with the
    live current value (so a fresh high reports 0.0 drawdown). Mirrors the
    backtest's portfolio-level circuit breaker (max_portfolio_dd). Returns 0.0
    when there is no history.
    """
    import sqlite3

    db = _journal_db()
    if not db.exists() or current_value <= 0:
        return 0.0
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT MAX(total_value) FROM portfolio_snapshots WHERE date(ts) <= ?",
        (asof_date,),
    ).fetchone()
    conn.close()
    hist_peak = float(row[0]) if row and row[0] is not None else 0.0
    peak = max(hist_peak, current_value)
    if peak <= 0:
        return 0.0
    return (peak - current_value) / peak


def _read_daily_equity(db_path) -> dict[str, float]:
    """Read portfolio_snapshots from *db_path* and return a {YYYY-MM-DD: value}
    dict using the same filtering logic as compute_live_metrics (only
    scheduled daily cycle_ids, last snapshot of each day wins).

    Returns an empty dict when the file is missing or the table is empty.
    """
    import re
    import sqlite3

    db = Path(db_path)
    if not db.exists():
        return {}
    _daily = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    conn = sqlite3.connect(str(db))
    rows = conn.execute(
        "SELECT cycle_id, ts, total_value FROM portfolio_snapshots ORDER BY ts"
    ).fetchall()
    conn.close()
    by_day: dict[str, float] = {}
    for cycle_id, _ts, val in rows:
        if cycle_id is not None and _daily.match(cycle_id) and val is not None:
            by_day[cycle_id] = val
    return by_day


def _metrics_from_values(values) -> dict:
    """Compute Sharpe / total-return / max-drawdown from an ordered equity
    sequence.  Reuses the same arithmetic as compute_live_metrics.

    Args:
        values: sequence of float equity values, oldest first.

    Returns:
        Dict with keys ``sharpe``, ``ret``, ``maxdd``.
    """
    import numpy as np

    arr = np.array(list(values), dtype=float)
    if len(arr) < 2:
        return {"sharpe": float("nan"), "ret": 0.0, "maxdd": 0.0}
    rets = np.diff(arr) / arr[:-1]
    if len(rets) > 1 and np.std(rets, ddof=1) > 0:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    maxdd = float(np.max((peak - cum) / peak)) if len(cum) else 0.0
    return {
        "sharpe": sharpe,
        "ret": float((arr[-1] - arr[0]) / arr[0]),
        "maxdd": maxdd,
    }


def compare_quant_hybrid(
    quant_db_path,
    hybrid_db_path,
    coins: list,
) -> dict:
    """Compare live equity curves of the quant and hybrid books.

    Reads each journal's ``portfolio_snapshots`` (filtered to scheduled daily
    cycle_ids, last snapshot of each day), aligns them to the overlapping date
    window (from the *later* of the two start dates to the shared end), then
    computes Sharpe / total-return / max-drawdown for each book and their
    deltas.

    Args:
        quant_db_path: path to the quant bot's ``trade_journal.db``.
        hybrid_db_path: path to the hybrid bot's ``trade_journal.db``.
        coins: list of coin names (e.g. ``["bitcoin", "ethereum"]``).
               Accepted for a future per-coin breakdown; currently recorded
               in the output but not used for sub-slicing.

    Returns:
        Dict with structure::

            {
              "quant":  {"sharpe": float, "ret": float, "maxdd": float},
              "hybrid": {"sharpe": float, "ret": float, "maxdd": float},
              "delta":  {"sharpe": float, "ret": float, "maxdd": float},
              "window": {"start": str, "end": str, "n": int, "coins": list},
            }

        ``delta`` = hybrid − quant for every metric.  ``window.n`` is the
        number of daily data points in the overlap.  All metrics use ``nan``
        / zero defaults when there are fewer than two overlapping snapshots.
    """
    quant_eq = _read_daily_equity(quant_db_path)
    hybrid_eq = _read_daily_equity(hybrid_db_path)

    # Overlap: dates present in both, restricted to the common window.
    shared_dates = sorted(set(quant_eq) & set(hybrid_eq))

    window_start = shared_dates[0] if shared_dates else ""
    window_end = shared_dates[-1] if shared_dates else ""

    quant_vals = [quant_eq[d] for d in shared_dates]
    hybrid_vals = [hybrid_eq[d] for d in shared_dates]

    q_metrics = _metrics_from_values(quant_vals)
    h_metrics = _metrics_from_values(hybrid_vals)

    import math

    def _delta(h, q):
        if math.isnan(h) or math.isnan(q):
            return float("nan")
        return h - q

    delta = {
        "sharpe": _delta(h_metrics["sharpe"], q_metrics["sharpe"]),
        "ret": _delta(h_metrics["ret"], q_metrics["ret"]),
        "maxdd": _delta(h_metrics["maxdd"], q_metrics["maxdd"]),
    }

    return {
        "quant": q_metrics,
        "hybrid": h_metrics,
        "delta": delta,
        "window": {
            "start": window_start,
            "end": window_end,
            "n": len(shared_dates),
            "coins": list(coins),
        },
    }


def run_weekly_parity(*, week_end, live_start_date, live_end_date,
                       output_dir, journal_db=None, sandbox=None,
                       kelly: float = 0.25, lookback_days: int = 1500) -> Path:
    """Run the V5 parity refetch-and-replay check and capture its verdict.

    Shells `scripts/parity_refetch_and_replay.py`, which prints a
    ``VERDICT: PASS|INVESTIGATE|FAIL`` line and the path to a markdown
    parity report. We persist a JSON summary alongside the live metrics.

    Args:
        week_end: ISO week label, e.g. "2026-W21".
        live_start_date / live_end_date: ISO dates ("YYYY-MM-DD"), passed
            straight through as the parity script's --start-date/--end-date.
        output_dir: where the `parity_<week_end>.json` summary is written.
        journal_db: live trade journal; defaults to `$DATA_DIR/trade_journal.db`.
        sandbox: scratch dir the parity script wipes + refetches into;
            defaults to `$DATA_DIR/parity_sandbox`.
        kelly: Kelly fraction for the replay (0.25 = V5 live).
        lookback_days: feature-history depth for the refetch.

    Returns:
        Path to the JSON summary.

    Note:
        The parity replay runs `baseline_v5_mix.py`, which consumes the four
        pre-generated walk-forward prediction CSV dirs (see that script's
        DEFAULT_ROUTING). Those must exist under the repo `data/` dir or the
        replay subprocess fails with `Missing prediction file`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(os.environ.get("DATA_DIR", "data"))
    journal_db = Path(journal_db) if journal_db else data_root / "trade_journal.db"
    sandbox = Path(sandbox) if sandbox else data_root / "parity_sandbox"

    live = compute_live_metrics(live_start_date, live_end_date)

    script = _REPO_ROOT / "scripts" / "parity_refetch_and_replay.py"
    # sys.executable, not bare "python" — the service user has no venv on PATH.
    # ISO dates: the parity script's --start-date/--end-date match the live
    # runner's cycle_id format directly (no YYYYMMDD conversion).
    cmd = [
        sys.executable, str(script),
        "--journal", str(journal_db),
        "--start-date", live_start_date,
        "--end-date", live_end_date,
        "--sandbox", str(sandbox),
        "--kelly", str(kelly),
        "--lookback-days", str(lookback_days),
    ]
    verdict = "ERROR"
    parity_report = ""
    stdout_tail = ""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout_tail = result.stdout[-2000:]
        verdict_m = re.search(r"VERDICT:\s*(\w+)", result.stdout)
        report_m = re.search(r"REPORT:\s*(\S+)", result.stdout)
        verdict = verdict_m.group(1) if verdict_m else "UNKNOWN"
        parity_report = report_m.group(1) if report_m else ""
    except subprocess.CalledProcessError as e:
        # Never raise: a failed parity run must still write a summary so the
        # operator sees ERROR rather than a silent missing report.
        stdout_tail = ((e.stdout or "") + "\n--- stderr ---\n" + (e.stderr or ""))[-2000:]
        logger.error("Parity script failed (exit %s)", e.returncode)

    report = {
        "week_end": week_end,
        "live_start_date": live_start_date,
        "live_end_date": live_end_date,
        "live": live,
        "verdict": verdict,
        "parity_report": parity_report,
        "stdout_tail": stdout_tail,
    }
    out_path = output_dir / f"parity_{week_end}.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Weekly parity %s → verdict %s", week_end, verdict)
    return out_path
