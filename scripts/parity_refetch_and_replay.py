#!/usr/bin/env python
"""V5 MIX live-vs-backtest parity check via historical refetch.

Refetches all 6 data sources fresh into a sandbox directory, replays the
backtest over the same cycle window as live trades, compares per-cycle
predictions / positions / PnL to the live journal.

Spec: docs/superpowers/specs/2026-05-15-v5-mix-live-deployment-design.md §7.

The replay regenerates the four V5 walk-forward prediction routes against
the freshly-refetched sandbox data (the committed CSVs are frozen at
backtest time and never cover post-deploy dates). This regeneration is the
heavy step — four pooled LGB walk-forwards, tens of minutes to hours.

Usage:
    python scripts/parity_refetch_and_replay.py \\
        --journal /opt/tradingagents/data/trade_journal.db \\
        --start-date 2026-05-16 --end-date 2026-05-22 \\
        --sandbox /home/malecada/parity_w1_sandbox \\
        --lookback-days 1500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _wipe_sandbox(sandbox: Path) -> None:
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    for sub in ("onchain", "derivatives", "derivatives_raw", "options",
                "ohlcv_cache", "preds"):
        (sandbox / sub).mkdir(parents=True, exist_ok=True)


def _run_script(name: str, args: list[str], env_extra: dict[str, str]) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / name)] + args
    env = os.environ.copy()
    env.update(env_extra)
    logger.info("Running %s with extra env %s", name, list(env_extra.keys()))
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, check=True)
    logger.info("  %s done in %.1fs", name, time.time() - t0)


def refetch_into_sandbox(sandbox: Path, start_date: str, lookback_days: int) -> None:
    """Re-pull every historical data source needed for V5 MIX into sandbox.

    ``start_date`` is an ISO date ("YYYY-MM-DD") — the same format the live
    runner writes into ``cycles.cycle_id``.
    """
    start_lookback = (datetime.strptime(start_date, "%Y-%m-%d")
                       - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    env_extra = {"TRADINGAGENTS_DATA_ROOT": str(sandbox)}

    # 1. OHLCV — Binance/CoinGecko cache populated on demand by build_pooled_dataset;
    #    let the backtest replay (run_replay step) trigger OHLCV fetches.

    # 2. CoinMetrics
    _run_script("refetch_coinmetrics_full.py",
                 ["--coins", "btc", "eth", "usdt", "usdc", "dai",
                  "usdt_eth", "usdc_eth", "usdt_trx",
                  "--since", start_lookback,
                  "--root", str(sandbox / "onchain")],
                 env_extra)

    # 3. DefiLlama
    _run_script("fetch_defillama_extensions.py",
                 ["--since", start_lookback, "--root", str(sandbox / "onchain")],
                 env_extra)

    # 4. Funding (writes raw + daily aggregate)
    _run_script("backfill_funding_history.py",
                 ["--symbols", "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
                  "--start", start_lookback,
                  "--cache-dir", str(sandbox / "derivatives_raw"),
                  "--daily-out-dir", str(sandbox / "derivatives")],
                 env_extra)

    # 5. Perp-spot basis
    _run_script("build_perp_spot_basis.py",
                 ["--symbols", "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
                  "--start", start_lookback,
                  "--cache-dir", str(sandbox / "derivatives_raw"),
                  "--daily-dir", str(sandbox / "derivatives")],
                 env_extra)

    # 6. Deribit DVOL
    _run_script("fetch_deribit_dvol.py",
                 ["--currencies", "BTC", "ETH",
                  "--start", start_lookback,
                  "--out-dir", str(sandbox / "options")],
                 env_extra)

    # 7. Coinglass (uses env to redirect to sandbox derivatives paths)
    _run_script("fetch_coinglass_history.py", [], env_extra)


# V5 MIX per-coin routing — each coin's predictions come from a distinct
# pooled walk-forward (see baseline_v5_mix.DEFAULT_ROUTING). To replay the
# *live* window we must regenerate these CSVs against the freshly-refetched
# sandbox data; the committed CSVs are frozen at backtest time and never
# cover post-deploy dates.
_PARITY_ROUTES = {
    "bitcoin":     dict(slug="bitcoin",     coins=["bitcoin", "ethereum"],                 pit=False),
    "ethereum":    dict(slug="ethereum",    coins=["bitcoin", "ethereum"],                 pit=True),
    "binancecoin": dict(slug="binancecoin", coins=["bitcoin", "ethereum", "binancecoin"],  pit=False),
    "solana":      dict(slug="solana",      coins=["bitcoin", "ethereum", "solana"],       pit=True),
}


def regenerate_predictions(sandbox: Path, end_date: str,
                            lookback_days: int) -> dict[str, str]:
    """Walk-forward-regenerate the four V5 routes against sandbox data.

    Runs ``evaluate_models_multi.py`` once per route with
    ``TRADINGAGENTS_DATA_ROOT`` pointed at the sandbox, so predictions are
    produced from the freshly-refetched data and extend through ``end_date``
    (the live window). Returns a routing dict {coin: absolute pred dir}
    suitable for ``baseline_v5_mix.py --routing-json``.

    This is the heavy step — four pooled LGB walk-forwards. Budget tens of
    minutes to a couple of hours depending on ``lookback_days``.
    """
    env_extra = {"TRADINGAGENTS_DATA_ROOT": str(sandbox)}
    routing: dict[str, str] = {}
    for coin, route in _PARITY_ROUTES.items():
        out_dir = (sandbox / "preds" / route["slug"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "--coins", *route["coins"],
            "--horizons", "7", "14",
            "--models", "lgb",
            "--days", str(lookback_days),
            "--trade-date", end_date,
            "--output-dir", str(out_dir),
        ]
        if route["pit"]:
            args.append("--onchain-pit")
        logger.info("Regenerating predictions: %s route → %s", coin, out_dir)
        _run_script("evaluate_models_multi.py", args, env_extra)
        routing[coin] = str(out_dir)
    return routing


def load_live_journal_rows(journal_db: str, start_cycle: str, end_cycle: str) -> dict:
    """Pull predictions, decisions, trades, portfolio_snapshots for [start, end]."""
    conn = sqlite3.connect(journal_db)
    cycles = pd.read_sql(
        "SELECT * FROM cycles WHERE cycle_id BETWEEN ? AND ?",
        conn, params=(start_cycle, end_cycle),
    )
    preds = pd.read_sql(
        "SELECT * FROM predictions WHERE cycle_id BETWEEN ? AND ?",
        conn, params=(start_cycle, end_cycle),
    )
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", conn,
    )["name"].values
    # The live journal records sizing decisions in `sizing` (there is no
    # `decisions` table — the old name silently yielded an empty frame).
    sizing = pd.read_sql(
        "SELECT * FROM sizing WHERE cycle_id BETWEEN ? AND ?",
        conn, params=(start_cycle, end_cycle),
    ) if "sizing" in tables else pd.DataFrame()
    trades = pd.read_sql(
        "SELECT * FROM trades WHERE cycle_id BETWEEN ? AND ?",
        conn, params=(start_cycle, end_cycle),
    ) if "trades" in tables else pd.DataFrame()
    # Portfolio snapshots drive the live-vs-replay equity-curve diff.
    snaps = pd.read_sql(
        "SELECT ts, total_value FROM portfolio_snapshots "
        "WHERE date(ts) BETWEEN ? AND ?",
        conn, params=(start_cycle, end_cycle),
    ) if "portfolio_snapshots" in tables else pd.DataFrame()
    # Shadow decisions are the per-cycle live-vs-backtest *signal* comparison —
    # the parity check that stays valid on testnet, where the live price feed
    # diverges from the mainnet replay so return parity is meaningless.
    shadow = pd.read_sql(
        "SELECT * FROM shadow_decisions WHERE cycle_id BETWEEN ? AND ?",
        conn, params=(start_cycle, end_cycle),
    ) if "shadow_decisions" in tables else pd.DataFrame()
    conn.close()
    return {"cycles": cycles, "predictions": preds, "sizing": sizing,
            "trades": trades, "portfolio_snapshots": snaps,
            "shadow_decisions": shadow}


# V5 sizing needs lookback history before it produces any position:
# realized-vol lookback=20, SMA30 trend filter, min_hold=7, vol-regime
# percentile mask. Replaying only the bare live window leaves every
# indicator NaN and every position 0. Replay from this many days before
# the live window so the sizing layer is warmed up by live_start.
_REPLAY_WARMUP_DAYS = 90


def run_replay(sandbox: Path, start_date: str, end_date: str, kelly: float,
               routing: dict[str, str]) -> Path:
    """Run baseline_v5_mix.py against sandbox; return its output dir.

    ``start_date`` / ``end_date`` are ISO dates bounding the live window.
    The replay actually starts ``_REPLAY_WARMUP_DAYS`` *before* ``start_date``
    so V5 sizing indicators are warmed up by the time the live window begins;
    :func:`compare` slices the result back to the live window. ``routing`` is
    the {coin: pred_dir} map from :func:`regenerate_predictions`, written to
    JSON and passed via ``--routing-json``.
    """
    out = sandbox / "replay"
    out.mkdir(exist_ok=True)
    routing_json = sandbox / "parity_routing.json"
    routing_json.write_text(json.dumps(routing, indent=2))
    replay_start = (datetime.strptime(start_date, "%Y-%m-%d")
                    - timedelta(days=_REPLAY_WARMUP_DAYS)).strftime("%Y-%m-%d")
    env = os.environ.copy()
    env["TRADINGAGENTS_DATA_ROOT"] = str(sandbox)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "baseline_v5_mix.py"),
        "--start", replay_start, "--end", end_date,
        "--kelly", str(kelly),
        "--data-root", str(sandbox),
        "--routing-json", str(routing_json),
        "--output-dir", str(out),
    ]
    subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, check=True)
    return out


def _window_metrics(returns: pd.Series) -> dict:
    """Sharpe / total return / max DD for a daily-return series."""
    r = returns.dropna()
    if len(r) < 2:
        return {"sharpe": float("nan"), "total_return": float("nan"),
                "max_drawdown": float("nan"), "n_bars": int(len(r))}
    sd = r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    return {
        "sharpe": float(r.mean() / sd * (252 ** 0.5)) if sd > 0 else 0.0,
        "total_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": dd,
        "n_bars": int(len(r)),
    }


# Divergence thresholds for the live-vs-replay equity-curve diff. The verdict
# is driven by how far the LIVE equity curve drifts from the REPLAY of the same
# strategy over the same window — that is the actual parity question, and it
# captures every sizing/signal/cost divergence as its net effect on returns.
_GAP_FAIL = 0.10        # cumulative-return gap (10 pp) -> FAIL
_GAP_INVESTIGATE = 0.03
_CORR_FAIL = 0.5        # daily-return correlation below this -> FAIL
_CORR_INVESTIGATE = 0.8
_MIN_PARITY_BARS = 10


def _live_daily_returns(snapshots: pd.DataFrame, live_start: str,
                        live_end: str) -> pd.Series:
    """Daily portfolio returns from portfolio_snapshots over the live window."""
    if snapshots is None or snapshots.empty or "total_value" not in snapshots:
        return pd.Series(dtype=float)
    s = snapshots.copy()
    # Snapshots are UTC-stamped (tz-aware); coerce to tz-naive UTC dates so they
    # compare against the tz-naive replay index / window bounds.
    s["date"] = pd.to_datetime(s["ts"], utc=True).dt.tz_localize(None).dt.normalize()
    s = s[(s["date"] >= pd.Timestamp(live_start))
          & (s["date"] <= pd.Timestamp(live_end))]
    if s.empty:
        return pd.Series(dtype=float)
    s = (s.sort_values("date").drop_duplicates("date", keep="last")
          .set_index("date")["total_value"].astype(float))
    return s.pct_change().dropna()


# Signal-parity verdict thresholds (used when the return-diff window is too
# short — the normal case for a testnet week, and the only valid parity there).
_SIGNAL_PASS = 1.0          # 100% agreement -> PASS
_SIGNAL_INVESTIGATE = 0.9   # >= 90% agreement -> INVESTIGATE; below -> FAIL


def _signal_parity(shadow: pd.DataFrame) -> dict:
    """Live-vs-shadow per-cycle signal agreement.

    `shadow_decisions` records, for each (cycle, coin), the live signal vs the
    backtest (shadow) signal and whether they agree. This is the parity check
    that stays valid on testnet — the live exchange's price feed diverges from
    the mainnet replay, so return parity is meaningless, but the *decision*
    the strategy makes from the same inputs must still match.
    """
    if shadow is None or shadow.empty or "agree" not in shadow.columns:
        return {"n": 0, "n_agree": 0, "agree_pct": float("nan"),
                "disagreements": []}
    agree = shadow["agree"].fillna(0).astype(int)
    n = int(len(agree))
    n_agree = int(agree.sum())
    cols = [c for c in ("cycle_id", "coin", "live_signal", "backtest_signal")
            if c in shadow.columns]
    disagreements = (shadow[agree == 0][cols].to_dict("records")
                     if cols else [])
    return {"n": n, "n_agree": n_agree,
            "agree_pct": (n_agree / n) if n else float("nan"),
            "disagreements": disagreements}


def compare(live: dict, replay_dir: Path, out_report: Path,
            live_start: str, live_end: str) -> str:
    """Generate parity_report.md. Returns PASS / INVESTIGATE / FAIL /
    INSUFFICIENT_WINDOW.

    The verdict is driven by the divergence between the LIVE equity curve
    (portfolio_snapshots) and the REPLAY of the same strategy over the same
    window — not by the replay's own Sharpe. The replay spans a warmup window
    before ``live_start``; it is sliced back to [``live_start``, ``live_end``]
    to line up with the live journal.
    """
    # baseline_v5_mix writes daily_returns.csv with an unnamed DatetimeIndex —
    # index_col=0 reads it back without assuming a "date" column name.
    replay_daily = pd.read_csv(replay_dir / "daily_returns.csv",
                                index_col=0, parse_dates=True)
    window = replay_daily.loc[
        (replay_daily.index >= pd.Timestamp(live_start))
        & (replay_daily.index <= pd.Timestamp(live_end))
    ]
    replay_port = (_window_metrics(window["portfolio"])
                   if "portfolio" in window.columns and not window.empty
                   else {"sharpe": float("nan"), "total_return": float("nan"),
                         "max_drawdown": float("nan"), "n_bars": 0})

    # --- live-vs-replay equity-curve diff (the parity check proper) ---
    live_rets = _live_daily_returns(
        live.get("portfolio_snapshots", pd.DataFrame()), live_start, live_end)
    replay_rets = (window["portfolio"].copy()
                   if "portfolio" in window.columns else pd.Series(dtype=float))
    if len(replay_rets):
        replay_rets.index = pd.to_datetime(replay_rets.index).normalize()
    aligned = pd.DataFrame({"live": live_rets, "replay": replay_rets}).dropna()
    n_common = len(aligned)
    if n_common >= 2:
        cum_live = float((1 + aligned["live"]).prod() - 1)
        cum_replay = float((1 + aligned["replay"]).prod() - 1)
        cum_gap = abs(cum_live - cum_replay)
        # Correlation is only meaningful when both curves actually move; a flat
        # curve's "correlation" is FP noise. Below this daily-vol floor, judge
        # parity on the cumulative-return gap alone (corr left NaN).
        _vol_floor = 1e-4
        if aligned["live"].std() > _vol_floor and aligned["replay"].std() > _vol_floor:
            corr = float(aligned["live"].corr(aligned["replay"]))
        else:
            corr = float("nan")
        daily_mae = float((aligned["live"] - aligned["replay"]).abs().mean())
    else:
        cum_live = cum_replay = cum_gap = corr = daily_mae = float("nan")

    # --- signal parity (live vs shadow backtest) — valid even on testnet ---
    sig = _signal_parity(live.get("shadow_decisions", pd.DataFrame()))

    live_total_trades = int(live["cycles"]["n_trades"].sum()) if not live["cycles"].empty else 0
    live_status_summary = (live["cycles"]["status"].value_counts().to_dict()
                            if not live["cycles"].empty else {})

    cycle_min = live['cycles']['cycle_id'].min() if not live['cycles'].empty else '?'
    cycle_max = live['cycles']['cycle_id'].max() if not live['cycles'].empty else '?'
    lines = [
        f"# V5 MIX parity report — cycles {cycle_min}..{cycle_max}",
        "",
        f"## Refetch summary",
        f"- Sandbox: `{replay_dir.parent}`",
        f"- Replay daily bars (incl. {_REPLAY_WARMUP_DAYS}d warmup): {len(replay_daily)}",
        f"- Live-window bars (sliced): {replay_port['n_bars']}",
        "",
        f"## Live journal summary",
        f"- Cycles: {len(live['cycles'])}",
        f"- Total trades executed: {live_total_trades}",
        f"- Status counts: {live_status_summary}",
        "",
        f"## Live-vs-replay equity diff ({n_common} aligned bars)",
        f"- Live cumulative return:   {cum_live:+.2%}" if n_common >= 2 else "- Live cumulative return:   n/a",
        f"- Replay cumulative return: {cum_replay:+.2%}" if n_common >= 2 else "- Replay cumulative return: n/a",
        f"- Cumulative-return gap:    {cum_gap:.2%}" if n_common >= 2 else "- Cumulative-return gap:    n/a",
        f"- Daily-return correlation: {corr:.3f}" if n_common >= 2 else "- Daily-return correlation: n/a",
        f"- Daily-return MAE:         {daily_mae:.4f}" if n_common >= 2 else "- Daily-return MAE:         n/a",
        "",
        f"## Signal parity (live vs shadow backtest)",
        (f"- Decisions: {sig['n']}  agree: {sig['n_agree']}  agreement: {sig['agree_pct']:.1%}"
         if sig["n"] else "- Decisions: 0 (no shadow_decisions in window)"),
        *[f"- DISAGREE: {d}" for d in sig["disagreements"][:20]],
        "",
        f"## Aggregate metrics (replay, live window only)",
        f"- Replay portfolio Sharpe: {replay_port.get('sharpe', float('nan')):.3f}",
        f"- Replay portfolio return: {replay_port.get('total_return', float('nan')):+.1%}",
        f"- Replay portfolio max DD: {replay_port.get('max_drawdown', float('nan')):.1%}",
        "",
    ]

    data_fail = any(k in live_status_summary
                    for k in ("predict_majority_fail", "critical_data_fail", "error"))
    if n_common >= _MIN_PARITY_BARS:
        # Enough aligned bars for a return-diff verdict (mainnet / long run).
        if cum_gap > _GAP_FAIL or (corr == corr and corr < _CORR_FAIL):
            verdict = "FAIL"
        elif data_fail:
            verdict = "INVESTIGATE"
        elif cum_gap > _GAP_INVESTIGATE or (corr == corr and corr < _CORR_INVESTIGATE):
            verdict = "INVESTIGATE"
        else:
            verdict = "PASS"
    elif sig["n"] > 0:
        # Return-diff window too short (the testnet-week case): fall back to
        # signal parity, which is the only valid parity on testnet prices.
        if data_fail:
            verdict = "INVESTIGATE"
        elif sig["agree_pct"] >= _SIGNAL_PASS:
            verdict = "PASS"
        elif sig["agree_pct"] >= _SIGNAL_INVESTIGATE:
            verdict = "INVESTIGATE"
        else:
            verdict = "FAIL"
    else:
        # No return-diff bars and no shadow decisions — nothing to compare.
        verdict = "INSUFFICIENT_WINDOW"
    lines.append(f"## Verdict: {verdict}")
    out_report.write_text("\n".join(lines))
    return verdict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--journal", required=True,
                    help="Path to live trade journal SQLite DB")
    p.add_argument("--start-date", required=True,
                    help="ISO date YYYY-MM-DD (matches live cycle_id format)")
    p.add_argument("--end-date", required=True,
                    help="ISO date YYYY-MM-DD (matches live cycle_id format)")
    p.add_argument("--sandbox", required=True, help="Sandbox directory (will be wiped)")
    p.add_argument("--lookback-days", type=int, default=1500)
    p.add_argument("--kelly", type=float, default=0.25,
                    help="Kelly fraction for replay (default 0.25 = V5 live)")
    p.add_argument("--skip-regen", action="store_true",
                    help="Skip the heavy walk-forward prediction regeneration "
                         "and replay from committed CSVs (debug only — will "
                         "not cover post-deploy live dates).")
    args = p.parse_args()

    sandbox = Path(args.sandbox)
    logger.info("=== V5 MIX parity check ===")
    logger.info("Sandbox: %s  (will be wiped)", sandbox)

    _wipe_sandbox(sandbox)
    refetch_into_sandbox(sandbox, args.start_date, args.lookback_days)

    live = load_live_journal_rows(args.journal, args.start_date, args.end_date)
    logger.info("Live journal: %d cycles, %d predictions",
                len(live["cycles"]), len(live["predictions"]))

    if args.skip_regen:
        logger.warning("--skip-regen: replaying from committed CSVs (will not "
                        "cover post-deploy live dates)")
        # Mirror of baseline_v5_mix.DEFAULT_ROUTING — committed frozen CSVs.
        routing = {
            "bitcoin":     str(PROJECT_ROOT / "data" / "multi_2coins_walkforward"),
            "ethereum":    str(PROJECT_ROOT / "data" / "multi_2coins_pit_wf"),
            "binancecoin": str(PROJECT_ROOT / "data" / "multi_3coins_bnb_wf"),
            "solana":      str(PROJECT_ROOT / "data" / "multi_3coins_sol_pit_wf"),
        }
    else:
        routing = regenerate_predictions(sandbox, args.end_date, args.lookback_days)

    replay_dir = run_replay(sandbox, args.start_date, args.end_date,
                            args.kelly, routing)
    logger.info("Replay output: %s", replay_dir)

    report = sandbox / "parity_report.md"
    verdict = compare(live, replay_dir, report, args.start_date, args.end_date)
    logger.info("Verdict: %s", verdict)
    logger.info("Report: %s", report)
    print(f"\nVERDICT: {verdict}\nREPORT: {report}\n")


if __name__ == "__main__":
    main()
