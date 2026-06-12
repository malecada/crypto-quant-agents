# Live Bot Monitoring UI

Read-only FastAPI dashboard for the dual-strategy (quant + hybrid) V5 MIX live
bot. Reads the bots' `trade_journal.db` (SQLite, `mode=ro`) and structured
cycle logs. Never writes to the bots' data.

## Run locally

```bash
TA_MONITOR_PASSWORD=somepw python -m tradingagents.monitor
# open http://127.0.0.1:8800  (user: admin)
```

To run with both strategies visible, also set `HYBRID_DATA_DIR`:

```bash
QUANT_DATA_DIR=data/quant \
HYBRID_DATA_DIR=data/hybrid \
HYBRID_BINANCE_API_KEY=xxx \
HYBRID_BINANCE_API_SECRET=yyy \
TA_MONITOR_PASSWORD=somepw \
  python -m tradingagents.monitor
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `TA_MONITOR_PASSWORD` | — (required) | Basic-auth password; app refuses to start if unset |
| `QUANT_DATA_DIR` | `$DATA_DIR` → `data` | Directory holding the quant bot's `trade_journal.db` |
| `HYBRID_DATA_DIR` | — (optional) | Directory holding the hybrid bot's `trade_journal.db`; hybrid pane is disabled when unset or equal to `QUANT_DATA_DIR` |
| `DATA_DIR` | `data` | Fallback data directory when `QUANT_DATA_DIR` is not set |
| `LOG_DIR` | `logs` | Directory holding `cycle_*.jsonl` (quant runner only) |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | — | Quant bot live-account credentials (follow quant runner's `LIVE_MODE` config) |
| `HYBRID_BINANCE_API_KEY` / `HYBRID_BINANCE_API_SECRET` | — | Hybrid bot testnet credentials (always queries testnet, regardless of quant `LIVE_MODE`) |
| `TA_MONITOR_ANCHOR_SR_QUANT` | `3.18` | Backtest Sharpe anchor for the quant strategy (shown on Performance tab) |
| `TA_MONITOR_ANCHOR_SR_HYBRID` | — (optional) | Backtest Sharpe anchor for the hybrid strategy |
| `TA_MONITOR_HOST` | `127.0.0.1` | Bind host (keep loopback; proxy terminates TLS) |
| `TA_MONITOR_PORT` | `8800` | Bind port |
| `TA_MONITOR_START_CAPITAL` | `10000` | Starting capital for equity reconstruction when no snapshots exist |

## Tabs

- **Performance** — equity curve vs backtest anchors (quant SR 3.18 default,
  hybrid optional), Sharpe, drawdown, rolling Sharpe, uPnL cards. The compare
  panel (quant vs hybrid delta) is shown only when hybrid is configured.
- **Positions** — open positions per strategy with entry/mark/leverage/uPnL,
  plus an allocation donut. Falls back to the journal snapshot (STALE badge)
  when the live Binance query fails.
- **Executions** — order execution log (entry price, slippage, status). V5 is a
  rebalancing strategy — the journal records executions only, never round-trip
  trades, so per-trade exit price / PnL / fees do not exist; realized PnL is the
  equity curve on the Performance tab.
- **Decisions** — per-cycle LGB predictions, sizing, risk checks, shadow
  decisions. The hybrid modulator panel (multiplier, reasoning) appears only
  when the hybrid strategy is configured.
- **Health** — cycle timeline, pipeline-step timings, recent errors, retrain
  history.

## Degradation contract

- **Hybrid pane** renders `null` (grayed out) when `HYBRID_DATA_DIR` is unset
  or matches `QUANT_DATA_DIR`.
- **STALE badge** appears on Position cards whenever the live Binance query
  fails (network error, IP ban, missing credentials); data falls back to the
  last journal snapshot.
- **Per-strategy isolation**: a missing or unreadable journal for one strategy
  yields `null` for that strategy only. The other strategy continues serving
  normally. This applies to `/api/performance`, `/api/positions`, and
  `/api/health`.

## React build workflow

The built React SPA is committed to the repo as `tradingagents/monitor/frontend/dist/`.
The VPS does **not** need Node.js installed — the dist is served directly by FastAPI.

To rebuild after frontend changes:

```bash
cd tradingagents/monitor/frontend
npm install          # first time only
npm run build        # writes to dist/; commit the result
```

## Deployment

`deploy/systemd/ta-monitor.service` runs it as a persistent service;
`deploy/Caddyfile` provides public HTTPS. See `deploy/deploy.sh`.

Secrets are loaded from `EnvironmentFile=/opt/tradingagents/secrets/.env.trading`
(quant Binance keys) and optionally
`EnvironmentFile=-/opt/tradingagents/secrets/.env.monitor` (monitor-specific
vars including `TA_MONITOR_PASSWORD`, `HYBRID_DATA_DIR`,
`HYBRID_BINANCE_API_KEY`, etc.).
