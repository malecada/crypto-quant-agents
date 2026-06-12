# Project Overview — crypto-quant-agents

Last updated: 2026-06-15.

---

## 1. What it is

`crypto-quant-agents` is a cryptocurrency trading system that combines two things:

1. **A hardened quantitative baseline** — a machine-learning signal (LightGBM on
   futures term-structure features) sized by a volatility-targeted Kelly rule and
   filtered by a trend filter. This is the workhorse that generates the returns.
2. **A team of LLM agents** — market, on-chain, sentiment, and ML-prediction
   analysts; bull/bear researchers; a risk-debate panel; and a portfolio manager,
   all orchestrated like a small trading desk. An optional **LLM modulator** layer
   scales the quant signal up or down per coin.

The central research question of the thesis: **does multi-agent LLM reasoning add
alpha on top of an already-strong quant baseline?** The answer so far is nuanced —
the quant baseline does the heavy lifting, the LLM layer adds robust value on some
coins (notably ETH) but not universally.

It began as a crypto adaptation of the academic
[TradingAgents](https://arxiv.org/abs/2412.20138) framework and grew into a
standalone research codebase. It trades Binance perpetual futures and is currently
**live on a testnet/VPS deployment** with a public monitoring dashboard (see §6).

---

## 2. Headline results

Walk-forward, out-of-sample, look-ahead-controlled backtests. Research only — not
financial advice.

| Strategy | Sharpe | Return | Max DD | Window |
|---|---|---|---|---|
| **V5 MIX — 4-coin, per-coin feature routing** | **+3.25** *(drift +3.18)* | +787% | −4.9% | 4.5-yr WF |
| V5 MIX — 8-coin expansion | +3.97 | +1053% | −4.8% | 4.5-yr WF |
| Hybrid V5 LLM modulator — ETH (Δ vs pure V5) | +1.10 | — | — | 1-yr WF |

- **V5 MIX** is the strongest result: per-coin feature routing (different feature
  sets per coin) plus equal-weight diversification across uncorrelated coins.
- **The LLM modulator** produced the first *robust* LLM-driven alpha on ETH
  (Δ Sharpe +1.10, bootstrap CI [+0.60, +1.56]). On BTC and other coins it was
  neutral or slightly negative — so it ships per-coin, not blanket.
- The honest negative results are documented too: a more complex "V3" architecture
  systematically *underperformed* V2, confirming that **signal quality, not
  architecture, is the binding constraint**.
- A random-entry placebo attributes roughly **90% of V5 MIX's risk-adjusted return
  to the sizing + diversification mechanics and ~10% to the ML signal** — a
  strength, since the edge is robust to prediction-model drift rather than fragile
  to it.

For scale: over the same 4.5-year window a $1 in V5 MIX (4-coin) grows to ~$8.6 at
Sharpe 3.18, against an equal-weight buy-and-hold of the same coins that ends below
its starting value at Sharpe 0.22. The full set of experiments — including the dead
ends — is inventoried in [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) under "Tested
approaches".

---

## 3. Feature areas

### Multi-agent analysis pipeline
Agents organized like a trading firm, orchestrated with LangGraph:

```
Analysts (parallel)            market · on-chain · sentiment · prediction
  → Bull / Bear Researchers    structured investment debate
    → Research Manager         synthesis
      → Trader                 BUY / HOLD / SELL proposal
        → Risk Panel           aggressive / conservative / neutral debate
          → Portfolio Manager  final 5-level rating → execution
```

- **Market analyst** — Binance/CoinGecko OHLCV + 150+ technical indicators.
- **On-chain analyst** — funding rates, TVL, gas, stablecoin supply, open interest,
  liquidations, long/short ratios (CoinMetrics, DefiLlama, Coinglass).
- **Sentiment analyst** — multi-source LLM sentiment (Alpha Vantage news, Reddit,
  Google News, macro). *Empirically dropped from production for BTC+ETH daily —
  three independent runs showed it adds noise.*
- **Prediction analyst** — ML price forecasts (Random Forest / ARIMA / LightGBM),
  multi-horizon (1/3/7/14-day), pooled multi-coin.

### Quant baseline (V2 / V5 MIX)
The production strategy. Sizing primitives in `tradingagents/strategies/`, shared
identically by backtest and live execution. V5 MIX adds per-coin feature routing
and multi-coin equal-weight allocation.

### Hybrid LLM modulator
The component the thesis exists to test. The LLM agents do **not** trade directly;
instead they output a per-coin multiplier that scales the quant signal up or down,
leaving the quant direction call intact — so the LLM can only adjust *conviction*,
never overrule the model. The multiplier is calibrated, gated, and validated
against a pure-quant control on the identical window, and ships per coin only where
it demonstrably helps. Its clearest win is ETH (Sharpe +3.59 → +4.68, Δ +1.10 over
one year), where it also nearly halves drawdown; on BTC it is neutral. A
leave-one-analyst-out study traces the gain to specific analysts (prediction =
backbone) and shows sentiment adds none.

### Backtesting & validation
- Walk-forward evaluation over 4.5 years, multi-coin, with cumulative-return and
  Sharpe comparisons against buy-and-hold.
- Combinatorial Purged Cross-Validation (CPCV) and Deflated Sharpe Ratio to guard
  against overfitting and multiple-testing bias.
- Random-entry placebo (signal-vs-mechanics attribution), bootstrap confidence
  intervals, regime breakdowns, and cost-sensitivity checks.
- Deterministic LLM **replay cache** so multi-agent backtests are reproducible and
  cost-bounded.

### Diversification — carry sleeve
A funding-carry sleeve harvests the perpetual-futures funding rate with a
price-neutral position; its return is near-uncorrelated with the momentum-driven
V5 MIX (≈ +0.003), so a 20% blend modestly improves the risk-adjusted result
(Sharpe 3.18 → 3.29).

### Interfaces & models
- An interactive **CLI** and a **Python API** for single-coin analysis and live
  runs.
- A pluggable **LLM provider factory** (OpenAI, Anthropic, Google, xAI, OpenRouter,
  Ollama); production runs on a cost-efficient model.

### Live execution
- Binance Futures wrapper (testnet by default), pre-trade risk checks
  (leverage caps, max open positions, min-notional, stop-loss).
- SQLite trade journal, fill reconciliation, robust order/position handling.
- Deployed to a Hetzner VPS via systemd timers (trading cycle, hybrid cycle,
  weekly re-backtest).

### Live monitoring web
A read-only dashboard over the live bot — see §6 for what it shows and how to
access it.

### Data layer
Point-in-time (PIT) feature store, ~166K rows, 50+ engineered features incl.
smart-money divergence, OI z-scores, liquidation z-scores. All look-ahead
controlled; backtests start after the LLM training cutoff to avoid memory-based
leakage.

---

## 4. Honest limitations (already documented in the thesis)

- LLM modulator alpha is coin-specific, not universal.
- The complex V3 architecture was a controlled negative result (retired).
- Results are testnet / backtest; no real-capital track record yet.
- A funding-cost approximation bug was found and corrected; impact documented in
  the thesis.

---

## 5. Repository map

```
tradingagents/      core package — agents, dataflows, models, strategies,
                    execution, backtesting, llm_clients, monitor
cli/                Typer interactive CLI
scripts/            model evaluation, backtests, baseline-strategy scripts
deploy/             VPS systemd units, deploy scripts, live monitor (Caddy) config
tests/              pytest suite
app.py              Streamlit analysis dashboard
report_assets/      figures embedded in PROGRESS_REPORT.md
PROGRESS_REPORT.md  narrative of how the project evolved
```

---

## 6. Live monitoring dashboard — how to access

A **read-only** web dashboard runs continuously on the VPS and shows the live bot's
state in real time. It never writes to the bot's data.

**URL:** <https://46.225.169.184.nip.io>
**Username:** `admin`
**Password:** `NtF4n7afA97pCplspQrxOS` *(included because this repository is private — repo access gates dashboard access; the dashboard is read-only).*

> The address is an automatic-HTTPS endpoint (Caddy + Let's Encrypt) reverse-proxying
> the dashboard. No software install needed — it opens in any browser.

It is a **dual-strategy** dashboard (quant bot vs hybrid LLM-modulated bot, shown
side-by-side). Five tabs:

| Tab | What it shows |
|---|---|
| **Performance** | Live equity curve vs the backtest Sharpe anchor (≈3.18), realized Sharpe, drawdown, rolling Sharpe, unrealized PnL. Quant-vs-hybrid delta when both run. |
| **Positions** | Open positions per strategy — entry / mark / leverage / uPnL — plus an allocation donut. Marked STALE if the live exchange query fails. |
| **Executions** | Order execution log: entry price, slippage, status. (V5 is a rebalancing strategy, so it records executions, not round-trip trades.) |
| **Decisions** | Per-cycle ML predictions, position sizing, risk checks, and — for the hybrid bot — the LLM modulator multiplier and its reasoning. |
| **Health** | Cycle timeline, pipeline-step timings, recent errors, model-retrain history. |

The dashboard degrades gracefully: if one strategy's data is unavailable the other
keeps serving, and live exchange-query failures fall back to the last journal
snapshot with a STALE badge.

**Current deployment:** Hetzner VPS, 8-coin V5 MIX live bot, monitoring UI v2
(React + FastAPI). Quant and hybrid bots both report into the dashboard.

---

## 7. Status & next steps

- **Done:** V5 MIX backtests (4- and 8-coin), hybrid modulator validation, live
  testnet deployment, dual-strategy monitoring dashboard, thesis write-up.
- **In progress:** live testnet evaluation window, judged against a pre-registered
  acceptance target (Sharpe ≥ +2.86 over 90 days at Kelly 0.25).
- **Codebase:** consolidated into this single repo with clean history; full test
  suite green.

---

*Questions about any feature? Each area maps to a directory under `tradingagents/`,
and [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) walks through how each piece came to
be.*
