# crypto-quant-agents

**Multi-agent LLM + quantitative-hybrid trading system for cryptocurrency, with live Binance Futures execution.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

`crypto-quant-agents` pairs a team of specialized LLM agents — market, on-chain, sentiment, and ML-prediction analysts; bull/bear researchers; an aggressive/conservative/neutral risk-debate panel; and a portfolio manager — with a hardened quantitative baseline and an optional LLM signal modulator. It trades crypto perpetual futures on Binance and ships with a full backtesting harness and a live monitoring dashboard.

It began as a crypto adaptation of the multi-agent [TradingAgents](https://arxiv.org/abs/2412.20138) framework and grew into a standalone research codebase for a master's thesis investigating whether multi-agent LLM reasoning adds alpha on top of a strong quant baseline.

## Headline results

Master's-thesis backtests — walk-forward, out-of-sample, look-ahead controlled. Research only; not financial advice.

| Strategy | Sharpe | Return | Max DD | Window |
| --- | --- | --- | --- | --- |
| V5 MIX — 4-coin, per-coin feature routing | **+3.25** | +787% | −4.9% | 4.5-yr WF |
| V5 MIX — 8-coin expansion | +3.97 | +1053% | −4.8% | 4.5-yr WF |
| Hybrid V5 LLM modulator — ETH, Δ vs pure V5 | +1.10 | — | — | 1-yr walk-forward |

> The **+3.25** figure is the published 4.5-yr headline; a later data-refresh drifted the canonical 4-coin baseline to ≈ **+3.18** (identical trade logic, refreshed price data).

The quant baseline (LightGBM term-structure consensus + vol-targeted Kelly sizing + SMA trend filter) is the workhorse. The LLM modulator adds robust alpha on ETH but not universally.

## Architecture

Agents are organized like a trading firm and orchestrated with LangGraph:

```
Analysts (parallel)            market · on-chain · sentiment · prediction
  → Bull / Bear Researchers    structured investment debate
    → Research Manager         synthesis
      → Trader                 BUY / HOLD / SELL proposal
        → Risk Panel           aggressive / conservative / neutral debate
          → Portfolio Manager  final 5-level rating → execution
```

- **Crypto analysts** — Binance/CoinGecko OHLCV + 150+ technical indicators; on-chain metrics (funding rates, TVL, gas, stablecoin supply); multi-source sentiment (Alpha Vantage news, Reddit, Google News, macro); ML price forecasts (Random Forest / ARIMA / LightGBM, multi-horizon, pooled multi-coin).
- **Quant baseline** — V2 / V5 MIX sizing primitives in `tradingagents/strategies/`, shared by backtest and live execution.
- **Hybrid modulator** — optional LLM layer that scales the quant signal per coin.
- **Execution** — Binance Futures wrapper (testnet by default), pre-trade risk checks, SQLite trade journal, and a live monitor UI.

## Installation

Requires Python ≥ 3.10.

```bash
# with uv (recommended — uses the committed lockfile)
uv sync

# or with pip
pip install -e .
```

Provide API keys by copying the example file:

```bash
cp .env.example .env
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, ALPHA_VANTAGE_API_KEY, ...
# BINANCE_API_KEY / BINANCE_API_SECRET for live trading
```

## Usage

**Interactive CLI:**
```bash
tradingagents          # or: python -m cli.main
```

**Python API — single crypto analysis:**
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["asset_class"] = "crypto"
ta = TradingAgentsGraph(
    selected_analysts=["market", "onchain", "prediction"],
    config=config,
)
final_state, signal = ta.propagate("bitcoin", "2025-01-15")
# signal: BUY | OVERWEIGHT | HOLD | UNDERWEIGHT | SELL
```

**Quant baseline backtest (V2 / V5 MIX):**
```bash
python scripts/evaluate_models_multi.py --coins bitcoin ethereum \
    --horizons 1 3 7 14 --models lgb --output-dir data/multi_2coins_v2
python scripts/baseline_strategy_v2.py --pred-dir data/multi_2coins_v2 --symmetric
```

**Live execution (testnet by default):**
```python
from tradingagents.execution.runner import LiveRunner

runner = LiveRunner(config={
    "asset_class": "crypto",
    "execution": {"live_mode": False, "dry_run": True},
})
signal, result = runner.run_single("bitcoin")
```

**Live monitor UI + VPS deployment:** see [`deploy/`](deploy/). Ad-hoc single-coin predictions run from the monitor UI's "Run Prediction" tab.

## Configuration

Behavior is controlled by `.env` and the config dict in
`tradingagents/default_config.py` — `asset_class`, `llm_provider`,
`deep_think_llm` / `quick_think_llm`, analyst selection, and execution limits.
Environment-variable overrides (`LIVE_MODE`, `STOP_LOSS_PCT`, `LEVERAGE`,
`MAX_OPEN_POSITIONS`, …) support container and cloud deployment. Full reference:
[`CLAUDE.md`](CLAUDE.md).

## Live monitoring dashboard

A **read-only** web dashboard runs continuously on the VPS and shows the live
bot's state in real time — equity vs the backtest anchor, open positions,
per-cycle decisions, and health — for the quant and hybrid strategies side by
side. It opens in any browser (automatic HTTPS, nothing to install).

- **URL:** <https://46.225.169.184.nip.io>
- **Username:** `admin`
- **Password:** `NtF4n7afA97pCplspQrxOS`

> Credentials are included here intentionally: this repository is private, so
> access to the repo gates access to the dashboard. The dashboard is read-only
> and cannot place or modify trades.

What each tab shows is documented in [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) §6.

## Project overview

A concise, feature-oriented summary of what has been built — including the live
monitoring dashboard and how to access it — is in
[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

For a progress report on how the project evolved (Krypto-v0's leaks → the live
V5 MIX desk), told as experiment → problem → solution, see
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).

## Repository layout

```
tradingagents/      core package — agents, dataflows, models, strategies, execution, backtesting, llm_clients
cli/                Typer CLI
scripts/            model evaluation, backtests, baseline-strategy scripts
deploy/             VPS systemd units, deploy scripts, live monitor UI
tests/              pytest suite
PROJECT_OVERVIEW.md feature summary + live monitor access
PROGRESS_REPORT.md  narrative of how the project evolved
```

## Origin & attribution

This project began as a crypto adaptation of **TradingAgents: Multi-Agents LLM
Financial Trading Framework** (Xiao, Sun, Luo & Wang, 2024) and remains a derivative
work distributed under the upstream Apache 2.0 license.

```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
  title  = {TradingAgents: Multi-Agents LLM Financial Trading Framework},
  author = {Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
  year   = {2025},
  eprint = {2412.20138},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.TR},
  url    = {https://arxiv.org/abs/2412.20138}
}
```

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

## Disclaimer

Research software built for a master's thesis. Cryptocurrency trading carries
substantial risk. Nothing in this repository is financial, investment, or trading
advice. Use at your own risk and test on paper/testnet before risking real capital.
