# crypto-quant-agents

Multi-agent LLM framework for cryptocurrency trading decisions. Uses a trading firm hierarchy of specialized AI agents (analysts, researchers, risk managers) built on LangGraph + LangChain. Adapted from the original stock-focused TradingAgents (arxiv.org/abs/2412.20138) with crypto data sources, on-chain analytics, ML price forecasting, and Binance Futures execution.

**IMPORTANT**: When new empirical findings are produced (model evaluations, backtest results, strategy comparisons), update `THESIS_FINDINGS.md` in the project root. That file is the persistent record of all experimental results for the master's thesis.

## Baseline Strategy (Finalized)

The quant baseline that the multi-agent LLM system must beat is `scripts/baseline_strategy_v2.py` with default settings:
- **Signal**: LightGBM term structure consensus — h=7 and h=14 must agree on direction
- **Training pool**: 2-coin (BTC+ETH) or 3-coin (BTC+ETH+target) — larger pools hurt due to altcoin noise
- **Sizing**: Vol-targeted Kelly + confidence-weighted + conditional leverage (1-3x) + SMA30 trend filter (1.5x aligned, 0.5x against)
- **Risk**: 7-day min hold with adaptive early exit, 3% stop-loss, 15% portfolio circuit breaker, 95th percentile vol cap
- **Performance**: 2-coin portfolio Sharpe **2.69** (+106% return); 3-coin portfolio Sharpe 2.58 (+155% return)

### Quant V3 (Built, Underperforms V2)

V3 (`tradingagents/strategies/v3/`) extends V2 with a Non-Homogeneous Hidden Markov Model (NH-HMM) regime detector, microstructure features (klines-proxy OFI, volume dispersion), open-interest and funding-rate derivatives, a multi-horizon LightGBM ensemble (h=3,7,14,21) in place of V2's h=7+h=14 consensus, CDAP drawdown-adaptive position control, vol-targeted Kelly sizing, and a Pydantic-typed signal contract layer. The architecture is complete (117+ unit tests, Pydantic contracts, CPCV harness) and V2 regression stays green throughout.

**Empirical result**: V3 is systematically inferior to V2 on every metric and every evaluation window tested. An 88-bar OOS A/B (2026-01-16 → 2026-04-15) produced portfolio Sharpe -0.73 vs V2 2.38. A 28-split CPCV over 2024-05 → 2026-04 yielded BTC mean Sharpe -2.40 (0/28 positive splits) and ETH mean Sharpe -2.92 (1/28 positive splits); Deflated Sharpe Ratio ≈ 0 for both coins. A 5-variant component ablation confirms V3 architecture is internally well-engineered (each component—multi-horizon horizons, regime detector, vol-target/CDAP—contributes positively within V3), but the LGB signal quality is the binding constraint: LGB probability estimates cluster in the 0.52–0.57 range and do not generate alpha on this OOS window. This reproduces the BT11 finding that V2's alpha is ~90% sizing+momentum and sophisticated ML modulation hurts BTC.

**Status**: V3 build complete; empirically inferior to V2 on current data. Architecture is sound (ablations confirm each component contributes positively), but LGB signal quality is the binding constraint. V2 remains the production quant baseline.

**Reference documents**:
- Spec: `docs/superpowers/specs/2026-05-08-quant-v3-design.md`
- Plan: `docs/superpowers/plans/2026-05-08-quant-v3.md`
- 88-bar A/B results: `data/multi_2coins_v3/metrics.json`
- CPCV results: `data/v3_cpcv/bitcoin/summary.json`, `data/v3_cpcv/ethereum/summary.json`
- Ablation results: `data/v3_ablations/ablations_metrics.json`
- Full empirical findings: `THESIS_FINDINGS.md` Section 12

**Reproduce V3 results**:
```bash
# 88-bar A/B evaluation
python scripts/baseline_strategy_v3.py --coins bitcoin ethereum \
    --start 2026-01-16 --end 2026-04-15 --output-dir data/multi_2coins_v3

# CPCV (28 splits × 2 coins)
python scripts/v3_cpcv.py --coins bitcoin ethereum \
    --start 2024-05-01 --end 2026-04-30 --n-splits 28 \
    --output-dir data/v3_cpcv

# Component ablation (5 variants)
python scripts/v3_ablation.py --coins bitcoin ethereum \
    --start 2026-01-16 --end 2026-04-15 \
    --output-dir data/v3_ablations
```

## Architecture

```
Analysts (parallel data collection)
  → Bull/Bear Researchers (investment debate)
    → Research Manager (synthesis)
      → Trader (decision)
        → Aggressive/Conservative/Neutral Risk Analysts (risk debate)
          → Portfolio Manager (final rating: Buy/Overweight/Hold/Underweight/Sell)
```

### Agent Teams

**Crypto Analysts** (optional, any combination):
- **Market** — crypto OHLCV from Binance/CoinGecko + 150+ technical indicators via stockstats (RSI, MACD, Bollinger, ATR, etc.)
- **On-Chain** — funding rates (Binance Futures), TVL (DeFiLlama), gas prices + stablecoin supply (Web3/EVM)
- **Crypto Sentiment** — multi-source: Alpha Vantage crypto news, Reddit crypto subreddits, Google News, global macro news. LLM-centric analysis (no HuggingFace NLP model)
- **Prediction Model** — Random Forest + ARIMA(2,1,2) + LightGBM price forecasts with 95% confidence intervals, plus on-chain Gradient Boosting (observational). Supports multi-horizon prediction (h=1,3,7,14 days) and pooled multi-coin training.

**Stock Analysts** (legacy, still available when `asset_class="stock"`):
- Market, Social Media, News, Fundamentals

**Debate & Decision** (always active):
- **Researchers**: Bull argues for investment, Bear argues against — configurable rounds
- **Trader**: Synthesizes research into BUY/HOLD/SELL proposal
- **Risk Management**: Three-way debate (aggressive/conservative/neutral)
- **Portfolio Manager**: Final 5-level rating

Each analyst uses LangChain tool-calling to fetch data. BM25-based memory retrieves similar past situations. All debate/decision agents are crypto-aware and reference on-chain + prediction reports.

## Project Structure

```
tradingagents/                    # Core package
  agents/
    analysts/
      market_analyst.py           # Crypto OHLCV + technical indicators (switches tools by asset_class)
      onchain_analyst.py          # On-chain metrics: funding rates, TVL, gas, stablecoin supply
      crypto_sentiment_analyst.py # Multi-source sentiment: Alpha Vantage + Reddit + Google News
      prediction_analyst.py       # RF/ARIMA/GBR forecasts (tools defined inline, not via vendor routing)
      social_media_analyst.py     # Stock social media (legacy)
      news_analyst.py             # Stock news (legacy)
      fundamentals_analyst.py     # Stock fundamentals (legacy)
    researchers/                  # Bull and bear researchers (crypto-adapted prompts)
    managers/                     # Research manager, portfolio manager (crypto-adapted)
    risk_mgmt/                    # Aggressive, conservative, neutral debators (crypto-adapted)
    trader/                       # Trader agent (crypto-adapted)
    utils/
      agent_states.py             # TypedDict state: includes onchain_report, prediction_report
      agent_utils.py              # All tool imports (stock + crypto)
      memory.py                   # BM25-based FinancialSituationMemory
      core_stock_tools.py         # get_stock_data (stock mode)
      technical_indicators_tools.py
      fundamental_data_tools.py
      news_data_tools.py          # get_news, get_global_news, get_insider_transactions
      crypto_market_tools.py      # get_crypto_data, get_crypto_indicators
      onchain_tools.py            # get_funding_rates, get_tvl_metrics, get_stablecoin_metrics, get_gas_metrics, get_stablecoin_supply
      crypto_sentiment_tools.py   # get_reddit_posts, get_crypto_google_news
  graph/
    trading_graph.py              # TradingAgentsGraph orchestrator; clears session cache per propagate()
    setup.py                      # Graph node/edge construction (supports onchain, prediction, crypto_sentiment analysts)
    conditional_logic.py          # Routing: tool loops, debate continuation, analyst sequencing
    propagation.py                # State initialization (includes onchain_report, prediction_report)
    reflection.py                 # Post-trade learning (includes on-chain + prediction in situation memory)
    signal_processing.py          # Extract trading signal from portfolio manager output
  dataflows/
    interface.py                  # Vendor routing with 7 categories: core_stock, technical_indicators, fundamental_data, news_data, crypto_market_data, onchain_data, crypto_sentiment
    config.py                     # Runtime config with env var override on init
    coingecko_binance.py          # CoinGecko + Binance OHLCV with disk + session cache
    onchain.py                    # Web3 (gas, stablecoin supply), Binance Futures (funding rates), DeFiLlama (TVL, stablecoin mcap)
    crypto_sentiment.py           # Reddit scraper + Google News fetcher (raw text for LLM analysis)
    y_finance.py                  # Yahoo Finance (stock mode)
    alpha_vantage*.py             # Alpha Vantage (stock mode + crypto news)
    stockstats_utils.py           # Technical indicator computation (works on any OHLCV data)
  models/
    rf_model.py                   # Random Forest (1000 trees) with 95% CI; forecast_next() + model_run() + model_run_pooled()
    arima_model.py                # ARIMA(2,1,2) with exogenous features; forecast_next() + model_run()
    onchain_model.py              # Gradient Boosting on on-chain features; forecast_next() + model_run()
    lgb_model.py                  # LightGBM for multi-horizon pooled prediction; model_run_pooled()
    model_utils.py                # data_transform, fetch_ohlcv_for_model, build_pooled_dataset, compute_metrics
    prediction.py                 # Prediction dataclass with to_report_string()
  backtesting/
    engine.py                     # run_backtest() with 5-level signal support, realistic costs (fees, slippage, short borrowing)
    strategies.py                 # FiveLevelSignal, ThresholdSignal, ModelConsensus; SignalLevel enum
    runner.py                     # evaluate_models(), generate_system_signals(), run_system_backtest()
    reporting.py                  # print_summary_table(), plot_equity_curves(), save_results_json()
  strategies/
    v2_sizing.py                  # V2 sizing primitives — single source of truth for backtest + live (signals, vol, sizing, leverage, trend filter)
    v3/                           # V3 quant stack (NH-HMM + microstructure + multi-horizon) — built, underperforms V2 (see THESIS_FINDINGS.md §12)
  execution/
    exchange.py                   # Binance Futures wrapper (testnet default); place_market_order, place_stop_loss
    risk.py                       # 4-tier pre-trade checks: confidence gate, daily loss limit, max positions, position sizing
    runner.py                     # LiveRunner: propagate() → risk check → execute → journal log
    logger.py                     # SQLite trade journal: trades, portfolio_snapshots, daily_summary, analyst_reports
  llm_clients/
    factory.py                    # LLM client factory (OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama)
    replay_cache.py               # CachedChatModel — SQLite-backed LLM response cache for deterministic backtests
    base_client.py, openai_client.py, anthropic_client.py, google_client.py
    model_catalog.py, validators.py
  default_config.py               # DEFAULT_CONFIG + apply_env_overrides()
cli/
  main.py                         # Typer CLI with asset class selection (crypto/stock)
  utils.py                        # get_crypto_ticker, select_asset_class, select_analysts(asset_class)
  models.py                       # AnalystType enum (market, social, news, fundamentals, onchain, prediction, crypto_sentiment)
scripts/
  evaluate_models.py              # Single-coin RF+ARIMA walk-forward eval (legacy)
  evaluate_models_multi.py        # Multi-coin multi-horizon pooled eval (LGB+ARIMA+RF) with cross-asset features
  backtest_models.py              # Simple strategy backtest on predictions (naive daily flip)
  backtest_system.py              # Full multi-agent system backtest (propagate() over date range)
  baseline_strategy.py            # V1 baseline: RF+ARIMA h=1 ensemble (superseded by V2)
  baseline_strategy_v2.py         # V2 baseline (PRODUCTION): LGB h=7/h=14 term structure consensus + SMA30 trend filter + adaptive hold
main.py                           # Example: crypto analysis of bitcoin
THESIS_FINDINGS.md                # Persistent record of all experimental results (keep updated!)
```

## Development Commands

```bash
# Install in development mode (requires Python 3.10+; runtime uses 3.9 with __future__ annotations)
pip install -e .

# Run the interactive CLI
tradingagents
# or: python -m cli.main

# Run example crypto analysis
python main.py

# Run tests
python -m pytest tests/

# Docker
docker compose run --rm tradingagents

# ── Model Evaluation & Backtesting ──────────────────────────────────

# === FINALIZED BASELINE PIPELINE ===
# Step 1: Multi-horizon pooled evaluation (LGB only — ARIMA proven useless in pooled setting)
python scripts/evaluate_models_multi.py --coins bitcoin ethereum --horizons 1 3 7 14 \
    --models lgb --days 730 --min-train 365 --output-dir data/multi_2coins_v2

# Step 2: Run V2+trend baseline strategy (current best — Sharpe 2.69 on 2-coin portfolio)
python scripts/baseline_strategy_v2.py --pred-dir data/multi_2coins_v2 --symmetric

# For trading a target altcoin, use "2+1" pool (BTC+ETH+target)
python scripts/evaluate_models_multi.py --coins bitcoin ethereum binancecoin \
    --horizons 1 3 7 14 --models lgb --output-dir data/multi_3coins_bnb
python scripts/baseline_strategy_v2.py --pred-dir data/multi_3coins_bnb --symmetric

# === LEGACY / DIAGNOSTIC ===
# Single-coin evaluation (legacy, not used for production strategy)
python scripts/evaluate_models.py --coin bitcoin --days 730 --models rf arima --min-train 365

# Simple strategy backtest (naive daily flip, replaced by V2)
python scripts/backtest_models.py --input data/eval_predictions.csv --threshold 0.01

# V1 baseline (RF+ARIMA ensemble, h=1 only — superseded by V2)
python scripts/baseline_strategy.py --input data/eval_predictions.csv

# Full multi-agent system backtest (expensive — uses LLM API calls)
python scripts/backtest_system.py --coin bitcoin --start 2024-05-01 --end 2025-03-01
```

## Python API Usage

### Crypto Analysis (primary use case)
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["asset_class"] = "crypto"
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"

ta = TradingAgentsGraph(
    selected_analysts=["market", "onchain", "crypto_sentiment", "prediction"],
    debug=True,
    config=config,
)
final_state, signal = ta.propagate("bitcoin", "2025-01-15")
# signal: "BUY" | "OVERWEIGHT" | "HOLD" | "UNDERWEIGHT" | "SELL"

ta.reflect_and_remember(returns_losses=1000)
```

### Live Execution
```python
from tradingagents.execution.runner import LiveRunner

runner = LiveRunner(config={
    "asset_class": "crypto",
    "execution": {"live_mode": False, "dry_run": True},  # testnet + dry run
})
signal, result = runner.run_single("bitcoin")
```

### Backtesting (Agent Signal-Based)
```python
from tradingagents.backtesting import run_backtest, FiveLevelSignal

result = run_backtest(
    dates=dates_series, actuals=price_array,
    agent_signals=signals_list,  # ["BUY", "HOLD", "SELL", ...]
    strategy=FiveLevelSignal(), ticker="BTC",
)
print(f"Sharpe: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.1%}")
```

### Model Evaluation (Walk-Forward)
```python
from tradingagents.backtesting.runner import evaluate_models

results = evaluate_models(
    coin="bitcoin", lookback_days=730, min_train_window=365,
    models=["rf", "arima"], output_dir="data/",
)
# Returns dict[str, ModelEvalResult] with metrics + dated predictions
```

### Multi-Horizon Pooled Evaluation
```bash
# Best results: 2-coin pool (BTC+ETH), LGB, h=14 achieves 84.6% DirAcc for BTC
python scripts/evaluate_models_multi.py --coins bitcoin ethereum \
    --horizons 1 3 7 14 --models lgb --output-dir data/multi_2coins_v2

# 2+1 approach for trading altcoins: BTC+ETH+target
python scripts/evaluate_models_multi.py --coins bitcoin ethereum binancecoin \
    --horizons 1 3 7 14 --models lgb --output-dir data/multi_3coins_bnb
```

### V2 Baseline Strategy (Production)
```bash
# 2-coin portfolio: Sharpe 2.69, return +106%, MaxDD 5.9%
python scripts/baseline_strategy_v2.py --pred-dir data/multi_2coins_v2 --symmetric

# 3-coin portfolio: Sharpe 2.58, return +156%, MaxDD 13%
python scripts/baseline_strategy_v2.py --pred-dir data/multi_3coins_bnb --symmetric

# Key defaults: h=7/h=14 consensus, SMA30 trend filter (1.5x multiplier), 7-day min hold
# with adaptive early exit, 10% target vol, half-Kelly, 3x max leverage, 3% stop-loss
```

### Stock Analysis (legacy)
```python
config["asset_class"] = "stock"
ta = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    config=config,
)
final_state, signal = ta.propagate("NVDA", "2025-01-15")
```

## Configuration

**Environment variables** (`.env` file, auto-loaded via python-dotenv):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`
- `ALPHA_VANTAGE_API_KEY` (optional, for Alpha Vantage crypto news)
- `WEB3_PROVIDER_URI_ETH`, `WEB3_PROVIDER_URI_BSC` (optional, for on-chain gas/stablecoin supply)
- `BINANCE_API_KEY`, `BINANCE_API_SECRET` (for live trading only, in `.env.trading`)

**Execution env var overrides** (for container/cloud deployment):
- `LIVE_MODE`, `DRY_RUN`, `MAX_POSITION_PCT`, `STOP_LOSS_PCT`, `MAX_DAILY_LOSS_PCT`, `MAX_OPEN_POSITIONS`, `MIN_CONFIDENCE`, `POSITION_SIZING`, `LEVERAGE`
- `TRADINGAGENTS_LLM_PROVIDER`, `TRADINGAGENTS_ASSET_CLASS`, `TRADINGAGENTS_DEEP_THINK_LLM`, `TRADINGAGENTS_QUICK_THINK_LLM`

**Config dict** (`tradingagents/default_config.py`):
- `asset_class`: `"crypto"` (default) or `"stock"`
- `llm_provider`: openai | anthropic | google | xai | openrouter | ollama
- `deep_think_llm` / `quick_think_llm`: Model IDs
- `max_debate_rounds` / `max_risk_discuss_rounds`: Debate iteration count (default: 1)
- `data_vendors`: Category-level vendor selection (7 categories)
- `web3_provider_eth`, `web3_provider_bsc`: Ethereum/BSC RPC URLs
- `use_onchain`: Enable on-chain data (default True, degrades gracefully)
- `prediction_models`: RF/ARIMA/GBR/LGB hyperparameters, checkpoint paths, lookback days
- `replay_cache`: Enable LLM response caching for deterministic backtest reruns (default: False)
- `replay_cache_db`: SQLite path for cached LLM responses (default: `./data/llm_replay_cache.db`)
- `execution`: live_mode, dry_run, max_position_pct, stop_loss_pct, position_sizing, leverage

## Code Conventions

- Python 3.9+ with `from __future__ import annotations` for PEP 604 union syntax; `Annotated[type, "description"]` for tool parameters
- `TypedDict` for LangGraph state schemas (see `agent_states.py`)
- snake_case for functions/variables, CamelCase for classes
- Google-style docstrings
- `@tool` decorator from `langchain_core.tools` for all tool functions
- Tools route through `route_to_vendor()` in `dataflows/interface.py` (except prediction tools which call model code directly)

## Key Patterns

- **Vendor routing**: `dataflows/interface.py` routes tool calls to vendor implementations with automatic fallback on rate limits
- **Asset class switching**: `config["asset_class"]` controls which tools Market Analyst binds (crypto vs stock) and which data vendors are used
- **Session cache**: `coingecko_binance.py` has in-memory `_session_cache` cleared per `propagate()` call to avoid redundant fetches within a single analysis run. Disk cache (CSV per symbol) persists across sessions.
- **Factory pattern**: `llm_clients/factory.py` creates provider-specific LLM clients
- **BM25 memory**: Lexical similarity retrieval of past trading situations; 5 separate memory instances
- **Tool-calling loops**: Analysts call tools via `bind_tools`, graph loops until no more tool calls
- **Env var overrides**: `apply_env_overrides()` in `default_config.py` overlays env vars on config at initialization
- **Prediction tools bypass vendor routing**: Defined inline in `prediction_analyst.py`, call model code directly (documented exception to the vendor pattern)
- **Graceful degradation**: On-chain data, Web3 metrics, Reddit all optional — system continues if any source is unavailable
- **Look-ahead bias prevention**: `set_prediction_trade_date()` binds prediction models to backtest date; OHLCV cache uses `min(curr_date, today)` as fetch boundary; `data["Date"] <= curr_date` filter applied before returning
- **LLM replay cache**: `CachedChatModel` wraps LangChain chat models with SQLite-backed prompt-hash caching. Enable via `config["replay_cache"] = True`. Mandatory for system backtests (determinism + cost control).
- **Multi-horizon pooled prediction**: `model_utils.build_pooled_dataset()` creates cross-coin features; `lgb_model.model_run_pooled()` runs walk-forward on pooled data with horizon as parameter. Best result: 2-coin BTC+ETH pool, h=14, BTC 84.6% / ETH 75.8% directional accuracy.
- **V2 strategy trend filter**: V2 sizing primitives (including `apply_trend_filter`) live in `tradingagents/strategies/v2_sizing.py`; `scripts/baseline_strategy_v2.py` imports them. SMA30-based position scaling — 1.5x when aligned with trend, 0.5x when against. Single highest-impact improvement (Sharpe 1.88 → 2.69).
- **"2+1" pooling pattern**: For trading a target altcoin, use a 3-coin pool {BTC, ETH, target} instead of larger universes. Preserves BTC/ETH quality while giving near-optimal DirAcc for the target coin.

## Gotchas

- **`asset_class` defaults to `"crypto"`** — set to `"stock"` explicitly for equity analysis
- **Prediction models require training** — run `scripts/train_models.py` before using the prediction analyst, or models will train on the fly (slow)
- **`LIVE_MODE` defaults to `False`** — testnet only; must be explicitly `True` for real money
- **`social` and `crypto_sentiment` analysts both write to `sentiment_report`** — they are mutually exclusive (don't select both)
- **On-chain Web3 metrics require RPC endpoints** — set `WEB3_PROVIDER_URI_ETH` / `WEB3_PROVIDER_URI_BSC` env vars; without them, gas/stablecoin supply tools return helpful error messages
- **On-chain model uses fallback features** — when called via prediction analyst, the GBR model receives OHLCV-derived features (not actual on-chain data), since the data pipeline doesn't merge on-chain metrics into the model DataFrame
- **CoinGecko free tier rate limits** — disk caching + backoff mitigates this; consider Pro API key for heavy usage
- **Reddit rate limiting** — exponential backoff + 30s delay on 429; Reddit data is additive, not required
- **Confidence parsed from LLM output** — regex-based extraction of confidence level from portfolio manager text; can be brittle if output format drifts
- **Max recursion limit** — with 4 crypto analysts + debates, the graph has many nodes; default `max_recur_limit=100` should suffice but increase if hitting limits

- **Pooled training universe matters**: Adding altcoins beyond BTC+ETH degrades directional accuracy by 12-22 pp. Optimal universe is 2-coin (BTC+ETH). See THESIS_FINDINGS.md for full comparison.
- **ARIMA R² is misleading**: ARIMA achieves R²>0.999 but ~50% directional accuracy (coin flip) in pooled settings. Always evaluate by directional accuracy and PnL, not regression metrics.
- **RF is extremely slow for pooled walk-forward**: 1000 trees × 365 iterations × 10 coins = hours. Use LightGBM for pooled experiments.

## Results Output

- **State logs**: `results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json` (includes all analyst reports, debate state, final decision)
- **Trade journal**: `data/trade_journal.db` (SQLite — trades, portfolio snapshots, daily summaries, full analyst reports)
- **Model checkpoints**: `data/checkpoints/` (RF, ARIMA, GBR joblib/pkl files)
- **Evaluation results (legacy, pre-DirAcc fix)**: `data/multi_2coins/`, `data/multi_5coins/`, `data/multi_full/`
- **Evaluation results (current, with ref_price)**: `data/multi_2coins_v2/`, `data/multi_3coins_bnb/`, `data/multi_5coins_v2/`, `data/multi_6coins/`
- **V2 strategy reports**: `data/multi_*/report_v2/` — per-universe detailed reports with per-coin plots, monthly returns, metrics.json
- **Backtest plots**: `data/*/baseline_v2_equity.png` (current), `data/baseline_equity.png` (V1 legacy)
- **Thesis findings**: `THESIS_FINDINGS.md` — persistent record of all experimental results (KEEP UPDATED)
