# Building reliable backtests for multi-agent LLM crypto trading

**Every major LLM trading paper published between 2023 and 2025 is likely contaminated by lookahead bias, and most don't even acknowledge it.** A systematic review of 164 papers found no single bias discussed in more than 28% of studies. The FINSABER framework demonstrated that when LLM trading strategies are evaluated properly — longer time horizons, survivorship-free universes, realistic costs — previously reported advantages evaporate entirely, with FinMem's MSFT returns flipping from +23.3% to **−22.0%**. For a master's thesis adapting TradingAgents to cryptocurrency, this means the backtesting methodology is not a secondary concern — it is the thesis. Getting it wrong renders every downstream result meaningless.

This report synthesizes the current academic literature, engineering patterns, and framework analysis needed to build an event-driven backtest system for multi-agent LLM crypto trading that avoids the pitfalls plaguing existing work. It covers LLM memorization and mitigation, bitemporal data architecture on Delta Lake, crypto-specific data quality hazards, and concrete detection strategies for lookahead bias.

---

## LLM memorization makes most financial backtests unfalsifiable

The most dangerous form of lookahead bias in LLM trading systems is not a coding bug — it is structural. **GPT-4o can recall exact S&P 500 closing prices with <1% error for dates within its training window** (Lopez-Lira, Tang & Zhu, 2025, arXiv:2504.14765). Errors explode for post-cutoff dates. This means any backtest conducted within an LLM's training period cannot distinguish genuine reasoning from memorization. Lopez-Lira et al. formalize this as a non-identification problem: when the model has seen realized values during training, its counterfactual forecasting ability is unrecoverable from outputs alone.

Sarkar & Vafa (2024, SSRN:4754678) provide direct tests confirming the problem. LLMs prompted with 2019 earnings calls to predict 2020 risks were **3.6× more likely to mention "Pandemic"** than those predicting 2019 risks from 2018 calls. When dates and firm names were masked, GPT-4 still inferred the year with **r = 0.79 correlation** and reconstructed firm identity with 70% accuracy. Prompting-based mitigations — instructions to "only use information available at time t" — are demonstrably insufficient.

The mitigation landscape has matured rapidly into five tiers, ranked by effectiveness:

**Chronologically consistent models** represent the gold standard. ChronoBERT/ChronoGPT (He et al., 2025, arXiv:2502.21206) provide 149M–1.5B parameter models with strict annual cutoffs from 1999–2024. DatedGPT (Yan et al., 2025, arXiv:2603.11838) offers twelve 1.3B-parameter models trained from scratch on ~100B tokens per annual cutoff. These are too small for frontier reasoning tasks but prove the concept. For a thesis, **the practical approach is to ensure backtest periods fall entirely after the LLM's verified training cutoff** — use GPT-4o only on post-April-2024 data, Claude 3.5 only on post-early-2025 data, and so on.

**Divergence Decoding** (Merchant & Levy, 2025, arXiv:2512.06607) offers a compelling middle ground. It uses two small auxiliary models — one fine-tuned on data to "forget" (post-cutoff) and one on data to "retain" (pre-cutoff) — to adjust the base model's logits at inference time. Even trigram language models work as auxiliaries, making the approach extremely low-cost while preserving model utility without retraining.

**Entity anonymization** helps but has severe trade-offs. Glasserman & Lin (2023, arXiv:2309.17322) found that anonymized headlines actually outperformed originals, suggesting the distraction effect from company knowledge exceeds lookahead bias. However, Wu, Yang, Ying & Zhou (2025, arXiv:2511.15364) showed that **information loss from anonymization is more pervasive than the bias it removes**, particularly when numerical entities are stripped. This creates a genuine dilemma: anonymization reduces signal quality more than it reduces bias.

**Statistical detection** should be standard practice. The **LAP test** (Gao, Jiang & Yan, 2025, arXiv:2512.23847) uses MIN-K% PROB from membership inference attacks to estimate whether prompts appeared in training data. A positive correlation between LAP and forecast accuracy formally indicates contamination. The TimeSPEC framework (arXiv:2602.17234) decomposes agent rationales into atomic claims and uses Shapley values to quantify how much decision-driving reasoning stems from leaked information, reducing leakage by **75–99%** compared to standard prompting.

---

## Event-driven architecture with bitemporal data is non-negotiable

For multi-agent LLM systems, vectorized backtesting is structurally inadequate. LLM agents are sequential decision-makers reacting to information as it arrives — they require an event-driven simulator that feeds data strictly as-of-time. The canonical architecture uses a FIFO event queue with typed events (`MarketEvent`, `SentimentEvent`, `OnChainEvent`, `NewsEvent`) processed through a two-loop structure: an outer heartbeat loop advancing time to the next event, and an inner loop draining all cascading events at each timestamp.

The critical design element is **distinguishing event time from knowledge time**. Every data point must carry both timestamps, and the simulator must advance based on knowledge time — when a system could have first known the information, not when the underlying event occurred. This is especially important for crypto on-chain data, where Glassnode demonstrated that **identical backtests produce dramatically different results** when run with point-in-time versus retroactively revised metrics. Their BTC exchange balance strategy showed competitive returns with buy-and-hold using revised data but notably worse performance using PIT data, missing key upticks in November 2024 and March 2025. On-chain analytics metrics are continuously revised as address clustering and entity labeling improve.

For heterogeneous data sources with different latencies, a **priority-queue merge pattern** (min-heap sorted by knowledge time) unifies all streams into a single event timeline:

| Source | Typical Latency | Update Frequency |
|--------|----------------|-----------------|
| OHLCV (exchange WebSocket) | ~100ms | Per-tick or 1-min bars |
| On-chain data (Glassnode) | 10 min–1 hr | 10 min–24 hr |
| News APIs | 1–30 seconds | Irregular |
| Social sentiment | 1–5 minutes | Irregular |
| Fundamental data | Hours to days | Quarterly |

**Bitemporal modeling** tracks every fact along two independent time dimensions: valid time (when the fact is true in reality) and transaction time (when the system recorded it). The HSTR framework (MDPI, 2026) implements this specifically for LLM financial agents, achieving a 97% reduction in context retrieval latency while guaranteeing strict temporal integrity. For Delta Lake, the schema pattern uses append-only Bronze tables with `valid_from`, `valid_to`, `sys_recorded_at` columns, with Silver-layer interval closure computed via `LEAD()` window functions.

Among open-source frameworks, **NautilusTrader** is the strongest foundation for this thesis. Its Rust-native core with Python bindings provides nanosecond-resolution timestamps, an Actor/Strategy pattern that maps naturally to multi-agent architectures, a MessageBus for pub/sub agent communication, and explicit `ts_event` versus `ts_init` separation for PIT correctness. It streams 5M+ rows/second and was designed with AI agent training as a stated goal. QuantConnect LEAN offers superior built-in PIT data handling across 40+ sources but uses a single-strategy paradigm less suited to multi-agent orchestration. VectorBT excels at rapid parameter sweeps (1M orders in 70–100ms) but is structurally incompatible with sequential LLM decision-making.

---

## TradingAgents' evaluation is far below minimum standards

The TradingAgents framework (Xiao, Sun, Luo & Wang, 2024, arXiv:2412.20138) simulates a professional trading firm with seven specialized LLM agents across five stages: four concurrent analysts (fundamental, sentiment, news, technical), bull/bear researchers engaged in structured debate, a trader agent, a risk management team with three risk profiles, and a fund manager for final approval. The architecture is well-conceived — built on LangGraph, using ReAct prompting, with dual-LLM routing (cheap models for data retrieval, expensive models for reasoning).

However, the evaluation is critically deficient. The paper's "Back Trading" covers **only 3 months on 5 large-cap tech stocks** (AAPL, AMZN, GOOGL, NVDA, TSLA). Each prediction requires 11 LLM calls and 20+ tool calls, which the authors cite as justification for the short period. The reported Sharpe ratios are exceptionally high — the authors themselves acknowledge they "exceed our expected empirical range." The framework has **no documented PIT correctness mechanism**: it fetches data from live APIs (FinnHub, Alpha Vantage) with no guarantee these APIs don't return revised data. There is no survivorship bias control, no transaction cost modeling, and no memory persistence between `propagate()` calls.

The **FINSABER framework** (Li et al., 2025, arXiv:2505.07078) provides the most devastating critique of LLM trading evaluation. Using a bias-aware pipeline with survivorship-free data spanning 2000–2024 across 100+ symbols, they found that **LLM strategies are overly conservative in bull markets and overly aggressive in bear markets**. Previously reported advantages collapsed under proper evaluation. FinMem's NFLX Sharpe ratio dropped from 2.017 to **−0.478**. Their conclusion: "LLM-derived alpha is likely a methodological artefact of narrow, biased evaluations."

A 2026 taxonomy paper (arXiv:2603.27539) evaluated 12 multi-agent systems and identified five pervasive evaluation failures that can "reverse the sign of reported returns": look-ahead bias, survivorship bias, backtesting overfitting, transaction cost neglect, and regime-shift blindness. The paper introduces the **Coordination Primacy Hypothesis** — inter-agent coordination protocol design drives decision quality more than model scaling — and the **Coordination Breakeven Spread** metric for determining whether multi-agent coordination adds genuine value net of costs. It also warns that structured debate, while improving accuracy over 2–4 rounds, risks **Degeneration-of-Thought**, where agents converge to a shared wrong answer through social pressure.

Agent memory creates a particularly insidious leakage vector. FinMem's layered memory persists across decisions by design — memories from training carry into testing. FinAgent's dual-level reflection module analyzes past trading decisions, potentially encoding future-aware insights. For any multi-agent system with persistent state, the rule is: **memory accumulated during the backtest period must contain only information available at each decision point**, and memory must be reset between independent backtest runs.

**LLM non-determinism** compounds these challenges. Temperature=0 does not guarantee deterministic outputs due to floating-point non-associativity in GPU parallel reductions. OpenAI's `seed` parameter provides only "best effort" determinism. The practical solution is **replay caching**: hash each prompt + model + parameters into a cache key, store responses in Delta Lake, and serve cached responses on identical requests. This makes warm reruns deterministic and free, reducing the cost problem (TradingAgents' ~2,772 LLM calls per stock per year at $10–40 per comprehensive test).

---

## Cryptocurrency data is an adversarial environment for backtesting

Crypto introduces data quality challenges that dwarf those in traditional finance. **95% of reported Bitcoin trading volume is fake** according to Bitwise's 2019 SEC report, with Cong et al. (2020, Management Science) confirming that over 70% of volume on unregulated exchanges consists of wash trades. Volume-based signals are unreliable unless sourced from regulated exchanges (Coinbase, Kraken) or cross-referenced with on-chain deposit/withdrawal volumes.

**Survivorship bias in crypto is catastrophic for equal-weighted strategies.** Ammann et al. (2022, SSRN:4287573) studied 3,904 cryptocurrencies from 2014–2021 and found the annualized survivorship bias was 0.93% for value-weighted portfolios but **62.19% for equal-weighted portfolios**. The size premium was overestimated by 50%. Robuxio demonstrated that a trend strategy on current top-10 crypto assets showed 4× more profit than trading the actual top-10-at-the-time. Building survivorship-free datasets requires CoinMarketCap's permanent numeric IDs (UCIDs) or CoinGecko's inactive coin tracking, reconstructing the tradeable universe as it actually existed at each rebalance date.

The 24/7 market structure eliminates natural session boundaries. There is no canonical "close" price — any daily candle boundary is an arbitrary choice. Liquidity varies dramatically by time-of-day and day-of-week. During the March 2020 crash, BTC traded at **$3,800 on some exchanges and $4,800 on others — a 25% spread**. Exchange outages during peak volatility prevent order execution entirely; the October 2025 flash crash ($19B in liquidations) saw API outages and oracle misfires across venues.

On-chain data presents unique PIT challenges. **Chain reorganizations** mean data near the chain tip is unreliable — Bitcoin requires 6 confirmations (~60 minutes), Ethereum achieves economic finality after 2 epochs (~13 minutes). Block timestamps can be manipulated by ±15 seconds on Ethereum and ±2 hours on Bitcoin. For backtesting, query only finalized blocks and use block numbers rather than timestamps for sub-minute ordering. Archive nodes (storing every historical state since genesis) are essential for reconstructing smart contract state — full nodes retain only the last 128 blocks (~28 minutes on Ethereum).

Stablecoin depegging adds quote-currency risk that most backtests ignore entirely. Moody's documented over **1,900 depeg events** between 2020 and mid-2023. A USDT depeg raises the probability of BTC price jumps nearly 5× within 5 minutes. If a strategy uses stablecoin-denominated prices, apparent profits during depeg events may be illusory.

---

## Delta Lake engineering patterns for PIT-correct backtesting

Delta Lake's time travel provides the foundational mechanism for PIT reconstruction, but requires deliberate configuration. The default retention is only 7 days — for backtesting, extend both `logRetentionDuration` and `deletedFileRetentionDuration` to match your full backtest horizon (minimum 365 days). Never run `VACUUM` with aggressive retention on critical tables. Enable **Change Data Feed** early in the pipeline — it captures row-level changes (`_change_type`, `_commit_version`, `_commit_timestamp`) but only after enablement, not retroactively.

The **Medallion architecture** maps naturally to PIT financial data:

**Bronze** stores append-only raw ingestion with `_ingested_at`, `_source` metadata — never transform, never delete. **Silver** performs deduplication, validation, and bitemporal interval computation (deriving `valid_to` via `LEAD()` window functions over `valid_from`). **Gold** materializes PIT-correct feature tables for backtesting, pre-computed indicators, and the tradeable universe at each timestamp. Data corrections follow the compensating events pattern: never update the original record, append a correction with a new `sys_recorded_at`, and let bitemporal queries resolve to the version known at any given simulation time.

For LLM-specific engineering, **replay caching in Delta Lake** is the critical pattern. Hash each LLM call's prompt, model, temperature, and max_tokens into a SHA-256 cache key. Store responses with cost tracking and prompt version metadata. On cache hit, serve the stored response — making warm reruns deterministic and free. **MLflow's Prompt Registry** (integrated with Databricks) provides git-inspired versioning with immutable prompt versions, aliases for deployment management, and automatic lineage linking to model versions. Every backtest run should log the prompt version, model ID, and cache key for full reproducibility.

Detecting lookahead bias after the fact requires multiple complementary strategies. **Freqtrade's lookahead-analysis pattern** compares full-data signals against truncated-data signals at each decision point — any difference indicates bias. The **delay sensitivity test** artificially shifts features by one bar; if performance collapses, lookahead is present. For time-series cross-validation, **Combinatorial Purged Cross-Validation** (de Prado, 2018) partitions observations into N ordered groups, tests k groups at a time with purging (removing training observations whose labels overlap test labels) and embargo periods (excluding observations immediately after test folds). The `skfolio` library implements this as `CombinatorialPurgedCV`. For LLM-specific contamination, apply the LAP test to estimate memorization propensity, and use the TimeSPEC Shapley decomposition to quantify what fraction of agent reasoning stems from leaked future information.

---

## Conclusion

Building a reliable backtest for multi-agent LLM crypto trading is fundamentally a **data integrity and epistemology problem**, not an architecture problem. The core contribution of a thesis in this space is not demonstrating that LLM agents can trade profitably — FINSABER suggests that claim may be entirely artifactual — but rather establishing a methodology that can distinguish genuine reasoning from memorization, genuine signal from data leakage.

Three non-obvious insights emerge from this research. First, entity anonymization may do more harm than good: the information loss from stripping identifiers exceeds the bias reduction in many settings, making Divergence Decoding a superior approach. Second, on-chain analytics data — widely assumed to be immutable because blockchains are immutable — is in fact continuously revised through entity relabeling, making Glassnode's PIT tier (or equivalent raw-blockchain-data approaches) essential rather than optional. Third, the Coordination Primacy Hypothesis suggests that for multi-agent systems, the inter-agent communication protocol matters more than the underlying model quality — a testable claim that could form the core of a thesis contribution.

The minimum viable methodology requires: backtest periods strictly after the LLM's training cutoff, bitemporal data modeling in Delta Lake with append-only Bronze storage, survivorship-free universe construction using CoinMarketCap UCIDs, replay caching for LLM call determinism, the LAP test or TimeSPEC for contamination detection, and Combinatorial Purged Cross-Validation for out-of-sample evaluation. NautilusTrader's Actor/MessageBus architecture provides the strongest open-source foundation for the event-driven simulation layer. Any result that shows suspiciously high Sharpe ratios or smooth equity curves should be treated as evidence of bias until proven otherwise.