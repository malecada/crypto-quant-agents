import os

from dotenv import load_dotenv

# Load .env files if present (project root or current directory)
load_dotenv(override=False)
load_dotenv(".env.trading", override=False)

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4-mini",
    "quick_think_llm": "gpt-5.4-nano",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 2,  # Phase 5 / Tier B7: 3-way Bull/Bear/Skeptic-Quant rotation
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        # Crypto-specific vendors
        "crypto_market_data": "coingecko_binance",
        "onchain_data": "onchain",  # "onchain" = realtime (funding/TVL/gas), "onchain_pit" = PIT store
        "crypto_sentiment": "crypto_sentiment",
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Asset class: "stock" or "crypto"
    "asset_class": "crypto",
    # Crypto-specific configuration
    "binance_symbol_map": {},  # Optional overrides: {"coingecko_id": "BINANCE_SYMBOL"}
    "web3_provider_eth": os.getenv("WEB3_PROVIDER_URI_ETH", ""),
    "web3_provider_bsc": os.getenv("WEB3_PROVIDER_URI_BSC", ""),
    "use_onchain": True,  # Enable on-chain data collection (degrades gracefully if no RPC)
    # Prediction model configuration
    "prediction_models": {
        "rf_n_estimators": 1000,
        "rf_max_depth": None,
        "rf_min_samples_split": 5,
        "rf_min_samples_leaf": 2,
        "arima_order": [2, 1, 2],
        "arima_max_iter": 500,
        "onchain_n_estimators": 500,
        "onchain_max_depth": 5,
        "onchain_learning_rate": 0.1,
        # LightGBM (used by lgb_model for pooled multi-horizon prediction)
        "lgb_n_estimators": 500,
        "lgb_max_depth": -1,
        "lgb_learning_rate": 0.05,
        "lgb_num_leaves": 31,
        "lgb_min_child": 20,
        "lag_features": 7,
        "lookback_days": 300,
        "checkpoint_dir": "./data/checkpoints/",
        "prediction_interval_alpha": 0.05,  # 95% confidence interval
    },
    # Multi-horizon pooled prediction configuration
    "coin_universe": [
        "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
        "cardano", "avalanche-2", "chainlink", "polkadot", "matic-network",
    ],
    "model_horizons": [1, 3, 7, 14],
    "pooled_lookback_days": 730,
    "pooled_min_train_window": 365,
    # LLM replay cache — enables deterministic backtest reruns by caching
    # LLM responses keyed by prompt hash.  Disabled by default for live use.
    "replay_cache": False,
    "replay_cache_db": "./data/llm_replay_cache.db",
    # Hybrid quant+LLM modulator (Phase 0 scaffolding; Phase 4 wires into graph).
    # Asset-agnostic single-path architecture — every coin runs the same Layer 1
    # → Layer 2 → Layer 3 stack.  LLM influence is a derived quantity from
    # (regime, uncertainty, rolling_llm_edge[coin], unlock_flag), NOT a per-coin
    # config knob.  See plans/i-want-to-start-recursive-candy.md.
    "regime_weighting": {
        "bull":     [0.2, 0.3],
        "sideways": [0.6, 0.8],
        "bear":     [0.4, 0.4],
    },
    "rolling_edge_window_days": 30,
    "uncertainty_dampener_k": 1.0,
    "edge_dampener_k": 1.0,
    "rolling_edge_min_trades": 10,  # cold-start threshold per coin
    "quant_pred_dir": "data/multi_2coins_v2",
    "regime_hmm_path_template": "data/checkpoints/regime_hmm_{coin}.pkl",
    # Asset-name anonymization (Tier A4 / Glasserman & Lin 2309.17322).
    # When True, build_instrument_context masks the coin name to "Asset_X"
    # for all analyst + debate agents; the Portfolio Manager un-masks at
    # the Layer 3 boundary. Configurable so V4 ablation can toggle it.
    "anonymize_assets": False,
    # Hybrid RAG (Tier B8) — extends BM25 memories with FAISS dense
    # retrieval + reciprocal-rank-fusion. Off by default; enable per
    # research-run since it requires sentence-transformers + faiss-cpu.
    "hybrid_rag": False,
    # Execution configuration (safe defaults — testnet only)
    "execution": {
        "live_mode": False,  # Must be explicitly True for real money
        "dry_run": False,    # When True, log trades but don't place orders
        "max_position_pct": 0.02,
        "stop_loss_pct": 0.03,
        "max_daily_loss_pct": 0.05,
        "max_open_positions": 3,
        "min_confidence": "medium",  # "high" / "medium" / "low"
        "position_sizing": "fixed_fraction",  # or "kelly"
        "kelly_fraction": 0.5,  # half-Kelly when position_sizing="kelly"
        "leverage": 1,
    },
}


def apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides to config dict.

    Supports the following env vars (matching Krypto-v0's pattern):

    LLM:
      TRADINGAGENTS_LLM_PROVIDER, TRADINGAGENTS_DEEP_THINK_LLM,
      TRADINGAGENTS_QUICK_THINK_LLM, TRADINGAGENTS_BACKEND_URL

    Crypto:
      TRADINGAGENTS_ASSET_CLASS, WEB3_PROVIDER_URI_ETH, WEB3_PROVIDER_URI_BSC

    Execution (safe defaults enforced):
      LIVE_MODE, DRY_RUN, MAX_POSITION_PCT, STOP_LOSS_PCT,
      MAX_DAILY_LOSS_PCT, MAX_OPEN_POSITIONS, MIN_CONFIDENCE,
      POSITION_SIZING, LEVERAGE, TRADING_STRATEGY
    """
    # Top-level overrides
    _env_str(config, "llm_provider", "TRADINGAGENTS_LLM_PROVIDER")
    _env_str(config, "deep_think_llm", "TRADINGAGENTS_DEEP_THINK_LLM")
    _env_str(config, "quick_think_llm", "TRADINGAGENTS_QUICK_THINK_LLM")
    _env_str(config, "backend_url", "TRADINGAGENTS_BACKEND_URL")
    _env_str(config, "asset_class", "TRADINGAGENTS_ASSET_CLASS")
    _env_str(config, "web3_provider_eth", "WEB3_PROVIDER_URI_ETH")
    _env_str(config, "web3_provider_bsc", "WEB3_PROVIDER_URI_BSC")
    _env_bool(config, "use_onchain", "TRADINGAGENTS_USE_ONCHAIN")

    # Execution overrides
    exec_cfg = config.setdefault("execution", {})
    _env_bool(exec_cfg, "live_mode", "LIVE_MODE")
    _env_bool(exec_cfg, "dry_run", "DRY_RUN")
    _env_float(exec_cfg, "max_position_pct", "MAX_POSITION_PCT")
    _env_float(exec_cfg, "stop_loss_pct", "STOP_LOSS_PCT")
    _env_float(exec_cfg, "max_daily_loss_pct", "MAX_DAILY_LOSS_PCT")
    _env_int(exec_cfg, "max_open_positions", "MAX_OPEN_POSITIONS")
    _env_str(exec_cfg, "min_confidence", "MIN_CONFIDENCE")
    _env_str(exec_cfg, "position_sizing", "POSITION_SIZING")
    _env_int(exec_cfg, "leverage", "LEVERAGE")

    return config


def _env_str(cfg: dict, key: str, env_var: str):
    val = os.getenv(env_var)
    if val is not None:
        cfg[key] = val


def _env_bool(cfg: dict, key: str, env_var: str):
    val = os.getenv(env_var)
    if val is not None:
        cfg[key] = val.lower() in ("true", "1", "yes")


def _env_float(cfg: dict, key: str, env_var: str):
    val = os.getenv(env_var)
    if val is not None:
        try:
            cfg[key] = float(val)
        except ValueError:
            pass


def _env_int(cfg: dict, key: str, env_var: str):
    val = os.getenv(env_var)
    if val is not None:
        try:
            cfg[key] = int(val)
        except ValueError:
            pass
