"""Live trading configuration loaded from environment variables."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field


# CoinGecko id → Binance base symbol. The model code uses CoinGecko ids
# (`bitcoin`, `ethereum`, `binancecoin`); the exchange uses Binance bases
# (`BTC`, `ETH`, `BNB`) plus the `USDT` quote suffix.
_COIN_TO_BINANCE_BASE = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "binancecoin": "BNB",
    "solana": "SOL",
    # 8-coin expansion satellites (V5 MIX 8-coin, THESIS §20).
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "tron": "TRX",
}


# V5 MIX per-coin feature-set routing (validated in THESIS_FINDINGS §17/§20).
# BTC and BNB use the canonical 78-feature set; ETH and SOL use the extended
# 193-feature set. The `pool` lists the coins included in each coin's
# training universe (2+1 pattern for altcoins).
_V5_DEFAULT_ROUTING: dict[str, dict[str, object]] = {
    "bitcoin":     {"feature_set": "78f",  "pool": ["bitcoin", "ethereum"]},
    "ethereum":    {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]},
    "binancecoin": {"feature_set": "78f",  "pool": ["bitcoin", "ethereum", "binancecoin"]},
    "solana":      {"feature_set": "193f", "pool": ["bitcoin", "ethereum", "solana"]},
    # 8-coin satellites (feature sets from scripts/baseline_v5_mix.DEFAULT_ROUTING:
    # XRP/DOGE/TRX = 78f canonical, ADA = 193f extended; 2+1 pools).
    "ripple":      {"feature_set": "78f",  "pool": ["bitcoin", "ethereum", "ripple"]},
    "dogecoin":    {"feature_set": "78f",  "pool": ["bitcoin", "ethereum", "dogecoin"]},
    "cardano":     {"feature_set": "193f", "pool": ["bitcoin", "ethereum", "cardano"]},
    "tron":        {"feature_set": "78f",  "pool": ["bitcoin", "ethereum", "tron"]},
}


# V5 MIX core/satellite portfolio weights — canonical source is
# scripts/baseline_v5_mix.py:PORTFOLIO_WEIGHTS (the run that produced the
# published SR +3.18). Core coins 15% each, satellites 10% each. These are
# renormalized over the active universe by `compute_portfolio_weights`, so a
# 4-core-coin book becomes 25% equal-weight — matching baseline_v5_mix's
# `portfolio_return`, which does `w = w / w.sum()` over present columns.
# A parity test (tests/execution/live/test_portfolio_weights.py) asserts this
# stays equal to the backtest constant.
_V5_PORTFOLIO_WEIGHTS: dict[str, float] = {
    "bitcoin": 0.15, "ethereum": 0.15, "binancecoin": 0.15, "solana": 0.15,
    "ripple": 0.10, "dogecoin": 0.10, "cardano": 0.10, "tron": 0.10,
}


def compute_portfolio_weights(universe: list[str]) -> dict[str, float]:
    """Per-coin portfolio weights renormalized over the active universe.

    Mirrors `scripts.baseline_v5_mix.portfolio_return`: restrict the canonical
    core/satellite weights to the coins actually traded, then divide by their
    sum so the live book allocates exactly the same fraction of equity to each
    coin as the validated backtest combines its per-coin sleeves. Without this,
    converting each coin's full-equity size fraction to a quantity over-levers
    the shared-margin account by ~N (one full sleeve per coin).
    """
    present = {c: _V5_PORTFOLIO_WEIGHTS[c] for c in universe if c in _V5_PORTFOLIO_WEIGHTS}
    total = sum(present.values())
    if total <= 0:
        raise ValueError(
            f"no weighted coins in universe {universe!r} "
            f"(known: {sorted(_V5_PORTFOLIO_WEIGHTS)})"
        )
    return {c: w / total for c, w in present.items()}


def to_binance_symbol(coin_id: str) -> str:
    """Convert a CoinGecko coin id to its Binance Futures USDT-pair symbol.

    Falls back to upper-casing the id if the coin is not in the known map —
    callers passing already-base-cased symbols (e.g. `BTC`) get `BTCUSDT`.
    """
    base = _COIN_TO_BINANCE_BASE.get(coin_id.lower(), coin_id.upper())
    return f"{base}USDT"


_BINANCE_BASE_TO_COIN = {v: k for k, v in _COIN_TO_BINANCE_BASE.items()}


def from_binance_symbol(symbol: str) -> str:
    """Inverse of :func:`to_binance_symbol`: Binance USDT pair → coin id.

    Strips the ``USDT`` quote and reverses the known base map; unknown bases
    fall back to the lower-cased base, so the mapping round-trips for any id.
    """
    base = symbol.upper()
    if base.endswith("USDT"):
        base = base[:-4]
    return _BINANCE_BASE_TO_COIN.get(base, base.lower())


@dataclass(frozen=True)
class LiveConfig:
    live_mode: bool
    binance_api_key: str
    binance_api_secret: str
    binance_base_url: str
    coinmetrics_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str
    max_leverage: float
    max_daily_loss_pct: float
    max_portfolio_dd: float
    stop_loss_pct: float
    max_open_positions: int
    target_vol: float
    kelly_fraction: float
    vol_lookback: int
    vol_cap_pct: float
    confidence_ref_return: float
    early_exit_loss: float
    min_hold: int
    trend_sma: int
    trend_multiplier: float
    horizons: list[int]
    symmetric: bool
    arima_filter: bool
    initial_capital: float
    coin_universe: list[str]
    # V5 routing fields (Task 3 — V5 MIX live deployment)
    routing: dict[str, dict[str, object]] = field(default_factory=dict)
    # Per-coin portfolio weights renormalized over `coin_universe` (C1 fix).
    # The runner scales each coin's size fraction by its weight before
    # converting to a quantity, so the shared-margin book matches the
    # validated backtest's combined gross exposure instead of running ~N x.
    portfolio_weights: dict[str, float] = field(default_factory=dict)
    coinglass_api_key: str = ""
    data_refresh_critical: set[str] = field(default_factory=set)
    data_root: str = "data"
    signal_threshold: float = 0.0  # not used by V2 (kept for back-compat)
    # S3265: minimum equity below which the runner aborts the cycle instead of
    # sizing every coin to zero (which would flatten the whole book). Guards a
    # garbled exchange response or a genuinely drained account.
    min_capital_floor: float = 100.0

    @classmethod
    def from_env(cls) -> "LiveConfig":
        """Load `LiveConfig` from environment variables (V5-aware).

        Thin alias for `load_config()` — V5 callers (retrain, predict,
        parity_refetch_and_replay) use this name to signal they expect the
        V5 routing/coinglass/data_root fields to be populated.
        """
        return load_config()


def _required(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise ValueError(f"Required env var {name} is not set")
    return val


def _bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def load_config() -> LiveConfig:
    # Binance creds are the most fundamental — check first so a fresh
    # operator sees the missing-creds error before V5-specific failures.
    binance_api_key = _required("BINANCE_API_KEY")
    binance_api_secret = _required("BINANCE_API_SECRET")

    # COINGLASS_API_KEY is V5-specific (193f-routed coins depend on
    # Coinglass refresh). Raise ValueError to match `_required()`'s
    # contract so `except ValueError` handles all required-env errors.
    coinglass_api_key = os.environ.get("COINGLASS_API_KEY", "").strip()
    if not coinglass_api_key:
        raise ValueError(
            "COINGLASS_API_KEY env var required for V5 live deployment "
            "(193f-routed coins depend on Coinglass refresh)"
        )

    coin_universe = [c.strip() for c in os.environ.get(
        "COIN_UNIVERSE",
        "bitcoin,ethereum,binancecoin,solana,ripple,dogecoin,cardano,tron",
    ).split(",") if c.strip()]

    # P5: single data-root source of truth. The runner reads the OHLCV cache +
    # journal from $DATA_DIR, while data_refresh / retrain WRITE to data_root.
    # If those diverge the read-side cache silently goes stale (observed live:
    # frozen /opt/.../data/ohlcv_cache vs fresh repo/data). Fall back to DATA_DIR
    # when TRADINGAGENTS_DATA_ROOT is unset, and refuse to start if both are set
    # and disagree.
    _explicit_root = os.environ.get("TRADINGAGENTS_DATA_ROOT", "").strip()
    _data_dir = os.environ.get("DATA_DIR", "").strip()
    if _explicit_root and _data_dir and _explicit_root != _data_dir:
        raise ValueError(
            f"TRADINGAGENTS_DATA_ROOT ({_explicit_root!r}) != DATA_DIR ({_data_dir!r}) "
            f"— set them equal or unset one (runner reads DATA_DIR, refresh writes data_root)"
        )
    data_root = _explicit_root or _data_dir or "data"

    # Validate that every coin in the universe has a routing entry —
    # otherwise downstream predict.py KeyErrors mid-cycle.
    for c in coin_universe:
        if c not in _V5_DEFAULT_ROUTING:
            raise ValueError(
                f"coin '{c}' in COIN_UNIVERSE has no routing entry — "
                f"add to _V5_DEFAULT_ROUTING or remove from COIN_UNIVERSE"
            )

    cfg = LiveConfig(
        live_mode=_bool("LIVE_MODE", "false"),
        binance_api_key=binance_api_key,
        binance_api_secret=binance_api_secret,
        binance_base_url=os.environ.get("BINANCE_BASE_URL", "https://testnet.binancefuture.com"),
        coinmetrics_api_key=os.environ.get("COINMETRICS_API_KEY", ""),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        max_leverage=_float("MAX_LEVERAGE", 3.0),
        max_daily_loss_pct=_float("MAX_DAILY_LOSS_PCT", 0.15),
        max_portfolio_dd=_float("MAX_PORTFOLIO_DD", 0.15),
        stop_loss_pct=_float("STOP_LOSS_PCT", 0.03),
        max_open_positions=_int("MAX_OPEN_POSITIONS", 8),
        target_vol=_float("TARGET_VOL", 0.10),
        kelly_fraction=_float("KELLY_FRACTION", 0.25),
        vol_lookback=_int("VOL_LOOKBACK", 20),
        vol_cap_pct=_float("VOL_CAP_PCT", 0.95),
        # Canonical V5 MIX signal config (matches scripts/baseline_v5_mix.py
        # V5_CONFIDENCE_REF / V5_ASYMMETRIC — the published SR +3.18 run).
        confidence_ref_return=_float("CONFIDENCE_REF_RETURN", 0.05),
        early_exit_loss=_float("EARLY_EXIT_LOSS", 0.015),
        min_hold=_int("MIN_HOLD", 7),
        trend_sma=_int("TREND_SMA", 30),
        trend_multiplier=_float("TREND_MULTIPLIER", 1.5),
        horizons=[int(x) for x in os.environ.get("HORIZONS", "7,14").split(",") if x.strip()],
        symmetric=_bool("SYMMETRIC", "false"),
        arima_filter=_bool("ARIMA_FILTER", "false"),
        initial_capital=_float("INITIAL_CAPITAL", 10000.0),
        coin_universe=coin_universe,
        # Deep-copy: the module constant is shared mutable state.
        # `@dataclass(frozen=True)` only freezes attribute rebinding;
        # without deep-copy, mutations through `cfg.routing` would
        # persist across cycles and corrupt subsequent loads.
        routing=copy.deepcopy(_V5_DEFAULT_ROUTING),
        portfolio_weights=compute_portfolio_weights(coin_universe),
        coinglass_api_key=coinglass_api_key,
        data_refresh_critical={"ohlcv", "coinmetrics"},
        data_root=data_root,
        min_capital_floor=_float("MIN_CAPITAL_FLOOR", 100.0),
    )
    if cfg.max_leverage <= 0:
        raise ValueError(f"MAX_LEVERAGE must be > 0, got {cfg.max_leverage}")
    if cfg.max_daily_loss_pct <= 0 or cfg.max_daily_loss_pct >= 1:
        raise ValueError(f"MAX_DAILY_LOSS_PCT must be in (0, 1), got {cfg.max_daily_loss_pct}")
    return cfg
