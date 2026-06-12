"""Daily incremental data refresh for the live trading cycle.

All three sources are append-only into Parquet stores keyed on
(metric, coin, valid_from). Re-running the same date is a no-op due to
dedupe keys in the on-chain store and a date-level deduplication on the
OHLCV cache.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.onchain import (
    fetch_coinmetrics_incremental,
    fetch_defillama_incremental,
)
from tradingagents.dataflows.coingecko_binance import fetch_binance_daily
from tradingagents.execution.live.config import to_binance_symbol

logger = logging.getLogger(__name__)


def upsert_onchain_rows(df: pd.DataFrame, root: Path) -> int:
    """Wrapper around the on-chain store upsert function.

    Defined on this module so tests can patch it directly. Delegates to
    ``tradingagents.dataflows.onchain_store.upsert_rows``.
    """
    from tradingagents.dataflows import onchain_store
    return onchain_store.upsert_rows(df, root=root)


def append_ohlcv(df: pd.DataFrame, coin: str, cache_root: Path) -> None:
    """Append rows to the per-coin OHLCV cache, deduping on the date column."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    out = cache_root / f"{coin}USDT_1d.parquet"
    if out.exists():
        existing = pd.read_parquet(out)
        merged = pd.concat([existing, df]).drop_duplicates(
            subset=["date"], keep="last"
        )
    else:
        merged = df
    merged.to_parquet(out, index=False)


def refresh_coinmetrics(coins: list[str], store_root: Path) -> None:
    df = fetch_coinmetrics_incremental(coins=coins, since=_yesterday_utc())
    if df.empty:
        logger.warning("CoinMetrics returned 0 rows")
        return
    n = upsert_onchain_rows(df, store_root)
    logger.info("CoinMetrics: upserted %d rows", n)


def refresh_defillama(coins: list[str], store_root: Path) -> None:
    df = fetch_defillama_incremental(coins=coins, since=_yesterday_utc())
    if df.empty:
        logger.warning("DefiLlama returned 0 rows")
        return
    n = upsert_onchain_rows(df, store_root)
    logger.info("DefiLlama: upserted %d rows", n)


def refresh_ohlcv(coin: str, cache_root: Path, min_history: int = 60) -> None:
    """Refresh OHLCV cache for ``coin`` (CoinGecko id or Binance base).

    Cold-start backfill: when the cache is missing or shorter than
    ``min_history`` rows, fetch ``min_history`` days; otherwise the cheap
    incremental 2-day fetch. The 60-day default ensures the first cycle
    after a fresh deploy has enough history for vol_lookback=20 and
    SMA30 computations.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    symbol = to_binance_symbol(coin)
    out = cache_root / f"{symbol}_1d.parquet"
    existing_rows = 0
    if out.exists():
        try:
            existing_rows = len(pd.read_parquet(out))
        except Exception:
            existing_rows = 0
    days = min_history if existing_rows < min_history else 2
    df = fetch_binance_daily(symbol=symbol, days=days)
    if df.empty:
        logger.warning("Binance OHLCV returned 0 rows for %s", symbol)
        return
    append_ohlcv(df, symbol.replace("USDT", ""), cache_root)
    logger.info(
        "OHLCV: appended %d rows for %s (cache had %d)",
        len(df), symbol, existing_rows,
    )


def _yesterday_utc() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


def refresh_coinglass(
    coins: list[str],
    derivatives_dir: Path,
    raw_dir: Path,
    api_key: str,
    structured_log: object | None,
) -> None:
    """Daily incremental refresh of Coinglass derivatives parquets.

    Wraps the §13 fetch helpers from ``scripts/fetch_coinglass_history.py``,
    appends new rows to ``{raw_dir}/{SYMBOL}_cg_*.parquet`` and merges
    everything into ``{derivatives_dir}/{coin}.parquet`` for V3/runner_v3 +
    V4-B PIT feature consumers.

    Idempotent: re-running over a date range already present is a no-op for
    the on-disk parquets.
    """
    if not api_key:
        raise RuntimeError("COINGLASS_API_KEY env var missing — required for V5 193f routes")

    # Late import to avoid pulling the heavy scripts package at module import time.
    from scripts.fetch_coinglass_history import (
        COIN_TO_SYMS, ENDPOINTS, fetch_oi_agg, fetch_liq_agg, fetch_ls_ratio,
        fetch_taker_vol, fetch_funding_weighted,
    )

    derivatives_dir = Path(derivatives_dir)
    raw_dir = Path(raw_dir)
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for coin in coins:
        if coin not in COIN_TO_SYMS:
            if structured_log is not None:
                structured_log.warn("coinglass_coin_unsupported", coin=coin)
            continue
        sym_base, pair = COIN_TO_SYMS[coin]

        # Fetch all 7 endpoints. Empty frames OK — leave the merge step to handle.
        frames = {
            "oi":              fetch_oi_agg(sym_base, api_key),
            "liq":             fetch_liq_agg(sym_base, api_key),
            "ls_global":       fetch_ls_ratio("ls_global", pair, api_key),
            "ls_top_position": fetch_ls_ratio("ls_top_position", pair, api_key),
            "ls_top_account":  fetch_ls_ratio("ls_top_account", pair, api_key),
            "taker":           fetch_taker_vol(pair, api_key),
            "funding_w":       fetch_funding_weighted(sym_base, api_key),
        }

        # Cache raw + merge into daily aggregate (matches fetch_coinglass_history.py logic).
        non_empty = []
        for name, df in frames.items():
            if df.empty:
                continue
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index).tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            raw_path = raw_dir / f"{pair}_cg_{name}.parquet"
            df.to_parquet(raw_path)  # full overwrite — idempotent
            non_empty.append(df)

        if not non_empty:
            continue
        merged_cg = pd.concat(non_empty, axis=1).sort_index()

        daily_file = derivatives_dir / f"{coin}.parquet"
        if daily_file.exists():
            existing = pd.read_parquet(daily_file)
            if existing.index.tz is None:
                existing.index = pd.to_datetime(existing.index).tz_localize("UTC")
            # Drop any pre-existing cg_* prefixed columns to avoid stale double-merge.
            existing = existing.loc[:, ~existing.columns.str.startswith(
                ("oi_", "liq_", "ls_", "taker_", "funding_oiw")
            )]
            out = existing.join(merged_cg, how="outer").sort_index()
        else:
            out = merged_cg
        out.to_parquet(daily_file)


def refresh_deribit_dvol(
    currencies: list[str],
    options_dir: Path,
    structured_log: object | None,
) -> None:
    """Daily incremental refresh of Deribit DVOL parquets.

    For each currency in ``currencies`` (e.g. ["BTC", "ETH"]) fetches yesterday's
    DVOL row from the Deribit public API and appends it to
    ``{options_dir}/{ccy_lower}_dvol.parquet``. Idempotent: existing rows
    are deduped on index.
    """
    from scripts.fetch_deribit_dvol import fetch_dvol
    import pandas as pd

    options_dir = Path(options_dir)
    options_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.utcnow().tz_convert("UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=3)  # 3-day window catches any small gaps

    for ccy in currencies:
        try:
            new_df = fetch_dvol(ccy, start, end)
        except Exception as exc:
            if structured_log is not None:
                structured_log.warn("dvol_fetch_failed", currency=ccy, err=str(exc))
            raise

        if new_df.empty:
            continue
        if new_df.index.tz is None:
            new_df.index = pd.to_datetime(new_df.index).tz_localize("UTC")

        out_file = options_dir / f"{ccy.lower()}_dvol.parquet"
        if out_file.exists():
            existing = pd.read_parquet(out_file)
            if existing.index.tz is None:
                existing.index = pd.to_datetime(existing.index).tz_localize("UTC")
            combined = pd.concat([existing, new_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df
        combined.to_parquet(out_file)


_PERP_URL = "https://fapi.binance.com/fapi/v1/klines"
_SPOT_URL = "https://api.binance.com/api/v3/klines"
_BASIS_SYM_TO_COIN = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum",
    "BNBUSDT": "binancecoin", "SOLUSDT": "solana",
    # 8-coin expansion satellites.
    "XRPUSDT": "ripple", "DOGEUSDT": "dogecoin",
    "ADAUSDT": "cardano", "TRXUSDT": "tron",
}


def refresh_perp_spot_basis(
    symbols: list[str],
    raw_dir: Path,
    daily_dir: Path,
    structured_log: object | None,
) -> None:
    """Daily incremental refresh of perp-spot basis.

    For each Binance symbol in ``symbols``, fetches yesterday's perp + spot
    daily klines, computes ``basis_annual = (perp_close - spot_close) /
    spot_close * 365``, appends to ``{raw_dir}/{SYMBOL}_basis.parquet``, and
    merges ``perp_price`` / ``spot_price`` / ``basis_annual`` columns into
    ``{daily_dir}/{coin}.parquet`` for downstream PIT feature builders.
    """
    from scripts.build_perp_spot_basis import fetch_klines
    import pandas as pd

    raw_dir = Path(raw_dir)
    daily_dir = Path(daily_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp.utcnow().tz_convert("UTC").normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=3)  # 3-day catch-up window

    for sym in symbols:
        coin = _BASIS_SYM_TO_COIN.get(sym)
        if coin is None:
            if structured_log is not None:
                structured_log.warn("basis_symbol_unsupported", symbol=sym)
            continue
        perp = fetch_klines(_PERP_URL, sym, start, end)
        spot = fetch_klines(_SPOT_URL, sym, start, end)
        if perp.empty or spot.empty:
            continue

        basis = pd.concat([
            perp["close"].rename("perp_price"),
            spot["close"].rename("spot_price"),
        ], axis=1).dropna()
        basis["basis_annual"] = (
            (basis["perp_price"] - basis["spot_price"]) / basis["spot_price"] * 365.0
        )
        if basis.index.tz is None:
            basis.index = pd.to_datetime(basis.index).tz_localize("UTC")

        raw_path = raw_dir / f"{sym}_basis.parquet"
        # Append to raw cache
        if raw_path.exists():
            existing = pd.read_parquet(raw_path)
            if existing.index.tz is None:
                existing.index = pd.to_datetime(existing.index).tz_localize("UTC")
            cached = pd.concat([existing, basis]).sort_index()
            cached = cached[~cached.index.duplicated(keep="last")]
        else:
            cached = basis
        cached.to_parquet(raw_path)

        # Merge into daily aggregate (overwrite cols if present)
        daily_file = daily_dir / f"{coin}.parquet"
        merge_cols = basis[["perp_price", "spot_price", "basis_annual"]]
        if daily_file.exists():
            d = pd.read_parquet(daily_file)
            if d.index.tz is None:
                d.index = pd.to_datetime(d.index).tz_localize("UTC")
            d = d.drop(columns=[c for c in ("perp_price", "spot_price", "basis_annual")
                                  if c in d.columns])
            out = d.join(merge_cols, how="outer").sort_index()
        else:
            out = merge_cols
        out.to_parquet(daily_file)


class CriticalDataRefreshError(RuntimeError):
    """Raised when any critical data source fails — cycle must abort."""

    def __init__(self, failures: list[tuple[str, Exception]]):
        self.failures = failures
        msg = "; ".join(f"{src}: {err}" for src, err in failures)
        super().__init__(f"critical data refresh failed: {msg}")


def refresh_all(cfg, structured_log) -> dict:
    """Run the 6 daily refreshers in two tiers.

    CRITICAL  : ohlcv, coinmetrics — failure raises CriticalDataRefreshError.
    SUPPLEMENTARY: defillama, coinglass, deribit_dvol, perp_spot_basis —
                    failure logs ``supplementary_data_stale`` warning,
                    cycle continues using last-good parquet.

    Returns ``{"critical_ok": True, "supplementary_failures": [(src, err), ...]}``
    on success. Raises on critical fail.
    """
    data_root = Path(cfg.data_root)
    store_root = data_root / "onchain"
    cache_root = data_root / "ohlcv_cache"
    deriv_dir = data_root / "derivatives"
    deriv_raw = data_root / "derivatives_raw"
    options_dir = data_root / "options"

    critical_failures: list[tuple[str, Exception]] = []
    # 1. OHLCV
    try:
        for coin in cfg.coin_universe:
            refresh_ohlcv(coin, cache_root=cache_root)
        if structured_log is not None:
            structured_log.info("refresh_ohlcv_ok", coins=cfg.coin_universe)
    except Exception as e:
        critical_failures.append(("ohlcv", e))

    # 2. CoinMetrics
    try:
        refresh_coinmetrics(cfg.coin_universe, store_root)
        if structured_log is not None:
            structured_log.info("refresh_coinmetrics_ok")
    except Exception as e:
        critical_failures.append(("coinmetrics", e))

    if critical_failures:
        raise CriticalDataRefreshError(critical_failures)

    # Supplementary
    supplementary_failures: list[tuple[str, Exception]] = []

    def _try(source: str, fn) -> None:
        try:
            fn()
        except Exception as e:
            supplementary_failures.append((source, e))
            if structured_log is not None:
                structured_log.warn("supplementary_data_stale", source=source, err=str(e))

    _try("defillama", lambda: refresh_defillama(cfg.coin_universe, store_root))
    _try("coinglass", lambda: refresh_coinglass(
        coins=cfg.coin_universe, derivatives_dir=deriv_dir, raw_dir=deriv_raw,
        api_key=cfg.coinglass_api_key, structured_log=structured_log,
    ))
    _try("deribit_dvol", lambda: refresh_deribit_dvol(
        currencies=["BTC", "ETH"], options_dir=options_dir, structured_log=structured_log,
    ))
    coin_to_sym = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT",
                   "binancecoin": "BNBUSDT", "solana": "SOLUSDT",
                   # 8-coin expansion satellites.
                   "ripple": "XRPUSDT", "dogecoin": "DOGEUSDT",
                   "cardano": "ADAUSDT", "tron": "TRXUSDT"}
    symbols = [coin_to_sym[c] for c in cfg.coin_universe if c in coin_to_sym]
    _try("perp_spot_basis", lambda: refresh_perp_spot_basis(
        symbols=symbols, raw_dir=deriv_raw, daily_dir=deriv_dir,
        structured_log=structured_log,
    ))

    return {"critical_ok": True, "supplementary_failures": supplementary_failures}
