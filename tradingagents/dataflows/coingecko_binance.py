"""CoinGecko + Binance data vendor for cryptocurrency OHLCV and technical indicators.

Primary source: Binance public API (no rate-limit issues, no day cap).
Fallback: CoinGecko (chunked if > 365 days).

Ported from Krypto-v0/src/scraping/coingecko_binance.py, adapted to match
TradingAgents' vendor interface pattern.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Annotated

import numpy as np
import pandas as pd
import requests
from pycoingecko import CoinGeckoAPI
from stockstats import wrap

from .config import get_config
from .stockstats_utils import _clean_dataframe

logger = logging.getLogger(__name__)

_cg = CoinGeckoAPI()
_BINANCE_BASE_URL = "https://api.binance.com/api/v3"
_BINANCE_KLINE_LIMIT = 1000

# ── Symbol resolution cache ──────────────────────────────────────────

_symbol_cache: dict[str, str] = {}

# Well-known CoinGecko ID → Binance USDT pairs (avoids API call)
_KNOWN_SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "cardano": "ADAUSDT",
    "dogecoin": "DOGEUSDT",
    "tron": "TRXUSDT",
    "polkadot": "DOTUSDT",
    "avalanche-2": "AVAXUSDT",
    "chainlink": "LINKUSDT",
    "polygon": "MATICUSDT",
    "litecoin": "LTCUSDT",
    "uniswap": "UNIUSDT",
    "stellar": "XLMUSDT",
    "near": "NEARUSDT",
    "aptos": "APTUSDT",
    "sui": "SUIUSDT",
    "arbitrum": "ARBUSDT",
    "optimism": "OPUSDT",
    "aave": "AAVEUSDT",
}


def _resolve_binance_symbol(coingecko_id: str) -> str | None:
    """Resolve a CoinGecko ID to a Binance USDT symbol."""
    if coingecko_id in _symbol_cache:
        return _symbol_cache[coingecko_id]

    # Check well-known mapping first
    if coingecko_id in _KNOWN_SYMBOLS:
        sym = _KNOWN_SYMBOLS[coingecko_id]
        _symbol_cache[coingecko_id] = sym
        return sym

    # Check config overrides
    config = get_config()
    overrides = config.get("binance_symbol_map", {})
    if coingecko_id in overrides:
        sym = overrides[coingecko_id]
        _symbol_cache[coingecko_id] = sym
        return sym

    # Resolve via CoinGecko API
    try:
        coin_info = _cg.get_coin_by_id(
            coingecko_id,
            localization=False, tickers=False,
            market_data=False, community_data=False, developer_data=False
        )
        sym = coin_info["symbol"].upper() + "USDT"
        _symbol_cache[coingecko_id] = sym
        logger.info(f"Resolved Binance symbol: {coingecko_id} → {sym}")
        return sym
    except Exception as e:
        logger.warning(f"Could not resolve Binance symbol for {coingecko_id}: {e}")
        return None


# ── Binance helpers ──────────────────────────────────────────────────


def _binance_klines(symbol_usdt: str, from_ms: int, to_ms: int) -> list:
    """Fetch daily klines from Binance public API."""
    url = f"{_BINANCE_BASE_URL}/klines"
    params = {
        "symbol": symbol_usdt,
        "interval": "1d",
        "startTime": int(from_ms),
        "endTime": int(to_ms),
        "limit": _BINANCE_KLINE_LIMIT,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Binance klines error for {symbol_usdt}: {e}")
        return []


def _binance_klines_chunked(symbol_usdt: str, from_ms: int, to_ms: int) -> list:
    """Fetch daily klines from Binance, paginating if range > 1000 days."""
    all_klines = []
    cursor = int(from_ms)
    while cursor < int(to_ms):
        klines = _binance_klines(symbol_usdt, cursor, to_ms)
        if not klines:
            break
        all_klines.extend(klines)
        last_open_ms = klines[-1][0]
        cursor = last_open_ms + 86_400_000  # +1 day in ms
        if len(klines) < _BINANCE_KLINE_LIMIT:
            break
        time.sleep(0.5)
    return all_klines


def fetch_binance_daily(symbol: str, days: int = 2) -> pd.DataFrame:
    """Fetch the most recent `days` daily Binance OHLCV bars for `symbol`.

    Returns a DataFrame with columns date, open, high, low, close, volume.
    Empty DataFrame on error.
    """
    if not symbol:
        return pd.DataFrame()
    now_ms = int(time.time() * 1000)
    from_ms = now_ms - int(days) * 86_400_000
    klines = _binance_klines_chunked(symbol, from_ms, now_ms)
    if not klines:
        return pd.DataFrame()
    rows = [
        {
            "date": pd.to_datetime(k[0], unit="ms").strftime("%Y-%m-%d"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
        for k in klines
    ]
    return pd.DataFrame(rows)


# ── CoinGecko helpers (fallback) ─────────────────────────────────────

_CG_MAX_RANGE_DAYS = 365


def _fetch_cg_range(coin_id: str, dt_start: datetime, dt_end: datetime) -> dict:
    """Fetch CoinGecko market chart data for a single <= 365-day window."""
    lookback_days = (dt_end - dt_start).days
    if lookback_days <= 90:
        return _cg.get_coin_market_chart_by_id(
            id=coin_id, vs_currency="usd",
            days=lookback_days, interval="daily"
        )
    else:
        from_unix = int(time.mktime(dt_start.timetuple()))
        to_unix = int(time.mktime(dt_end.timetuple()))
        return _cg.get_coin_market_chart_range_by_id(
            id=coin_id, vs_currency="usd",
            from_timestamp=from_unix, to_timestamp=to_unix
        )


def _fetch_cg_chunked(coin_id: str, dt_start: datetime, dt_end: datetime) -> dict:
    """Fetch CoinGecko data in <= 365-day chunks, concatenating results."""
    all_prices, all_mcaps, all_vols = [], [], []
    chunk_start = dt_start
    while chunk_start < dt_end:
        chunk_end = min(chunk_start + timedelta(days=_CG_MAX_RANGE_DAYS), dt_end)
        data = _fetch_cg_range(coin_id, chunk_start, chunk_end)
        all_prices.extend(data.get("prices", []))
        all_mcaps.extend(data.get("market_caps", []))
        all_vols.extend(data.get("total_volumes", []))
        chunk_start = chunk_end
        if chunk_start < dt_end:
            time.sleep(2)

    # Deduplicate by timestamp
    seen = set()
    prices = [(ts, p) for ts, p in all_prices if ts not in seen and not seen.add(ts)]
    seen_mc = set()
    mcaps = [(ts, mc) for ts, mc in all_mcaps if ts not in seen_mc and not seen_mc.add(ts)]
    seen_vol = set()
    vols = [(ts, v) for ts, v in all_vols if ts not in seen_vol and not seen_vol.add(ts)]
    return {"prices": prices, "market_caps": mcaps, "total_volumes": vols}


# ── Session cache (in-memory, cleared between propagate() calls) ─────

_session_cache: dict[str, pd.DataFrame] = {}


def clear_session_cache():
    """Clear the in-memory session cache. Call between propagate() runs."""
    _session_cache.clear()


# ── OHLCV data loading with disk + session cache ─────────────────────


def _load_crypto_ohlcv(coingecko_id: str, curr_date: str) -> pd.DataFrame:
    """Fetch crypto OHLCV data with hybrid caching, filtered to prevent look-ahead bias.

    Layer 1 (session cache): In-memory dict keyed by (coingecko_id, curr_date).
    Prevents re-fetching within a single propagate() call when multiple analysts
    need the same OHLCV data.

    Layer 2 (disk cache): CSV per symbol. Persists across sessions.
    Prevents re-downloading 300+ days of history on every run.

    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume
    matching the format expected by stockstats.
    """
    cache_key = f"{coingecko_id}:{curr_date}"
    if cache_key in _session_cache:
        logger.debug(f"Session cache hit for {cache_key}")
        return _session_cache[cache_key].copy()

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    # Use curr_date (not today) as the upper fetch boundary so the disk
    # cache never contains data beyond the requested date.  This prevents
    # even the *cache file itself* from holding future prices — important
    # for backtesting auditability.
    today = pd.Timestamp.today()
    fetch_end = min(curr_date_dt, today)
    lookback_years = int(config.get("ohlcv_lookback_years", 7))
    start_date = fetch_end - pd.DateOffset(years=lookback_years)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = fetch_end.strftime("%Y-%m-%d")

    cache_dir = config.get("data_cache_dir", "dataflows/data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Canonical, date-independent cache filename. Append-only updates so
    # every cycle does NOT redownload the full history (previous behavior
    # tripped Binance -1003 IP-ban on cumulative klines weight).
    data_file = os.path.join(cache_dir, f"{coingecko_id}-crypto-ohlcv.csv")
    legacy_glob = f"{coingecko_id}-crypto-"

    data = pd.DataFrame()
    if os.path.exists(data_file):
        try:
            data = pd.read_csv(data_file, on_bad_lines="skip")
        except Exception as e:
            logger.warning(f"Cache read failed for {data_file}: {e}; refetching")
            data = pd.DataFrame()
    else:
        # One-time seed from legacy dated cache files (largest range wins).
        try:
            candidates = [
                f for f in os.listdir(cache_dir)
                if f.startswith(legacy_glob) and f.endswith(".csv")
            ]
            if candidates:
                candidates.sort(key=lambda f: os.path.getsize(os.path.join(cache_dir, f)),
                                reverse=True)
                seed = os.path.join(cache_dir, candidates[0])
                data = pd.read_csv(seed, on_bad_lines="skip")
                data.to_csv(data_file, index=False)
                logger.info(f"Seeded canonical cache {data_file} from {seed} "
                            f"({len(data)} rows)")
        except Exception as e:
            logger.debug(f"Legacy cache seed skipped: {e}")

    if not data.empty and "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        cache_last = data["Date"].max()
    else:
        cache_last = None

    need_fetch_start = start_date
    if cache_last is not None and cache_last >= fetch_end:
        # Cache already covers the requested end → no network call.
        logger.debug(f"OHLCV cache covers {coingecko_id} through {cache_last.date()}, "
                     f"skipping fetch")
    else:
        if cache_last is not None:
            # Incremental: only fetch bars strictly after the last cached date.
            need_fetch_start = cache_last + pd.Timedelta(days=1)
            if need_fetch_start > fetch_end:
                need_fetch_start = fetch_end  # no-op; loop below will exit

        dt_start = need_fetch_start.to_pydatetime() if hasattr(need_fetch_start, "to_pydatetime") else need_fetch_start
        dt_end = fetch_end.to_pydatetime()
        from_ms = int(time.mktime(dt_start.timetuple())) * 1000
        # `to_ms` is exclusive in Binance's chunked loop (`while cursor < to_ms`),
        # so push it past midnight of fetch_end to include that day's own bar.
        # The post-fetch `data["Date"] <= curr_date_dt` filter trims any overshoot.
        to_ms = int(time.mktime(dt_end.timetuple())) * 1000 + 86_400_000

        binance_symbol = _resolve_binance_symbol(coingecko_id)
        dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []

        if binance_symbol and from_ms < to_ms:
            klines = _binance_klines_chunked(binance_symbol, from_ms, to_ms)
            for k in klines:
                dates.append(pd.to_datetime(k[0], unit="ms"))
                opens.append(float(k[1]))
                highs.append(float(k[2]))
                lows.append(float(k[3]))
                closes.append(float(k[4]))
                volumes.append(float(k[5]))
            if dates:
                kind = "appended" if cache_last is not None else "fetched"
                logger.info(f"Binance: {kind} {len(dates)} daily candles for {binance_symbol}")

        if not dates and cache_last is None:
            logger.info(f"Binance unavailable, trying CoinGecko for {coingecko_id}...")
            try:
                lookback_days = (dt_end - dt_start).days
                if lookback_days <= _CG_MAX_RANGE_DAYS:
                    cg_data = _fetch_cg_range(coingecko_id, dt_start, dt_end)
                else:
                    cg_data = _fetch_cg_chunked(coingecko_id, dt_start, dt_end)

                for ts_ms, p in cg_data.get("prices", []):
                    dates.append(pd.to_datetime(ts_ms, unit="ms"))
                    closes.append(p)
                    opens.append(p)
                    highs.append(p)
                    lows.append(p)
                for ts_ms, vol in cg_data.get("total_volumes", []):
                    volumes.append(vol)
                while len(volumes) < len(dates):
                    volumes.append(0)
                logger.info(f"CoinGecko: fetched {len(dates)} daily prices for {coingecko_id}")
            except Exception as e:
                logger.error(f"CoinGecko historical also failed: {e}")

        if dates:
            new_rows = pd.DataFrame({
                "Date": dates, "Open": opens, "High": highs, "Low": lows,
                "Close": closes, "Volume": volumes,
            })
            if data.empty:
                data = new_rows
            else:
                data = pd.concat([data, new_rows], ignore_index=True)
                data = (data.drop_duplicates(subset="Date", keep="last")
                            .sort_values("Date").reset_index(drop=True))
            data.to_csv(data_file, index=False)
        elif data.empty:
            return pd.DataFrame()

    data = _clean_dataframe(data)

    # Filter to curr_date to prevent look-ahead bias
    data = data[data["Date"] <= curr_date_dt]

    # Store in session cache
    _session_cache[cache_key] = data.copy()

    return data


# ── Public vendor interface functions ────────────────────────────────


def get_crypto_data(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch OHLCV data for a cryptocurrency. Returns formatted CSV string.

    This is the crypto equivalent of get_YFin_data_online / get_stock_data.
    """
    data = _load_crypto_ohlcv(symbol, end_date)

    if data.empty:
        return f"No data found for cryptocurrency '{symbol}' between {start_date} and {end_date}"

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    data = data[(data["Date"] >= start_dt) & (data["Date"] <= end_dt)]

    if data.empty:
        return f"No data found for cryptocurrency '{symbol}' between {start_date} and {end_date}"

    # Round numerical values
    for col in ["Open", "High", "Low", "Close"]:
        if col in data.columns:
            data[col] = data[col].round(2)

    csv_string = data.to_csv(index=False)
    header = f"# Crypto OHLCV data for {symbol} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_crypto_indicators(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"],
    indicator: Annotated[str, "technical indicator to calculate (e.g., 'rsi', 'macd', 'boll')"],
    curr_date: Annotated[str, "Current trading date in YYYY-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back for indicator values"],
) -> str:
    """Compute technical indicators for a cryptocurrency using stockstats.

    Reuses TradingAgents' stockstats infrastructure on crypto OHLCV data.
    This is the crypto equivalent of get_stock_stats_indicators_window.
    """
    best_ind_params = {
        "close_50_sma": (
            "50 SMA: Medium-term trend indicator. "
            "Usage: Identify trend direction and dynamic support/resistance."
        ),
        "close_200_sma": (
            "200 SMA: Long-term trend benchmark. "
            "Usage: Confirm overall trend and identify golden/death cross."
        ),
        "close_10_ema": (
            "10 EMA: Responsive short-term average. "
            "Usage: Capture quick momentum shifts."
        ),
        "macd": (
            "MACD: Momentum via EMA differences. "
            "Usage: Look for crossovers and divergence."
        ),
        "macds": (
            "MACD Signal: EMA smoothing of MACD. "
            "Usage: Crossovers with MACD line trigger trades."
        ),
        "macdh": (
            "MACD Histogram: Gap between MACD and signal. "
            "Usage: Visualize momentum strength."
        ),
        "rsi": (
            "RSI: Momentum oscillator for overbought/oversold. "
            "Usage: 70/30 thresholds; in crypto, 80/20 may be more appropriate due to higher volatility."
        ),
        "boll": (
            "Bollinger Middle: 20 SMA basis for Bollinger Bands."
        ),
        "boll_ub": (
            "Bollinger Upper Band: 2 std dev above middle. "
            "Usage: Signals potential overbought and breakout zones."
        ),
        "boll_lb": (
            "Bollinger Lower Band: 2 std dev below middle. "
            "Usage: Indicates potential oversold conditions."
        ),
        "atr": (
            "ATR: Average True Range volatility measure. "
            "Usage: Set stop-loss levels and position sizes. Especially important in crypto due to high volatility."
        ),
        "vwma": (
            "VWMA: Volume-weighted moving average. "
            "Usage: Confirm trends with volume data."
        ),
        "mfi": (
            "MFI: Money Flow Index combining price and volume. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions."
        ),
    }

    if indicator not in best_ind_params:
        return (
            f"Indicator '{indicator}' is not supported. "
            f"Available: {', '.join(best_ind_params.keys())}"
        )

    data = _load_crypto_ohlcv(symbol, curr_date)
    if data.empty:
        return f"No data available for {symbol} to compute {indicator}"

    try:
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        # Trigger stockstats calculation
        df[indicator]

        # Build date→value mapping
        indicator_data = {}
        for _, row in df.iterrows():
            date_str = row["Date"]
            val = row[indicator]
            indicator_data[date_str] = "N/A" if pd.isna(val) else str(val)

        # Generate lookback range
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        before = curr_date_dt - timedelta(days=look_back_days)

        ind_string = ""
        current_dt = curr_date_dt
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            value = indicator_data.get(date_str, "N/A: No data for this date")
            ind_string += f"{date_str}: {value}\n"
            current_dt -= timedelta(days=1)

        result_str = (
            f"## {indicator} values for {symbol} from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + ind_string
            + "\n\n"
            + best_ind_params.get(indicator, "No description available.")
        )
        return result_str

    except Exception as e:
        return f"Error computing {indicator} for {symbol}: {e}"
