"""On-chain and derivatives data vendor for cryptocurrency analysis.

Three tiers of data:
  1. Web3.py  -- gas price, tx count, stablecoin supply (Ethereum + BSC)
  2. Binance Futures -- funding rate (public endpoints, no auth)
  3. DeFiLlama      -- TVL, stablecoin market cap (free, no auth)

Ported from Krypto-v0/src/scraping/onchain.py, adapted to TradingAgents'
vendor interface (functions return formatted strings).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from functools import reduce
from typing import Annotated

import pandas as pd
import requests

from .config import get_config

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_REQUEST_TIMEOUT = 15
_MAX_RETRIES = 3
_RETRY_DELAY = 2

_BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
_DEFILLAMA_BASE_URL = "https://api.llama.fi"
_DEFILLAMA_STABLECOINS_URL = "https://stablecoins.llama.fi"

# ERC-20 minimal ABI (only totalSupply)
_ERC20_TOTAL_SUPPLY_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    }
]

_CHAIN_CONFIG = {
    1: {  # Ethereum mainnet
        "name": "Ethereum",
        "usdt": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "usdc": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "decimals": 6,
    },
    56: {  # BSC mainnet
        "name": "BSC",
        "usdt": "0x55d398326f99059fF775485246999027B3197955",
        "usdc": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
        "decimals": 18,
    },
}


# ── Helpers ──────────────────────────────────────────────────────────


def _ts_to_date(ts_ms: int | float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _unix_to_date(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _request_with_retry(url: str, params: dict | None = None, label: str = "") -> requests.Response:
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAY * (attempt + 1)
                logger.warning(f"Retry {attempt + 1}/{_MAX_RETRIES} for {label or url[:50]}: {exc}")
                time.sleep(delay)
            else:
                raise


def _parse_date_range(start_date: str, end_date: str) -> tuple[date, date]:
    """Parse date strings to date objects."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    return start, end


def _resolve_futures_symbol(coingecko_id: str) -> str:
    """Map CoinGecko ID to Binance Futures symbol."""
    from .coingecko_binance import _resolve_binance_symbol
    sym = _resolve_binance_symbol(coingecko_id)
    return sym or "BTCUSDT"


# ── Tier 1: Web3 metrics ────────────────────────────────────────────


def _find_block_by_timestamp(w3, target_ts: int) -> int | None:
    """Binary-search for the block closest to target_ts (Unix seconds)."""
    try:
        latest = w3.eth.get_block("latest")
        lo, hi = 1, latest["number"]
        if target_ts > latest["timestamp"]:
            return latest["number"]
        for _ in range(50):
            if lo >= hi:
                break
            mid = (lo + hi) // 2
            mid_block = w3.eth.get_block(mid)
            if mid_block["timestamp"] < target_ts:
                lo = mid + 1
            else:
                hi = mid
        return lo
    except Exception:
        return None


def _scrape_web3_chain(provider_uri: str, past: date, today: date, prefix: str = "") -> pd.DataFrame:
    """Fetch gas price, tx count, stablecoin supply from a single EVM chain."""
    try:
        from web3 import Web3
    except ImportError:
        logger.warning("web3 not installed, skipping web3 metrics")
        return pd.DataFrame()

    w3 = None
    for attempt in range(_MAX_RETRIES):
        try:
            w3 = Web3(Web3.HTTPProvider(provider_uri, request_kwargs={"timeout": _REQUEST_TIMEOUT}))
            if w3.is_connected():
                break
            w3 = None
        except Exception:
            w3 = None
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_DELAY * (attempt + 1))

    if w3 is None or not w3.is_connected():
        logger.error(f"web3 provider unreachable after {_MAX_RETRIES} attempts")
        return pd.DataFrame()

    try:
        chain_id = w3.eth.chain_id
    except Exception:
        chain_id = 0
    chain_cfg = _CHAIN_CONFIG.get(chain_id)
    chain_name = chain_cfg["name"] if chain_cfg else f"chain {chain_id}"
    logger.info(f"web3 connected to {chain_name} (chain_id={chain_id})")

    usdt = usdc = None
    stablecoin_divisor = 1
    if chain_cfg:
        stablecoin_divisor = 10 ** chain_cfg["decimals"]
        usdt = w3.eth.contract(
            address=Web3.to_checksum_address(chain_cfg["usdt"]),
            abi=_ERC20_TOTAL_SUPPLY_ABI,
        )
        usdc = w3.eth.contract(
            address=Web3.to_checksum_address(chain_cfg["usdc"]),
            abi=_ERC20_TOTAL_SUPPLY_ABI,
        )

    n_days = (today - past).days + 1
    dates = [(past + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]

    def _fetch_day(d: str) -> dict | None:
        dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        target_ts = int(dt.timestamp())
        for attempt in range(3):
            try:
                block_num = _find_block_by_timestamp(w3, target_ts)
                if block_num is None:
                    return None
                block = w3.eth.get_block(block_num)
                gas_price = w3.eth.gas_price / 1e9  # gwei
                tx_count = len(block.get("transactions", []))
                row = {
                    "date": d,
                    f"{prefix}avg_gas_price": gas_price,
                    f"{prefix}daily_tx_count": tx_count,
                }
                if usdt is not None:
                    row[f"{prefix}usdt_supply"] = usdt.functions.totalSupply().call(
                        block_identifier=block_num
                    ) / stablecoin_divisor
                if usdc is not None:
                    row[f"{prefix}usdc_supply"] = usdc.functions.totalSupply().call(
                        block_identifier=block_num
                    ) / stablecoin_divisor
                return row
            except Exception as exc:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    logger.warning(f"web3 fetch failed for {d} on {chain_name}: {exc}")
        return None

    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = pool.map(_fetch_day, dates)
        rows = [r for r in results if r is not None]

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ── Tier 2: Binance Futures ──────────────────────────────────────────


def _scrape_funding_rates(past: date, today: date, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Fetch funding rate history from Binance Futures."""
    past_ms = int(datetime.combine(past, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
    today_ms = int(datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_funding = []
    start = past_ms
    while start < today_ms:
        resp = _request_with_retry(
            f"{_BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate",
            params={"symbol": symbol, "startTime": start, "limit": 1000},
            label=f"Binance funding {symbol}",
        )
        data = resp.json()
        if not data:
            break
        all_funding.extend(data)
        start = data[-1]["fundingTime"] + 1

    if not all_funding:
        return pd.DataFrame()

    df = pd.DataFrame(all_funding)
    df["date"] = df["fundingTime"].apply(_ts_to_date)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.groupby("date")["fundingRate"].mean().rename("funding_rate").to_frame()
    df.index.name = "date"
    return df


# ── Tier 3: DeFiLlama ───────────────────────────────────────────────


def _scrape_total_tvl(past: date, today: date) -> pd.DataFrame:
    """Fetch total DeFi TVL from DeFiLlama."""
    past_str = past.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    resp = _request_with_retry(f"{_DEFILLAMA_BASE_URL}/v2/historicalChainTvl", label="DeFiLlama total TVL")
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = df["date"].apply(_unix_to_date)
    df = df.set_index("date")[["tvl"]].rename(columns={"tvl": "total_tvl"})
    df = df.loc[(df.index >= past_str) & (df.index <= today_str)]
    return df


def _scrape_chain_tvl(chain: str, past: date, today: date) -> pd.DataFrame:
    """Fetch chain-specific TVL from DeFiLlama."""
    past_str = past.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    resp = _request_with_retry(
        f"{_DEFILLAMA_BASE_URL}/v2/historicalChainTvl/{chain}",
        label=f"DeFiLlama {chain} TVL",
    )
    data = resp.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = df["date"].apply(_unix_to_date)
    col_name = f"{chain.lower()}_tvl"
    df = df.set_index("date")[["tvl"]].rename(columns={"tvl": col_name})
    df = df.loc[(df.index >= past_str) & (df.index <= today_str)]
    return df


def _scrape_stablecoin_mcap() -> float:
    """Fetch current total stablecoin market cap from DeFiLlama."""
    resp = _request_with_retry(
        f"{_DEFILLAMA_STABLECOINS_URL}/stablecoins",
        label="DeFiLlama stablecoins",
    )
    data = resp.json()
    if data and "peggedAssets" in data:
        return sum(
            s.get("circulating", {}).get("peggedUSD", 0)
            for s in data["peggedAssets"]
        )
    return 0.0


def _scrape_stablecoin_mcap_history(past: date, today: date) -> pd.DataFrame:
    """Fetch historical total stablecoin market cap from DeFiLlama charts API."""
    past_str = past.strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    resp = _request_with_retry(
        f"{_DEFILLAMA_STABLECOINS_URL}/stablecoincharts/all",
        label="DeFiLlama stablecoin history",
    )
    data = resp.json()
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        ts = int(item["date"])
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        mcap = item.get("totalCirculatingUSD", {}).get("peggedUSD", 0)
        rows.append({"date": d, "stablecoin_mcap": mcap})

    df = pd.DataFrame(rows).set_index("date")
    df = df.loc[(df.index >= past_str) & (df.index <= today_str)]
    return df


# ── Public vendor interface functions ────────────────────────────────
# Each returns a formatted string for the LLM analyst to consume.


def get_funding_rates(
    symbol: Annotated[str, "CoinGecko ID (e.g., 'bitcoin') or Binance futures symbol (e.g., 'BTCUSDT')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch funding rate history from Binance Futures perpetual swaps.

    Funding rates indicate the cost of holding long/short positions:
    - Positive rate: longs pay shorts (bullish market)
    - Negative rate: shorts pay longs (bearish market)
    - Extreme rates (>0.1% or <-0.1%): signal overleveraged market
    """
    past, today = _parse_date_range(start_date, end_date)

    # Resolve to Binance futures symbol if CoinGecko ID given
    if not symbol.endswith("USDT"):
        symbol = _resolve_futures_symbol(symbol)

    try:
        df = _scrape_funding_rates(past, today, symbol)
        if df.empty:
            return f"No funding rate data available for {symbol} between {start_date} and {end_date}"

        header = f"# Funding Rate History for {symbol}\n"
        header += f"# Period: {start_date} to {end_date}\n"
        header += f"# Records: {len(df)}\n"
        header += "# Interpretation: positive = longs pay shorts (bullish bias), negative = shorts pay longs (bearish bias)\n\n"

        # Summary statistics
        avg = df["funding_rate"].mean()
        latest = df["funding_rate"].iloc[-1] if len(df) > 0 else 0
        header += f"Average funding rate: {avg:.6f}\n"
        header += f"Latest funding rate: {latest:.6f}\n\n"

        return header + df.to_csv()
    except Exception as e:
        return f"Error fetching funding rates for {symbol}: {e}"


def get_tvl_metrics(
    protocol: Annotated[str, "Protocol or chain name (e.g., 'Ethereum', 'total' for all chains)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Total Value Locked (TVL) from DeFiLlama.

    TVL measures the total capital deposited in DeFi protocols:
    - Rising TVL: growing confidence and capital inflow
    - Falling TVL: capital flight, potential risk-off environment
    """
    past, today = _parse_date_range(start_date, end_date)

    try:
        if protocol.lower() == "total":
            df = _scrape_total_tvl(past, today)
            label = "Total DeFi"
        else:
            df = _scrape_chain_tvl(protocol, past, today)
            label = protocol

        if df.empty:
            return f"No TVL data available for {label} between {start_date} and {end_date}"

        col = df.columns[0]
        latest = df[col].iloc[-1] if len(df) > 0 else 0
        earliest = df[col].iloc[0] if len(df) > 0 else 0
        change_pct = ((latest - earliest) / earliest * 100) if earliest > 0 else 0

        header = f"# TVL History for {label}\n"
        header += f"# Period: {start_date} to {end_date}\n"
        header += f"# Records: {len(df)}\n\n"
        header += f"Latest TVL: ${latest:,.0f}\n"
        header += f"Period change: {change_pct:+.1f}%\n\n"

        return header + df.to_csv()
    except Exception as e:
        return f"Error fetching TVL for {protocol}: {e}"


def get_stablecoin_metrics(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch stablecoin market cap history from DeFiLlama.

    Stablecoin market cap is a liquidity indicator:
    - Growing stablecoin supply: more dry powder available for crypto purchases
    - Shrinking supply: capital leaving the crypto ecosystem
    """
    past, today = _parse_date_range(start_date, end_date)

    try:
        df = _scrape_stablecoin_mcap_history(past, today)
        if df.empty:
            # Fallback to current snapshot
            mcap = _scrape_stablecoin_mcap()
            result = f"# Stablecoin Market Cap (Current Snapshot)\n"
            result += f"# Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            result += f"Total stablecoin market cap: ${mcap:,.0f}\n"
            return result

        latest = df["stablecoin_mcap"].iloc[-1]
        earliest = df["stablecoin_mcap"].iloc[0]
        change_pct = ((latest - earliest) / earliest * 100) if earliest > 0 else 0

        header = f"# Stablecoin Market Cap History\n"
        header += f"# Period: {start_date} to {end_date}\n"
        header += f"# Records: {len(df)}\n\n"
        header += f"Latest stablecoin mcap: ${latest:,.0f}\n"
        header += f"Period change: {change_pct:+.2f}%\n\n"

        return header + df.to_csv()
    except Exception as e:
        return f"Error fetching stablecoin metrics: {e}"


def get_gas_metrics(
    chain: Annotated[str, "Blockchain name: 'ethereum' or 'bsc'"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch gas prices and transaction counts from EVM chains via Web3.

    Gas metrics indicate network activity and demand:
    - High gas: heavy network usage, high demand for block space
    - Low gas: reduced activity
    - Rising tx count: growing adoption/usage
    """
    config = get_config()
    past, today = _parse_date_range(start_date, end_date)

    if chain.lower() == "ethereum":
        provider_uri = config.get("web3_provider_eth", "")
        prefix = ""
    elif chain.lower() == "bsc":
        provider_uri = config.get("web3_provider_bsc", "")
        prefix = "bsc_"
    else:
        return f"Unsupported chain: {chain}. Use 'ethereum' or 'bsc'."

    if not provider_uri:
        return (
            f"No Web3 provider configured for {chain}. "
            f"Set 'web3_provider_{chain.lower()[:3]}' in config or "
            f"WEB3_PROVIDER_URI_{chain.upper()[:3]} environment variable."
        )

    try:
        df = _scrape_web3_chain(provider_uri, past, today, prefix=prefix)
        if df.empty:
            return f"No gas data available for {chain} between {start_date} and {end_date}"

        gas_col = f"{prefix}avg_gas_price"
        tx_col = f"{prefix}daily_tx_count"

        header = f"# Gas Metrics for {chain.capitalize()}\n"
        header += f"# Period: {start_date} to {end_date}\n"
        header += f"# Records: {len(df)}\n\n"

        if gas_col in df.columns:
            header += f"Latest gas price: {df[gas_col].iloc[-1]:.2f} gwei\n"
            header += f"Average gas price: {df[gas_col].mean():.2f} gwei\n"
        if tx_col in df.columns:
            header += f"Latest daily tx count: {df[tx_col].iloc[-1]:,.0f}\n"

        header += "\n"
        return header + df.to_csv()
    except Exception as e:
        return f"Error fetching gas metrics for {chain}: {e}"


# ── Incremental fetchers for live data refresh ──────────────────────
# Thin wrappers that return canonical bitemporal long-format rows ready
# for onchain_store.upsert_rows(). Used by
# tradingagents.execution.live.data_refresh.

_DEFILLAMA_CHAIN_BY_COIN = {"btc": [], "eth": ["Ethereum"], "bnb": ["BSC"]}
_STABLE_LAG = timedelta(days=1)


def fetch_coinmetrics_incremental(coins: list[str], since: str) -> pd.DataFrame:
    """Fetch CoinMetrics rows with event_ts >= since (UTC, YYYY-MM-DD).

    Returns canonical long-format frame: event_ts, as_of_ts, coin, metric,
    value, source, status.
    """
    from . import coinmetrics as _cm

    # Drive metric selection from the central SUPPORTED catalog so new free
    # metrics added there are picked up automatically.
    cm_asset_map = {
        "btc": "btc", "eth": "eth", "bitcoin": "btc", "ethereum": "eth",
        "usdt": "usdt", "usdc": "usdc", "dai": "dai",
        "usdt_eth": "usdt_eth", "usdc_eth": "usdc_eth", "usdt_trx": "usdt_trx",
    }

    start_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc)

    frames: list[pd.DataFrame] = []
    for coin in coins:
        asset = cm_asset_map.get(coin.lower())
        if asset is None:
            logger.warning("CM incremental: %s not supported, skipping", coin)
            continue
        metrics = sorted(_cm.SUPPORTED.get(asset, frozenset()))
        if not metrics:
            logger.warning("CM incremental: no supported metrics for %s", asset)
            continue
        df = _cm.fetch_asset_metrics_df(asset, metrics, start_dt, end_dt)
        if df.empty:
            continue
        df["coin"] = asset
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_defillama_incremental(coins: list[str], since: str) -> pd.DataFrame:
    """Fetch DefiLlama TVL + stablecoin rows with event_ts >= since.

    Returns canonical long-format frame matching onchain_store SCHEMA_COLS.
    """
    from . import onchain_store as _store

    start_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.now(timezone.utc) + timedelta(days=1)

    rows: list[dict] = []
    for coin in coins:
        chains = _DEFILLAMA_CHAIN_BY_COIN.get(coin.lower(), [])
        for chain in chains:
            try:
                resp = _request_with_retry(
                    f"{_DEFILLAMA_BASE_URL}/v2/historicalChainTvl/{chain}",
                    label=f"DefiLlama {chain} TVL",
                )
                payload = resp.json()
            except Exception as exc:
                logger.warning("DefiLlama TVL %s fetch failed: %s", chain, exc)
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                ts = datetime.fromtimestamp(item["date"], tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                if not (start_dt <= ts < end_dt):
                    continue
                rows.append({
                    "event_ts": ts,
                    "as_of_ts": ts + _STABLE_LAG,
                    "coin": coin.lower(),
                    "metric": f"tvl_{chain.lower()}",
                    "value": float(item["tvl"]),
                    "source": "defillama",
                    "status": "final",
                })
            time.sleep(0.3)

    # Stablecoin market cap (global, written once)
    try:
        resp = _request_with_retry(
            f"{_DEFILLAMA_STABLECOINS_URL}/stablecoincharts/all",
            label="DefiLlama stablecoin history",
        )
        stable_payload = resp.json()
    except Exception as exc:
        logger.warning("DefiLlama stablecoin history fetch failed: %s", exc)
        stable_payload = None
    if isinstance(stable_payload, list):
        for item in stable_payload:
            try:
                ts = datetime.fromtimestamp(int(item["date"]), tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            except (KeyError, ValueError, TypeError):
                continue
            if not (start_dt <= ts < end_dt):
                continue
            peg = item.get("totalCirculatingUSD")
            if isinstance(peg, dict):
                peg = peg.get("peggedUSD")
            if peg is None:
                continue
            rows.append({
                "event_ts": ts,
                "as_of_ts": ts + _STABLE_LAG,
                "coin": "global",
                "metric": "stablecoin_mcap_total",
                "value": float(peg),
                "source": "defillama",
                "status": "final",
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows, columns=_store.SCHEMA_COLS)


def get_stablecoin_supply(
    chain: Annotated[str, "Blockchain name: 'ethereum' or 'bsc'"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch USDT and USDC supply on a specific chain via Web3 ERC-20 calls.

    On-chain stablecoin supply indicates capital availability:
    - Rising supply on a chain: capital inflow, bullish signal
    - Falling supply: capital outflow or bridge to other chains
    """
    config = get_config()
    past, today = _parse_date_range(start_date, end_date)

    if chain.lower() == "ethereum":
        provider_uri = config.get("web3_provider_eth", "")
        prefix = ""
    elif chain.lower() == "bsc":
        provider_uri = config.get("web3_provider_bsc", "")
        prefix = "bsc_"
    else:
        return f"Unsupported chain: {chain}. Use 'ethereum' or 'bsc'."

    if not provider_uri:
        return (
            f"No Web3 provider configured for {chain}. "
            f"Set 'web3_provider_{chain.lower()[:3]}' in config."
        )

    try:
        df = _scrape_web3_chain(provider_uri, past, today, prefix=prefix)
        if df.empty:
            return f"No stablecoin supply data for {chain} between {start_date} and {end_date}"

        # Filter to stablecoin supply columns only
        supply_cols = [c for c in df.columns if "supply" in c]
        if not supply_cols:
            return f"No stablecoin supply data available for {chain}"

        df_supply = df[supply_cols]

        header = f"# Stablecoin Supply on {chain.capitalize()}\n"
        header += f"# Period: {start_date} to {end_date}\n"
        header += f"# Records: {len(df_supply)}\n\n"

        for col in supply_cols:
            latest = df_supply[col].iloc[-1] if len(df_supply) > 0 else 0
            header += f"Latest {col}: ${latest:,.0f}\n"

        header += "\n"
        return header + df_supply.to_csv()
    except Exception as e:
        return f"Error fetching stablecoin supply for {chain}: {e}"
