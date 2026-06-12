from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.onchain_pit import build_pit_onchain_summary


@tool
def get_funding_rates(
    symbol: Annotated[str, "CoinGecko ID (e.g., 'bitcoin') or Binance futures symbol (e.g., 'BTCUSDT')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch funding rate history from Binance Futures perpetual swaps.

    Funding rates indicate the cost of holding long/short positions:
    - Positive rate: longs pay shorts (bullish market bias)
    - Negative rate: shorts pay longs (bearish market bias)
    - Extreme rates (>0.1% or <-0.1%): signal overleveraged market
    """
    return route_to_vendor("get_funding_rates", symbol, start_date, end_date)


@tool
def get_tvl_metrics(
    protocol: Annotated[str, "Protocol or chain name (e.g., 'Ethereum', 'total' for all chains)"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Total Value Locked (TVL) from DeFiLlama.

    TVL measures total capital deposited in DeFi protocols.
    Rising TVL indicates growing confidence; falling TVL signals capital flight.
    """
    return route_to_vendor("get_tvl_metrics", protocol, start_date, end_date)


@tool
def get_stablecoin_metrics(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch stablecoin market cap from DeFiLlama.

    Stablecoin supply is a liquidity indicator for the crypto ecosystem.
    Growing supply means more dry powder for purchases; shrinking means capital leaving.
    """
    return route_to_vendor("get_stablecoin_metrics", start_date, end_date)


@tool
def get_gas_metrics(
    chain: Annotated[str, "Blockchain name: 'ethereum' or 'bsc'"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch gas prices and transaction counts from EVM chains.

    Gas metrics indicate network activity and demand.
    High gas = heavy usage; rising tx count = growing adoption.
    """
    return route_to_vendor("get_gas_metrics", chain, start_date, end_date)


@tool
def get_stablecoin_supply(
    chain: Annotated[str, "Blockchain name: 'ethereum' or 'bsc'"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch USDT and USDC supply on a specific chain via Web3.

    On-chain stablecoin supply indicates capital availability on the chain.
    Rising supply = capital inflow; falling = outflow or bridging to other chains.
    """
    return route_to_vendor("get_stablecoin_supply", chain, start_date, end_date)


@tool
def get_onchain_pit(
    coin: Annotated[str, "CoinGecko ID (e.g., 'bitcoin', 'ethereum', 'binancecoin')"],
    trade_date: Annotated[str, "Decision date in yyyy-mm-dd. All returned "
                               "values respect the PIT rule (as_of_ts <= "
                               "trade_date) and are safe for backtests."],
    lookback_days: Annotated[int, "Window for 7d/30d derivations. "
                                  "Default 30."] = 30,
) -> str:
    """Return a PIT-safe on-chain summary for the given coin at trade_date.

    Data sources: CoinMetrics Community (MVRV, exchange flows, active
    addresses, hash rate, issuance) for BTC + ETH; DefiLlama TVL for
    BSC (BNB) and Ethereum. Bitemporal store enforces no look-ahead.

    Includes MVRV regime classification, Puell Multiple regime,
    exchange net-flow z-score, active-address z-score, and DeFi TVL
    snapshots. Coverage for BNB is thin (DefiLlama BSC TVL +
    stablecoin mcap only) — documented in output.
    """
    return build_pit_onchain_summary(coin, trade_date, lookback_days=lookback_days)
