"""PIT on-chain summary for LLM analysts.

Reads the bitemporal onchain_store and produces a Markdown block that a
language model can reason over: current MVRV regime, Puell multiple,
exchange net-flow, active-address z-score, TVL snapshot. Enforces the
PIT rule via `as_of_ts <= trade_date`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import onchain_features, onchain_store


def _fmt(v: float | None, pct: bool = False, scale: int | None = None) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if scale is not None:
        v = v / scale
    if pct:
        return f"{v:+.2%}"
    if abs(v) >= 1e9:
        return f"{v/1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:,.2f}M"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3f}"


def _mvrv_regime(z: float | None) -> str:
    if z is None or pd.isna(z):
        return "unknown"
    if z >= 2.0:
        return "overheated (MVRV-Z ≥ 2) — historical cycle tops"
    if z >= 0.5:
        return "expensive — above 1y mean"
    if z >= -0.5:
        return "neutral — near 1y mean"
    if z >= -1.5:
        return "discounted — below 1y mean"
    return "deeply discounted (MVRV-Z ≤ -1.5) — historical accumulation zone"


def _puell_regime(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "unknown"
    if p >= 3.0:
        return "miner income > 3x trend — historical distribution signal"
    if p >= 1.2:
        return "above trend — miners selling into strength"
    if p >= 0.8:
        return "near trend"
    if p >= 0.5:
        return "below trend — miner squeeze possible"
    return "deep below-trend — historical bottom signal"


def _flow_regime(z: float | None, net: float | None) -> str:
    if z is None or pd.isna(z):
        return "unknown"
    direction = "inflow" if (net is not None and not pd.isna(net) and net > 0) else "outflow"
    if z >= 2:
        return f"extreme {direction} (30d-z ≥ 2) — distribution / selling pressure"
    if z >= 1:
        return f"elevated {direction}"
    if z <= -2:
        return f"extreme {direction} (30d-z ≤ -2) — accumulation / withdrawal"
    if z <= -1:
        return f"elevated {direction}"
    return f"{direction}, within normal range"


def build_pit_onchain_summary(
    coin: str,
    trade_date: str | datetime,
    lookback_days: int = 30,
    root: Path = onchain_store.DEFAULT_ROOT,
) -> str:
    """Return a Markdown summary of PIT on-chain signals for a coin.

    All values are masked by the PIT rule (`as_of_ts <= trade_date`). No
    look-ahead. Safe for use inside a backtest.
    """
    if isinstance(trade_date, str):
        td = datetime.strptime(trade_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        td = trade_date if trade_date.tzinfo else trade_date.replace(tzinfo=timezone.utc)

    start = td - timedelta(days=lookback_days)
    dates = pd.date_range(start=start, end=td, freq="D", tz="UTC")

    feats = onchain_features.build_pit_onchain_features(
        coin=coin, dates=dates, root=root,
    )
    if feats.empty or feats.dropna(how="all").empty:
        return (
            f"**On-chain PIT Summary — {coin.upper()} @ {td.date()}**\n\n"
            f"No PIT-safe on-chain data available for this coin/date "
            f"(coverage thin — CoinMetrics community tier does not publish {coin})."
        )

    last = feats.ffill().iloc[-1]

    def g(col):
        v = last.get(col)
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)

    price = g("oc_PriceUSD")
    mvrv = g("oc_CapMVRVCur")
    mvrv_z = g("oc_mvrv_z_1y")
    puell = g("oc_puell_multiple")
    net_flow = g("oc_net_flow_usd")
    flow_z = g("oc_net_flow_z_30d")
    addr_z = g("oc_active_addr_z_30d")
    addr = g("oc_AdrActCnt")
    hash_rate = g("oc_HashRate")
    tx_cnt = g("oc_TxCnt")
    tvl_eth = g("oc_tvl_ethereum")
    tvl_bsc = g("oc_tvl_bsc")
    tvl_eth_chg = g("oc_tvl_ethereum_chg_7d")
    tvl_bsc_chg = g("oc_tvl_bsc_chg_7d")
    sc_mcap = g("oc_stablecoin_mcap_total")
    sc_chg = g("oc_stablecoin_mcap_total_chg_7d")

    # 7-day change on active addresses for extra context
    addr_7d = feats["oc_AdrActCnt"].ffill().iloc[-1] if "oc_AdrActCnt" in feats else None
    addr_prior = feats["oc_AdrActCnt"].ffill().iloc[-8] if "oc_AdrActCnt" in feats and len(feats) >= 8 else None
    addr_chg = (addr_7d / addr_prior - 1) if (addr_7d and addr_prior and addr_prior > 0) else None

    lines: list[str] = []
    lines.append(f"### On-chain PIT Summary — {coin.upper()} @ {td.date()}")
    lines.append(f"All values enforce `as_of_ts <= {td.date()}` — no look-ahead.\n")

    lines.append("**Valuation Regime**")
    lines.append(
        f"- MVRV: {_fmt(mvrv)} | MVRV-Z (1y): {_fmt(mvrv_z)} → {_mvrv_regime(mvrv_z)}"
    )
    if puell is not None:
        lines.append(
            f"- Puell Multiple: {_fmt(puell)} → {_puell_regime(puell)}"
        )
    lines.append("")

    lines.append("**Exchange Flows (Binance + other CEXs, CM `Flow*ExUSD`)**")
    lines.append(
        f"- Net flow (in − out) 24h: {_fmt(net_flow)} USD | 30d z-score: {_fmt(flow_z)} → {_flow_regime(flow_z, net_flow)}"
    )
    lines.append("")

    lines.append("**Network Activity**")
    if addr is not None:
        lines.append(
            f"- Active addresses (24h): {_fmt(addr)} | 30d z-score: {_fmt(addr_z)}"
            + (f" | 7d change: {_fmt(addr_chg, pct=True)}" if addr_chg is not None else "")
        )
    if tx_cnt is not None:
        lines.append(f"- Transaction count (24h): {_fmt(tx_cnt)}")
    if hash_rate is not None:
        lines.append(f"- Hash rate (PoW only): {_fmt(hash_rate)} H/s")
    lines.append("")

    lines.append("**DeFi / Liquidity**")
    if tvl_eth is not None:
        lines.append(
            f"- TVL Ethereum: {_fmt(tvl_eth)} USD | 7d change: {_fmt(tvl_eth_chg, pct=True)}"
        )
    if tvl_bsc is not None:
        lines.append(
            f"- TVL BSC: {_fmt(tvl_bsc)} USD | 7d change: {_fmt(tvl_bsc_chg, pct=True)}"
        )
    if sc_mcap is not None:
        lines.append(
            f"- Stablecoin market cap (global): {_fmt(sc_mcap)} USD | 7d change: {_fmt(sc_chg, pct=True)}"
        )
    lines.append("")

    if price is not None:
        lines.append(f"**Reference price (CM PriceUSD):** {_fmt(price)} USD\n")

    lines.append(
        "**Interpretation cues:** MVRV-Z extremes flag cycle tops/bottoms. "
        "Persistent exchange outflows with elevated Puell signal accumulation by "
        "informed holders. Watch for divergence: flat price + heavy outflows often "
        "precede directional moves."
    )

    return "\n".join(lines)
