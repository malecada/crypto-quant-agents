"""PIT-correct on-chain feature construction from the bitemporal store.

Reads raw metrics written by ``backfill_onchain.py`` and returns a wide,
date-indexed DataFrame whose every row at date t only contains values that
would have been visible to a caller at as_of_ts <= t. Achieved via
``pandas.merge_asof`` on as_of_ts.

Derived features (rolling) are computed on the PIT-aligned series, so no
look-ahead leaks.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from . import onchain_store

# Coin → CM asset / DefiLlama slug mapping for feature lookup.
COIN_ALIAS = {
    "bitcoin": "btc",
    "btc": "btc",
    "ethereum": "eth",
    "eth": "eth",
    "binancecoin": "bnb",
    "bnb": "bnb",
}

# Raw metrics pulled per coin (what we expect to exist in the store).
_CM_COMMON_RAW = [
    "AdrActCnt", "AdrBalCnt", "BlkCnt",
    "CapMVRVCur", "CapMrktCurUSD", "CapMrktEstUSD",
    "FeeTotNtv",
    "FlowInExNtv", "FlowInExUSD", "FlowOutExNtv", "FlowOutExUSD",
    "HashRate",
    "IssTotNtv", "IssTotUSD",
    "PriceUSD",
    "ROI1yr", "ROI30d",
    "SplyCur", "SplyExNtv", "SplyExUSD",
    "TxCnt", "TxTfrCnt",
    "volume_reported_spot_usd_1d",
]

RAW_METRICS_BY_COIN = {
    "btc": list(_CM_COMMON_RAW),
    "eth": list(_CM_COMMON_RAW) + ["tvl_ethereum"],
    "bnb": ["tvl_bsc"],
}

GLOBAL_METRICS = [
    "stablecoin_mcap_total",
    "stable_usdt_mcap", "stable_usdc_mcap", "stable_dai_mcap", "stable_usde_mcap",
    "tvl_arbitrum", "tvl_solana", "tvl_polygon", "tvl_base", "tvl_op-mainnet",
    "dex_vol_total_7d",
]

# Per-stablecoin assets in CM with SplyCur (multi-chain + per-chain rows).
STABLECOIN_ASSETS = ["usdt", "usdc", "dai", "usdt_eth", "usdc_eth", "usdt_trx"]


def _load_metric_series(
    coin: str, metric: str, root: Path,
) -> pd.DataFrame:
    """Load raw rows for (coin, metric) sorted by as_of_ts ascending.

    Returns empty DataFrame if nothing present.
    """
    glob = f"{root}/*/*.parquet"
    import duckdb
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(f"CREATE VIEW onchain AS SELECT * FROM read_parquet('{glob}')")
        except duckdb.IOException:
            return pd.DataFrame(columns=["event_ts", "as_of_ts", "value"])
        sql = """
        SELECT event_ts, as_of_ts, value
        FROM onchain
        WHERE coin = ? AND metric = ?
        ORDER BY as_of_ts ASC, event_ts ASC
        """
        return con.execute(sql, [coin.lower(), metric]).fetchdf()
    finally:
        con.close()


def _pit_align(
    dates: pd.DatetimeIndex, series: pd.DataFrame, col_name: str,
) -> pd.Series:
    """Align a (event_ts, as_of_ts, value) series to a DatetimeIndex of dates.

    Each output date t gets the value whose as_of_ts is the maximum
    as_of_ts <= t. If no such row exists, NaN.
    """
    if series.empty:
        return pd.Series(index=dates, dtype="float64", name=col_name)
    left = pd.DataFrame({"date": dates})
    # merge_asof needs sorted keys. as_of_ts is already ascending per loader.
    right = series[["as_of_ts", "value"]].copy()
    # Normalize both sides to datetime64[ns, UTC] — Parquet/DuckDB returns
    # microsecond precision which otherwise triggers merge_asof dtype errors.
    right["as_of_ts"] = pd.to_datetime(right["as_of_ts"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    left["date"] = pd.to_datetime(left["date"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    merged = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("as_of_ts"),
        left_on="date", right_on="as_of_ts",
        direction="backward",
    )
    out = pd.Series(merged["value"].values, index=merged["date"], name=col_name)
    out.index = dates
    return out


def build_pit_onchain_features(
    coin: str,
    dates: Iterable[datetime],
    metrics: Optional[Iterable[str]] = None,
    include_global: bool = True,
    include_derived: bool = True,
    include_stablecoin_context: bool = True,
    include_options: bool = True,
    include_derivatives: bool = True,
    options_dir: Optional[Path] = None,
    derivatives_dir: Optional[Path] = None,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a wide, date-indexed PIT on-chain feature frame for a coin.

    Each row at date t has only values with as_of_ts <= t (strict PIT).
    Rolling derived features (z-scores, Puell Multiple) are computed on
    the full PIT-aligned series so long windows can stabilize even when
    the caller requests only a short slice of dates.

    Path defaults honor the ``TRADINGAGENTS_DATA_ROOT`` env var at call
    time when the corresponding parameter is None, so a single Python
    process can switch sandboxes between calls (e.g. parity-replay).
    """
    _data_root = Path(os.environ.get("TRADINGAGENTS_DATA_ROOT", "data"))
    if options_dir is None:
        options_dir = _data_root / "options"
    if derivatives_dir is None:
        derivatives_dir = _data_root / "derivatives"
    if root is None:
        root = onchain_store.DEFAULT_ROOT
    alias = COIN_ALIAS.get(coin.lower(), coin.lower())
    if metrics is None:
        metric_list = list(RAW_METRICS_BY_COIN.get(alias, []))
    else:
        metric_list = list(metrics)

    idx = pd.DatetimeIndex(
        [pd.to_datetime(d, utc=True) for d in dates]
    ).sort_values()
    idx.name = "date"

    # Build the full PIT-aligned frame over union(stored event_ts, query dates)
    # so rolling derivations see all history, then reindex to requested dates.
    metric_series: dict[str, pd.DataFrame] = {}
    all_as_of: list[pd.Timestamp] = []
    for m in metric_list:
        s = _load_metric_series(alias, m, root)
        metric_series[f"oc_{m}"] = s
        if not s.empty:
            all_as_of.extend(pd.to_datetime(s["as_of_ts"], utc=True).tolist())
    if include_global:
        for gm in GLOBAL_METRICS:
            s = _load_metric_series("global", gm, root)
            metric_series[f"oc_{gm}"] = s
            if not s.empty:
                all_as_of.extend(pd.to_datetime(s["as_of_ts"], utc=True).tolist())
    if include_stablecoin_context:
        for stable in STABLECOIN_ASSETS:
            s = _load_metric_series(stable, "SplyCur", root)
            metric_series[f"oc_stable_{stable}_supply"] = s
            if not s.empty:
                all_as_of.extend(pd.to_datetime(s["as_of_ts"], utc=True).tolist())

    if all_as_of:
        as_of_idx = pd.DatetimeIndex(sorted(set(all_as_of)))
    else:
        as_of_idx = pd.DatetimeIndex([], tz="UTC")

    full_idx = idx.union(as_of_idx).sort_values()
    full_idx = full_idx.tz_convert("UTC") if full_idx.tz is not None else full_idx.tz_localize("UTC")
    full_idx = full_idx.astype("datetime64[ns, UTC]")
    full_idx.name = "date"

    wide = pd.DataFrame(index=full_idx)
    for col, series in metric_series.items():
        wide[col] = _pit_align(full_idx, series, col).astype(float)

    if include_derived:
        wide = _add_derived(wide, alias)

    # Append DVOL (options-implied vol index) — coin-specific.
    if include_options:
        dvol_path = Path(options_dir) / f"{alias}_dvol.parquet"
        if dvol_path.exists():
            dvol = pd.read_parquet(dvol_path)
            dvol.index = pd.to_datetime(dvol.index, utc=True)
            wide["oc_dvol_close"] = dvol["dvol_close"].reindex(full_idx, method="ffill").astype(float)
            wide["oc_dvol_chg_7d"] = wide["oc_dvol_close"].pct_change(7)

    # Append derivatives daily aggregates (funding, basis, OI, liq, L/S, taker)
    # coin-specific. Loads every column in the daily parquet under oc_* prefix.
    if include_derivatives:
        coin_to_sym = {"btc": "bitcoin", "eth": "ethereum", "bnb": "binancecoin"}
        deriv_coin = coin_to_sym.get(alias)
        if deriv_coin is not None:
            deriv_path = Path(derivatives_dir) / f"{deriv_coin}.parquet"
            if deriv_path.exists():
                d = pd.read_parquet(deriv_path)
                if d.index.tz is None:
                    d.index = pd.to_datetime(d.index, utc=True)
                else:
                    d.index = d.index.tz_convert("UTC")
                for col in d.columns:
                    if col.startswith("perp_price") or col.startswith("spot_price"):
                        # raw prices not informative as features
                        continue
                    wide[f"oc_{col}"] = d[col].reindex(full_idx, method="ffill").astype(float)
                if include_derived:
                    wide = _add_derivatives_derived(wide)

    # Return only the dates the caller asked for, but with all columns.
    return wide.reindex(idx)


def _add_derived(df: pd.DataFrame, alias: str) -> pd.DataFrame:
    """Attach rolling / composite features. All derivations use the
    PIT-aligned columns already in `df`, so no leakage introduced here."""
    out = df.copy()

    mvrv_col = "oc_CapMVRVCur"
    if mvrv_col in out.columns:
        roll_1y = out[mvrv_col].rolling(window=365, min_periods=60)
        out["oc_mvrv_z_1y"] = (out[mvrv_col] - roll_1y.mean()) / roll_1y.std()
        # 4-yr (Glassnode-style) MVRV Z-Score — full crypto cycle
        roll_4y = out[mvrv_col].rolling(window=1460, min_periods=180)
        out["oc_mvrv_z_4y"] = (out[mvrv_col] - roll_4y.mean()) / roll_4y.std()

    fi, fo = "oc_FlowInExUSD", "oc_FlowOutExUSD"
    if fi in out.columns and fo in out.columns:
        out["oc_net_flow_usd"] = out[fi] - out[fo]
        nf_roll = out["oc_net_flow_usd"].rolling(window=30, min_periods=5)
        out["oc_net_flow_z_30d"] = (
            (out["oc_net_flow_usd"] - nf_roll.mean()) / nf_roll.std()
        )

    fin_ntv, fout_ntv = "oc_FlowInExNtv", "oc_FlowOutExNtv"
    if fin_ntv in out.columns and fout_ntv in out.columns:
        out["oc_net_flow_ntv"] = out[fin_ntv] - out[fout_ntv]

    iss = "oc_IssTotUSD"
    if iss in out.columns:
        iss_ma = out[iss].rolling(window=365, min_periods=60).mean()
        out["oc_puell_multiple"] = out[iss] / iss_ma

    aa = "oc_AdrActCnt"
    if aa in out.columns:
        aa_roll = out[aa].rolling(window=30, min_periods=5)
        out["oc_active_addr_z_30d"] = (
            (out[aa] - aa_roll.mean()) / aa_roll.std()
        )

    # Exchange supply ratio (SplyExNtv / SplyCur) — classic reserve-on-exchange signal
    ex_ntv, sply = "oc_SplyExNtv", "oc_SplyCur"
    if ex_ntv in out.columns and sply in out.columns:
        denom = out[sply].replace(0.0, float("nan"))
        out["oc_ex_supply_ratio"] = out[ex_ntv] / denom
        out["oc_ex_supply_ratio_chg_30d"] = out["oc_ex_supply_ratio"].pct_change(30)

    # Holder growth (AdrBalCnt 30-day pct change) — long-term holder accumulation
    bal = "oc_AdrBalCnt"
    if bal in out.columns:
        out["oc_holder_growth_30d"] = out[bal].pct_change(30)

    # Transfer count momentum (TxTfrCnt 30-day pct change) — economic throughput
    tft = "oc_TxTfrCnt"
    if tft in out.columns:
        out["oc_tfr_cnt_chg_30d"] = out[tft].pct_change(30)

    # Spot volume z-score — turnover / liquidity proxy
    vol = "oc_volume_reported_spot_usd_1d"
    if vol in out.columns:
        v_roll = out[vol].rolling(window=30, min_periods=5)
        out["oc_spot_vol_z_30d"] = (out[vol] - v_roll.mean()) / v_roll.std()

    # Hash rate momentum (BTC only — ETH stops post-merge Sep 2022)
    hr = "oc_HashRate"
    if hr in out.columns:
        out["oc_hashrate_chg_30d"] = out[hr].pct_change(30)

    # TVL % change (DefiLlama)
    for tvl_col in (
        "oc_tvl_ethereum", "oc_tvl_bsc",
        "oc_tvl_arbitrum", "oc_tvl_solana", "oc_tvl_polygon",
        "oc_tvl_base", "oc_tvl_op-mainnet",
    ):
        if tvl_col in out.columns:
            out[f"{tvl_col}_chg_7d"] = out[tvl_col].pct_change(7)

    # Stablecoin mcap % change (aggregate + per-token)
    for sc in (
        "oc_stablecoin_mcap_total",
        "oc_stable_usdt_mcap", "oc_stable_usdc_mcap",
        "oc_stable_dai_mcap", "oc_stable_usde_mcap",
    ):
        if sc in out.columns:
            out[f"{sc}_chg_7d"] = out[sc].pct_change(7)

    # CM-derived stablecoin supply aggregates (SplyCur, per-chain)
    usdt = "oc_stable_usdt_supply"
    usdc = "oc_stable_usdc_supply"
    dai = "oc_stable_dai_supply"
    if usdt in out.columns and usdc in out.columns:
        total = out[usdt].fillna(0.0) + out[usdc].fillna(0.0)
        if dai in out.columns:
            total = total + out[dai].fillna(0.0)
        out["oc_stable_total_supply"] = total
        out["oc_stable_total_chg_7d"] = total.pct_change(7)
        out["oc_stable_total_chg_30d"] = total.pct_change(30)
        denom = total.replace(0.0, float("nan"))
        out["oc_usdt_dominance"] = out[usdt] / denom

    usdt_eth = "oc_stable_usdt_eth_supply"
    usdt_trx = "oc_stable_usdt_trx_supply"
    usdc_eth = "oc_stable_usdc_eth_supply"
    if usdt_eth in out.columns and usdt in out.columns:
        denom_t = out[usdt].replace(0.0, float("nan"))
        out["oc_usdt_eth_share"] = out[usdt_eth] / denom_t
        if usdt_trx in out.columns:
            out["oc_usdt_trx_share"] = out[usdt_trx] / denom_t
    if usdt_eth in out.columns and usdc_eth in out.columns:
        out["oc_stable_eth_chain_supply"] = out[usdt_eth].fillna(0.0) + out[usdc_eth].fillna(0.0)
        out["oc_stable_eth_chain_chg_7d"] = out["oc_stable_eth_chain_supply"].pct_change(7)

    # DEX volume momentum
    dex = "oc_dex_vol_total_7d"
    if dex in out.columns:
        out["oc_dex_vol_chg_30d"] = out[dex].pct_change(30)

    return out


def _add_derivatives_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived features over the Coinglass-augmented derivatives columns.

    Expects columns produced by ``scripts/fetch_coinglass_history.py`` after
    being prefixed with ``oc_`` by the loader. All transforms are causal.
    """
    import numpy as np
    out = df.copy()

    # OI momentum (cross-exchange aggregated)
    oi_close = "oc_oi_close"
    if oi_close in out.columns:
        log_oi = np.log(out[oi_close].replace(0.0, np.nan))
        out["oc_oi_chg_1d"] = log_oi.diff(1)
        out["oc_oi_chg_7d"] = log_oi.diff(7)
        oi_roll = out[oi_close].rolling(window=30, min_periods=5)
        out["oc_oi_z_30d"] = (out[oi_close] - oi_roll.mean()) / oi_roll.std()

    # OI / market cap (leverage proxy)
    if oi_close in out.columns and "oc_CapMrktCurUSD" in out.columns:
        out["oc_oi_to_mcap"] = out[oi_close] / out["oc_CapMrktCurUSD"].replace(0.0, np.nan)

    # Liquidation asymmetry z-score
    if "oc_liq_asym_24h" in out.columns:
        la_roll = out["oc_liq_asym_24h"].rolling(window=30, min_periods=5)
        out["oc_liq_asym_z_30d"] = (out["oc_liq_asym_24h"] - la_roll.mean()) / la_roll.std()
    if "oc_liq_total_usd" in out.columns:
        lt_roll = out["oc_liq_total_usd"].rolling(window=30, min_periods=5)
        out["oc_liq_total_z_30d"] = (out["oc_liq_total_usd"] - lt_roll.mean()) / lt_roll.std()

    # Smart-money divergence: top trader position ratio − global retail account ratio.
    # Positive = top traders more bullish than retail; classic contrarian-when-extreme signal.
    top_pos = "oc_ls_top_position_top_position_long_short_ratio"
    retail = "oc_ls_global_global_account_long_short_ratio"
    if top_pos in out.columns and retail in out.columns:
        out["oc_smart_money_diff"] = out[top_pos] - out[retail]
        smd_roll = out["oc_smart_money_diff"].rolling(window=30, min_periods=5)
        out["oc_smart_money_z_30d"] = (out["oc_smart_money_diff"] - smd_roll.mean()) / smd_roll.std()

    # Taker buy/sell asymmetry z-score (aggressive flow direction)
    if "oc_taker_asym" in out.columns:
        ta_roll = out["oc_taker_asym"].rolling(window=30, min_periods=5)
        out["oc_taker_asym_z_30d"] = (out["oc_taker_asym"] - ta_roll.mean()) / ta_roll.std()

    # Cross-exchange OI-weighted funding z-score (replaces Binance-only funding_z for breadth)
    if "oc_funding_oiw_close" in out.columns:
        fw_roll = out["oc_funding_oiw_close"].rolling(window=30, min_periods=5)
        out["oc_funding_oiw_z_30d"] = (out["oc_funding_oiw_close"] - fw_roll.mean()) / fw_roll.std()

    # Binance funding z-score (existing data, just adding the z here for symmetry)
    if "oc_funding_rate" in out.columns:
        f_roll = out["oc_funding_rate"].rolling(window=30, min_periods=5)
        out["oc_funding_z_30d"] = (out["oc_funding_rate"] - f_roll.mean()) / f_roll.std()

    # Basis z-score
    if "oc_basis_annual" in out.columns:
        b_roll = out["oc_basis_annual"].rolling(window=30, min_periods=5)
        out["oc_basis_z_30d"] = (out["oc_basis_annual"] - b_roll.mean()) / b_roll.std()

    return out
