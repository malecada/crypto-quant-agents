"""PIT-enforced crypto sentiment tool implementations.

Registered as vendor 'crypto_sentiment_pit' in dataflows.interface.
When data_vendors['crypto_sentiment'] = 'crypto_sentiment_pit', agent tool
calls route here instead of the today-relative live implementations.

Phase 2 adds multi-source aggregation:
  * Alpaca News (Benzinga-curated, dense for BTC/ETH)
  * GDELT DOC 2.0 (global news, broad coverage, title-only)
  * HuggingFace edaschau/bitcoin_news (historical BTC corpus)
  * alternative.me Fear & Greed daily index
All sources PIT-enforced via bitemporal (event_ts, as_of_ts) columns.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd

from tradingagents.dataflows import fng_store, sentiment_store

# Source store roots. New stores land under data/sentiment/{source}/...
# Resolved at call time (not import) so tests can monkeypatch sentiment_store.DEFAULT_ROOT
# and fng_store.DEFAULT_ROOT, and the tool picks the patched values up.
GDELT_ROOT = Path("data/sentiment/gdelt")
HF_ROOT = Path("data/sentiment/hf_btc")


def get_crypto_news_pit(
    coin_name: Annotated[str, "Cryptocurrency name (e.g., 'Bitcoin', 'Ethereum')"],
    start_date: Annotated[str, "Start date yyyy-mm-dd (inclusive)"],
    end_date: Annotated[str, "End date yyyy-mm-dd (inclusive); acts as PIT cutoff"],
) -> str:
    """Fetch Alpaca News articles with strict PIT enforcement.

    Signature matches the live ``get_crypto_google_news`` tool so the vendor
    router can dispatch positionally. ``end_date`` is treated as end-of-day
    UTC (inclusive of its own 23:59:59.999999) and is also used as the
    ``as_of`` cutoff — no article with ``as_of_ts`` beyond end-of-day(end_date)
    is returned. This is consistent with the OHLCV ``Date <= curr_date``
    convention in ``coingecko_binance.py``.

    Returns raw headlines and article content for the LLM analyst to
    interpret sentiment. Every row satisfies as_of_ts <= end-of-day(end_date),
    so there is no look-ahead.
    """
    coin = coin_name.lower()
    if coin not in sentiment_store.COIN_TO_SYMBOL:
        return f"Sentiment store error: Unsupported coin for sentiment store: {coin_name!r}"
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return f"Invalid date format: start={start_date!r} end={end_date!r} (expected yyyy-mm-dd)."

    # End-of-day UTC: end_date is inclusive of its own 23:59:59.999999.
    ts_end = end_dt + timedelta(days=1) - timedelta(microseconds=1)
    ts_start = start_dt

    def _query(root: Path) -> pd.DataFrame:
        if not Path(root).exists():
            return pd.DataFrame(columns=sentiment_store.SCHEMA_COLS)
        try:
            return sentiment_store.query_news(
                coin=coin,
                ts_start=ts_start, ts_end=ts_end, as_of=ts_end,
                limit=50, root=root,
            )
        except ValueError as e:
            if "Unsupported coin" in str(e):
                raise
            return pd.DataFrame(columns=sentiment_store.SCHEMA_COLS)

    alpaca_df = _query(sentiment_store.DEFAULT_ROOT)
    gdelt_df = _query(GDELT_ROOT)
    hf_df = _query(HF_ROOT)

    fng_df = fng_store.query_fng(
        trade_date=ts_end, lookback_days=(end_dt - start_dt).days + 1,
        root=fng_store.DEFAULT_ROOT,
    )

    sections: list[str] = [
        f"# PIT Sentiment: {coin_name}",
        f"# Window: {start_date} → {end_date}",
        "",
    ]

    # Fear & Greed time series — cheap, always first
    if not fng_df.empty:
        latest = fng_df.iloc[-1]
        sections.append("## Fear & Greed Index (alternative.me)")
        sections.append(f"**Latest {latest['event_ts'].strftime('%Y-%m-%d')}:** "
                        f"{latest['value']} ({latest['classification']})")
        if len(fng_df) > 1:
            first = fng_df.iloc[0]
            delta = int(latest['value']) - int(first['value'])
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            sections.append(f"**Trend over window:** {first['value']} "
                            f"({first['classification']}) {direction} {latest['value']} "
                            f"({latest['classification']}) [Δ {delta:+d}]")
        sections.append("")

    def _render_articles(label: str, df: pd.DataFrame, show_body: bool = True) -> None:
        if df.empty:
            return
        sections.append(f"## {label} — {len(df)} articles")
        for i, row in enumerate(df.itertuples(index=False), 1):
            event_ts_str = pd.Timestamp(row.event_ts).strftime("%Y-%m-%d %H:%M UTC")
            sections.append(f"### {label} Article {i} — {row.source}")
            sections.append(f"**Date:** {event_ts_str}")
            sections.append(f"**Headline:** {row.headline}")
            if show_body:
                if row.summary:
                    sections.append(f"**Summary:** {row.summary}")
                elif row.content:
                    sections.append(f"**Content:** {row.content[:600]}")
            if row.url:
                sections.append(f"**URL:** {row.url}")
            sections.append("")

    _render_articles("Alpaca News (Benzinga)", alpaca_df, show_body=True)
    _render_articles("GDELT Global News", gdelt_df, show_body=False)  # GDELT has no body
    _render_articles("Historical Corpus (HF)", hf_df, show_body=True)

    if len(sections) <= 3:
        return (
            f"No sentiment signals found for {coin_name} in window "
            f"{start_date} → {end_date} across any source."
        )

    return "\n".join(sections)


def get_reddit_posts_pit_stub(
    coin_name: Annotated[str, "Cryptocurrency name"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """P1 stub: Reddit PIT data is not available (Phase 3).

    Returning an explicit message (instead of no impl) prevents the vendor
    router from silently falling back to the today-relative live Reddit tool.
    """
    return (
        f"Reddit PIT data is not available in P1 (no Arctic Shift/Pushshift ingest yet). "
        f"Sentiment analysis should rely on Alpaca News for {coin_name}."
    )
