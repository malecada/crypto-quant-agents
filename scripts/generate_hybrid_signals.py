#!/usr/bin/env python
"""Generate per-coin hybrid quant+LLM signals over a date range.

Drop-in cousin of ``scripts/generate_agent_signals.py`` for the
asset-agnostic hybrid graph. Captures the Layer 2 ``ModulatedPosition``
emitted by the new Modulator node:

  date, coin, regime, regime_confidence, hurst, quant_direction,
  quant_magnitude, llm_multiplier, llm_confidence, llm_uncertainty,
  effective_weight, position, unlock_flag, rolling_llm_edge, narrative

Usage:
    python scripts/generate_hybrid_signals.py \\
        --coins bitcoin ethereum \\
        --start 2026-01-16 --end 2026-04-15 \\
        --analysts market onchain crypto_sentiment prediction \\
        --output-dir data/hybrid_signals_p1
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument(
        "--analysts",
        nargs="+",
        default=["market", "onchain", "crypto_sentiment", "prediction"],
    )
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--deep-think", default="gpt-4o-mini")
    p.add_argument("--quick-think", default="gpt-4o-mini")
    p.add_argument("--output-dir", default="data/hybrid_signals_p1")
    p.add_argument("--anonymize", action="store_true",
                   help="Enable asset-name anonymization (Tier A4)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quant-version", choices=("v2", "v3"), default="v2",
                   help="Quant signal version. v3 requires per-coin regime + "
                        "multi-horizon bundles (pickles) and OHLCV prices "
                        "(parquet/CSV) to be present in --v3-state-dir.")
    p.add_argument(
        "--v3-state-dir",
        default="data/checkpoints",
        help="Directory containing V3 per-coin pickles: "
             "regime_hmm_v3_{coin}.pkl and v3_models_{coin}.pkl. "
             "Also searched for {coin}_ohlcv.parquet / prices.parquet for price series. "
             "Only used when --quant-version v3.",
    )
    p.add_argument(
        "--v3-micro-dir",
        default="data/microstructure",
        help="Directory for optional microstructure parquets ({coin}.parquet).",
    )
    p.add_argument(
        "--v3-deriv-dir",
        default="data/derivatives",
        help="Directory for optional derivatives parquets ({coin}.parquet).",
    )
    return p.parse_args()


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    """Return parquet as DataFrame, or empty DataFrame if the file is missing."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_required_pickle(path: Path):
    """Load a required pickle file; raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required V3 bundle missing: {path}. "
            "Run the V3 training pipeline first or pass --quant-version v2."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_prices_for_coin(coin: str, state_dir: Path) -> pd.Series:
    """Load close-price series for a coin from parquet or CSV.

    Tried in order:
      1. {state_dir}/{coin}_ohlcv.parquet  (multi-coin pipeline output)
      2. {state_dir}/prices.parquet        (single-file store keyed by coin)
      3. data/multi_2coins_v2/{coin}_predictions.parquet  (V2 side-effect)
    Raises FileNotFoundError if none found.
    """
    candidates = [
        state_dir / f"{coin}_ohlcv.parquet",
        state_dir / "prices.parquet",
        PROJECT_ROOT / "data" / "multi_2coins_v2" / f"{coin}_predictions.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "close" in df.columns:
            return df["close"]
        if coin in df.columns:
            return df[coin]
        # prices.parquet may have per-coin columns
        for col in df.columns:
            if coin.lower() in col.lower() or col.lower() in ("close", "price"):
                return df[col]
    raise FileNotFoundError(
        f"Could not find price series for coin={coin!r}. "
        f"Searched: {[str(c) for c in candidates]}. "
        "Provide OHLCV data in --v3-state-dir or use --quant-version v2."
    )


def _inject_v3_state_for_coins(
    coins: list[str],
    state_dir: str,
    micro_dir: str,
    deriv_dir: str,
    log: logging.Logger,
) -> None:
    """Load V3 bundles per coin and register them in the module-level state.

    Called once at startup when ``--quant-version v3`` is active.
    """
    from tradingagents.strategies.quant_signal_provider import set_v3_provider_state
    from tradingagents.strategies.v3.config import V3Config

    sd = Path(state_dir)
    md = Path(micro_dir)
    dd = Path(deriv_dir)

    config = V3Config()

    for coin in coins:
        log.info("V3: loading state for coin=%s", coin)

        regime_path = sd / f"regime_hmm_v3_{coin}.pkl"
        models_path = sd / f"v3_models_{coin}.pkl"

        regime_bundle = _load_required_pickle(regime_path)
        mh_bundle = _load_required_pickle(models_path)

        prices = _load_prices_for_coin(coin, sd)

        micro = _load_optional_parquet(md / f"{coin}.parquet")
        deriv = _load_optional_parquet(dd / f"{coin}.parquet")

        set_v3_provider_state(
            coin=coin,
            prices=prices,
            regime_bundle=regime_bundle,
            multi_horizon_bundle=mh_bundle,
            microstructure_features=micro,
            derivatives_features=deriv,
            config=config,
        )
        log.info("V3: registered state for coin=%s (prices len=%d)", coin, len(prices))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)
    t0 = time.time()

    from tradingagents.strategies.quant_signal_provider import set_active_quant_version
    set_active_quant_version(args.quant_version)

    # If V3 is requested, eagerly load per-coin bundles before any agent runs.
    if args.quant_version == "v3":
        log.info("V3 mode: loading per-coin state from %s", args.v3_state_dir)
        _inject_v3_state_for_coins(
            coins=args.coins,
            state_dir=args.v3_state_dir,
            micro_dir=args.v3_micro_dir,
            deriv_dir=args.v3_deriv_dir,
            log=log,
        )

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = args.llm_provider
    cfg["deep_think_llm"] = args.deep_think
    cfg["quick_think_llm"] = args.quick_think
    cfg["asset_class"] = "crypto"
    cfg["replay_cache"] = True
    cfg["anonymize_assets"] = bool(args.anonymize)

    print(f"\n{'=' * 60}")
    print(f"  Hybrid Signal Generation (Layer 1 + Modulator)")
    print(f"{'=' * 60}")
    print(f"  Coins      : {', '.join(args.coins)}")
    print(f"  Period     : {args.start} -> {args.end}")
    print(f"  Analysts   : {', '.join(args.analysts)}")
    print(f"  LLM        : {args.deep_think} / {args.quick_think}")
    print(f"  Anonymize  : {args.anonymize}")
    print(f"  Output     : {args.output_dir}")
    print()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ta = TradingAgentsGraph(
        selected_analysts=args.analysts, debug=False, config=cfg,
    )

    dates = pd.date_range(start=args.start, end=args.end, freq="D")

    for coin in args.coins:
        csv_path = out_dir / f"{coin}_{args.start}_{args.end}.csv"
        cached: list[dict] = []
        have: set[str] = set()
        if csv_path.exists() and not args.force:
            df_old = pd.read_csv(csv_path, parse_dates=["date"])
            cached = df_old.to_dict(orient="records")
            have = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in df_old["date"]}
            log.info(f"{coin}: resuming with {len(cached)} cached rows")

        rows = list(cached)
        for i, dt in enumerate(dates):
            ds = dt.strftime("%Y-%m-%d")
            if ds in have:
                continue
            try:
                final_state, mp, qs, narrative = ta.propagate_with_modulator(coin, ds)
            except Exception as exc:
                log.error(f"{coin} @ {ds}: {exc}")
                rows.append({"date": dt, "coin": coin, "error": str(exc)[:200]})
                continue
            row = {
                "date": dt,
                "coin": coin,
                "regime": (qs or {}).get("regime"),
                "regime_confidence": (qs or {}).get("regime_confidence"),
                "hurst": (qs or {}).get("hurst"),
                "quant_direction": (qs or {}).get("direction"),
                "quant_magnitude": (qs or {}).get("magnitude"),
                "llm_multiplier": (mp or {}).get("llm_multiplier"),
                "llm_confidence": (mp or {}).get("llm_confidence"),
                "llm_uncertainty": (mp or {}).get("llm_uncertainty"),
                "effective_weight": (mp or {}).get("effective_weight"),
                "position": (mp or {}).get("position"),
                "unlock_flag": (mp or {}).get("unlock_flag"),
                "rolling_llm_edge": (mp or {}).get("rolling_llm_edge"),
                "narrative": (narrative or "")[:500],
            }
            rows.append(row)
            tmp = csv_path.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(csv_path)
            if (i + 1) % 5 == 0:
                log.info(f"{coin}: {i+1}/{len(dates)} -> {csv_path}")

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        log.info(f"{coin}: saved {len(df)} rows to {csv_path}")

    print(f"\n  Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
