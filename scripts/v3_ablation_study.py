#!/usr/bin/env python
"""V3 ablation study — runs the 88-bar window with components selectively disabled.

Ablations:
  - full       : full V3 baseline (control)
  - no_micro   : drop microstructure features
  - h7_h14     : restrict to h=7+h=14 (V2-like multi-horizon)
  - flat_regime: skip regime conditioning (use uniform horizon weights)
  - v2_sizing  : skip vol-target + CDAP (use fixed position scaled by direction × confidence)

Usage:
    python scripts/v3_ablation_study.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.backtesting.engine import run_backtest
from tradingagents.backtesting.strategies import FiveLevelSignal, SignalLevel
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv
from tradingagents.strategies.v3.backtest.runner_v3 import (
    _build_v3_features_at,
    _position_to_signal,
)
from tradingagents.strategies.v3.config import V3Config
from tradingagents.strategies.v3.contracts import RegimeState
from tradingagents.strategies.v3.models.multi_horizon import consensus_signal
from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3
from tradingagents.strategies.v3.sizing.vol_target import cdap_adjust, vol_target_position

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

START = pd.Timestamp("2026-01-16")
END = pd.Timestamp("2026-04-15")
COINS = ["bitcoin", "ethereum"]


def _load_prices(coin: str) -> tuple[pd.Series, pd.Series]:
    df = _load_crypto_ohlcv(coingecko_id=coin, curr_date=END.strftime("%Y-%m-%d"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df["close"], df["close"].pct_change().fillna(0.0)


def _flat_regime() -> RegimeState:
    """Trivial regime: sideways with neutral hurst — used by flat_regime ablation."""
    return RegimeState(
        label="sideways",
        confidence=0.5,
        hurst=0.5,
        changepoint_alert=False,
        posterior={"bull": 0.33, "sideways": 0.34, "bear": 0.33},
    )


def _extract_expected_features(mh_bundle) -> list[str]:
    """Extract expected feature names from fitted MultiHorizonEnsemble.

    Returns empty list if features are generic Column_N names or unavailable.
    """
    import re
    _GENERIC_COL = re.compile(r"^Column_\d+$")

    for _h, ph in mh_bundle._models.items():
        members = getattr(ph.ensemble, "_fitted_members", None)
        if not members:
            break
        first = next(iter(members.values()))
        fn = getattr(first, "feature_name_", None)
        if fn is not None and fn != "auto" and len(fn) > 0:
            if not all(_GENERIC_COL.match(str(f)) for f in fn):
                return list(fn)
        fn2 = getattr(first, "feature_names_in_", None)
        if fn2 is not None and len(fn2) > 0:
            if not all(_GENERIC_COL.match(str(f)) for f in fn2):
                return list(fn2)
        break
    return []


def run_ablation(
    coin: str,
    variant: str,
    prices: pd.Series,
    returns: pd.Series,
    micro: pd.DataFrame,
    deriv: pd.DataFrame,
    regime_bundle,
    mh_bundle,
    config: V3Config,
):
    bars = prices.loc[START:END].index
    agent_signals = []
    DEADBAND = 0.02

    expected_features = _extract_expected_features(mh_bundle)

    for as_of in bars:
        # Build features — with or without micro
        if variant == "no_micro":
            micro_arg = pd.DataFrame(index=micro.index)
        else:
            micro_arg = micro

        feat_df = _build_v3_features_at(prices, micro_arg, deriv, as_of)
        if feat_df.empty:
            agent_signals.append(SignalLevel.HOLD.value)
            continue

        # Align columns to training schema when we have explicit feature names
        if expected_features:
            for col in expected_features:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[expected_features]

        try:
            probas_dict = mh_bundle.predict_proba(feat_df)
        except Exception:
            logger.exception("predict_proba failed at %s; falling back to HOLD", as_of)
            agent_signals.append(SignalLevel.HOLD.value)
            continue

        # predict_proba returns dict[int, np.ndarray] — extract scalar per horizon
        scalar_probas: dict[int, float] = {
            h: float(arr[0]) for h, arr in probas_dict.items()
        }

        # h7_h14: keep only h=7 and h=14 (V2-like multi-horizon)
        if variant == "h7_h14":
            scalar_probas = {h: scalar_probas[h] for h in (7, 14) if h in scalar_probas}

        # Regime detection
        if variant == "flat_regime":
            regime = _flat_regime()
        else:
            try:
                regime = detect_regime_v3(prices=prices, bundle=regime_bundle, as_of=as_of)
            except Exception:
                logger.exception("detect_regime_v3 failed at %s; falling back to HOLD", as_of)
                agent_signals.append(SignalLevel.HOLD.value)
                continue

        direction, confidence = consensus_signal(
            scalar_probas, regime, config, deadband=DEADBAND
        )

        # Realized annualized vol from simple returns (21-bar rolling)
        sub_rets = returns.loc[returns.index <= as_of].iloc[-21:]
        rv = float(sub_rets.std() * np.sqrt(252)) if len(sub_rets) > 1 else 0.15

        if variant == "v2_sizing":
            # Fixed sizing: direction × confidence, clipped to [-max_leverage, +max_leverage]
            # (matches V2 approach: no vol-target scaling, no CDAP)
            position = float(direction) * float(confidence)
            position = max(-config.max_leverage, min(config.max_leverage, position))
        else:
            position = vol_target_position(
                direction=direction,
                confidence=confidence,
                realized_vol_annual=rv,
                target_vol_annual=config.target_annual_vol,
                max_leverage=config.max_leverage,
            )
            position = cdap_adjust(
                position=position,
                portfolio_dd_pct=0.0,
                regime=regime,
                config=config,
            )

        # _position_to_signal internally applies low_vol_scale=10.0
        agent_signals.append(_position_to_signal(position))

    # Safety: pad / truncate to exactly len(bars)
    while len(agent_signals) < len(bars):
        agent_signals.append(SignalLevel.HOLD.value)
    agent_signals = agent_signals[: len(bars)]

    actuals = prices.loc[bars].values
    dates = pd.Series(bars)
    result = run_backtest(
        dates=dates,
        actuals=actuals,
        agent_signals=agent_signals,
        strategy=FiveLevelSignal(),
        ticker=coin.upper(),
    )
    return result


def main() -> None:
    out_dir = Path("data/v3_ablations")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = V3Config()

    coin_data = {}
    for coin in COINS:
        logger.info("Loading %s", coin)
        prices, returns = _load_prices(coin)
        micro = pd.read_parquet(f"data/microstructure/{coin}.parquet")
        if micro.index.tz is not None:
            micro.index = micro.index.tz_localize(None)
        deriv_file = Path(f"data/derivatives/{coin}.parquet")
        if deriv_file.exists():
            deriv = pd.read_parquet(deriv_file)
            if deriv.index.tz is not None:
                deriv.index = deriv.index.tz_localize(None)
        else:
            deriv = pd.DataFrame()
        with open(f"data/checkpoints/regime_hmm_v3_{coin}.pkl", "rb") as f:
            regime_bundle = pickle.load(f)
        with open(f"data/checkpoints/v3_models_{coin}.pkl", "rb") as f:
            mh_bundle = pickle.load(f)
        coin_data[coin] = (prices, returns, micro, deriv, regime_bundle, mh_bundle)

    variants = ["full", "no_micro", "h7_h14", "flat_regime", "v2_sizing"]
    all_results: dict[str, dict] = {}

    for v in variants:
        logger.info("=" * 50)
        logger.info("Variant: %s", v)
        all_results[v] = {}
        for coin in COINS:
            prices, returns, micro, deriv, regime_bundle, mh_bundle = coin_data[coin]
            try:
                result = run_ablation(
                    coin, v, prices, returns, micro, deriv, regime_bundle, mh_bundle, cfg
                )
                m = result.metrics
                all_results[v][coin] = {
                    "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
                    "total_return": float(m.get("total_return", 0.0)),
                    "max_drawdown": float(m.get("max_drawdown", 0.0)),
                }
                logger.info(
                    "  %s: Sharpe=%.2f Return=%.2f%% MaxDD=%.2f%%",
                    coin,
                    m.get("sharpe_ratio", 0.0),
                    m.get("total_return", 0.0) * 100,
                    m.get("max_drawdown", 0.0) * 100,
                )
            except Exception:
                logger.exception("Failed %s/%s", v, coin)
                all_results[v][coin] = {
                    "sharpe_ratio": float("nan"),
                    "total_return": float("nan"),
                    "max_drawdown": float("nan"),
                }

    out_path = out_dir / "ablations_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Print summary table
    print("\n=== V3 Ablation Study (88-bar window: 2026-01-16 → 2026-04-15) ===")
    print(f"{'Variant':<14} {'BTC SR':>8} {'BTC Ret%':>10} {'BTC DD%':>9} {'ETH SR':>8} {'ETH Ret%':>10} {'ETH DD%':>9}")
    print("-" * 80)
    for v in variants:
        b = all_results[v].get("bitcoin", {})
        e = all_results[v].get("ethereum", {})
        print(
            f"{v:<14} "
            f"{b.get('sharpe_ratio', float('nan')):>8.2f} "
            f"{b.get('total_return', float('nan')):>9.2%} "
            f"{b.get('max_drawdown', float('nan')):>8.2%} "
            f"{e.get('sharpe_ratio', float('nan')):>8.2f} "
            f"{e.get('total_return', float('nan')):>9.2%} "
            f"{e.get('max_drawdown', float('nan')):>8.2%}"
        )

    # Highlight best ablation per coin
    print("\n--- Δ Sharpe vs 'full' baseline ---")
    full_b_sr = all_results.get("full", {}).get("bitcoin", {}).get("sharpe_ratio", 0.0)
    full_e_sr = all_results.get("full", {}).get("ethereum", {}).get("sharpe_ratio", 0.0)
    print(f"{'Variant':<14} {'ΔBTC SR':>9} {'ΔETH SR':>9}")
    print("-" * 36)
    for v in variants:
        if v == "full":
            continue
        b_sr = all_results[v].get("bitcoin", {}).get("sharpe_ratio", 0.0)
        e_sr = all_results[v].get("ethereum", {}).get("sharpe_ratio", 0.0)
        delta_b = b_sr - full_b_sr
        delta_e = e_sr - full_e_sr
        print(f"{v:<14} {delta_b:>+9.2f} {delta_e:>+9.2f}")


if __name__ == "__main__":
    main()
