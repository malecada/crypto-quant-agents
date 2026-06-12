"""Re-derive the executed V5 base fraction for one coin.

Mirrors the runner's size+hold sequence (runner.py:373-464) so the hybrid
composes against the same V5 base the quant logic would produce — but using
the HYBRID journal's own prior HoldState, so the hybrid runs its own min-hold
discipline on its own book. Zero dependency on the quant runner.
"""
from __future__ import annotations

from tradingagents.execution.live import sizer
from tradingagents.execution.live.hold_sizer import HoldState, step_hold_state


def derive_base(
    *, coin: str, prediction: dict, price_history,
    prev_state: HoldState, cfg: dict, asof: str,
) -> tuple[float, HoldState, sizer.SizingResult | None]:
    """Return (held_fraction, new_state, sizing_result_or_None).

    Re-derives the executed V5 base by calling sizer.compute_size then
    hold_sizer.step_hold_state exactly as the live runner does
    (runner.py:385-464), but against the caller-supplied prior HoldState
    (the hybrid journal's own). Returns (0.0, prev_state, None) when the
    price history is too short to compute a valid vol estimate.

    held_fraction == base_target * sz.sma_multiplier (the runner's executed
    base, runner.py:464).

    Args:
        coin: coin identifier (e.g. "bitcoin").
        prediction: dict with keys ref_price, pred_h7, pred_h14 (and any
            extra horizons per cfg["horizons"]).
        price_history: DataFrame with lowercase columns "date" + "close".
        prev_state: prior HoldState from the hybrid journal.
        cfg: sizing/hold config dict; required keys: horizons, symmetric,
            target_vol, kelly_fraction, max_leverage, vol_lookback,
            vol_cap_pct, confidence_ref_return, trend_sma, trend_multiplier,
            min_hold, early_exit_loss.
        asof: ISO date string; bars after this date are excluded (P4 partial-
            bar guard, matching the runner).

    Returns:
        (held_fraction, new_state, sz) where sz is a SizingResult or None.
    """
    history = sizer.bars_through(price_history, asof)
    if len(history) < int(cfg["vol_lookback"]):
        return 0.0, prev_state, None

    sz = sizer.compute_size(
        coin=coin,
        prediction=prediction,
        price_history=history,
        horizons=cfg["horizons"],
        symmetric=cfg["symmetric"],
        target_vol=cfg["target_vol"],
        kelly_fraction=cfg["kelly_fraction"],
        max_leverage=cfg["max_leverage"],
        vol_lookback=cfg["vol_lookback"],
        vol_cap_pct=cfg["vol_cap_pct"],
        confidence_ref=cfg["confidence_ref_return"],
        trend_sma=cfg["trend_sma"],
        trend_multiplier=cfg["trend_multiplier"],
    )
    new_state, base_target = step_hold_state(
        prev_state,
        sig=sz.signal,
        vol_ok=sz.vol_ok,
        fresh_base=sz.leverage,  # pre-trend sized position (runner.py:450)
        price=prediction["ref_price"],
        min_hold=cfg["min_hold"],
        early_exit_loss=cfg["early_exit_loss"],
    )
    held_fraction = base_target * sz.sma_multiplier
    return float(held_fraction), new_state, sz
