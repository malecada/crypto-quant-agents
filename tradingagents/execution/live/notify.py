"""Telegram bot — daily summary + immediate alerts.

Outbound only; failures are logged, never raised (Telegram outage must not
abort a trading cycle).
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _post_telegram(*, token: str, chat_id: str, text: str):
    resp = requests.post(
        _TELEGRAM_API.format(token=token),
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )
    # AL1: a 4xx/5xx (bad token, chat not found, Markdown parse error) does NOT
    # raise on its own — without this the alert silently vanishes. Promote it to
    # an exception so the callers' except-blocks log it loudly.
    resp.raise_for_status()
    return resp


def send_daily_summary(*, bot_token, chat_id, cycle_id,
                        portfolio_before, portfolio_after, trades,
                        agreement_rate, peak_value=0.0,
                        initial_capital=0.0) -> None:
    pnl = portfolio_after - portfolio_before
    pnl_pct = pnl / portfolio_before if portfolio_before else 0
    # peak_value can be 0 on the very first cycle (no prior snapshot); treat
    # the current value as the peak in that case so the drawdown line reads 0%.
    peak = max(peak_value, portfolio_after)
    dd_from_peak = (portfolio_after - peak) / peak if peak else 0
    lines = [
        f"*Cycle {cycle_id}*",
        f"Portfolio: {portfolio_before:.2f} → {portfolio_after:.2f} ({pnl_pct:+.2%})",
        f"Peak: {peak:.2f}  DD-from-peak: {dd_from_peak:+.2%}",
    ]
    if initial_capital:
        cum_pnl_pct = (portfolio_after - initial_capital) / initial_capital
        lines.append(
            f"Cumulative vs initial ({initial_capital:.0f}): {cum_pnl_pct:+.2%}"
        )
    lines.append(f"Trades: {len(trades)}")
    lines.append(f"Shadow agreement: {agreement_rate:.1%}")
    for t in trades:
        lines.append(f"  {t['coin']} {t['side']} {t['qty']:.6f} @ {t['price']:.2f}")
    text = "\n".join(lines)
    try:
        _post_telegram(token=bot_token, chat_id=chat_id, text=text)
    except Exception as e:
        logger.error("Telegram delivery failed (non-fatal): %s", e)


def send_alert(*, bot_token, chat_id, severity: str, message: str) -> None:
    text = f"🚨 *{severity}*\n{message}"
    try:
        _post_telegram(token=bot_token, chat_id=chat_id, text=text)
    except Exception as e:
        logger.error("Telegram alert failed (non-fatal): %s", e)
