"""Risk management: pre-trade checks, position sizing, and stop-loss computation.

Ported from Krypto-v0's ``src_live/risk.py`` and adapted for TradingAgents'
five-level signal system (BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL).

The four-tier pre-trade check:
  1. Confidence gate  (min confidence threshold)
  2. Daily loss limit (% of portfolio)
  3. Max open positions
  4. Position sizing  (fixed_fraction or kelly)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from tradingagents.dataflows.config import get_config
from tradingagents.execution.exchange import ExchangeClient
from tradingagents.execution.logger import TradeJournal

logger = logging.getLogger(__name__)

# Confidence levels -- ordinal mapping for the gate check.
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class RiskCheckResult:
    """Outcome of a pre-trade risk check."""

    approved: bool
    reason: str
    position_size: float = 0.0    # USDT value of the intended position
    quantity: float = 0.0         # asset quantity (after rounding)
    stop_loss_price: float = 0.0


# -- Signal mapping -----------------------------------------------------------
# TradingAgents produces: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL.
# We map these to a Binance *side* for order placement.

_SIGNAL_TO_SIDE: dict[str, str | None] = {
    "BUY": "BUY",
    "OVERWEIGHT": "BUY",
    "SELL": "SELL",
    "UNDERWEIGHT": "SELL",
    "HOLD": None,   # no execution
}


def signal_to_side(signal: str) -> str | None:
    """Map a TradingAgents signal to Binance order side (or ``None`` for HOLD)."""
    return _SIGNAL_TO_SIDE.get(signal.upper())


class RiskManager:
    """Enforces pre-trade risk limits and computes position sizing.

    All configuration is read from ``get_config()["execution"]``.
    """

    def __init__(
        self,
        exchange: ExchangeClient,
        journal: TradeJournal,
    ):
        self.exchange = exchange
        self.journal = journal

    # -- helpers to read config once per call ----------------------------------

    def _cfg(self) -> dict:
        return get_config().get("execution", {})

    # -- Main entry point ------------------------------------------------------

    def pre_trade_check(
        self,
        symbol: str,
        signal: str,
        current_price: float,
        confidence: str = "medium",
    ) -> RiskCheckResult:
        """Run all pre-trade risk checks.  Returns approval with position size.

        Parameters
        ----------
        symbol : str
            Binance symbol, e.g. ``"BTCUSDT"``.
        signal : str
            TradingAgents signal: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL.
        current_price : float
            Current market price of the asset.
        confidence : str
            ``"high"`` / ``"medium"`` / ``"low"``.

        Returns
        -------
        RiskCheckResult
            ``approved=True`` with sizing info, or ``approved=False`` with reason.
        """
        cfg = self._cfg()
        side = signal_to_side(signal)

        # HOLD signals need no execution
        if side is None:
            return RiskCheckResult(
                approved=False,
                reason=f"Signal is {signal.upper()} -- no execution needed",
            )

        # 1. Confidence gate
        min_conf = cfg.get("min_confidence", "medium")
        if not self._check_confidence_gate(confidence, min_conf):
            return RiskCheckResult(
                approved=False,
                reason=f"Confidence '{confidence}' below minimum '{min_conf}'",
            )

        # 2. Daily loss limit
        max_daily_loss_pct = cfg.get("max_daily_loss_pct", 0.05)
        within_limit, daily_pnl = self._check_daily_loss_limit(max_daily_loss_pct)
        if not within_limit:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss limit reached: {daily_pnl:.2f} USDT",
            )

        # 3. Max open positions (only for new entries, not for closing shorts)
        max_positions = cfg.get("max_open_positions", 3)
        open_count = self.exchange.get_open_position_count()
        if side == "BUY" and open_count >= max_positions:
            return RiskCheckResult(
                approved=False,
                reason=f"Max positions reached: {open_count}/{max_positions}",
            )

        # 4. Position sizing
        portfolio_value = self.exchange.get_total_portfolio_value()
        if portfolio_value <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Portfolio value is zero or negative",
            )

        size_usdt = self._compute_position_size(portfolio_value, cfg)
        quantity = size_usdt / current_price
        quantity = self.exchange.round_quantity(symbol, quantity)

        if quantity <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Computed quantity rounds to zero",
            )

        stop_loss = self._compute_stop_loss_price(
            current_price, side, cfg.get("stop_loss_pct", 0.03),
        )

        logger.info(
            "Risk check APPROVED: %s %s size=%.2f USDT qty=%.8f stop=%.2f",
            side, symbol, size_usdt, quantity, stop_loss,
        )
        return RiskCheckResult(
            approved=True,
            reason="All checks passed",
            position_size=size_usdt,
            quantity=quantity,
            stop_loss_price=stop_loss,
        )

    # -- Individual checks -----------------------------------------------------

    @staticmethod
    def _check_confidence_gate(confidence: str, min_confidence: str) -> bool:
        """True if *confidence* meets the minimum threshold."""
        min_level = _CONFIDENCE_ORDER.get(min_confidence.lower(), 1)
        actual_level = _CONFIDENCE_ORDER.get(confidence.lower(), 0)
        return actual_level >= min_level

    def _check_daily_loss_limit(
        self, max_daily_loss_pct: float,
    ) -> tuple[bool, float]:
        """Check whether today's P&L is within the daily loss limit.

        Uses the trade journal to compute today's realized P&L.
        """
        daily_pnl = self._get_daily_pnl()
        portfolio_value = self.exchange.get_usdt_balance()
        if portfolio_value <= 0:
            return True, daily_pnl

        max_loss = portfolio_value * max_daily_loss_pct
        within_limit = daily_pnl > -max_loss
        return within_limit, daily_pnl

    def _get_daily_pnl(self) -> float:
        """Compute today's realized P&L from the trade journal.

        Approximation: SELL inflows minus BUY outflows for EXECUTED trades.
        """
        today = date.today().isoformat()
        trades = self.journal.get_trades(start_date=today, end_date=today)
        pnl = 0.0
        for t in trades:
            if t.get("status") != "EXECUTED":
                continue
            value = t.get("quantity", 0) * t.get("price", 0)
            if t.get("side") == "SELL":
                pnl += value
            elif t.get("side") == "BUY":
                pnl -= value
        return pnl

    # -- Position sizing -------------------------------------------------------

    def _compute_position_size(self, portfolio_value: float, cfg: dict) -> float:
        """Compute position size in USDT.

        Modes:
        - ``fixed_fraction``: ``portfolio_value * max_position_pct``
        - ``kelly``: half-Kelly, capped at ``max_position_pct``
        """
        max_pct = cfg.get("max_position_pct", 0.02)
        sizing = cfg.get("position_sizing", "fixed_fraction")

        if sizing == "kelly":
            kelly_fraction = cfg.get("kelly_fraction", 0.5)
            metrics = self._get_performance_metrics(days=30)
            win_rate = metrics["win_rate"]
            ratio = metrics["avg_win_loss_ratio"]
            if ratio > 0:
                kelly_pct = win_rate - (1 - win_rate) / ratio
            else:
                kelly_pct = 0.0
            kelly_pct = max(0.0, kelly_pct) * kelly_fraction
            pct = min(kelly_pct, max_pct)
        else:
            # fixed_fraction (default)
            pct = max_pct

        return portfolio_value * pct

    def _get_performance_metrics(self, days: int = 30) -> dict:
        """Compute win_rate and avg_win_loss_ratio from recent trades.

        Used for Kelly criterion position sizing.
        """
        from datetime import timedelta
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=days)).isoformat()
        trades = self.journal.get_trades(start_date=start_date, end_date=end_date)

        if not trades:
            return {"win_rate": 0.5, "avg_win_loss_ratio": 1.0, "total_trades": 0}

        executed = [t for t in trades if t.get("status") == "EXECUTED"]
        sells = [t for t in executed if t.get("side") == "SELL"]
        buys = [t for t in executed if t.get("side") == "BUY"]

        if not sells or not buys:
            return {"win_rate": 0.5, "avg_win_loss_ratio": 1.0,
                    "total_trades": len(executed)}

        # Pair buys and sells chronologically for P&L per round-trip
        n_pairs = min(len(buys), len(sells))
        pnls = []
        for i in range(n_pairs):
            sell_value = sells[i].get("quantity", 0) * sells[i].get("price", 0)
            buy_value = buys[i].get("quantity", 0) * buys[i].get("price", 0)
            pnls.append(sell_value - buy_value)

        if not pnls:
            return {"win_rate": 0.5, "avg_win_loss_ratio": 1.0, "total_trades": 0}

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1.0
        ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        return {
            "win_rate": win_rate,
            "avg_win_loss_ratio": ratio,
            "total_trades": len(pnls),
        }

    # -- Stop-loss -------------------------------------------------------------

    @staticmethod
    def _compute_stop_loss_price(
        entry_price: float, side: str, stop_loss_pct: float,
    ) -> float:
        """Compute stop-loss price given entry and side."""
        if side == "BUY":
            return entry_price * (1 - stop_loss_pct)
        else:
            return entry_price * (1 + stop_loss_pct)
