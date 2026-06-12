"""Live trading runner: bridges TradingAgents graph output to Binance execution.

Ported from Krypto-v0's ``src_live/runner.py`` and adapted for TradingAgents'
multi-agent debate framework and five-level signal system.

Usage::

    from tradingagents.execution.runner import LiveRunner

    runner = LiveRunner()
    signal, result = runner.run_single("BTCUSDT")
    runner.run_multi(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Optional

from tradingagents.dataflows.config import get_config
from tradingagents.execution.exchange import ExchangeClient
from tradingagents.execution.logger import TradeJournal
from tradingagents.execution.risk import RiskManager, signal_to_side

logger = logging.getLogger(__name__)

_SANITY_THRESHOLD = 0.50  # skip if predicted price deviates >50% from current


class LiveRunner:
    """Bridges TradingAgents' graph output to Binance Futures execution.

    Lifecycle per symbol:
      1. Run ``ta.propagate(symbol, trade_date)`` to get signal
      2. Run risk checks
      3. Execute on Binance if approved
      4. Log to trade journal
    """

    def __init__(self, config: dict | None = None):
        if config is not None:
            from tradingagents.dataflows.config import set_config
            set_config(config)

        cfg = get_config()
        exec_cfg = cfg.get("execution", {})

        # -- Exchange client ---------------------------------------------------
        self.exchange = ExchangeClient()

        # -- Trade journal (SQLite) -------------------------------------------
        db_path = exec_cfg.get("trade_log_db")
        self.journal = TradeJournal(db_path)

        # -- Risk manager -----------------------------------------------------
        self.risk_mgr = RiskManager(self.exchange, self.journal)

        # -- TradingAgents graph (lazy-init on first use) ---------------------
        self._ta = None
        self._ta_config = cfg
        self._dry_run = exec_cfg.get("dry_run", False)

    # -- Lazy graph init -------------------------------------------------------

    def _get_graph(self):
        """Lazily build the TradingAgentsGraph so that import-time cost is
        deferred and the runner can be instantiated without GPU/LLM deps
        (useful for testing the execution layer alone).
        """
        if self._ta is None:
            from tradingagents.graph.trading_graph import TradingAgentsGraph

            analysts = self._ta_config.get(
                "selected_analysts",
                ["market", "social", "news", "fundamentals"],
            )
            self._ta = TradingAgentsGraph(
                selected_analysts=analysts,
                debug=False,
                config=self._ta_config,
            )
        return self._ta

    # -- Public API ------------------------------------------------------------

    def run_single(
        self,
        symbol: str,
        trade_date: str | None = None,
    ) -> tuple[str, dict]:
        """Run the full analysis-to-execution pipeline for one symbol.

        Parameters
        ----------
        symbol : str
            Binance futures symbol, e.g. ``"BTCUSDT"``.
        trade_date : str, optional
            ISO date (``"YYYY-MM-DD"``).  Defaults to today.

        Returns
        -------
        tuple[str, dict]
            ``(signal, execution_result)`` where *signal* is one of
            BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL and
            *execution_result* contains status and order details.
        """
        trade_date = trade_date or date.today().isoformat()
        exec_cfg = get_config().get("execution", {})

        logger.info("== %s  date=%s ==", symbol, trade_date)

        # -- Frequency guard: max once per day per symbol ---------------------
        today = date.today().isoformat()
        today_trades = self.journal.get_trades(
            symbol=symbol, start_date=today, end_date=today,
        )
        executed_today = [t for t in today_trades if t.get("status") == "EXECUTED"]
        if executed_today:
            logger.info(
                "Frequency guard: %s already executed today (%d time(s)) -- skipping",
                symbol, len(executed_today),
            )
            return "HOLD", {"status": "SKIPPED", "reason": "frequency_guard"}

        # -- Set leverage -----------------------------------------------------
        leverage = exec_cfg.get("leverage", 1)
        try:
            self.exchange.set_leverage(symbol, leverage)
        except Exception as e:
            logger.warning(
                "Could not set leverage for %s: %s -- using exchange default",
                symbol, e,
            )

        # -- Run TradingAgents graph ------------------------------------------
        try:
            ta = self._get_graph()
            final_state, signal = ta.propagate(symbol, trade_date)
        except Exception as e:
            logger.error("TradingAgents graph failed for %s: %s", symbol, e)
            self.journal.log_trade(
                symbol=symbol,
                side="HOLD",
                status="FAILED",
                risk_check_reason=f"Graph error: {e}",
            )
            return "HOLD", {"status": "FAILED", "reason": str(e)}

        signal = signal.strip().upper()
        logger.info("Signal for %s: %s", symbol, signal)

        # -- Extract confidence from final state (best-effort) ----------------
        confidence = self._extract_confidence(final_state)

        # -- Prediction sanity check ------------------------------------------
        current_price = self._safe_get_price(symbol)
        if current_price and not self._is_prediction_sane(
            symbol, final_state, current_price,
        ):
            self.journal.log_trade(
                symbol=symbol,
                side=signal,
                signal=signal,
                confidence=confidence,
                price=current_price,
                status="SKIPPED",
                risk_check_reason="Prediction sanity check failed",
            )
            return signal, {"status": "SKIPPED", "reason": "sanity_check"}

        # -- Log analyst reports alongside the trade --------------------------
        # We log reports for every run, even HOLDs, for thesis observability.

        # -- HOLD needs no execution ------------------------------------------
        side = signal_to_side(signal)
        if side is None:
            trade_id = self.journal.log_trade(
                symbol=symbol,
                side="HOLD",
                signal=signal,
                confidence=confidence,
                price=current_price or 0.0,
                status="HOLD",
                risk_check_reason="HOLD -- no trade needed",
            )
            self._safe_log_reports(trade_id, final_state)
            return signal, {"status": "HOLD"}

        # -- Risk checks ------------------------------------------------------
        if current_price is None or current_price <= 0:
            current_price = self.exchange.get_ticker_price(symbol)

        risk_result = self.risk_mgr.pre_trade_check(
            symbol, signal, current_price, confidence,
        )

        if not risk_result.approved:
            logger.warning("Risk REJECTED: %s", risk_result.reason)
            trade_id = self.journal.log_trade(
                symbol=symbol,
                side=side,
                signal=signal,
                confidence=confidence,
                price=current_price,
                status="REJECTED",
                risk_check_reason=risk_result.reason,
            )
            self._safe_log_reports(trade_id, final_state)
            return signal, {"status": "REJECTED", "reason": risk_result.reason}

        # -- Dry run -----------------------------------------------------------
        if self._dry_run:
            logger.info(
                "DRY RUN: would %s %.8f %s (%.2f USDT)",
                side, risk_result.quantity, symbol, risk_result.position_size,
            )
            trade_id = self.journal.log_trade(
                symbol=symbol,
                side=side,
                signal=signal,
                confidence=confidence,
                quantity=risk_result.quantity,
                price=current_price,
                stop_loss=risk_result.stop_loss_price,
                status="DRY_RUN",
                risk_check_reason=risk_result.reason,
            )
            self._safe_log_reports(trade_id, final_state)
            return signal, {"status": "DRY_RUN", "quantity": risk_result.quantity}

        # -- Execute on Binance ------------------------------------------------
        return self._execute_trade(
            symbol=symbol,
            side=side,
            signal=signal,
            confidence=confidence,
            current_price=current_price,
            risk_result=risk_result,
            final_state=final_state,
        )

    def run_multi(self, symbols: list[str]) -> list[tuple[str, dict]]:
        """Run the pipeline for multiple symbols sequentially.

        Includes the frequency guard (max once per day per symbol).

        Parameters
        ----------
        symbols : list[str]
            Binance futures symbols, e.g. ``["BTCUSDT", "ETHUSDT"]``.

        Returns
        -------
        list[tuple[str, dict]]
            One ``(signal, execution_result)`` per symbol.
        """
        results = []
        for i, symbol in enumerate(symbols, 1):
            logger.info("[%d/%d] %s", i, len(symbols), symbol)
            try:
                result = self.run_single(symbol)
                results.append(result)
            except Exception as e:
                logger.error("Failed on %s: %s", symbol, e)
                results.append(("HOLD", {"status": "FAILED", "reason": str(e)}))
        return results

    def close(self):
        """Clean up resources."""
        self.journal.close()

    # -- Private helpers -------------------------------------------------------

    def _execute_trade(
        self,
        symbol: str,
        side: str,
        signal: str,
        confidence: str,
        current_price: float,
        risk_result,
        final_state: dict,
    ) -> tuple[str, dict]:
        """Place market order + stop-loss, log results."""
        try:
            # Close any opposing position first
            self._close_opposing_position(symbol, side)

            order = self.exchange.place_market_order(
                symbol, side, risk_result.quantity,
            )
            order_id = str(order.get("orderId", ""))
            exec_price = float(order.get("avgPrice", current_price))
            logger.info(
                "Order executed: %s (id=%s, price=%.2f)",
                side, order_id, exec_price,
            )

            # Stop-loss: opposite side to the trade
            stop_side = "SELL" if side == "BUY" else "BUY"
            stop_status = "EXECUTED"
            try:
                stop_order = self.exchange.place_stop_loss(
                    symbol, risk_result.quantity,
                    risk_result.stop_loss_price, stop_side,
                )
                logger.info(
                    "Stop-loss at %.2f (id=%s)",
                    risk_result.stop_loss_price,
                    stop_order.get("orderId"),
                )
            except Exception as e:
                logger.error(
                    "CRITICAL: Failed to place stop-loss for %s after %s: %s "
                    "-- position UNPROTECTED",
                    symbol, side, e,
                )
                stop_status = "UNPROTECTED"

            trade_id = self.journal.log_trade(
                symbol=symbol,
                side=side,
                signal=signal,
                confidence=confidence,
                quantity=risk_result.quantity,
                price=exec_price,
                stop_loss=risk_result.stop_loss_price,
                order_id=order_id,
                status=stop_status,
                risk_check_reason=risk_result.reason,
            )
            self._safe_log_reports(trade_id, final_state)

            return signal, {
                "status": stop_status,
                "order_id": order_id,
                "exec_price": exec_price,
                "quantity": risk_result.quantity,
            }

        except Exception as e:
            logger.error("Order failed: %s", e)
            trade_id = self.journal.log_trade(
                symbol=symbol,
                side=side,
                signal=signal,
                confidence=confidence,
                quantity=risk_result.quantity,
                price=current_price,
                status="FAILED",
                risk_check_reason=str(e),
            )
            self._safe_log_reports(trade_id, final_state)
            return signal, {"status": "FAILED", "reason": str(e)}

    def _close_opposing_position(self, symbol: str, intended_side: str) -> None:
        """Close any existing position that opposes *intended_side*."""
        try:
            pos = self.exchange.get_current_position(symbol)
        except Exception as e:
            logger.warning(
                "Could not query position for %s: %s -- skipping close",
                symbol, e,
            )
            return

        if pos == 0:
            return

        # Close if position opposes intended direction
        is_long = pos > 0
        if (intended_side == "BUY" and not is_long) or (
            intended_side == "SELL" and is_long
        ):
            close_side = "SELL" if is_long else "BUY"
            qty = abs(pos)
            try:
                self.exchange.place_market_order(symbol, close_side, qty)
                logger.info(
                    "Closed existing %s position of %.8f %s",
                    "long" if is_long else "short", qty, symbol,
                )
            except Exception as e:
                logger.error(
                    "Failed to close position for %s: %s -- proceeding anyway",
                    symbol, e,
                )

    def _safe_get_price(self, symbol: str) -> float | None:
        """Try to get the current price from the exchange; return None on failure."""
        try:
            return self.exchange.get_ticker_price(symbol)
        except Exception as e:
            logger.warning("Could not fetch price for %s: %s", symbol, e)
            return None

    def _safe_log_reports(self, trade_id: int, final_state: dict) -> None:
        """Log analyst reports; swallow errors so they never block execution."""
        try:
            self.journal.log_analyst_reports(trade_id, final_state)
        except Exception as e:
            logger.warning("Could not log analyst reports for trade #%d: %s", trade_id, e)

    @staticmethod
    def _extract_confidence(final_state: dict) -> str:
        """Best-effort extraction of confidence from the graph's final state.

        The portfolio manager's ``final_trade_decision`` text often contains
        a confidence qualifier.  We look for common patterns; default to
        ``"medium"`` if nothing is found.
        """
        decision_text = final_state.get("final_trade_decision", "")
        if not isinstance(decision_text, str):
            return "medium"

        lower = decision_text.lower()
        # Look for explicit confidence markers
        if "high confidence" in lower or "confidence: high" in lower:
            return "high"
        if "low confidence" in lower or "confidence: low" in lower:
            return "low"
        # "Strong" buy/sell implies high confidence
        if "strong" in lower:
            return "high"
        return "medium"

    @staticmethod
    def _is_prediction_sane(
        symbol: str, final_state: dict, current_price: float,
    ) -> bool:
        """Check whether the prediction report contains values that deviate
        too far from the current price.

        Since TradingAgents uses a multi-agent debate rather than a single
        numeric prediction, we check the *prediction_report* field if it
        contains numeric forecasts.  If no numeric prediction is available,
        we skip the sanity check (pass by default).
        """
        prediction_report = final_state.get("prediction_report", "")
        if not prediction_report or not isinstance(prediction_report, str):
            return True  # no numeric prediction to check

        # Match patterns like "$65,000" or "65000.50" near "predict" or "forecast"
        price_pattern = re.compile(
            r"(?:predict|forecast|target|expected)[^\d]*"
            r"\$?([\d,]+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        matches = price_pattern.findall(prediction_report)

        for match in matches:
            try:
                pred_value = float(match.replace(",", ""))
                if current_price > 0:
                    deviation = abs(pred_value - current_price) / current_price
                    if deviation > _SANITY_THRESHOLD:
                        logger.warning(
                            "Sanity check FAILED for %s: prediction=%.2f "
                            "deviates %.1f%% from price=%.2f -- skipping",
                            symbol, pred_value, deviation * 100, current_price,
                        )
                        return False
            except ValueError:
                continue

        return True
