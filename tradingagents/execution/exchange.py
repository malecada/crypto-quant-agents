"""Binance Futures (USDT-M) client wrapper with testnet default and retry logic.

Ported from Krypto-v0's ``src_live/exchange.py`` and adapted for TradingAgents.
Reads configuration from ``get_config().get("execution", {})``, and API
credentials from environment variables ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET``.
"""

from __future__ import annotations

import math
import os
import re
import time
import uuid
import logging
from typing import Optional

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_S = 2  # exponential backoff base

# Errors that leave order execution status UNKNOWN (the order may or may not
# have reached the matching engine): a -1007 backend timeout, or a transport
# failure after the request was sent. Both must reconcile, never blind-retry.
_UNKNOWN_EXEC_NETWORK_ERRORS = (
    BinanceRequestException,
    requests.exceptions.RequestException,  # ConnectionError, ReadTimeout, ...
)

# Match "banned until <ms-timestamp>" anywhere in the -1003 message body.
_BAN_UNTIL_RE = re.compile(r"banned until (\d{10,16})")


class BinanceIPBan(Exception):
    """Raised when Binance returns -1003 (IP banned). Carries ban-expiry epoch (ms).

    The retry loop does NOT retry on this — repeated requests during a ban
    extend the cooldown. Callers (e.g. runner.run_cycle) catch this, alert,
    and exit cleanly so systemd does not enter a restart loop.
    """

    def __init__(self, until_ms: int, raw_message: str = ""):
        self.until_ms = int(until_ms)
        self.raw_message = raw_message
        self.seconds_remaining = max(0.0, self.until_ms / 1000.0 - time.time())
        super().__init__(
            f"Binance IP banned until {self.until_ms} "
            f"(~{self.seconds_remaining:.0f}s remaining): {raw_message}"
        )


class BinanceOrderTimeoutUnknown(Exception):
    """Raised on -1007 when reconciliation can't determine if order landed.

    Binance -1007 "Timeout waiting for response from backend server" carries
    EXPLICITLY unknown execution status — the order may or may not have been
    placed. Blind retry risks double-fill. After this exception is raised,
    a human MUST verify position state before the next cycle places more
    orders for the same symbol; the runner emits a RECONCILE_NEEDED alert.

    `state` ∈ {"unknown", "open_order_canceled"}:
      - "unknown": reconciliation found neither fill nor open order, but
        repeated -1007 prevents safe retry.
      - "open_order_canceled": an order was placed and was still open;
        reconciliation canceled it. Treat as "not filled, do not retry".
    """

    def __init__(self, symbol: str, side: str, qty: float,
                 state: str = "unknown", raw_message: str = ""):
        self.symbol = symbol
        self.side = side
        self.qty = float(qty)
        self.state = state
        self.raw_message = raw_message
        super().__init__(
            f"Binance order timeout ({state}): {side} {symbol} qty={qty} — "
            f"{raw_message}"
        )


class ExchangeClient:
    """Thin wrapper around ``python-binance`` for USDT-M Futures trading.

    By default connects to the Binance **testnet** so that no real funds
    are at risk.  Set ``execution.live_mode: True`` in config (and supply
    real API keys) to trade with real money.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        testnet: bool | None = None,
        base_url: str | None = None,
    ):
        cfg = get_config().get("execution", {})

        self._api_key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("BINANCE_API_SECRET", "")

        # Testnet unless explicitly opted into live mode
        if testnet is None:
            self.testnet = not cfg.get("live_mode", False)
        else:
            self.testnet = testnet

        self._client = Client(self._api_key, self._api_secret, testnet=self.testnet)

        # Configure Futures endpoint
        if base_url:
            self._client.FUTURES_URL = base_url.rstrip("/") + "/fapi"
        elif self.testnet:
            self._client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

        self._symbol_info_cache: dict = {}

    # -- Account ---------------------------------------------------------------

    def get_balances(self) -> dict[str, float]:
        """Return ``{asset: available_balance}`` for all futures wallet assets."""
        account = self._retry(self._client.futures_account)
        return {
            a["asset"]: float(a["availableBalance"])
            for a in account["assets"]
            if float(a["walletBalance"]) > 0 or float(a["availableBalance"]) > 0
        }

    def get_usdt_balance(self) -> float:
        """Return available USDT balance in futures wallet."""
        balances = self.get_balances()
        return balances.get("USDT", 0.0)

    def get_current_position(self, symbol: str) -> float:
        """Return net position size for *symbol*.

        Positive = long, negative = short, 0 = flat.
        """
        positions = self._retry(
            self._client.futures_position_information, symbol=symbol,
        )
        for pos in positions:
            if pos["symbol"] == symbol:
                return float(pos["positionAmt"])
        return 0.0

    def get_open_positions(self) -> list[dict]:
        """Return all non-flat positions in a single API call.

        One ``futures_position_information`` request (no symbol) covers the
        whole account. Each entry: ``{symbol, qty, usd}`` where ``qty`` is the
        signed ``positionAmt`` (negative = short) and ``usd`` is the signed
        notional (``markPrice * qty`` when Binance omits ``notional``).
        """
        positions = self._retry(self._client.futures_position_information)
        out: list[dict] = []
        for pos in positions:
            qty = float(pos["positionAmt"])
            if qty == 0:
                continue
            notional = pos.get("notional")
            usd = float(notional) if notional is not None \
                else qty * float(pos["markPrice"])
            out.append({"symbol": pos["symbol"], "qty": qty, "usd": usd})
        return out

    def get_position_details(self) -> list[dict]:
        """All non-flat positions with the fields the monitor UI shows.

        Read-only superset of get_open_positions (kept separate so the
        runner's hot path is untouched). qty/notional are signed.
        """
        positions = self._retry(self._client.futures_position_information)
        out: list[dict] = []
        for pos in positions:
            qty = float(pos["positionAmt"])
            if qty == 0:
                continue
            notional = pos.get("notional")
            out.append({
                "symbol": pos["symbol"],
                "qty": qty,
                "entry_price": float(pos["entryPrice"]),
                "mark_price": float(pos["markPrice"]),
                "upnl": float(pos["unRealizedProfit"]),
                "leverage": float(pos.get("leverage") or 0),
                "liq_price": float(pos.get("liquidationPrice") or 0),
                "notional": float(notional) if notional is not None
                else qty * float(pos["markPrice"]),
            })
        return out

    def income_history(self, *, start_time_ms: int | None = None,
                       income_type: str | None = None,
                       limit: int = 1000) -> list[dict]:
        """Futures income records (REALIZED_PNL / COMMISSION / FUNDING_FEE...).

        Caller aggregates; this is a thin retry wrapper. Binance caps one
        page at 1000 records — enough for the testnet A/B volumes; the
        monitor labels totals as 'last 1000 records' rather than paginating.
        """
        params: dict = {"limit": limit}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if income_type is not None:
            params["incomeType"] = income_type
        return self._retry(self._client.futures_income_history, **params)

    def get_position_value(self, symbol: str) -> float:
        """Return absolute USDT value of current position for *symbol*."""
        pos_amt = self.get_current_position(symbol)
        if pos_amt == 0:
            return 0.0
        price = self.get_ticker_price(symbol)
        return abs(pos_amt) * price

    def get_total_portfolio_value(self) -> float:
        """Total futures wallet value (wallet balance + unrealised PnL).

        Raises ValueError when the account response lacks ``totalMarginBalance``
        (e.g. an error envelope that still parses as a dict). The old
        ``.get(..., 0.0)`` silently returned 0.0, which made the runner size
        every coin to zero and flatten the entire book with no alert (S3265).
        Treating a missing key as a fetch failure lets the runner abort the
        cycle instead.
        """
        account = self._retry(self._client.futures_account)
        if not isinstance(account, dict) or "totalMarginBalance" not in account:
            raise ValueError(
                "Binance futures_account response missing 'totalMarginBalance' "
                f"(type={type(account).__name__}, "
                f"keys={sorted(account)[:8] if isinstance(account, dict) else 'n/a'}) "
                "— treating as a fetch failure rather than sizing the book to "
                "zero (S3265 floor)."
            )
        return float(account["totalMarginBalance"])

    def get_open_position_count(self) -> int:
        """Count symbols with non-zero futures positions."""
        positions = self._retry(self._client.futures_position_information)
        return sum(1 for p in positions if float(p["positionAmt"]) != 0)

    # -- Leverage --------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for *symbol*."""
        logger.info("Setting leverage for %s to %dx", symbol, leverage)
        return self._retry(
            self._client.futures_change_leverage,
            symbol=symbol,
            leverage=leverage,
        )

    # -- Orders ----------------------------------------------------------------

    @staticmethod
    def _new_client_order_id() -> str:
        """Unique, Binance-valid newClientOrderId (<=36 chars, [A-Za-z0-9])."""
        return "ta" + uuid.uuid4().hex  # 34 chars

    def _reconcile_by_client_id(
        self, symbol: str, client_order_id: str, qty: float,
    ) -> tuple[str, dict | None]:
        """Determine an order's actual state after an unknown-execution event.

        DETERMINISTIC: queries the exact order we tagged with
        ``client_order_id`` via ``futures_get_order(origClientOrderId=...)``,
        rather than guessing from a side+qty userTrades scan. This is
        partial-fill safe — ``executedQty`` on the order is the cumulative
        filled size across all maker/taker fills — and cannot mis-attribute an
        unrelated same-side / same-qty order (the old heuristic's R2 bug).

        Returns ``(state, payload)``:
        - ``filled``: ``executedQty > 0`` (FILLED or PARTIALLY_FILLED). The
          synthesized envelope carries the real ``executedQty`` / ``avgPrice``
          so the caller records the position that actually exists.
        - ``open``: status NEW, nothing filled — order resting on the book.
        - ``not_placed``: order unknown to the exchange, or CANCELED / EXPIRED /
          REJECTED → never established a position → safe to retry.

        A 3s pre-query sleep lets the matching engine settle.
        """
        time.sleep(3)
        try:
            o = self._client.futures_get_order(
                symbol=symbol, origClientOrderId=client_order_id,
            )
        except Exception as e:  # noqa: BLE001 — reconciliation must not raise
            # -2013 "Order does not exist" → never reached the matching engine.
            logger.warning("Reconcile get_order failed for %s/%s: %s",
                           symbol, client_order_id, e)
            return ("not_placed", None)

        o = o or {}
        status = o.get("status", "")
        try:
            executed = float(o.get("executedQty", 0) or 0)
        except (TypeError, ValueError):
            executed = 0.0

        if executed > 0:
            return ("filled", {
                "orderId": o.get("orderId"),
                "clientOrderId": o.get("clientOrderId", client_order_id),
                "symbol": symbol,
                "side": o.get("side"),
                "status": "FILLED",
                "origQty": o.get("origQty", str(qty)),
                "executedQty": str(executed),
                "avgPrice": o.get("avgPrice"),
                "_reconciled": True,
            })
        if status == "NEW":
            return ("open", o)
        # CANCELED / EXPIRED / REJECTED / empty → no position established.
        return ("not_placed", None)

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> dict:
        """Place a futures MARKET order.  *side* is ``'BUY'`` or ``'SELL'``.

        When *reduce_only* is True, the order is constrained to reduce or
        close the existing position. Binance allows below-min-notional
        orders only when this flag is set.

        On Binance -1007 (timeout, execution status unknown) this method
        runs :meth:`_reconcile_after_timeout`, treats a matched fill as
        success, retries once if reconciliation confirms "not placed",
        and raises :class:`BinanceOrderTimeoutUnknown` only when state
        cannot be resolved (an open order is canceled in that path so the
        next cycle starts from a known state).
        """
        quantity = self.round_quantity(symbol, quantity)
        logger.info(
            "FUTURES MARKET %s %s qty=%.8f%s",
            side, symbol, quantity, " reduceOnly" if reduce_only else "",
        )

        for attempt in range(2):
            client_order_id = self._new_client_order_id()
            kwargs = dict(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                newClientOrderId=client_order_id,
            )
            if reduce_only:
                kwargs["reduceOnly"] = "true"
            try:
                return self._retry(self._client.futures_create_order, **kwargs)
            except BinanceAPIException as e:
                if getattr(e, "code", None) != -1007 and "-1007" not in str(e):
                    raise
                reason = f"-1007 timeout: {e}"
            except _UNKNOWN_EXEC_NETWORK_ERRORS as e:
                reason = f"network error: {e!r}"

            # Unknown execution status (-1007 or transport failure): reconcile
            # the tagged order before deciding whether to retry.
            logger.warning(
                "Unknown execution on %s %s qty=%s (attempt %d/2): %s; reconciling…",
                side, symbol, quantity, attempt + 1, reason,
            )
            state, payload = self._reconcile_by_client_id(
                symbol, client_order_id, quantity,
            )
            if state == "filled":
                logger.info("Reconcile: %s %s qty=%s FILLED (orderId=%s)",
                            side, symbol, payload.get("executedQty"),
                            payload.get("orderId"))
                return payload
            if state == "open":
                logger.warning("Reconcile: %s %s qty=%s open on book; "
                               "canceling for safety", side, symbol, quantity)
                try:
                    self._client.futures_cancel_order(
                        symbol=symbol, orderId=payload["orderId"],
                    )
                except Exception as cancel_err:  # noqa: BLE001
                    logger.error("Cancel of stranded order %s failed: %s",
                                 payload.get("orderId"), cancel_err)
                raise BinanceOrderTimeoutUnknown(
                    symbol=symbol, side=side, qty=quantity,
                    state="open_order_canceled", raw_message=reason,
                )
            # state == "not_placed" → safe to retry once with a fresh id.
            if attempt == 0:
                continue
            raise BinanceOrderTimeoutUnknown(
                symbol=symbol, side=side, qty=quantity,
                state="unknown", raw_message=reason,
            )

    def place_stop_loss(
        self,
        symbol: str,
        quantity: float,
        stop_price: float,
        side: str = "SELL",
    ) -> dict:
        """Place a futures STOP_MARKET order that reduces the position.

        *side* should be ``'SELL'`` for long positions and ``'BUY'`` for shorts.
        Uses ``reduceOnly=true`` + explicit quantity instead of
        ``closePosition=true`` so partial-position stops work and the order
        avoids the TIF GTE position-presence check that ``closePosition``
        triggers on testnet (APIError -4509).

        Same -1007 reconciliation as :meth:`place_market_order`.
        """
        stop_price = self.round_price(symbol, stop_price)
        quantity = self.round_quantity(symbol, quantity)
        logger.info(
            "FUTURES STOP_MARKET %s %s qty=%.8f stop=%.2f",
            symbol, side, quantity, stop_price,
        )

        for attempt in range(2):
            client_order_id = self._new_client_order_id()
            kwargs = dict(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                quantity=quantity,
                reduceOnly="true",
                newClientOrderId=client_order_id,
            )
            try:
                resp = self._retry(self._client.futures_create_order, **kwargs)
                # Binance routes STOP_MARKET through the conditional/algo system:
                # a success carries `algoId` (CONDITIONAL), not `orderId`. Treat
                # EITHER as success; a response with neither is a silent failure
                # (observed on testnet) — raise so arm_stop_loss marks the
                # position UNPROTECTED + alerts instead of recording a phantom id.
                if not (resp.get("orderId") or resp.get("algoId")):
                    raise RuntimeError(
                        f"STOP_MARKET {symbol} returned no orderId/algoId "
                        f"(silent rejection): {resp!r}"
                    )
                return resp
            except BinanceAPIException as e:
                if getattr(e, "code", None) != -1007 and "-1007" not in str(e):
                    raise
                reason = f"-1007 timeout: {e}"
            except _UNKNOWN_EXEC_NETWORK_ERRORS as e:
                reason = f"network error: {e!r}"

            logger.warning(
                "Unknown execution on STOP %s %s qty=%s (attempt %d/2): %s; reconciling…",
                side, symbol, quantity, attempt + 1, reason,
            )
            state, payload = self._reconcile_by_client_id(
                symbol, client_order_id, quantity,
            )
            if state == "filled":
                return payload
            if state == "open":
                # Stop resting on the book — that's the intended end state.
                logger.info("Reconcile: STOP %s %s resting (orderId=%s)",
                            side, symbol, payload.get("orderId"))
                return payload
            if attempt == 0:
                continue
            raise BinanceOrderTimeoutUnknown(
                symbol=symbol, side=side, qty=quantity,
                state="unknown", raw_message=reason,
            )

    def get_user_trades(self, symbol: str, order_id) -> list[dict]:
        """Return Binance Futures fills for one order (`/fapi/v1/userTrades`).

        Each fill carries `commission` (with `commissionAsset`) and
        `realizedPnl` which the placement response does NOT include. The
        live runner calls this immediately after a successful
        :meth:`place_market_order` and feeds the summed values into
        ``journal.update_trade_fills`` so `trades.fees` + `trades.pnl`
        stop being NULL.

        `order_id` may be ``str`` (the journal stores it as text) or ``int``.
        """
        return list(self._retry(
            self._client.futures_account_trades,
            symbol=symbol, orderId=int(order_id),
        ))

    def list_open_stops(self, symbol: str) -> list[dict]:
        """Open protective STOP_MARKET orders for *symbol*.

        Binance places STOP_MARKET as CONDITIONAL/algo orders, which do NOT
        appear in ``futures_get_open_orders`` — they live in the algo-orders
        endpoint and are keyed by ``algoId`` (not ``orderId``), with
        ``triggerPrice``/``quantity`` instead of ``stopPrice``/``origQty``.
        Query BOTH and normalize to the shape ``arm_stop_loss`` expects
        (``orderId`` / ``stopPrice`` / ``origQty``), tagging algo rows so
        ``cancel_order`` can route them. Without the algo half, the runner is
        blind to existing stops → no monotonic-keep (R5) and orphan stops
        accumulate every cycle.
        """
        out: list[dict] = []
        # Regular STOP_MARKET (mainnet/legacy path).
        try:
            for o in self._retry(self._client.futures_get_open_orders, symbol=symbol):
                if o.get("type") == "STOP_MARKET":
                    out.append({**o, "_algo": False})
        except Exception as e:  # noqa: BLE001 — never let a query failure block arming
            logger.warning("futures_get_open_orders failed for %s: %s", symbol, e)
        # Conditional/algo STOP_MARKET (the actual path on current Binance).
        try:
            algo = self._retry(self._client.futures_get_open_algo_orders)
            for o in algo:
                if o.get("symbol") != symbol or o.get("orderType") != "STOP_MARKET":
                    continue
                out.append({
                    "orderId": o.get("algoId"),
                    "stopPrice": o.get("triggerPrice"),
                    "origQty": o.get("quantity"),
                    "type": "STOP_MARKET",
                    "_algo": True,
                })
        except Exception as e:  # noqa: BLE001
            logger.warning("futures_get_open_algo_orders failed for %s: %s", symbol, e)
        return out

    def cancel_order(self, symbol: str, order_id) -> dict:
        """Cancel a single open futures order by id.

        Tries the regular order endpoint first; STOP_MARKET stops are algo
        orders, so on an unknown-order error fall back to the algo cancel
        (``order_id`` is then the ``algoId``).
        """
        try:
            return self._retry(
                self._client.futures_cancel_order, symbol=symbol, orderId=order_id,
            )
        except BinanceAPIException as e:
            code = getattr(e, "code", None)
            if code not in (-2011, -2013) and "Unknown order" not in str(e) \
                    and "does not exist" not in str(e):
                raise
            return self._retry(
                self._client.futures_cancel_algo_order, algoId=order_id,
            )

    def cancel_all_orders(self, symbol: str) -> list[dict]:
        """Cancel all open futures orders for *symbol*."""
        open_orders = self._retry(
            self._client.futures_get_open_orders, symbol=symbol,
        )
        results = []
        for order in open_orders:
            try:
                r = self._retry(
                    self._client.futures_cancel_order,
                    symbol=symbol,
                    orderId=order["orderId"],
                )
                results.append(r)
            except BinanceAPIException as e:
                logger.warning(
                    "Failed to cancel order %s: %s", order["orderId"], e,
                )
        return results

    # -- Market data -----------------------------------------------------------

    def get_ticker_price(self, symbol: str) -> float:
        """Current futures mark price."""
        data = self._retry(self._client.futures_symbol_ticker, symbol=symbol)
        return float(data["price"])

    def get_symbol_info(self, symbol: str) -> dict:
        """Trading rules for *symbol* (cached from futures exchange info)."""
        if symbol not in self._symbol_info_cache:
            info = self._retry(self._client.futures_exchange_info)
            for s in info["symbols"]:
                self._symbol_info_cache[s["symbol"]] = s
        if symbol not in self._symbol_info_cache:
            raise ValueError(f"Symbol {symbol} not found in futures exchange info")
        return self._symbol_info_cache[symbol]

    # -- Rounding helpers ------------------------------------------------------

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round *quantity* down to the symbol's LOT_SIZE step."""
        info = self.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                precision = int(round(-math.log10(step)))
                return math.floor(quantity * 10**precision) / 10**precision
        return quantity

    def round_price(self, symbol: str, price: float) -> float:
        """Round *price* to the symbol's PRICE_FILTER tick size."""
        info = self.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
                precision = int(round(-math.log10(tick)))
                return round(price, precision)
        return price

    def min_notional(self, symbol: str) -> float:
        """Symbol's MIN_NOTIONAL filter — minimum order value in USDT.

        Binance rejects non-reduceOnly orders below this (-4164/-1013); the
        live runner uses it to skip dust rebalance deltas instead of logging
        FAILED trades. Defaults to 5.0 (the Futures floor) when absent.
        """
        info = self.get_symbol_info(symbol)
        for f in info["filters"]:
            if f["filterType"] == "MIN_NOTIONAL":
                return float(f.get("notional", f.get("minNotional", 5.0)))
        return 5.0

    # -- Retry logic -----------------------------------------------------------

    @staticmethod
    def _retry(func, *args, max_retries: int = _MAX_RETRIES, **kwargs):
        """Call *func* with exponential backoff on retryable errors.

        On Binance error -1003 (IP banned, HTTP 418/429 + "banned until ..."),
        parse the ban-expiry timestamp and raise BinanceIPBan immediately
        without retrying — additional requests during a ban extend it.
        """
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                if getattr(e, "code", None) == -1003 or "-1003" in str(e):
                    m = _BAN_UNTIL_RE.search(str(e))
                    if m:
                        raise BinanceIPBan(until_ms=int(m.group(1)), raw_message=str(e))
                    # -1003 without parsable timestamp: still surface as ban.
                    raise BinanceIPBan(until_ms=0, raw_message=str(e))
                # -1007 "execution status unknown" is NOT safely retryable by
                # _retry — bubble it to the caller (place_market_order /
                # place_stop_loss) which runs reconciliation before deciding
                # whether to retry. HTTP 504 alone (without -1007) is safe.
                if getattr(e, "code", None) == -1007 or "-1007" in str(e):
                    raise
                if e.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    delay = _RETRY_DELAY_S * (2 ** attempt)
                    logger.warning(
                        "Binance %d -- retrying in %ds (attempt %d/%d)",
                        e.status_code, delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise
