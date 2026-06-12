"""Backtesting engine for evaluating TradingAgents strategies on historical data.

Ported from Krypto-v0's backtesting engine and adapted for TradingAgents'
5-level signal output (BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL).

Key adaptations from Krypto-v0:
- Position sizes are continuous [-1.0, 1.0] rather than discrete {-1, 0, +1}.
- Transaction costs scale with the absolute position size.
- Short borrowing cost applies proportionally to the short fraction.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from tradingagents.backtesting.strategies import Signal, Strategy
from tradingagents.dataflows.config import get_config


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single executed trade entry in the trade log.

    Attributes:
        date: Trade date.
        signal_level: The 5-level signal string (e.g. "BUY").
        position: Position weight applied ([-1.0, 1.0]).
        entry_price: Price at entry (previous actual close).
        exit_price: Price at exit (current actual close).
        gross_return: Return before costs.
        cost: Total transaction cost incurred.
        net_return: Return after costs.
        equity_after: Equity value after this trade settles.
    """

    date: Any
    signal_level: str
    position: float
    entry_price: float
    exit_price: float
    gross_return: float
    cost: float
    net_return: float
    equity_after: float


@dataclass
class BacktestResult:
    """Container for a complete backtest run.

    Attributes:
        strategy_name: Name of the strategy used.
        ticker: Ticker / asset identifier.
        dates: List of trade dates.
        positions: Position weight per day.
        daily_returns: Net daily return per day.
        equity_curve: Cumulative equity (length = len(dates) + 1 for the
            initial capital entry).
        metrics: Dict of computed performance metrics.
        trade_log: Detailed per-trade records.
        config_snapshot: Copy of the config dict at the time of the run.
    """

    strategy_name: str
    ticker: str
    dates: list
    positions: list[float]
    daily_returns: list[float]
    equity_curve: list[float]
    metrics: dict[str, float]
    trade_log: list[TradeRecord] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    daily_returns: list[float],
    positions: list[float],
    initial_capital: float,
    equity_curve: list[float],
    risk_free_rate: float = 0.045,
) -> dict[str, float]:
    """Compute standard backtest performance metrics.

    Args:
        daily_returns: Sequence of net daily returns.
        positions: Position weight per day (used to identify traded days).
        initial_capital: Starting equity.
        equity_curve: Cumulative equity series (length = len(daily_returns) + 1).
        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        Dict with keys: total_return, annualized_return, sharpe_ratio,
        max_drawdown, win_rate, n_trades, profit_factor.
    """
    returns = np.array(daily_returns, dtype=np.float64)
    pos = np.array(positions, dtype=np.float64)

    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital

    n_days = len(returns)
    if n_days > 0:
        ann_return = (1 + total_return) ** (252 / n_days) - 1
    else:
        ann_return = 0.0

    # Sharpe ratio: only on days with a non-zero position.
    traded_mask = np.abs(pos) > 1e-9
    traded_returns = returns[traded_mask]
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    if len(traded_returns) > 1:
        excess = traded_returns - daily_rf
        std_excess = np.std(excess, ddof=1)
        sharpe = (
            float(np.mean(excess) / std_excess * np.sqrt(252))
            if std_excess > 0
            else 0.0
        )
    else:
        sharpe = 0.0

    # Max drawdown.
    eq = np.array(equity_curve, dtype=np.float64)
    running_max = np.maximum.accumulate(eq)
    drawdowns = np.where(running_max > 0, (running_max - eq) / running_max, 0.0)
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Win rate.
    n_trades = int(traded_mask.sum())
    wins = int(np.sum(traded_returns > 0))
    win_rate = wins / n_trades if n_trades > 0 else 0.0

    # Profit factor.
    gross_profit = float(np.sum(traded_returns[traded_returns > 0]))
    gross_loss = float(np.abs(np.sum(traded_returns[traded_returns < 0])))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def run_backtest(
    dates: pd.Series,
    actuals: np.ndarray,
    agent_signals: list[str],
    strategy: Strategy,
    ticker: str = "",
    initial_capital: float = 10_000.0,
    predictions: Optional[np.ndarray] = None,
    predictions_other: Optional[np.ndarray] = None,
    actuals_other: Optional[np.ndarray] = None,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    short_cost: float = 0.0003,
    position_size: float = 1.0,
    risk_free_rate: float = 0.045,
) -> BacktestResult:
    """Run a backtest for a strategy over a historical period.

    The engine iterates day-by-day starting from index 1.  For each day *i*:

    1. The strategy produces a ``Signal`` from the agent signal (and
       optionally model predictions).
    2. The position weight from the signal is scaled by ``position_size``.
    3. Gross return = ``position * (actual[i] - actual[i-1]) / actual[i-1]``.
    4. Transaction costs (fees, slippage, short borrowing) are deducted
       proportionally to the absolute position weight.
    5. Equity is updated: ``equity[i] = equity[i-1] * (1 + net_return)``.

    Cost model (per trade, proportional to ``|position|``):
        - Round-trip fees: ``2 * fee_rate``
        - Slippage: ``slippage``
        - Short borrowing: ``short_cost`` (only when position < 0)

    Args:
        dates: Pandas Series of dates aligned with ``actuals``.
        actuals: Array of actual prices (e.g. close prices).
        agent_signals: List of 5-level signal strings, one per date.
            Must have the same length as ``dates``.
        strategy: A ``Strategy`` instance that converts signals into positions.
        ticker: Asset ticker (stored in the result for identification).
        initial_capital: Starting equity in currency units.
        predictions: Optional array of model price predictions (same length
            as ``dates``).  Passed to the strategy for consensus logic.
        predictions_other: Optional secondary model predictions.
        actuals_other: Optional actuals aligned with the secondary model.
        fee_rate: One-way fee rate (applied twice for round trip).
        slippage: Slippage cost per trade.
        short_cost: Daily borrowing cost for short positions.
        position_size: Global scaling factor for all positions.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.

    Returns:
        A ``BacktestResult`` containing the equity curve, metrics, and
        trade log.
    """
    config = get_config()

    positions: list[float] = []
    daily_returns: list[float] = []
    equity: list[float] = [initial_capital]
    trade_dates: list = []
    trade_log: list[TradeRecord] = []

    for i in range(1, len(dates)):
        actual_prev = actuals[i - 1]
        actual_i = actuals[i]

        # Skip if prices are invalid.
        if (
            np.isnan(actual_prev)
            or np.isnan(actual_i)
            or actual_prev == 0
        ):
            positions.append(0.0)
            daily_returns.append(0.0)
            equity.append(equity[-1])
            trade_dates.append(dates.iloc[i])
            continue

        # Build optional kwargs for the strategy.
        kwargs: dict[str, Any] = {}
        if predictions is not None:
            pred_i = predictions[i]
            if np.isnan(pred_i):
                positions.append(0.0)
                daily_returns.append(0.0)
                equity.append(equity[-1])
                trade_dates.append(dates.iloc[i])
                continue
            kwargs["prediction"] = pred_i
            kwargs["actual_prev"] = actual_prev

        if predictions_other is not None and actuals_other is not None:
            kwargs["prediction_other"] = predictions_other[i]
            kwargs["actual_prev_other"] = actuals_other[i - 1]

        # Generate signal through the strategy.
        signal: Signal = strategy.generate_signal(
            agent_signal=agent_signals[i],
            **kwargs,
        )

        # Scale by global position_size.
        effective_position = signal.position * position_size
        positions.append(effective_position)

        if abs(effective_position) < 1e-9:
            # No trade: equity unchanged.
            daily_returns.append(0.0)
            equity.append(equity[-1])
            trade_dates.append(dates.iloc[i])
            trade_log.append(
                TradeRecord(
                    date=dates.iloc[i],
                    signal_level=signal.level.value,
                    position=effective_position,
                    entry_price=actual_prev,
                    exit_price=actual_i,
                    gross_return=0.0,
                    cost=0.0,
                    net_return=0.0,
                    equity_after=equity[-1],
                )
            )
            continue

        # Gross return: directional price move weighted by position.
        price_return = (actual_i - actual_prev) / actual_prev
        gross_ret = effective_position * price_return

        # Transaction costs scale with absolute position size.
        abs_pos = abs(effective_position)
        cost = (2 * fee_rate + slippage) * abs_pos
        # Additional borrowing cost for short exposure.
        if effective_position < 0:
            cost += short_cost * abs_pos

        net_ret = gross_ret - cost

        daily_returns.append(net_ret)
        equity.append(equity[-1] * (1 + net_ret))
        trade_dates.append(dates.iloc[i])

        trade_log.append(
            TradeRecord(
                date=dates.iloc[i],
                signal_level=signal.level.value,
                position=effective_position,
                entry_price=actual_prev,
                exit_price=actual_i,
                gross_return=gross_ret,
                cost=cost,
                net_return=net_ret,
                equity_after=equity[-1],
            )
        )

    metrics = compute_metrics(
        daily_returns,
        positions,
        initial_capital,
        equity,
        risk_free_rate=risk_free_rate,
    )

    return BacktestResult(
        strategy_name=strategy.name,
        ticker=ticker,
        dates=trade_dates,
        positions=positions,
        daily_returns=daily_returns,
        equity_curve=equity,
        metrics=metrics,
        trade_log=trade_log,
        config_snapshot=config,
    )
