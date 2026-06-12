"""Trading strategy definitions for backtesting with 5-level signal support.

Adapts Krypto-v0's strategy pattern for TradingAgents' 5-level signal output:
BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SignalLevel(str, Enum):
    """The five signal levels produced by TradingAgents' portfolio manager."""

    BUY = "BUY"
    OVERWEIGHT = "OVERWEIGHT"
    HOLD = "HOLD"
    UNDERWEIGHT = "UNDERWEIGHT"
    SELL = "SELL"


# Canonical mapping from signal level to position weight.
# BUY = full long, SELL = full short, intermediates are fractional.
SIGNAL_POSITION_MAP: dict[SignalLevel, float] = {
    SignalLevel.BUY: 1.0,
    SignalLevel.OVERWEIGHT: 0.5,
    SignalLevel.HOLD: 0.0,
    SignalLevel.UNDERWEIGHT: -0.5,
    SignalLevel.SELL: -1.0,
}


@dataclass(frozen=True)
class Signal:
    """A single trading signal.

    Attributes:
        level: The 5-level signal (BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL).
        position: Continuous position weight in [-1.0, 1.0].
            +1.0 = full long, -1.0 = full short, 0.0 = flat.
    """

    level: SignalLevel
    position: float


class Strategy(ABC):
    """Base class for all backtesting strategies.

    Subclasses translate raw inputs (agent signals, model predictions)
    into a ``Signal`` that the backtesting engine can execute.
    """

    name: str

    @abstractmethod
    def generate_signal(
        self,
        agent_signal: str,
        prediction: Optional[float] = None,
        actual_prev: Optional[float] = None,
        prediction_other: Optional[float] = None,
        actual_prev_other: Optional[float] = None,
    ) -> Signal:
        """Produce a trading signal for one time step.

        Args:
            agent_signal: The 5-level string emitted by the portfolio manager
                (e.g. "BUY", "UNDERWEIGHT").
            prediction: Model price prediction for the current step (optional).
            actual_prev: Previous actual price (optional).
            prediction_other: Secondary model prediction (optional, for
                consensus strategies).
            actual_prev_other: Previous actual price used by secondary model
                (optional).

        Returns:
            A ``Signal`` with the resolved level and position weight.
        """
        ...


def _parse_signal(raw: str) -> SignalLevel:
    """Normalise a raw signal string into a ``SignalLevel`` enum.

    Strips whitespace and upper-cases so that ``" buy "`` still matches.
    Falls back to HOLD for unrecognised values.
    """
    cleaned = raw.strip().upper()
    try:
        return SignalLevel(cleaned)
    except ValueError:
        return SignalLevel.HOLD


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class FiveLevelSignal(Strategy):
    """Maps 5-level agent signals directly to position sizes.

    This is the most straightforward strategy: each of the five signal
    levels maps to a fixed position weight via ``SIGNAL_POSITION_MAP``.
    """

    name = "FiveLevelSignal"

    def generate_signal(
        self,
        agent_signal: str,
        prediction: Optional[float] = None,
        actual_prev: Optional[float] = None,
        **kwargs,
    ) -> Signal:
        level = _parse_signal(agent_signal)
        return Signal(level=level, position=SIGNAL_POSITION_MAP[level])


class ThresholdSignal(Strategy):
    """Only trades on strong signals (BUY / SELL).

    OVERWEIGHT, UNDERWEIGHT, and HOLD are all treated as *no action*
    (position = 0).  Useful when you only want the engine to act on
    high-conviction calls.
    """

    name = "ThresholdSignal"

    def generate_signal(
        self,
        agent_signal: str,
        prediction: Optional[float] = None,
        actual_prev: Optional[float] = None,
        **kwargs,
    ) -> Signal:
        level = _parse_signal(agent_signal)
        if level == SignalLevel.BUY:
            return Signal(level=level, position=1.0)
        elif level == SignalLevel.SELL:
            return Signal(level=level, position=-1.0)
        # Anything else (including OVERWEIGHT / UNDERWEIGHT) is treated as flat.
        return Signal(level=SignalLevel.HOLD, position=0.0)


class ModelConsensus(Strategy):
    """Trades only when the prediction model and the agent signal agree.

    Agreement is defined as both pointing in the same directional sense:
    * Agent says BUY or OVERWEIGHT **and** model predicts price > previous
      actual  -->  long with the agent's weight.
    * Agent says SELL or UNDERWEIGHT **and** model predicts price < previous
      actual  -->  short with the agent's weight.
    * Otherwise (disagreement or HOLD)  -->  flat.

    When a secondary model prediction is provided, *both* models must agree
    with the agent signal for a trade to trigger.
    """

    name = "ModelConsensus"

    def generate_signal(
        self,
        agent_signal: str,
        prediction: Optional[float] = None,
        actual_prev: Optional[float] = None,
        prediction_other: Optional[float] = None,
        actual_prev_other: Optional[float] = None,
        **kwargs,
    ) -> Signal:
        level = _parse_signal(agent_signal)

        # HOLD always means no trade regardless of model output.
        if level == SignalLevel.HOLD:
            return Signal(level=level, position=0.0)

        # Determine the agent's directional bias.
        agent_bullish = level in (SignalLevel.BUY, SignalLevel.OVERWEIGHT)
        agent_bearish = level in (SignalLevel.SELL, SignalLevel.UNDERWEIGHT)

        # If no prediction is available, fall back to the agent signal alone.
        if prediction is None or actual_prev is None:
            return Signal(level=level, position=SIGNAL_POSITION_MAP[level])

        model_bullish = prediction > actual_prev
        model_bearish = prediction < actual_prev

        # Check primary model agreement.
        primary_agrees = (
            (agent_bullish and model_bullish)
            or (agent_bearish and model_bearish)
        )

        if not primary_agrees:
            return Signal(level=SignalLevel.HOLD, position=0.0)

        # If a secondary model is supplied, it must also agree.
        if prediction_other is not None and actual_prev_other is not None:
            other_bullish = prediction_other > actual_prev_other
            other_bearish = prediction_other < actual_prev_other
            secondary_agrees = (
                (agent_bullish and other_bullish)
                or (agent_bearish and other_bearish)
            )
            if not secondary_agrees:
                return Signal(level=SignalLevel.HOLD, position=0.0)

        return Signal(level=level, position=SIGNAL_POSITION_MAP[level])
