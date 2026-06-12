"""Structured prediction output shared by all forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Prediction:
    """A single model forecast with optional confidence interval."""

    value: float
    model_name: str
    timestamp: datetime
    lower: Optional[float] = None   # 95% CI lower bound
    upper: Optional[float] = None   # 95% CI upper bound
    features_used: list[str] = field(default_factory=list)

    @property
    def has_interval(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def interval_width(self) -> float:
        if not self.has_interval:
            return 0.0
        return self.upper - self.lower

    @property
    def direction(self) -> str:
        """Return 'up' or 'down' relative to the lower bound (proxy for current price)."""
        if self.has_interval and self.value > self.lower:
            return "up"
        return "down"

    def to_report_string(self, current_price: Optional[float] = None) -> str:
        """Format prediction as a human-readable report string for LLM consumption."""
        lines = [
            f"Model: {self.model_name}",
            f"Forecast Date: {self.timestamp.strftime('%Y-%m-%d')}",
            f"Predicted Price: ${self.value:,.2f}",
        ]
        if self.has_interval:
            lines.append(f"95% Confidence Interval: ${self.lower:,.2f} - ${self.upper:,.2f}")
            lines.append(f"Interval Width: ${self.interval_width:,.2f}")
        if current_price is not None and current_price > 0:
            pct_change = ((self.value - current_price) / current_price) * 100
            direction = "UP" if pct_change > 0 else "DOWN"
            lines.append(f"Direction: {direction} ({pct_change:+.2f}% from current ${current_price:,.2f})")
        else:
            lines.append(f"Direction: {self.direction.upper()}")
        if self.features_used:
            lines.append(f"Features Used: {len(self.features_used)}")
        return "\n".join(lines)
