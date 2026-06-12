"""Pre-trade risk gates. Each returns (passed: bool, reason: str)."""
from __future__ import annotations


def check_leverage(size: float, max_leverage: float) -> tuple[bool, str]:
    if abs(size) > max_leverage:
        return False, f"|size|={abs(size):.3f} > max_leverage={max_leverage}"
    return True, "ok"


def check_daily_loss(pnl_today_pct: float, max_loss_pct: float) -> tuple[bool, str]:
    if pnl_today_pct < -max_loss_pct:
        return False, (f"daily PnL {pnl_today_pct:.2%} breached -{max_loss_pct:.2%} — "
                       f"KILL SWITCH")
    return True, "ok"


def check_drawdown(dd_from_peak: float, max_dd_pct: float) -> tuple[bool, str]:
    if dd_from_peak >= max_dd_pct:
        return False, (f"drawdown {dd_from_peak:.2%} >= max {max_dd_pct:.2%} — "
                       f"KILL SWITCH")
    return True, "ok"


def check_max_positions(current_open: int, max_open: int, opening_new: bool) -> tuple[bool, str]:
    if opening_new and current_open >= max_open:
        return False, f"already at MAX_OPEN_POSITIONS={max_open}"
    return True, "ok"


def check_frequency_guard(coin: str, trades_today_count: int) -> tuple[bool, str]:
    if trades_today_count > 0:
        return False, f"{coin} already traded today ({trades_today_count} time(s))"
    return True, "ok"
