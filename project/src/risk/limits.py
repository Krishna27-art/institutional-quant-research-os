"""
Risk Limits Module
Implements position limits, stop losses, high VIX halts, drawdown trailing limits,
and daily/weekly circuit breakers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def calculate_position_limits(
    positions: List[Any],
    capital: float,
    max_position_pct: float = 0.05,
    sector_limits: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Calculate position limits and sector limits excesses."""
    limits = {}
    if sector_limits is None:
        sector_limits = {}
        
    for pos in positions:
        position_value = pos.quantity * pos.current_price
        exposure_pct = position_value / capital
        limit_excess = max(0.0, exposure_pct - max_position_pct)
        limits[f"limit_{pos.symbol}"] = limit_excess
        
    sector_exposures = {}
    for pos in positions:
        sector = pos.sector
        position_value = pos.quantity * pos.current_price
        sector_exposures[sector] = sector_exposures.get(sector, 0.0) + position_value
        
    for sector, exposure in sector_exposures.items():
        limit = sector_limits.get(sector, 0.20)
        exposure_pct = exposure / capital
        limit_excess = max(0.0, exposure_pct - limit)
        limits[f"limit_sector_{sector}"] = limit_excess
        
    return limits


def should_stop_trading_high_vix(
    vix: float,
    enable_high_vix_stop: bool = True,
    high_vix_threshold: float = 25.0,
    high_vix_reduction: float = 0.25
) -> Tuple[bool, float]:
    """Check if trading should be scaled down during periods of high VIX."""
    if not enable_high_vix_stop:
        return False, 1.0
    if vix > high_vix_threshold:
        return True, high_vix_reduction
    return False, 1.0


def check_trailing_drawdown_limit(
    current_equity: float,
    capital: float,
    current_peak_equity: float,
    in_recovery_mode: bool,
    enable_trailing_dd_limit: bool = True,
    max_dd_from_peak_pct: float = 0.10
) -> Tuple[bool, float, float, bool]:
    """
    Check if trailing drawdown limit is breached.
    Returns (should_stop, drawdown_pct, new_peak, in_recovery)
    """
    if not enable_trailing_dd_limit:
        return False, 0.0, current_peak_equity, in_recovery_mode
        
    new_peak = current_peak_equity
    new_recovery = in_recovery_mode
    
    if current_equity > current_peak_equity:
        new_peak = current_equity
        new_recovery = False
        
    drawdown_pct = (new_peak - current_equity) / new_peak if new_peak > 0 else 0.0
    
    if drawdown_pct > max_dd_from_peak_pct:
        new_recovery = True
        return True, drawdown_pct, new_peak, new_recovery
        
    return new_recovery, drawdown_pct, new_peak, new_recovery


def check_circuit_breaker(
    daily_pnl: float,
    capital: float,
    max_daily_loss_pct: float = 0.03,
    weekly_pnl: Optional[float] = None,
    max_weekly_loss_pct: float = 0.08,
    circuit_breaker_active: bool = False,
    circuit_breaker_recovery_days: int = 0
) -> Tuple[bool, str, int]:
    """
    Check daily and weekly circuit breakers.
    Returns (triggered, reason, recovery_days)
    """
    daily_pnl_pct = daily_pnl / capital if capital > 0 else 0.0
    weekly_pnl_pct = weekly_pnl / capital if (weekly_pnl is not None and capital > 0) else 0.0

    if daily_pnl_pct <= -max_daily_loss_pct:
        return True, f"Daily loss limit breached ({daily_pnl_pct:.2%})", max(circuit_breaker_recovery_days, 1)

    if weekly_pnl is not None and weekly_pnl_pct <= -max_weekly_loss_pct:
        return True, f"Weekly loss limit breached ({weekly_pnl_pct:.2%})", max(circuit_breaker_recovery_days, 3)

    if circuit_breaker_active:
        return True, f"Circuit breaker active (remaining recovery: {circuit_breaker_recovery_days} days)", circuit_breaker_recovery_days

    return False, "", 0


def calculate_stop_loss(
    entry_price: float,
    atr: float,
    stop_loss_atr_multiplier: float = 2.0,
    direction: str = "long",
    enable_stop_losses: bool = True
) -> Optional[float]:
    """Calculate ATR-based stop loss level."""
    if not enable_stop_losses or atr <= 0:
        return None
    stop_distance = atr * stop_loss_atr_multiplier
    if direction.lower() == "long":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance
