"""Enhanced Risk Engine with VaR, Circuit Breakers, Correlation Checks
Architecture V2 - Quantitative Trading System for Indian Markets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

from .allocator import PortfolioAllocation


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allowed: bool
    capped_allocations: list[PortfolioAllocation]
    reason: str
    var_99_1day: Optional[float] = None
    cvar_95_1day: Optional[float] = None
    portfolio_heat: Optional[float] = None
    leverage: Optional[float] = None
    circuit_breaker_triggered: bool = False


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    var_99_1day_pct: float  # 99% VaR 1-day
    cvar_95_1day_pct: float  # 95% CVaR 1-day
    portfolio_heat: float  # Correlation-weighted exposure
    leverage: float  # Current leverage
    max_drawdown: float  # Current max drawdown
    daily_pnl_pct: float  # Daily PnL percentage
    weekly_pnl_pct: float  # Weekly PnL percentage


class RiskManagerV2:
    """
    Enhanced Risk Manager for Architecture V2
    
    Features:
    - Pre-trade: Position limits, sector limits, VaR, correlation checks
    - Intraday: Trailing stops, circuit breakers, leverage monitoring
    - Post-trade: Sharpe, VaR, drawdown tracking, Kelly adjustment
    - Tail risk: OTM put hedge when VIX < 12
    """
    
    def __init__(
        self,
        # Pre-trade limits
        max_position_pct: float = 0.05,  # 5% of AUM
        max_sector_exposure_pct: float = 0.30,  # 30% of AUM
        var_99_1day_cap_pct: float = 0.02,  # 2% of AUM
        correlation_heat_threshold: float = 0.7,
        
        # Intraday controls
        trailing_stop_atr_pct: float = 0.10,  # 10% ATR
        daily_circuit_breaker_pct: float = -0.03,  # -3% daily
        weekly_circuit_breaker_pct: float = -0.08,  # -8% weekly
        leverage_warning_threshold: float = 3.0,
        leverage_hard_stop: float = 4.0,
        
        # Post-trade
        kelly_adjustment_frequency: str = "monthly",
        
        # Tail risk
        vix_threshold: float = 12.0,
        tail_hedge_cost_pct_aum: float = 0.01,  # 1% of AUM/year
    ) -> None:
        # Pre-trade limits
        self.max_position_pct = max_position_pct
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.var_99_1day_cap_pct = var_99_1day_cap_pct
        self.correlation_heat_threshold = correlation_heat_threshold
        
        # Intraday controls
        self.trailing_stop_atr_pct = trailing_stop_atr_pct
        self.daily_circuit_breaker_pct = daily_circuit_breaker_pct
        self.weekly_circuit_breaker_pct = weekly_circuit_breaker_pct
        self.leverage_warning_threshold = leverage_warning_threshold
        self.leverage_hard_stop = leverage_hard_stop
        
        # Post-trade
        self.kelly_adjustment_frequency = kelly_adjustment_frequency
        
        # Tail risk
        self.vix_threshold = vix_threshold
        self.tail_hedge_cost_pct_aum = tail_hedge_cost_pct_aum
        
        # State tracking
        self.daily_pnl_history: List[float] = []
        self.weekly_pnl_history: List[float] = []
        self.position_returns: Dict[str, List[float]] = {}
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_start_time: Optional[datetime] = None
        self.last_kelly_adjustment: Optional[datetime] = None
        self.current_kelly_fraction: float = 0.15  # Default 15%
        
    def calculate_var_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99
    ) -> Tuple[float, float]:
        """
        Calculate VaR and CVaR using historical simulation.
        
        Args:
            returns: Historical returns array
            confidence_level: Confidence level for VaR (e.g., 0.99 for 99%)
            
        Returns:
            (VaR, CVaR) tuple
        """
        if len(returns) < 30:
            return 0.02, 0.03  # Default values
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Calculate VaR
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[var_index]
        
        # Calculate CVaR (average of returns beyond VaR)
        cvar_returns = sorted_returns[:var_index]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var
        
        return abs(var), abs(cvar)
    
    def calculate_portfolio_heat(
        self,
        weights: Dict[str, float],
        correlation_matrix: np.ndarray
    ) -> float:
        """
        Calculate portfolio heat (correlation-weighted exposure).
        
        Heat = Σ(w_i × |corr_i|) where corr_i is average correlation with other positions
        """
        if not weights or correlation_matrix is None:
            return 0.0
        
        symbols = list(weights.keys())
        n = len(symbols)
        if n < 2:
            return 0.0
        
        heat = 0.0
        for i, symbol in enumerate(symbols):
            weight = weights[symbol]
            # Get average correlation with other positions
            correlations = correlation_matrix[i, :]
            correlations[i] = 0  # Exclude self-correlation
            avg_corr = np.mean(np.abs(correlations))
            heat += weight * avg_corr
        
        return heat
    
    def calculate_leverage(
        self,
        positions: Dict[str, float],
        capital: float
    ) -> float:
        """Calculate current portfolio leverage"""
        if capital == 0:
            return 0.0
        
        total_exposure = sum(abs(pos) for pos in positions.values())
        return total_exposure / capital
    
    def check_circuit_breaker(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float
    ) -> Tuple[bool, str]:
        """
        Check if circuit breaker should be triggered.
        
        Returns:
            (triggered, reason) tuple
        """
        # Daily circuit breaker
        if daily_pnl_pct <= self.daily_circuit_breaker_pct:
            return True, f"daily_circuit_breaker: {daily_pnl_pct:.2%} <= {self.daily_circuit_breaker_pct:.2%}"
        
        # Weekly circuit breaker
        if weekly_pnl_pct <= self.weekly_circuit_breaker_pct:
            return True, f"weekly_circuit_breaker: {weekly_pnl_pct:.2%} <= {self.weekly_circuit_breaker_pct:.2%}"
        
        return False, "ok"
    
    def check_leverage(
        self,
        leverage: float
    ) -> Tuple[bool, str]:
        """
        Check leverage limits.
        
        Returns:
            (warning_triggered, message) tuple
        """
        if leverage >= self.leverage_hard_stop:
            return True, f"leverage_hard_stop: {leverage:.2f}x >= {self.leverage_hard_stop:.2f}x"
        
        if leverage >= self.leverage_warning_threshold:
            return True, f"leverage_warning: {leverage:.2f}x >= {self.leverage_warning_threshold:.2f}x"
        
        return False, "ok"
    
    def check_tail_hedge(
        self,
        vix: float,
        capital: float
    ) -> Tuple[bool, float]:
        """
        Check if tail hedge is needed.
        
        Returns:
            (hedge_needed, hedge_cost) tuple
        """
        if vix < self.vix_threshold:
            hedge_cost = capital * self.tail_hedge_cost_pct_aum / 252  # Daily cost
            return True, hedge_cost
        
        return False, 0.0
    
    def pre_trade_check(
        self,
        capital: float,
        allocations: Iterable[PortfolioAllocation],
        position_returns_history: Dict[str, List[float]],
        correlation_matrix: Optional[np.ndarray] = None,
        sector_exposures: Optional[Dict[str, float]] = None,
    ) -> RiskDecision:
        """
        Pre-trade risk check.
        
        Checks:
        - Position size limits
        - Sector exposure limits
        - VaR limit
        - Correlation heat
        """
        allocations_list = list(allocations)
        
        # Calculate portfolio returns for VaR
        all_returns = []
        for returns in position_returns_history.values():
            all_returns.extend(returns)
        
        portfolio_returns = np.array(all_returns) if all_returns else np.array([])
        
        # Calculate VaR and CVaR
        var_99, cvar_95 = self.calculate_var_cvar(portfolio_returns, 0.99)
        
        # Check VaR limit
        if var_99 > self.var_99_1day_cap_pct:
            return RiskDecision(
                allowed=False,
                capped_allocations=[],
                reason=f"var_limit_exceeded: {var_99:.2%} > {self.var_99_1day_cap_pct:.2%}",
                var_99_1day=var_99,
                cvar_95_1day=cvar_95
            )
        
        # Calculate portfolio heat
        weights = {alloc.symbol: alloc.weight for alloc in allocations_list}
        portfolio_heat = 0.0
        if correlation_matrix is not None and weights:
            portfolio_heat = self.calculate_portfolio_heat(weights, correlation_matrix)
            
            # Check correlation heat
            if portfolio_heat > self.correlation_heat_threshold:
                return RiskDecision(
                    allowed=False,
                    capped_allocations=[],
                    reason=f"correlation_heat_exceeded: {portfolio_heat:.2f} > {self.correlation_heat_threshold:.2f}",
                    var_99_1day=var_99,
                    cvar_95_1day=cvar_95,
                    portfolio_heat=portfolio_heat
                )
        
        # Check position size limits
        capped: list[PortfolioAllocation] = []
        for allocation in allocations_list:
            cap = capital * self.max_position_pct
            if allocation.capital > cap:
                capped.append(
                    PortfolioAllocation(
                        symbol=allocation.symbol,
                        weight=allocation.weight,
                        capital=cap,
                        score=allocation.score,
                    )
                )
            else:
                capped.append(allocation)
        
        # Check sector exposure limits
        if sector_exposures:
            for sector, exposure in sector_exposures.items():
                if exposure > self.max_sector_exposure_pct:
                    return RiskDecision(
                        allowed=False,
                        capped_allocations=[],
                        reason=f"sector_limit_exceeded: {sector} {exposure:.2%} > {self.max_sector_exposure_pct:.2%}",
                        var_99_1day=var_99,
                        cvar_95_1day=cvar_95,
                        portfolio_heat=portfolio_heat
                    )
        
        return RiskDecision(
            allowed=True,
            capped_allocations=capped,
            reason="ok",
            var_99_1day=var_99,
            cvar_95_1day=cvar_95,
            portfolio_heat=portfolio_heat
        )
    
    def intraday_check(
        self,
        daily_pnl_pct: float,
        weekly_pnl_pct: float,
        leverage: float,
        positions: Dict[str, float],
        atr_values: Dict[str, float],
        current_prices: Dict[str, float],
        entry_prices: Dict[str, float]
    ) -> Tuple[bool, str, List[str]]:
        """
        Intraday risk check.
        
        Returns:
            (should_halt, reason, stop_signals) tuple
        """
        stop_signals = []
        
        # Check circuit breaker
        circuit_triggered, circuit_reason = self.check_circuit_breaker(daily_pnl_pct, weekly_pnl_pct)
        if circuit_triggered:
            self.circuit_breaker_active = True
            self.circuit_breaker_start_time = datetime.now()
            return True, circuit_reason, stop_signals
        
        # Check leverage
        leverage_warning, leverage_reason = self.check_leverage(leverage)
        if "hard_stop" in leverage_reason:
            return True, leverage_reason, stop_signals
        
        # Check trailing stops
        for symbol, current_price in current_prices.items():
            if symbol not in entry_prices or symbol not in atr_values:
                continue
            
            entry_price = entry_prices[symbol]
            atr = atr_values[symbol]
            
            if atr == 0:
                continue
            
            # Calculate stop loss price
            stop_price = entry_price * (1 - self.trailing_stop_atr_pct)
            
            if current_price <= stop_price:
                stop_signals.append(f"stop_loss_triggered: {symbol} @ {current_price:.2f}")
        
        return False, "ok", stop_signals
    
    def post_trade_update(
        self,
        daily_pnl_pct: float,
        realized_sharpe: float,
        realized_var: float,
        current_date: datetime
    ) -> Dict[str, float]:
        """
        Post-trade risk metrics update and Kelly adjustment.
        
        Returns:
            Dictionary of updated risk parameters
        """
        # Update PnL history
        self.daily_pnl_history.append(daily_pnl_pct)
        if len(self.daily_pnl_history) > 252:
            self.daily_pnl_history = self.daily_pnl_history[-252:]
        
        # Calculate weekly PnL
        if len(self.daily_pnl_history) >= 5:
            weekly_pnl = sum(self.daily_pnl_history[-5:])
            self.weekly_pnl_history.append(weekly_pnl)
            if len(self.weekly_pnl_history) > 52:
                self.weekly_pnl_history = self.weekly_pnl_history[-52:]
        
        # Kelly adjustment (monthly)
        if self.kelly_adjustment_frequency == "monthly":
            if self.last_kelly_adjustment is None or (current_date - self.last_kelly_adjustment).days >= 30:
                # Adjust Kelly based on realized Sharpe
                # Kelly fraction = 15% of (Sharpe^2)
                new_kelly = 0.15 * (realized_sharpe ** 2)
                self.current_kelly_fraction = min(0.50, max(0.05, new_kelly))
                self.last_kelly_adjustment = current_date
        
        return {
            "kelly_fraction": self.current_kelly_fraction,
            "daily_pnl_pct": daily_pnl_pct,
            "realized_sharpe": realized_sharpe,
            "realized_var": realized_var
        }
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after cooldown period"""
        if self.circuit_breaker_active and self.circuit_breaker_start_time:
            # 5-day cooldown
            if (datetime.now() - self.circuit_breaker_start_time).days >= 5:
                self.circuit_breaker_active = False
                self.circuit_breaker_start_time = None
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        daily_pnl = self.daily_pnl_history[-1] if self.daily_pnl_history else 0.0
        weekly_pnl = sum(self.daily_pnl_history[-5:]) if len(self.daily_pnl_history) >= 5 else 0.0
        
        # Calculate max drawdown
        if len(self.daily_pnl_history) > 1:
            cumulative = np.cumsum(self.daily_pnl_history)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max)
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0.0
        
        return RiskMetrics(
            var_99_1day_pct=0.0,  # Calculated per trade
            cvar_95_1day_pct=0.0,
            portfolio_heat=0.0,
            leverage=0.0,
            max_drawdown=max_drawdown,
            daily_pnl_pct=daily_pnl,
            weekly_pnl_pct=weekly_pnl
        )


# Legacy RiskManager for backward compatibility
class RiskManager:
    def __init__(self, daily_loss_limit_pct: float = 0.02, max_position_pct: float = 0.05, monthly_drawdown_cut_pct: float = 0.06) -> None:
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_position_pct = max_position_pct
        self.monthly_drawdown_cut_pct = monthly_drawdown_cut_pct

    def evaluate(self, capital: float, allocations: Iterable[PortfolioAllocation], daily_pnl_pct: float = 0.0, monthly_drawdown_pct: float = 0.0) -> RiskDecision:
        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            return RiskDecision(False, [], "daily_loss_limit_hit")

        scale = 0.5 if monthly_drawdown_pct >= self.monthly_drawdown_cut_pct else 1.0
        capped: list[PortfolioAllocation] = []
        for allocation in allocations:
            cap = capital * self.max_position_pct * scale
            capped.append(
                PortfolioAllocation(
                    symbol=allocation.symbol,
                    weight=allocation.weight,
                    capital=min(allocation.capital, cap),
                    score=allocation.score,
                )
            )
        return RiskDecision(True, capped, "ok")
