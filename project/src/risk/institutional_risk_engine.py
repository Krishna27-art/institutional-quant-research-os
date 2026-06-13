"""
Institutional Risk Engine (VaR, CVaR, Kelly)
Refactored to delegate implementation details to limits.py, compliance.py, and metrics.py.
"""

import os
import sys
import signal
import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .limits import (
    calculate_position_limits as limits_calculate_position_limits,
    should_stop_trading_high_vix as limits_should_stop_trading_high_vix,
    check_trailing_drawdown_limit as limits_check_trailing_drawdown_limit,
    check_circuit_breaker as limits_check_circuit_breaker,
    calculate_stop_loss as limits_calculate_stop_loss
)
from .metrics import (
    calculate_portfolio_returns as metrics_calculate_portfolio_returns,
    calculate_moments as metrics_calculate_moments,
    calculate_var as metrics_calculate_var,
    calculate_var_historical as metrics_calculate_var_historical,
    calculate_var_evt as metrics_calculate_var_evt,
    calculate_cvar as metrics_calculate_cvar,
    calculate_liquidity_adjusted_var as metrics_calculate_liquidity_adjusted_var,
    calculate_volatility_target_multiplier as metrics_calculate_volatility_target_multiplier,
    calculate_kelly_fraction as metrics_calculate_kelly_fraction,
    calculate_portfolio_heat as metrics_calculate_portfolio_heat,
    calculate_tail_risk as metrics_calculate_tail_risk
)
from .sebi_algo_compliance import SEBIAlgoCompliance

logger = logging.getLogger(__name__)


class CircuitBreakerTrigger(Enum):
    """Reason for circuit breaker trigger"""
    DAILY_LOSS = "daily_loss_exceeded"
    WEEKLY_LOSS = "weekly_loss_exceeded"
    DRAWDOWN = "drawdown_exceeded"
    VIX_SPIKE = "vix_spike"
    MANUAL = "manual"
    DATA_QUALITY = "data_quality_failure"
    SYSTEM_ERROR = "system_error"


@dataclass
class CircuitBreakerState:
    """Current state of circuit breaker"""
    is_active: bool
    trigger_reason: Optional[str]
    trigger_time: Optional[datetime]
    trigger_value: Optional[float]
    threshold: Optional[float]
    recovery_days_remaining: int
    positions_closed: bool = False
    orders_cancelled: bool = False


class HardCircuitBreaker:
    """
    Hard Circuit Breaker that enforces system shutdown.
    """
    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        max_weekly_loss_pct: float = 0.08,
        max_drawdown_pct: float = 0.10,
        vix_threshold: float = 35.0,
        recovery_days: int = 3,
        state_file: str = "circuit_breaker_state.json",
        shutdown_callback: Optional[Callable] = None,
        close_positions: bool = True,
        allow_manual_override: bool = False
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.vix_threshold = vix_threshold
        self.recovery_days = recovery_days
        self.state_file = Path(state_file)
        self.shutdown_callback = shutdown_callback
        self.close_positions = close_positions
        self.allow_manual_override = allow_manual_override
        self.state = self._load_state()
        if self.state.is_active:
            self._check_recovery()
    
    def _load_state(self) -> CircuitBreakerState:
        if not self.state_file.exists():
            return CircuitBreakerState(
                is_active=False,
                trigger_reason=None,
                trigger_time=None,
                trigger_value=None,
                threshold=None,
                recovery_days_remaining=0
            )
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            return CircuitBreakerState(
                is_active=data['is_active'],
                trigger_reason=data['trigger_reason'],
                trigger_time=datetime.fromisoformat(data['trigger_time']) if data['trigger_time'] else None,
                trigger_value=data['trigger_value'],
                threshold=data['threshold'],
                recovery_days_remaining=data['recovery_days_remaining'],
                positions_closed=data.get('positions_closed', False),
                orders_cancelled=data.get('orders_cancelled', False)
            )
        except Exception as e:
            logger.error(f"Failed to load circuit breaker state: {e}")
            return CircuitBreakerState(
                is_active=False, trigger_reason=None, trigger_time=None, trigger_value=None, threshold=None, recovery_days_remaining=0
            )
    
    def _save_state(self) -> None:
        try:
            data = {
                'is_active': self.state.is_active,
                'trigger_reason': self.state.trigger_reason,
                'trigger_time': self.state.trigger_time.isoformat() if self.state.trigger_time else None,
                'trigger_value': self.state.trigger_value,
                'threshold': self.state.threshold,
                'recovery_days_remaining': self.state.recovery_days_remaining,
                'positions_closed': self.state.positions_closed,
                'orders_cancelled': self.state.orders_cancelled,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save circuit breaker state: {e}")
    
    def _check_recovery(self) -> None:
        if not self.state.is_active or self.state.trigger_time is None:
            return
        days_since_trigger = (datetime.now() - self.state.trigger_time).days
        if days_since_trigger >= self.recovery_days:
            logger.info("Recovery period expired. Circuit breaker can be reset.")
            self.state.recovery_days_remaining = 0
            self._save_state()
        else:
            self.state.recovery_days_remaining = self.recovery_days - days_since_trigger
            self._save_state()
    
    def trigger(
        self,
        reason: CircuitBreakerTrigger,
        value: float,
        threshold: float,
        message: str = ""
    ) -> bool:
        if self.state.is_active:
            logger.warning("Circuit breaker already active. Cannot trigger again.")
            return False
        logger.critical("=" * 80)
        logger.critical("CIRCUIT BREAKER TRIGGERED - HARD SHUTDOWN INITIATED")
        logger.critical("=" * 80)
        logger.critical(f"Reason: {reason.value}")
        logger.critical(f"Value: {value:.4f}")
        logger.critical(f"Threshold: {threshold:.4f}")
        logger.critical(f"Message: {message}")
        logger.critical(f"Time: {datetime.now().isoformat()}")
        logger.critical("=" * 80)
        
        self.state.is_active = True
        self.state.trigger_reason = reason.value
        self.state.trigger_time = datetime.now()
        self.state.trigger_value = value
        self.state.threshold = threshold
        self.state.recovery_days_remaining = self.recovery_days
        self._save_state()
        self._execute_shutdown()
        return True
    
    def _execute_shutdown(self) -> None:
        logger.critical("Executing shutdown sequence...")
        logger.critical("Step 1: Stopping new order acceptance...")
        logger.critical("Step 2: Cancelling pending orders...")
        self.state.orders_cancelled = True
        if self.close_positions:
            logger.critical("Step 3: Closing all positions...")
            self.state.positions_closed = True
        if self.shutdown_callback:
            logger.critical("Step 4: Calling shutdown callback...")
            try:
                self.shutdown_callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
        self._save_state()
        logger.critical("Shutdown sequence complete. System halted.")
    
    def reset(self, force: bool = False) -> bool:
        if not self.state.is_active:
            return True
        if not force and self.state.recovery_days_remaining > 0:
            logger.error(f"Cannot reset circuit breaker. {self.state.recovery_days_remaining} recovery days remaining.")
            return False
        if force and not self.allow_manual_override:
            logger.error("Manual override not allowed.")
            return False
        logger.info("Resetting circuit breaker...")
        self.state = CircuitBreakerState(
            is_active=False, trigger_reason=None, trigger_time=None, trigger_value=None, threshold=None, recovery_days_remaining=0
        )
        self._save_state()
        logger.info("Circuit breaker reset successfully.")
        return True
    
    def is_trading_allowed(self) -> bool:
        if self.state.is_active:
            logger.warning(f"Trading NOT allowed. Circuit breaker active. Reason: {self.state.trigger_reason}")
            return False
        return True



@dataclass
class Position:
    """Position representation"""
    symbol: str
    sector: str
    quantity: int
    entry_price: float
    current_price: float
    side: str


@dataclass
class RiskMetrics:
    """Risk metrics output"""
    var: float  # Value at Risk (Parametric Cornish-Fisher)
    var_historical: float  # Historical Simulation VaR
    var_evt: float  # Extreme Value Theory VaR
    cvar: float  # Conditional Value at Risk
    l_var: float  # Liquidity-adjusted VaR
    vol_target_multiplier: float
    kelly_fractions: Dict[str, float]
    position_limits: Dict[str, float]
    portfolio_heat: float
    tail_risk: float
    circuit_breaker_active: bool
    daily_pnl_pct: float  # Daily PnL percentage
    weekly_pnl_pct: float  # Weekly PnL percentage


class InstitutionalRiskEngine:
    """
    Institutional Risk Engine for Indian Markets (Architecture V2).
    Delegates calculation implementation to metrics.py and limits.py modules.
    """
    
    def __init__(
        self,
        capital: float = 2.5e8,  # ₹25 Crore (Architecture V2 target)
        risk_target: float = 0.15,  # 15% annual vol
        var_confidence: float = 0.99,
        cvar_confidence: float = 0.95,
        max_leverage: float = 1.0,
        confidence_level: Optional[float] = None,
        shutdown_callback: Optional[Callable] = None
    ):
        self.capital = capital
        self.risk_target = risk_target
        self.var_confidence = confidence_level if confidence_level is not None else var_confidence
        self.cvar_confidence = cvar_confidence
        self.max_leverage = max_leverage
        
        # Average daily volume for liquidity adjustment (₹)
        self.adv_data = {
            'NIFTY': 5e10,  # ₹5000 Cr
            'BANKNIFTY': 3e10,  # ₹3000 Cr
            'RELIANCE': 2e9,  # ₹200 Cr
            'HDFCBANK': 1.5e9,  # ₹150 Cr
            'INFY': 1e9,  # ₹100 Cr
        }
        
        # Sector limits (Architecture V2)
        self.sector_limits = {
            'BANKNIFTY': 0.30,
            'NIFTY': 0.30,
            'IT': 0.30,
            'PHARMA': 0.30,
            'AUTO': 0.30,
            'FMCG': 0.30,
            'ENERGY': 0.30,
            'METALS': 0.30
        }
        
        # Position limits (Architecture V2)
        self.max_position_pct = 0.05  # 5% per position
        self.max_strategy_weight = 0.50  # 50% max single strategy weight
        self.warn_leverage = 3.0  # Warn at 3x
        
        # Risk limits (Architecture V2)
        self.risk_per_trade = 0.005  # 0.5% risk per trade
        self.risk_per_strategy = 0.05  # 5% risk per strategy
        self.total_portfolio_risk = 0.15  # 15% total portfolio at risk
        self.max_daily_loss_pct = 0.03  # 3% daily circuit breaker
        self.max_weekly_loss_pct = 0.08  # 8% weekly circuit breaker
        self.var_cap = 0.02  # VaR cap at 2% of AUM
        
        # Tail hedging parameters
        self.enable_tail_hedging = True
        self.vix_threshold = 12.0  # Buy OTM puts when VIX < 12
        self.tail_hedge_pct = 0.01  # 1% of AUM for tail hedge
        
        # High VIX stop trading (CRITICAL FIX)
        self.enable_high_vix_stop = True
        self.high_vix_threshold = 25.0  # Stop trading when VIX > 25
        self.high_vix_reduction = 0.25  # Reduce position size to 25% (75% reduction)
        
        # Trailing max drawdown limit (CRITICAL FIX)
        self.enable_trailing_dd_limit = True
        self.max_dd_from_peak_pct = 0.10  # 10% max drawdown from peak
        self.current_peak_equity = self.capital
        self.in_recovery_mode = False
        
        # Stop losses (CRITICAL FIX)
        self.enable_stop_losses = True
        self.stop_loss_atr_multiplier = 2.0  # 2x ATR from entry
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_recovery_days = 0
        self.daily_pnl_history = []
        self.weekly_pnl_history = []

        # Hard circuit breaker
        self.hard_breaker = HardCircuitBreaker(
            max_daily_loss_pct=self.max_daily_loss_pct,
            max_weekly_loss_pct=self.max_weekly_loss_pct,
            max_drawdown_pct=self.max_dd_from_peak_pct,
            vix_threshold=self.high_vix_threshold,
            shutdown_callback=shutdown_callback,
            state_file="circuit_breaker_state.json"
        )

    def compute_returns(self, prices) -> pd.Series:
        """
        Compute finite log returns from a price series.
        
        CRITICAL FIX: Handles multiple input types (DataFrame, Series, ndarray, list)
        and converts (n,1) shaped arrays to (n,) to avoid pandas shape errors.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Convert input to Series, handling different shapes
        if isinstance(prices, pd.DataFrame):
            if len(prices.columns) > 1:
                raise ValueError(f"compute_returns expects single-column DataFrame, got {len(prices.columns)} columns")
            # Extract single column and squeeze to 1D
            prices = prices.iloc[:, 0].squeeze()
            logger.debug(f"Converted DataFrame column to Series, shape: {prices.shape}")
        elif isinstance(prices, pd.Series):
            prices = prices.squeeze()
            logger.debug(f"Input is Series, shape: {prices.shape}")
        elif isinstance(prices, np.ndarray):
            if prices.ndim == 2:
                if prices.shape[1] > 1:
                    raise ValueError(f"compute_returns expects 2D array with shape (n,1), got {prices.shape}")
                prices = prices.squeeze()
                logger.debug(f"Squeezed ndarray from {prices.shape} to 1D")
            elif prices.ndim == 1:
                logger.debug(f"Input is 1D ndarray, shape: {prices.shape}")
            else:
                raise ValueError(f"compute_returns expects 1D or 2D array, got {prices.ndim}D")
        elif isinstance(prices, list):
            prices = np.array(prices).squeeze()
            logger.debug(f"Converted list to ndarray, shape: {prices.shape}")
        else:
            raise TypeError(f"compute_returns expects DataFrame, Series, ndarray, or list, got {type(prices)}")
        
        # Convert to Series and validate
        prices = pd.Series(prices).astype(float)
        if len(prices.shape) != 1:
            raise ValueError(f"After conversion, prices should be 1D, got shape {prices.shape}")
        
        # Compute returns
        returns = np.log(prices / prices.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        logger.debug(f"Computed {len(returns)} returns from {len(prices)} prices")
        return returns

    def _clean_returns(self, returns: pd.Series) -> pd.Series:
        returns = pd.Series(returns).astype(float)
        return returns.replace([np.inf, -np.inf], np.nan).dropna()

    def _loss_from_return(self, portfolio_value: float, return_quantile: float) -> float:
        if return_quantile >= 0:
            return 0.0
        return float(portfolio_value * (1 - np.exp(return_quantile)))

    def compute_var(self, portfolio_value: float, returns: pd.Series) -> float:
        """Calculate Value at Risk using Cornish-Fisher expansion."""
        returns_arr = np.asarray(returns, dtype=float)
        confidence = getattr(self, "var_confidence", 0.99)
        return metrics_calculate_var(returns_arr, portfolio_value, confidence, use_cornish_fisher=True)

    def compute_historical_var(self, portfolio_value: float, returns: pd.Series) -> float:
        """Calculate Historical Simulation VaR."""
        returns_arr = np.asarray(returns, dtype=float)
        confidence = getattr(self, "var_confidence", 0.99)
        return metrics_calculate_var_historical(returns_arr, portfolio_value, confidence)

    def compute_weibull_var(self, portfolio_value: float, returns: pd.Series) -> float:
        """Heavy-tail VaR fitted to negative returns with a mirrored Weibull tail."""
        from scipy import stats
        returns = self._clean_returns(returns)
        tail = -returns[returns < 0]
        if len(tail) < 50:
            confidence = getattr(self, "var_confidence", 0.99)
            scaled_returns = returns * np.sqrt(1)
            var_percentile = np.percentile(scaled_returns, 100 * (1 - confidence))
            return self._loss_from_return(portfolio_value, var_percentile)

        shape, loc, scale = stats.weibull_min.fit(tail, floc=0)
        tail_loss = stats.weibull_min.ppf(getattr(self, "var_confidence", 0.99), shape, loc=loc, scale=scale)
        return float(portfolio_value * (1 - np.exp(-tail_loss * np.sqrt(1))))

    def compute_cvar(self, portfolio_value: float, returns: pd.Series) -> float:
        """Conditional VaR (expected shortfall)."""
        returns = self._clean_returns(returns)
        confidence = getattr(self, "var_confidence", 0.99)
        if len(returns) < 100:
            mu = returns.mean()
            sigma = returns.std()
            from scipy import stats
            var_fraction = mu + stats.norm.ppf(1 - confidence) * sigma
            return self._loss_from_return(portfolio_value, var_fraction)
        threshold = np.percentile(returns, 100 * (1 - confidence))
        tail_returns = returns[returns <= threshold]
        if len(tail_returns) == 0:
            mu = returns.mean()
            sigma = returns.std()
            from scipy import stats
            var_fraction = mu + stats.norm.ppf(1 - confidence) * sigma
            return self._loss_from_return(portfolio_value, var_fraction)
        cvar_fraction = tail_returns.mean()
        return self._loss_from_return(portfolio_value, cvar_fraction)

    def tail_risk(self, portfolio_value: float, returns: pd.Series, tail_percent=0.05) -> float:
        """Average loss in worst tail_percent of days."""
        returns = self._clean_returns(returns)
        confidence = getattr(self, "var_confidence", 0.99)
        if len(returns) < 100:
            mu = returns.mean()
            sigma = returns.std()
            from scipy import stats
            var_fraction = mu + stats.norm.ppf(1 - confidence) * sigma
            return self._loss_from_return(portfolio_value, var_fraction)
        n_tail = max(1, int(len(returns) * tail_percent))
        tail = returns.nsmallest(n_tail)
        tail_loss = tail.mean()
        return self._loss_from_return(portfolio_value, tail_loss)

    def update_daily_pnl(self, daily_pnl: float) -> None:
        self.daily_pnl_history.append(daily_pnl / self.capital)
    
    def calculate_portfolio_returns(
        self,
        positions: List[Position],
        market_data: pd.DataFrame
    ) -> np.ndarray:
        return metrics_calculate_portfolio_returns(positions, market_data, self.capital)
    
    def calculate_moments(self, returns: np.ndarray) -> Tuple[float, float, float, float]:
        return metrics_calculate_moments(returns)
    
    def calculate_var(
        self,
        returns: np.ndarray,
        use_cornish_fisher: bool = True
    ) -> float:
        return metrics_calculate_var(returns, self.capital, self.var_confidence, use_cornish_fisher)
    
    def calculate_var_historical(self, returns: np.ndarray) -> float:
        return metrics_calculate_var_historical(returns, self.capital, self.var_confidence)
    
    def calculate_var_evt(self, returns: np.ndarray, threshold_percentile: float = 0.90) -> float:
        return metrics_calculate_var_evt(returns, self.capital, self.var_confidence, threshold_percentile)
    
    def calculate_cvar(self, returns: np.ndarray) -> float:
        return metrics_calculate_cvar(returns, self.capital, self.cvar_confidence)
    
    def calculate_liquidity_adjusted_var(
        self,
        positions: List[Position],
        returns: np.ndarray
    ) -> float:
        return metrics_calculate_liquidity_adjusted_var(positions, returns, self.capital, self.var_confidence, self.adv_data)
    
    def calculate_volatility_target_multiplier(
        self,
        returns: np.ndarray
    ) -> float:
        return metrics_calculate_volatility_target_multiplier(returns, self.risk_target)
    
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        win_rate: float,
        win_loss_ratio: float,
        entry_price: float,
        signal_strength: float
    ) -> float:
        return metrics_calculate_kelly_fraction(
            expected_return=expected_return,
            win_rate=win_rate,
            avg_win=win_loss_ratio * 0.015,
            avg_loss=0.015,
            volatility=0.15
        )
    
    def calculate_position_limits(
        self,
        positions: List[Position]
    ) -> Dict[str, float]:
        return limits_calculate_position_limits(positions, self.capital, self.max_position_pct, self.sector_limits)
    
    def calculate_portfolio_heat(
        self,
        positions: List[Position],
        market_data: pd.DataFrame
    ) -> float:
        return metrics_calculate_portfolio_heat(positions, market_data)
    
    def calculate_tail_risk(self, returns: np.ndarray, percentile: float = 0.15) -> float:
        return metrics_calculate_tail_risk(returns, self.capital, percentile)
    
    def should_tail_hedge(self, vix: float) -> Tuple[bool, float]:
        if not self.enable_tail_hedging:
            return False, 0.0
        if vix < self.vix_threshold:
            hedge_size = self.capital * self.tail_hedge_pct
            return True, hedge_size
        return False, 0.0
    
    def get_tail_hedge_signal(self, vix: float, underlying_price: float = 20000) -> Dict:
        should_hedge, hedge_size = self.should_tail_hedge(vix)
        if not should_hedge:
            return {
                "should_hedge": False,
                "reason": f"VIX ({vix:.2f}) above threshold ({self.vix_threshold})"
            }
        otm_pct = 0.05
        put_strike = underlying_price * (1 - otm_pct)
        premium = hedge_size * 0.02
        return {
            "should_hedge": True,
            "reason": f"VIX ({vix:.2f}) below threshold ({self.vix_threshold})",
            "hedge_size": hedge_size,
            "hedge_type": "OTM_PUT",
            "strike": put_strike,
            "premium": premium,
            "otm_pct": otm_pct,
            "underlying_price": underlying_price
        }
    
    def should_stop_trading_high_vix(self, vix: float) -> Tuple[bool, float]:
        return limits_should_stop_trading_high_vix(vix, self.enable_high_vix_stop, self.high_vix_threshold, self.high_vix_reduction)
    
    def check_trailing_drawdown_limit(self, current_equity: float) -> Tuple[bool, float]:
        should_stop, drawdown_pct, new_peak, in_recovery = limits_check_trailing_drawdown_limit(
            current_equity, self.capital, self.current_peak_equity, self.in_recovery_mode,
            self.enable_trailing_dd_limit, self.max_dd_from_peak_pct
        )
        self.current_peak_equity = new_peak
        self.in_recovery_mode = in_recovery
        
        if should_stop:
            self.hard_breaker.trigger(
                reason=CircuitBreakerTrigger.DRAWDOWN,
                value=drawdown_pct,
                threshold=self.max_dd_from_peak_pct,
                message=f"Drawdown of {drawdown_pct:.2%} exceeds threshold of {self.max_dd_from_peak_pct:.2%}"
            )
            self.circuit_breaker_active = True
            self.circuit_breaker_recovery_days = self.hard_breaker.state.recovery_days_remaining
            
        return should_stop, drawdown_pct
 
    def check_circuit_breaker(self, daily_pnl: float, weekly_pnl: Optional[float] = None) -> Tuple[bool, str]:
        daily_pnl_pct = daily_pnl / self.capital
        weekly_pnl_pct = weekly_pnl / self.capital if weekly_pnl is not None else 0.0
        
        triggered = False
        reason = ""
        
        if daily_pnl_pct < -self.max_daily_loss_pct:
            triggered = self.hard_breaker.trigger(
                reason=CircuitBreakerTrigger.DAILY_LOSS,
                value=daily_pnl_pct,
                threshold=-self.max_daily_loss_pct,
                message=f"Daily loss of {daily_pnl_pct:.2%} exceeds threshold of {-self.max_daily_loss_pct:.2%}"
            )
            reason = "daily_loss_exceeded"
        elif weekly_pnl is not None and weekly_pnl_pct < -self.max_weekly_loss_pct:
            triggered = self.hard_breaker.trigger(
                reason=CircuitBreakerTrigger.WEEKLY_LOSS,
                value=weekly_pnl_pct,
                threshold=-self.max_weekly_loss_pct,
                message=f"Weekly loss of {weekly_pnl_pct:.2%} exceeds threshold of {-self.max_weekly_loss_pct:.2%}"
            )
            reason = "weekly_loss_exceeded"
            
        if triggered or self.hard_breaker.state.is_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_recovery_days = self.hard_breaker.state.recovery_days_remaining
            return True, self.hard_breaker.state.trigger_reason or reason
            
        # Fallback to legacy checks
        triggered_legacy, reason_legacy, recovery_days = limits_check_circuit_breaker(
            daily_pnl, self.capital, self.max_daily_loss_pct, weekly_pnl, self.max_weekly_loss_pct,
            self.circuit_breaker_active, self.circuit_breaker_recovery_days
        )
        if triggered_legacy:
            self.circuit_breaker_active = True
            self.circuit_breaker_recovery_days = recovery_days
            self.daily_pnl_history.append(daily_pnl / self.capital)
        return triggered_legacy, reason_legacy

    def end_of_day_update(self) -> None:
        if self.circuit_breaker_active and self.circuit_breaker_recovery_days > 0:
            self.circuit_breaker_recovery_days -= 1
            if self.circuit_breaker_recovery_days == 0:
                self.circuit_breaker_active = False

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str = "long"
    ) -> Optional[float]:
        return limits_calculate_stop_loss(entry_price, atr, self.stop_loss_atr_multiplier, direction, self.enable_stop_losses)
    
    def calculate_risk_metrics(
        self,
        positions: List[Position],
        market_data: pd.DataFrame,
        daily_pnl: float = 0.0,
        weekly_pnl: Optional[float] = None
    ) -> RiskMetrics:
        portfolio_returns = self.calculate_portfolio_returns(positions, market_data)
        
        if len(portfolio_returns) == 0:
            return RiskMetrics(
                var=0.0, var_historical=0.0, var_evt=0.0, cvar=0.0, l_var=0.0,
                vol_target_multiplier=1.0, kelly_fractions={}, position_limits={},
                portfolio_heat=0.0, tail_risk=0.0, circuit_breaker_active=False,
                daily_pnl_pct=0.0, weekly_pnl_pct=0.0
            )
        
        var = min(self.calculate_var(portfolio_returns), self.capital * self.var_cap)
        var_historical = min(self.calculate_var_historical(portfolio_returns), self.capital * self.var_cap)
        var_evt = min(self.calculate_var_evt(portfolio_returns), self.capital * self.var_cap * 1.2)
        l_var = min(self.calculate_liquidity_adjusted_var(positions, portfolio_returns), self.capital * self.var_cap * 1.5)
        cvar = self.calculate_cvar(portfolio_returns)
        vol_mult = self.calculate_volatility_target_multiplier(portfolio_returns)
        
        kelly_fractions = {}
        for pos in positions:
            kelly_fractions[pos.symbol] = self.calculate_kelly_fraction(0.001, 0.55, 1.33, 45000, 1.0)
            
        position_limits = self.calculate_position_limits(positions)
        portfolio_heat = self.calculate_portfolio_heat(positions, market_data)
        tail_risk = self.calculate_tail_risk(portfolio_returns)
        
        daily_pnl_pct = daily_pnl / self.capital
        weekly_pnl_pct = weekly_pnl / self.capital if weekly_pnl is not None else 0.0
        
        return RiskMetrics(
            var=var,
            var_historical=var_historical,
            var_evt=var_evt,
            cvar=cvar,
            l_var=l_var,
            vol_target_multiplier=vol_mult,
            kelly_fractions=kelly_fractions,
            position_limits=position_limits,
            portfolio_heat=portfolio_heat,
            tail_risk=tail_risk,
            circuit_breaker_active=self.circuit_breaker_active,
            daily_pnl_pct=daily_pnl_pct,
            weekly_pnl_pct=weekly_pnl_pct
        )
