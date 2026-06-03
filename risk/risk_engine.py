"""
Enhanced Institutional Risk Management Engine
Based on Blueprint V1.0

Implements:
- VaR (Value at Risk) - Parametric and Historical
- CVaR (Conditional Value at Risk) - Expected Shortfall
- Kelly Criterion - Optimal position sizing
- Volatility Targeting - Dynamic position scaling
- Risk Parity - Equal risk contribution
- Stress Testing - Scenario analysis
- Circuit Breakers - Automatic risk limits
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    APPROVE = "approve"
    REDUCE_SIZE = "reduce_size"
    REJECT = "reject"
    FORCE_LIQUIDATE = "force_liquidate"


@dataclass
class RiskCheckResult:
    action: RiskAction
    reasons: List[str]
    adjusted_quantity: Optional[float] = None
    risk_metrics: Optional[Dict] = None


class RiskEngine:
    """
    Enhanced Institutional Risk Management Engine.
    
    Features:
    - VaR (Parametric & Historical)
    - CVaR (Conditional Value at Risk / Expected Shortfall)
    - Kelly Criterion position sizing
    - Volatility targeting
    - Risk parity optimization
    - Stress testing
    - Circuit breakers
    - Sector concentration limits
    - Drawdown monitoring
    - Correlation limits
    """

    def __init__(self, config: dict):
        self.config = config
        risk_config = config.get("risk", {})

        # VaR limits
        self.max_portfolio_var_95 = risk_config.get("max_portfolio_var_95", 0.02)
        self.max_portfolio_var_99 = risk_config.get("max_portfolio_var_99", 0.03)
        self.max_cvar_95 = risk_config.get("max_cvar_95", 0.025)
        
        # Position limits
        self.max_position_size_pct = risk_config.get("max_position_size_pct", 0.05)
        self.max_gross_exposure = risk_config.get("max_gross_exposure", 3.0)
        self.max_net_exposure = risk_config.get("max_net_exposure", 1.5)
        
        # Volatility targeting
        self.volatility_target = risk_config.get("volatility_target", 0.15)
        self.max_leverage = risk_config.get("max_leverage", 4.0)
        
        # Drawdown limits
        self.max_drawdown = risk_config.get("max_drawdown", 0.10)
        self.max_daily_drawdown = risk_config.get("max_daily_drawdown", 0.05)
        self.max_rolling_drawdown_20d = risk_config.get("max_rolling_drawdown_20d", 0.15)
        
        # Kelly Criterion
        self.use_kelly = risk_config.get("use_kelly", True)
        self.kelly_fraction_cap = risk_config.get("kelly_fraction_cap", 0.25)
        
        # Correlation and sector limits
        self.correlation_limit = risk_config.get("correlation_limit", 0.7)
        self.sector_concentration = risk_config.get("sector_concentration", 0.25)
        self.max_single_symbol = risk_config.get("max_single_symbol", 0.10)
        
        # Daily loss limits
        self.daily_loss_limit = risk_config.get("daily_loss_limit", 0.03)
        self.weekly_loss_limit = risk_config.get("weekly_loss_limit", 0.10)
        
        # Circuit breakers
        self.consecutive_losses_limit = risk_config.get("consecutive_losses_limit", 5)
        self.consecutive_losses = 0
        self.weekly_loss = 0.0

        # Portfolio state
        self.portfolio_value = risk_config.get("initial_capital", 1_000_000.0)
        self.current_positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.peak_equity = self.portfolio_value
        self.current_drawdown = 0.0

        # Historical data for risk calculations
        self._historical_returns: List[float] = []
        self._sector_exposure: Dict[str, float] = {}
        self._covariance_matrix: Optional[np.ndarray] = None
        self._asset_returns: Dict[str, List[float]] = {}

    def pre_trade_check(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        sector: str = "Unknown",
        strategy: str = "",
        regime_multiplier: float = 1.0
    ) -> Tuple[RiskAction, Dict]:
        """
        Perform pre-trade risk checks.

        Args:
            symbol: Stock symbol
            direction: "long" or "short"
            quantity: Order quantity
            price: Entry price
            sector: Stock sector
            strategy: Strategy name
            regime_multiplier: Regime-based position size multiplier

        Returns:
            (action, info) tuple
        """
        reasons = []
        adjusted_quantity = quantity
        risk_metrics = {}

        # 1. Position size limit
        position_value = quantity * price
        max_position_value = self.portfolio_value * self.max_position_size_pct

        if position_value > max_position_value:
            adjusted_quantity = max_position_value / price
            reasons.append(f"Position size reduced from {quantity:.0f} to {adjusted_quantity:.0f} due to limit")

        # 2. Sector concentration limit
        current_sector_exposure = self._sector_exposure.get(sector, 0.0)
        new_position_value = adjusted_quantity * price
        new_sector_exposure = current_sector_exposure + new_position_value
        max_sector_exposure = self.portfolio_value * self.sector_concentration

        if new_sector_exposure > max_sector_exposure:
            allowed_sector_value = max_sector_exposure - current_sector_exposure
            adjusted_quantity = min(adjusted_quantity, allowed_sector_value / price)
            reasons.append(f"Position size reduced due to sector concentration limit")

        # 3. Daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.portfolio_value if self.daily_pnl < 0 else 0
        if daily_loss_pct > self.daily_loss_limit:
            return RiskAction.REJECT, {"reason": "Daily loss limit exceeded"}

        # 4. Drawdown limit
        if self.current_drawdown > self.max_drawdown:
            return RiskAction.FORCE_LIQUIDATE, {"reason": "Max drawdown exceeded"}

        # 5. Regime adjustment
        adjusted_quantity *= regime_multiplier

        # 6. Correlation check (if we have existing positions)
        if self.current_positions:
            correlation_risk = self._check_correlation(symbol, sector)
            if correlation_risk > self.correlation_limit:
                adjusted_quantity *= 0.5  # Reduce size by 50%
                reasons.append(f"Position size reduced due to high correlation")

        # 7. Portfolio VaR check
        portfolio_var = self._calculate_portfolio_var()
        if portfolio_var > self.max_portfolio_var:
            adjusted_quantity *= 0.8  # Reduce size by 20%
            reasons.append(f"Position size reduced due to portfolio VaR limit")

        # Determine action
        if adjusted_quantity <= 0:
            return RiskAction.REJECT, {"reason": "Risk checks failed", "reasons": reasons}

        if adjusted_quantity < quantity * 0.5:
            return RiskAction.REDUCE_SIZE, {
                "reasons": reasons,
                "adjusted_quantity": adjusted_quantity,
                "risk_metrics": risk_metrics
            }

        return RiskAction.APPROVE, {
            "adjusted_quantity": adjusted_quantity,
            "risk_metrics": risk_metrics
        }

    def volatility_target_sizing(
        self,
        base_quantity: float,
        asset_volatility: float,
        regime_multiplier: float = 1.0
    ) -> float:
        """
        Adjust position size based on volatility targeting.

        Args:
            base_quantity: Base position quantity
            asset_volatility: Annualized volatility of the asset
            regime_multiplier: Regime-based multiplier

        Returns:
            Adjusted quantity
        """
        if asset_volatility == 0:
            return base_quantity

        # Volatility scaling: target_vol / asset_vol
        vol_scale = self.volatility_target / asset_volatility
        adjusted_quantity = base_quantity * vol_scale * regime_multiplier

        return max(adjusted_quantity, 1)  # Minimum 1 share

    def _calculate_portfolio_var(self, confidence_level: float = 0.95, method: str = "historical") -> float:
        """
        Calculate portfolio Value at Risk using parametric or historical method.
        
        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)
            method: "historical" or "parametric"
            
        Returns:
            Portfolio VaR as percentage of portfolio value
        """
        if not self._historical_returns or len(self._historical_returns) < 20:
            return 0.0
        
        returns = np.array(self._historical_returns[-252:])
        if len(returns) < 20:
            returns = np.array(self._historical_returns)
        
        if method == "historical":
            # Historical simulation VaR
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return abs(var)
        elif method == "parametric":
            # Parametric VaR (assuming normal distribution)
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean - z_score * std
            return abs(var)
        else:
            return 0.0
    
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        CVaR is the average loss beyond the VaR threshold.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
            
        Returns:
            CVaR as percentage of portfolio value
        """
        if not self._historical_returns or len(self._historical_returns) < 20:
            return 0.0
        
        returns = np.array(self._historical_returns[-252:])
        if len(returns) < 20:
            returns = np.array(self._historical_returns)
        
        # Calculate VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        
        # CVaR = average of returns below VaR threshold
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = abs(np.mean(tail_losses))
        return cvar
    
    def calculate_kelly_fraction(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction for optimal position sizing.
        
        Kelly Formula: f* = (bp - q) / b
        where:
        - b = avg_win / avg_loss (win/loss ratio)
        - p = win probability
        - q = 1 - p (loss probability)
        
        Args:
            win_prob: Probability of winning (0 to 1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive)
            
        Returns:
            Kelly fraction (capped at self.kelly_fraction_cap)
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Cap at configured maximum
        kelly_fraction = max(0, min(kelly_fraction, self.kelly_fraction_cap))
        
        return kelly_fraction
    
    def calculate_kelly_from_history(self, returns: List[float]) -> float:
        """
        Calculate Kelly fraction from historical returns.
        
        Args:
            returns: List of historical returns
            
        Returns:
            Kelly fraction
        """
        if len(returns) < 10:
            return 0.01  # Conservative default
        
        returns_array = np.array(returns)
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.01
        
        win_prob = len(wins) / len(returns_array)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return self.calculate_kelly_fraction(win_prob, avg_win, avg_loss)

    def _check_correlation(self, symbol: str, sector: str) -> float:
        """
        Check correlation with existing positions.

        Returns:
            Maximum correlation with existing positions
        """
        # Simplified: use sector as proxy for correlation
        # In production, use actual correlation matrix
        sector_exposure = self._sector_exposure.get(sector, 0.0)
        total_exposure = sum(self._sector_exposure.values())

        if total_exposure == 0:
            return 0.0

        sector_concentration = sector_exposure / total_exposure
        return sector_concentration

    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        sector: str = "Unknown"
    ) -> None:
        """
        Update position after trade execution.

        Args:
            symbol: Stock symbol
            quantity: Position quantity (positive for long, negative for short)
            price: Current price
            sector: Stock sector
        """
        position_value = abs(quantity) * price

        if symbol in self.current_positions:
            old_quantity = self.current_positions[symbol]["quantity"]
            old_sector = self.current_positions[symbol]["sector"]
            old_value = abs(old_quantity) * price

            # Update sector exposure
            self._sector_exposure[old_sector] = self._sector_exposure.get(old_sector, 0.0) - old_value

        # Update position
        self.current_positions[symbol] = {
            "quantity": quantity,
            "price": price,
            "sector": sector,
            "value": position_value,
        }

        # Update sector exposure
        self._sector_exposure[sector] = self._sector_exposure.get(sector, 0.0) + position_value

    def close_position(self, symbol: str, price: float) -> None:
        """
        Close a position.

        Args:
            symbol: Stock symbol
            price: Closing price
        """
        if symbol not in self.current_positions:
            return

        position = self.current_positions[symbol]
        quantity = position["quantity"]
        sector = position["sector"]
        old_price = position["price"]

        # Calculate PnL
        if quantity > 0:  # Long
            pnl = (price - old_price) * quantity
        else:  # Short
            pnl = (old_price - price) * abs(quantity)

        # Update daily PnL
        self.daily_pnl += pnl
        self.weekly_loss += pnl

        # Update consecutive losses counter
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update portfolio value
        self.portfolio_value += pnl

        # Update sector exposure
        position_value = abs(quantity) * price
        self._sector_exposure[sector] = self._sector_exposure.get(sector, 0.0) - position_value

        # Remove position
        del self.current_positions[symbol]

        # Update drawdown
        if self.portfolio_value > self.peak_equity:
            self.peak_equity = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - self.portfolio_value) / self.peak_equity

        logger.info(
            f"Closed position {symbol}: PnL={pnl:.2f}, "
            f"Portfolio={self.portfolio_value:.2f}, Drawdown={self.current_drawdown:.2%}"
        )

    def update_daily_returns(self, portfolio_return: float) -> None:
        """
        Update historical returns for VaR calculation.

        Args:
            portfolio_return: Daily portfolio return
        """
        self._historical_returns.append(portfolio_return)

        # Keep only last 252 returns (1 year)
        if len(self._historical_returns) > 252:
            self._historical_returns = self._historical_returns[-252:]

    def reset_daily(self) -> None:
        """Reset daily metrics (called at start of trading day)."""
        self.daily_pnl = 0.0
    
    def reset_weekly(self) -> None:
        """Reset weekly metrics (called at start of trading week)."""
        self.weekly_loss = 0.0
        self.consecutive_losses = 0
    
    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """
        Check if any circuit breakers should be triggered.
        
        Returns:
            (should_halt, reason) tuple
        """
        # Consecutive losses
        if self.consecutive_losses >= self.consecutive_losses_limit:
            return True, f"Circuit breaker: {self.consecutive_losses} consecutive losses"
        
        # Daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.portfolio_value if self.daily_pnl < 0 else 0
        if daily_loss_pct > self.daily_loss_limit:
            return True, f"Circuit breaker: Daily loss {daily_loss_pct:.2%} exceeds limit {self.daily_loss_limit:.2%}"
        
        # Weekly loss limit
        weekly_loss_pct = abs(self.weekly_loss) / self.portfolio_value if self.weekly_loss < 0 else 0
        if weekly_loss_pct > self.weekly_loss_limit:
            return True, f"Circuit breaker: Weekly loss {weekly_loss_pct:.2%} exceeds limit {self.weekly_loss_limit:.2%}"
        
        # Max drawdown
        if self.current_drawdown > self.max_drawdown:
            return True, f"Circuit breaker: Drawdown {self.current_drawdown:.2%} exceeds limit {self.max_drawdown:.2%}"
        
        return False, ""
    
    def stress_test(self, scenarios: Dict[str, float]) -> Dict[str, float]:
        """
        Perform stress testing on portfolio.
        
        Args:
            scenarios: Dictionary of scenario_name -> shock_percentage
                       e.g., {"crash": -0.20, "rally": 0.15}
                       
        Returns:
            Dictionary of scenario_name -> portfolio_value_after_shock
        """
        results = {}
        
        for scenario_name, shock in scenarios.items():
            # Apply shock to all positions
            shocked_value = self.portfolio_value * (1 + shock)
            results[scenario_name] = shocked_value
        
        return results
    
    def calculate_risk_parity_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk parity weights (equal risk contribution).
        
        Args:
            cov_matrix: Covariance matrix of asset returns
            
        Returns:
            Array of risk parity weights
        """
        n = cov_matrix.shape[0]
        
        def risk_parity_objective(weights):
            """Objective function for risk parity optimization."""
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib
            contrib_pct = contrib / portfolio_vol
            target = 1.0 / n
            return np.sum((contrib_pct - target) ** 2)
        
        # Constraints: weights sum to 1, non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess: equal weights
        w0 = np.ones(n) / n
        
        result = minimize(
            risk_parity_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else w0

    def get_risk_report(self) -> Dict:
        """
        Get comprehensive risk report with all enhanced metrics.

        Returns:
            Dictionary with all risk metrics
        """
        total_exposure = sum(abs(p["quantity"]) * p["price"] for p in self.current_positions.values())
        gross_exposure = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Calculate net exposure
        net_exposure = sum(p["quantity"] * p["price"] for p in self.current_positions.values())
        net_exposure_pct = net_exposure / self.portfolio_value if self.portfolio_value > 0 else 0

        # Circuit breaker check
        should_halt, halt_reason = self.check_circuit_breakers()

        return {
            "portfolio_value": self.portfolio_value,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            "weekly_loss": self.weekly_loss,
            "weekly_loss_pct": self.weekly_loss / self.portfolio_value if self.portfolio_value > 0 else 0,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure_pct,
            "num_positions": len(self.current_positions),
            "sector_exposure": self._sector_exposure.copy(),
            "portfolio_var_95": self._calculate_portfolio_var(0.95),
            "portfolio_var_99": self._calculate_portfolio_var(0.99),
            "portfolio_cvar_95": self.calculate_cvar(0.95),
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_triggered": should_halt,
            "circuit_breaker_reason": halt_reason,
            "positions": self.current_positions.copy(),
        }

    def force_liquidate_all(self) -> None:
        """Force liquidate all positions (emergency)."""
        logger.warning("FORCE LIQUIDATING ALL POSITIONS")
        self.current_positions.clear()
        self._sector_exposure.clear()
