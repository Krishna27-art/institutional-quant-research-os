"""
Risk Management Engine for the quantitative trading system.
Implements VaR, sector limits, drawdown control, and position sizing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    Comprehensive risk management engine.

    Features:
    - Portfolio VaR calculation
    - Position size limits
    - Sector concentration limits
    - Drawdown monitoring
    - Correlation limits
    - Daily loss limits
    - Volatility targeting
    """

    def __init__(self, config: dict):
        self.config = config
        risk_config = config.get("risk", {})

        self.max_portfolio_var = risk_config.get("max_portfolio_var", 0.02)
        self.max_position_size_pct = risk_config.get("max_position_size_pct", 0.05)
        self.volatility_target = risk_config.get("volatility_target", 0.15)
        self.max_drawdown = risk_config.get("max_drawdown", 0.10)
        self.correlation_limit = risk_config.get("correlation_limit", 0.7)
        self.sector_concentration = risk_config.get("sector_concentration", 0.25)
        self.daily_loss_limit = risk_config.get("daily_loss_limit", 0.03)

        self.portfolio_value = 1_000_000.0  # Default initial capital
        self.current_positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.peak_equity = self.portfolio_value
        self.current_drawdown = 0.0

        self._historical_returns: List[float] = []
        self._sector_exposure: Dict[str, float] = {}

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

    def _calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate portfolio Value at Risk.

        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)

        Returns:
            Portfolio VaR as percentage of portfolio value
        """
        if not self._historical_returns or len(self._historical_returns) < 20:
            return 0.0

        returns = np.array(self._historical_returns[-252:])  # Last year of returns
        if len(returns) < 20:
            returns = self._historical_returns

        # Historical VaR
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)

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

    def get_risk_report(self) -> Dict:
        """
        Get comprehensive risk report.

        Returns:
            Dictionary with all risk metrics
        """
        total_exposure = sum(abs(p["quantity"]) * p["price"] for p in self.current_positions.values())
        gross_exposure = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0

        return {
            "portfolio_value": self.portfolio_value,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
            "gross_exposure": gross_exposure,
            "num_positions": len(self.current_positions),
            "sector_exposure": self._sector_exposure.copy(),
            "portfolio_var": self._calculate_portfolio_var(),
            "positions": self.current_positions.copy(),
        }

    def force_liquidate_all(self) -> None:
        """Force liquidate all positions (emergency)."""
        logger.warning("FORCE LIQUIDATING ALL POSITIONS")
        self.current_positions.clear()
        self._sector_exposure.clear()
