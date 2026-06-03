"""
Robust Transaction Cost Model

Based on Comprehensive Upgrade Analysis - Tier 1 Upgrade (#6)
Expected Sharpe improvement: +0.2–0.3
Realistic slippage prevents over-trading

Methodology:
- Model commission, fees, and taxes (NSE-specific)
- Model market impact (nonlinear in order size)
- Model bid-ask spread cost
- Model timing risk (delay between signal and execution)
- Regime-dependent cost estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class TransactionCostConfig:
    """Configuration for Transaction Cost Model"""
    # NSE-specific costs
    broker_commission_bps: float = 5.0  # 5 bps broker commission
    stt_buy_bps: float = 0.0  # STT not applicable on buy
    stt_sell_bps: float = 10.0  # 10 bps STT on sell
    transaction_charge_bps: float = 1.0  # 1 bps transaction charge
    gst_bps: float = 18.0  # 18% GST on broker commission
    stamp_duty_bps: float = 0.003  # 0.003% stamp duty
    
    # Market impact parameters
    impact_alpha: float = 0.1  # Linear impact coefficient
    impact_beta: float = 0.5  # Nonlinear impact exponent
    impact_model: str = "sqrt"  # "sqrt" or "power"
    adv_window: int = 20  # Days for ADV calculation
    
    # Spread cost
    avg_spread_bps: float = 5.0  # Average bid-ask spread
    
    # Timing risk
    timing_decay_minutes: int = 5  # Signal decay over 5 minutes
    timing_cost_per_minute_bps: float = 1.0  # 1 bps per minute delay
    
    # Regime adjustments
    high_vol_multiplier: float = 1.5  # Cost multiplier in high vol regime
    low_vol_multiplier: float = 0.8  # Cost multiplier in low vol regime


@dataclass
class Order:
    """Order details"""
    symbol: str
    side: Side
    quantity: int
    price: float
    order_type: OrderType
    timestamp: datetime
    adv: float  # Average daily volume


@dataclass
class TransactionCost:
    """Transaction cost breakdown"""
    total_cost_bps: float
    commission_bps: float
    stt_bps: float
    transaction_charge_bps: float
    gst_bps: float
    stamp_duty_bps: float
    market_impact_bps: float
    spread_cost_bps: float
    timing_cost_bps: float
    regime_multiplier: float


class TransactionCostModel:
    """
    Robust Transaction Cost Model
    
    Models all costs associated with trading on NSE.
    Prevents over-trading by using realistic cost estimates.
    
    Components:
    1. Fixed costs (commission, fees, taxes)
    2. Market impact (nonlinear in order size)
    3. Spread cost
    4. Timing risk
    5. Regime-dependent adjustments
    """
    
    def __init__(self, config: TransactionCostConfig):
        self.config = config
        
        # Cost history
        self.cost_history: List[TransactionCost] = []
        
        # Regime state
        self.current_regime: str = "normal"
    
    def set_regime(self, regime: str) -> None:
        """Set current market regime"""
        self.current_regime = regime
    
    def compute_cost(self, order: Order, execution_delay_minutes: float = 0) -> TransactionCost:
        """
        Compute total transaction cost for an order
        
        Args:
            order: Order details
            execution_delay_minutes: Delay between signal and execution
            
        Returns:
            Transaction cost breakdown
        """
        # 1. Fixed costs (NSE-specific)
        commission = self._compute_commission(order)
        stt = self._compute_stt(order)
        transaction_charge = self._compute_transaction_charge(order)
        gst = self._compute_gst(commission)
        stamp_duty = self._compute_stamp_duty(order)
        
        # 2. Market impact (nonlinear)
        market_impact = self._compute_market_impact(order)
        
        # 3. Spread cost
        spread_cost = self._compute_spread_cost(order)
        
        # 4. Timing cost
        timing_cost = self._compute_timing_cost(order, execution_delay_minutes)
        
        # 5. Regime multiplier
        regime_multiplier = self._get_regime_multiplier()
        
        # Apply regime multiplier to variable costs
        market_impact *= regime_multiplier
        spread_cost *= regime_multiplier
        timing_cost *= regime_multiplier
        
        # Total cost
        total_cost = (commission + stt + transaction_charge + gst + 
                     stamp_duty + market_impact + spread_cost + timing_cost)
        
        cost = TransactionCost(
            total_cost_bps=total_cost,
            commission_bps=commission,
            stt_bps=stt,
            transaction_charge_bps=transaction_charge,
            gst_bps=gst,
            stamp_duty_bps=stamp_duty,
            market_impact_bps=market_impact,
            spread_cost_bps=spread_cost,
            timing_cost_bps=timing_cost,
            regime_multiplier=regime_multiplier
        )
        
        self.cost_history.append(cost)
        
        return cost
    
    def _compute_commission(self, order: Order) -> float:
        """Compute broker commission"""
        return self.config.broker_commission_bps
    
    def _compute_stt(self, order: Order) -> float:
        """Compute Securities Transaction Tax"""
        if order.side == Side.BUY:
            return self.config.stt_buy_bps
        else:
            return self.config.stt_sell_bps
    
    def _compute_transaction_charge(self, order: Order) -> float:
        """Compute exchange transaction charge"""
        return self.config.transaction_charge_bps
    
    def _compute_gst(self, commission: float) -> float:
        """Compute GST on broker commission"""
        return commission * self.config.gst_bps / 100.0
    
    def _compute_stamp_duty(self, order: Order) -> float:
        """Compute stamp duty"""
        return self.config.stamp_duty_bps
    
    def _compute_market_impact(self, order: Order) -> float:
        """
        Compute market impact (nonlinear in order size)
        
        Formula (sqrt model): impact = alpha * sqrt(participation_rate)
        Formula (power model): impact = alpha * (participation_rate)^beta
        
        where participation_rate = order_size / ADV
        """
        participation_rate = order.quantity / order.adv
        
        if self.config.impact_model == "sqrt":
            # Square-root impact model (recommended by brutal diagnosis)
            impact = self.config.impact_alpha * np.sqrt(participation_rate)
        else:
            # Power law model
            impact = self.config.impact_alpha * (participation_rate ** self.config.impact_beta)
        
        return impact * 100  # Convert to bps
    
    def _compute_spread_cost(self, order: Order) -> float:
        """Compute spread cost (half the spread for market orders)"""
        if order.order_type == OrderType.MARKET:
            return self.config.avg_spread_bps / 2.0
        elif order.order_type == OrderType.LIMIT:
            # Limit orders may avoid spread cost if filled at mid
            return self.config.avg_spread_bps * 0.25  # Assume some spread cost
        else:
            # VWAP/TWAP: partial spread cost
            return self.config.avg_spread_bps * 0.5
    
    def _compute_timing_cost(self, order: Order, delay_minutes: float) -> float:
        """
        Compute timing cost due to execution delay
        
        Signal decays over time, leading to worse execution prices.
        """
        if delay_minutes <= 0:
            return 0.0
        
        # Linear decay model
        decay_factor = min(delay_minutes / self.config.timing_decay_minutes, 1.0)
        timing_cost = decay_factor * self.config.timing_cost_per_minute_bps * delay_minutes
        
        return timing_cost
    
    def _get_regime_multiplier(self) -> float:
        """Get regime-dependent cost multiplier"""
        if self.current_regime == "high_vol":
            return self.config.high_vol_multiplier
        elif self.current_regime == "low_vol":
            return self.config.low_vol_multiplier
        else:
            return 1.0
    
    def get_average_cost(self, n_recent: int = 100) -> Optional[float]:
        """Get average transaction cost over recent trades"""
        if not self.cost_history:
            return None
        
        recent_costs = self.cost_history[-n_recent:]
        avg_cost = np.mean([c.total_cost_bps for c in recent_costs])
        
        return avg_cost
    
    def get_cost_breakdown(self, n_recent: int = 100) -> Dict[str, float]:
        """Get average cost breakdown over recent trades"""
        if not self.cost_history:
            return {}
        
        recent_costs = self.cost_history[-n_recent:]
        
        breakdown = {
            "commission_bps": np.mean([c.commission_bps for c in recent_costs]),
            "stt_bps": np.mean([c.stt_bps for c in recent_costs]),
            "transaction_charge_bps": np.mean([c.transaction_charge_bps for c in recent_costs]),
            "gst_bps": np.mean([c.gst_bps for c in recent_costs]),
            "stamp_duty_bps": np.mean([c.stamp_duty_bps for c in recent_costs]),
            "market_impact_bps": np.mean([c.market_impact_bps for c in recent_costs]),
            "spread_cost_bps": np.mean([c.spread_cost_bps for c in recent_costs]),
            "timing_cost_bps": np.mean([c.timing_cost_bps for c in recent_costs])
        }
        
        return breakdown


class PortfolioTransactionCost:
    """
    Portfolio-level transaction cost analysis
    
    Aggregates costs across all trades in a portfolio.
    """
    
    def __init__(self, cost_model: TransactionCostModel):
        self.cost_model = cost_model
        
        # Portfolio-level metrics
        self.total_trades: int = 0
        self.total_cost_bps: float = 0.0
        self.total_turnover: float = 0.0
    
    def execute_order(self, order: Order, execution_delay_minutes: float = 0) -> TransactionCost:
        """
        Execute order and record cost
        
        Args:
            order: Order details
            execution_delay_minutes: Execution delay
            
        Returns:
            Transaction cost
        """
        cost = self.cost_model.compute_cost(order, execution_delay_minutes)
        
        self.total_trades += 1
        self.total_cost_bps += cost.total_cost_bps
        self.total_turnover += order.quantity
        
        return cost
    
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio-level cost metrics"""
        if self.total_trades == 0:
            return {}
        
        avg_cost = self.total_cost_bps / self.total_trades
        
        return {
            "total_trades": self.total_trades,
            "average_cost_bps": avg_cost,
            "total_cost_bps": self.total_cost_bps,
            "total_turnover": self.total_turnover,
            "cost_per_turnover_bps": self.total_cost_bps / self.total_turnover if self.total_turnover > 0 else 0
        }


def simulate_trading(cost_model: TransactionCostModel, n_trades: int = 100) -> Dict:
    """Simulate trading to test cost model"""
    portfolio_cost = PortfolioTransactionCost(cost_model)
    
    # Simulate trades
    for i in range(n_trades):
        side = Side.BUY if np.random.random() > 0.5 else Side.SELL
        order_type = np.random.choice([OrderType.MARKET, OrderType.LIMIT, OrderType.VWAP])
        
        order = Order(
            symbol="RELIANCE",
            side=side,
            quantity=int(np.random.exponential(1000)),
            price=100.0 + np.random.randn(),
            order_type=order_type,
            timestamp=datetime.now(),
            adv=1000000  # 1M daily volume
        )
        
        execution_delay = np.random.exponential(2)  # Average 2 minute delay
        portfolio_cost.execute_order(order, execution_delay)
    
    return portfolio_cost.get_portfolio_metrics()


if __name__ == "__main__":
    # Example usage
    config = TransactionCostConfig(
        broker_commission_bps=5.0,
        stt_sell_bps=10.0,
        impact_alpha=0.1,
        impact_beta=0.5,
        avg_spread_bps=5.0
    )
    
    cost_model = TransactionCostModel(config)
    
    # Set regime
    cost_model.set_regime("normal")
    
    # Simulate a single order
    print("Computing cost for single order...")
    order = Order(
        symbol="RELIANCE",
        side=Side.BUY,
        quantity=10000,
        price=2500.0,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(),
        adv=5000000  # 5M daily volume
    )
    
    cost = cost_model.compute_cost(order, execution_delay_minutes=1.0)
    print(f"\nTransaction Cost Breakdown:")
    print(f"  Total Cost: {cost.total_cost_bps:.2f} bps")
    print(f"  Commission: {cost.commission_bps:.2f} bps")
    print(f"  STT: {cost.stt_bps:.2f} bps")
    print(f"  Transaction Charge: {cost.transaction_charge_bps:.2f} bps")
    print(f"  GST: {cost.gst_bps:.2f} bps")
    print(f"  Stamp Duty: {cost.stamp_duty_bps:.4f} bps")
    print(f"  Market Impact: {cost.market_impact_bps:.2f} bps")
    print(f"  Spread Cost: {cost.spread_cost_bps:.2f} bps")
    print(f"  Timing Cost: {cost.timing_cost_bps:.2f} bps")
    print(f"  Regime Multiplier: {cost.regime_multiplier:.2f}x")
    
    # Simulate multiple trades
    print(f"\nSimulating 100 trades...")
    metrics = simulate_trading(cost_model, n_trades=100)
    print(f"\nPortfolio Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test regime effects
    print(f"\n=== Regime Effects ===")
    cost_model.set_regime("high_vol")
    cost_high = cost_model.compute_cost(order, execution_delay_minutes=1.0)
    print(f"High Vol Regime: {cost_high.total_cost_bps:.2f} bps")
    
    cost_model.set_regime("low_vol")
    cost_low = cost_model.compute_cost(order, execution_delay_minutes=1.0)
    print(f"Low Vol Regime: {cost_low.total_cost_bps:.2f} bps")
