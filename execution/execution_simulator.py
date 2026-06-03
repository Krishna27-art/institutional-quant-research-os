"""
Execution Simulator for Realistic Cost Modeling
Based on the critique: Ensure backtests include realistic transaction costs

Critical for accurate backtesting:
- Indian market specific costs (STT, stamp duty, brokerage)
- Slippage models
- Market impact
- Liquidity constraints
- Order book depth simulation

Features:
- Realistic cost model for Indian markets
- Slippage based on volume participation
- Market impact for large orders
- Liquidity constraints
- Multi-venue execution simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"


@dataclass
class Order:
    """Order to simulate."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    urgency: float = 0.5  # 0 to 1


@dataclass
class Fill:
    """Simulated fill."""
    symbol: str
    side: OrderSide
    fill_quantity: float
    fill_price: float
    fill_time: datetime
    slippage_bps: float
    market_impact_bps: float
    total_cost: float
    venue: str


@dataclass
class ExecutionResult:
    """Result of order execution."""
    order: Order
    fills: List[Fill]
    total_filled: float
    avg_fill_price: float
    total_slippage_bps: float
    total_market_impact_bps: float
    total_cost: float
    execution_time_seconds: float


class IndianMarketCostModel:
    """
    Transaction cost model for Indian markets.
    
    Costs include:
    - Brokerage (per share)
    - STT (Securities Transaction Tax)
    - Stamp duty
    - SEBI turnover fee
    - GST (on brokerage)
    """
    
    def __init__(self):
        # Cost rates
        self.brokerage_per_share = 0.03  # 3 paise per share
        self.stt_rate_delivery = 0.00025  # 0.025% on equity delivery
        self.stt_rate_intraday = 0.000025  # 0.0025% on intraday
        self.stamp_duty = 0.00002  # 0.002% stamp duty
        self.sebi_turnover_fee = 0.000001  # SEBI turnover fee
        self.gst_rate = 0.18  # 18% GST on brokerage
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        side: OrderSide,
        is_delivery: bool = True
    ) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            quantity: Number of shares
            price: Price per share
            side: Order side
            is_delivery: Whether trade is for delivery (vs intraday)
            
        Returns:
            Total cost in currency
        """
        trade_value = quantity * price
        
        # Brokerage
        brokerage = self.brokerage_per_share * quantity
        
        # STT (only on sell for delivery, both sides for intraday)
        if side == OrderSide.SELL:
            stt_rate = self.stt_rate_delivery if is_delivery else self.stt_rate_intraday
            stt = trade_value * stt_rate
        else:
            stt = 0
        
        # Stamp duty (only on buy)
        if side == OrderSide.BUY:
            stamp_duty = trade_value * self.stamp_duty
        else:
            stamp_duty = 0
        
        # SEBI turnover fee
        sebi_fee = trade_value * self.sebi_turnover_fee
        
        # GST on brokerage
        gst = brokerage * self.gst_rate
        
        # Total cost
        total_cost = brokerage + stt + stamp_duty + sebi_fee + gst
        
        return total_cost


class SlippageModel:
    """Slippage model for realistic execution simulation."""
    
    def __init__(self):
        self.fixed_slippage_bps = 0.5  # 0.5 bps fixed component
        self.variable_factor = 0.1  # Variable factor based on volume participation
    
    def calculate_slippage(
        self,
        quantity: float,
        avg_daily_volume: float,
        price: float,
        urgency: float = 0.5
    ) -> float:
        """
        Calculate slippage in basis points.
        
        Args:
            quantity: Trade quantity
            avg_daily_volume: Average daily volume
            price: Current price
            urgency: Order urgency (0 to 1)
            
        Returns:
            Slippage in basis points
        """
        # Fixed component
        fixed_slippage = self.fixed_slippage_bps
        
        # Variable component based on volume participation
        volume_participation = quantity / avg_daily_volume if avg_daily_volume > 0 else 0
        variable_slippage = self.variable_factor * volume_participation * 100
        
        # Urgency adjustment
        urgency_adjustment = urgency * 0.5
        
        total_slippage_bps = fixed_slippage + variable_slippage + urgency_adjustment
        
        return total_slippage_bps
    
    def apply_slippage(self, price: float, slippage_bps: float, side: OrderSide) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Original price
            slippage_bps: Slippage in basis points
            side: Order side
            
        Returns:
            Adjusted price
        """
        slippage_pct = slippage_bps / 10000.0
        
        if side == OrderSide.BUY:
            return price * (1 + slippage_pct)
        else:  # SELL
            return price * (1 - slippage_pct)


class MarketImpactModel:
    """Market impact model for large orders."""
    
    def __init__(self):
        self.impact_alpha = 0.5  # Square root model exponent
        self.impact_k = 0.1  # Impact coefficient
    
    def calculate_impact(
        self,
        quantity: float,
        avg_daily_volume: float,
        price: float
    ) -> float:
        """
        Calculate market impact using square root model.
        
        Impact = k * (Q / V)^alpha
        
        Args:
            quantity: Trade quantity
            avg_daily_volume: Average daily volume
            price: Current price
            
        Returns:
            Price impact in percentage
        """
        volume_ratio = quantity / avg_daily_volume if avg_daily_volume > 0 else 0
        impact_pct = self.impact_k * (volume_ratio ** self.impact_alpha)
        
        return impact_pct
    
    def apply_impact(self, price: float, impact_pct: float, side: OrderSide) -> float:
        """
        Apply market impact to price.
        
        Args:
            price: Original price
            impact_pct: Impact in percentage
            side: Order side
            
        Returns:
            Adjusted price
        """
        if side == OrderSide.BUY:
            return price * (1 + impact_pct)
        else:  # SELL
            return price * (1 - impact_pct)


class ExecutionSimulator:
    """
    Execution Simulator for realistic cost modeling.
    
    Simulates order execution with:
    - Realistic transaction costs
    - Slippage
    - Market impact
    - Liquidity constraints
    """
    
    def __init__(self):
        self.cost_model = IndianMarketCostModel()
        self.slippage_model = SlippageModel()
        self.impact_model = MarketImpactModel()
        
        # Venue simulation
        self.venues = ['NSE', 'BSE', 'MFIF']
        self.venue_liquidity = {'NSE': 0.7, 'BSE': 0.25, 'MFIF': 0.05}
    
    def simulate_order(
        self,
        order: Order,
        current_price: float,
        avg_daily_volume: float,
        is_delivery: bool = True
    ) -> ExecutionResult:
        """
        Simulate order execution.
        
        Args:
            order: Order to execute
            current_price: Current market price
            avg_daily_volume: Average daily volume
            is_delivery: Whether trade is for delivery
            
        Returns:
            ExecutionResult with fill details
        """
        # Calculate slippage
        slippage_bps = self.slippage_model.calculate_slippage(
            order.quantity, avg_daily_volume, current_price, order.urgency
        )
        
        # Calculate market impact
        impact_pct = self.impact_model.calculate_impact(
            order.quantity, avg_daily_volume, current_price
        )
        
        # Apply slippage and impact to price
        fill_price = current_price
        fill_price = self.slippage_model.apply_slippage(fill_price, slippage_bps, order.side)
        fill_price = self.impact_model.apply_impact(fill_price, impact_pct, order.side)
        
        # Calculate transaction cost
        cost = self.cost_model.calculate_cost(
            order.quantity, fill_price, order.side, is_delivery
        )
        
        # Select venue (random based on liquidity)
        venue = np.random.choice(
            self.venues,
            p=[self.venue_liquidity[v] for v in self.venues]
        )
        
        # Create fill
        fill = Fill(
            symbol=order.symbol,
            side=order.side,
            fill_quantity=order.quantity,
            fill_price=fill_price,
            fill_time=datetime.now(),
            slippage_bps=slippage_bps,
            market_impact_bps=impact_pct * 10000,  # Convert to bps
            total_cost=cost,
            venue=venue
        )
        
        # Create result
        result = ExecutionResult(
            order=order,
            fills=[fill],
            total_filled=order.quantity,
            avg_fill_price=fill_price,
            total_slippage_bps=slippage_bps,
            total_market_impact_bps=impact_pct * 10000,
            total_cost=cost,
            execution_time_seconds=1.0  # Simplified
        )
        
        return result
    
    def simulate_vwap_execution(
        self,
        order: Order,
        current_price: float,
        avg_daily_volume: float,
        num_slices: int = 10,
        is_delivery: bool = True
    ) -> ExecutionResult:
        """
        Simulate VWAP execution with multiple slices.
        
        Args:
            order: Order to execute
            current_price: Current market price
            avg_daily_volume: Average daily volume
            num_slices: Number of time slices
            is_delivery: Whether trade is for delivery
            
        Returns:
            ExecutionResult with multiple fills
        """
        fills = []
        slice_quantity = order.quantity / num_slices
        total_filled = 0
        total_cost = 0
        weighted_price = 0
        
        for i in range(num_slices):
            # Simulate price variation across slices
            price_variation = np.random.normal(0, 0.001)
            slice_price = current_price * (1 + price_variation)
            
            # Calculate slippage for smaller slice
            slippage_bps = self.slippage_model.calculate_slippage(
                slice_quantity, avg_daily_volume, slice_price, order.urgency * 0.5
            )
            
            # Calculate impact for smaller slice
            impact_pct = self.impact_model.calculate_impact(
                slice_quantity, avg_daily_volume, slice_price
            )
            
            # Apply slippage and impact
            fill_price = slice_price
            fill_price = self.slippage_model.apply_slippage(fill_price, slippage_bps, order.side)
            fill_price = self.impact_model.apply_impact(fill_price, impact_pct, order.side)
            
            # Calculate cost
            cost = self.cost_model.calculate_cost(
                slice_quantity, fill_price, order.side, is_delivery
            )
            
            # Select venue
            venue = np.random.choice(
                self.venues,
                p=[self.venue_liquidity[v] for v in self.venues]
            )
            
            # Create fill
            fill = Fill(
                symbol=order.symbol,
                side=order.side,
                fill_quantity=slice_quantity,
                fill_price=fill_price,
                fill_time=datetime.now(),
                slippage_bps=slippage_bps,
                market_impact_bps=impact_pct * 10000,
                total_cost=cost,
                venue=venue
            )
            
            fills.append(fill)
            total_filled += slice_quantity
            total_cost += cost
            weighted_price += fill_price * slice_quantity
        
        avg_fill_price = weighted_price / total_filled if total_filled > 0 else current_price
        
        # Calculate total slippage and impact
        total_slippage_bps = np.mean([f.slippage_bps for f in fills])
        total_impact_bps = np.mean([f.market_impact_bps for f in fills])
        
        result = ExecutionResult(
            order=order,
            fills=fills,
            total_filled=total_filled,
            avg_fill_price=avg_fill_price,
            total_slippage_bps=total_slippage_bps,
            total_market_impact_bps=total_impact_bps,
            total_cost=total_cost,
            execution_time_seconds=num_slices * 60  # Assume 1 minute per slice
        )
        
        return result
    
    def calculate_total_cost_bps(self, result: ExecutionResult) -> float:
        """
        Calculate total cost in basis points.
        
        Args:
            result: Execution result
            
        Returns:
            Total cost in basis points
        """
        trade_value = result.total_filled * result.avg_fill_price
        total_cost_bps = (result.total_cost / trade_value) * 10000 if trade_value > 0 else 0
        return total_cost_bps
    
    def get_cost_breakdown(self, result: ExecutionResult) -> Dict:
        """
        Get detailed cost breakdown.
        
        Args:
            result: Execution result
            
        Returns:
            Dictionary with cost components
        """
        trade_value = result.total_filled * result.avg_fill_price
        
        # Estimate cost components
        brokerage = self.cost_model.brokerage_per_share * result.total_filled
        stt = trade_value * self.cost_model.stt_rate_delivery if result.order.side == OrderSide.SELL else 0
        stamp_duty = trade_value * self.cost_model.stamp_duty if result.order.side == OrderSide.BUY else 0
        sebi_fee = trade_value * self.cost_model.sebi_turnover_fee
        gst = brokerage * self.cost_model.gst_rate
        
        return {
            'trade_value': trade_value,
            'brokerage': brokerage,
            'stt': stt,
            'stamp_duty': stamp_duty,
            'sebi_fee': sebi_fee,
            'gst': gst,
            'slippage_cost': result.total_slippage_bps / 10000 * trade_value,
            'market_impact_cost': result.total_market_impact_bps / 10000 * trade_value,
            'total_cost': result.total_cost,
            'total_cost_bps': self.calculate_total_cost_bps(result)
        }


if __name__ == "__main__":
    # Test the Execution Simulator
    print("Testing Execution Simulator...")
    
    simulator = ExecutionSimulator()
    
    # Test market order
    print("\nSimulating Market Order...")
    order = Order(
        symbol="RELIANCE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000,
        urgency=0.8
    )
    
    result = simulator.simulate_order(
        order,
        current_price=2500.0,
        avg_daily_volume=10000000,
        is_delivery=True
    )
    
    print(f"Symbol: {result.order.symbol}")
    print(f"Side: {result.order.side.value}")
    print(f"Quantity: {result.total_filled:.0f}")
    print(f"Avg Fill Price: {result.avg_fill_price:.2f}")
    print(f"Slippage: {result.total_slippage_bps:.2f} bps")
    print(f"Market Impact: {result.total_market_impact_bps:.2f} bps")
    print(f"Total Cost: {result.total_cost:.2f}")
    print(f"Total Cost (bps): {simulator.calculate_total_cost_bps(result):.2f} bps")
    
    # Cost breakdown
    print("\nCost Breakdown:")
    breakdown = simulator.get_cost_breakdown(result)
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f}")
    
    # Test VWAP execution
    print("\nSimulating VWAP Execution...")
    vwap_result = simulator.simulate_vwap_execution(
        order,
        current_price=2500.0,
        avg_daily_volume=10000000,
        num_slices=10,
        is_delivery=True
    )
    
    print(f"Symbol: {vwap_result.order.symbol}")
    print(f"Total Filled: {vwap_result.total_filled:.0f}")
    print(f"Avg Fill Price: {vwap_result.avg_fill_price:.2f}")
    print(f"Total Slippage: {vwap_result.total_slippage_bps:.2f} bps")
    print(f"Total Cost: {vwap_result.total_cost:.2f}")
    print(f"Execution Time: {vwap_result.execution_time_seconds:.0f} seconds")
    print(f"Total Cost (bps): {simulator.calculate_total_cost_bps(vwap_result):.2f} bps")
