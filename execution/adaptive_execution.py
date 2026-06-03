"""
Adaptive Execution Engine with Regime-Based Algorithm Selection
Based on Blueprint V1.0

Architecture:
- Regime detection for algorithm selection
- Multiple execution algorithms (TWAP, VWAP, POV, Implementation Shortfall)
- Adaptive speed adjustment based on fill quality
- Smart order routing
- Real-time market impact monitoring

Algorithms:
- Low Vol: TWAP (Time-Weighted Average Price)
- High Vol: VWAP (Volume-Weighted Average Price)
- Trending: POV (Percentage of Volume)
- Panic: Implementation Shortfall (IS)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "is"
    ADAPTIVE = "adaptive"


class RegimeType(Enum):
    """Market regime types for execution."""
    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"
    TRENDING = "trending"
    PANIC = "panic"
    NORMAL = "normal"


@dataclass
class Order:
    """Order to be executed."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    limit_price: Optional[float] = None
    urgency: float = 0.5  # 0 to 1
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ExecutionSlice:
    """Single slice of an order."""
    symbol: str
    side: str
    quantity: float
    target_time: datetime
    algorithm: ExecutionAlgorithm


@dataclass
class Fill:
    """Fill information."""
    symbol: str
    side: str
    fill_quantity: float
    fill_price: float
    fill_time: datetime
    slippage_bps: float
    market_impact_bps: float


@dataclass
class ExecutionResult:
    """Result of order execution."""
    order_id: str
    symbol: str
    total_filled: float
    avg_fill_price: float
    total_slippage_bps: float
    total_market_impact_bps: float
    execution_time_seconds: float
    algorithm_used: ExecutionAlgorithm
    slices: List[ExecutionSlice]
    fills: List[Fill]


class TWAPSlicer:
    """Time-Weighted Average Price execution."""
    
    def __init__(self, num_slices: int = 10):
        self.num_slices = num_slices
    
    def generate_slices(
        self,
        order: Order,
        start_time: datetime,
        end_time: datetime
    ) -> List[ExecutionSlice]:
        """Generate equal time-weighted slices."""
        slices = []
        time_delta = (end_time - start_time) / self.num_slices
        slice_quantity = order.quantity / self.num_slices
        
        for i in range(self.num_slices):
            slice_time = start_time + time_delta * (i + 0.5)
            slices.append(ExecutionSlice(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_quantity,
                target_time=slice_time,
                algorithm=ExecutionAlgorithm.TWAP
            ))
        
        return slices


class VWAPSlicer:
    """Volume-Weighted Average Price execution."""
    
    def __init__(self, participation_rate: float = 0.1):
        self.participation_rate = participation_rate
    
    def generate_slices(
        self,
        order: Order,
        start_time: datetime,
        end_time: datetime,
        volume_profile: Optional[pd.Series] = None
    ) -> List[ExecutionSlice]:
        """Generate volume-weighted slices."""
        slices = []
        
        if volume_profile is None:
            # Fall back to TWAP if no volume profile
            twap = TWAPSlicer()
            return twap.generate_slices(order, start_time, end_time)
        
        # Normalize volume profile
        total_volume = volume_profile.sum()
        if total_volume == 0:
            twap = TWAPSlicer()
            return twap.generate_slices(order, start_time, end_time)
        
        # Generate slices based on volume
        cumulative_volume = 0
        remaining_quantity = order.quantity
        
        for time, volume in volume_profile.items():
            if start_time <= time <= end_time and remaining_quantity > 0:
                target_volume = volume * self.participation_rate
                slice_quantity = min(target_volume, remaining_quantity)
                
                if slice_quantity > 0:
                    slices.append(ExecutionSlice(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=slice_quantity,
                        target_time=time,
                        algorithm=ExecutionAlgorithm.VWAP
                    ))
                    remaining_quantity -= slice_quantity
        
        return slices


class POVSlicer:
    """Percentage of Volume execution."""
    
    def __init__(self, participation_rate: float = 0.08):
        self.participation_rate = participation_rate
    
    def generate_slices(
        self,
        order: Order,
        start_time: datetime,
        end_time: datetime,
        real_time_volume: Optional[float] = None
    ) -> List[ExecutionSlice]:
        """Generate POV slices (adaptive to real-time volume)."""
        slices = []
        
        # For POV, we generate fewer slices but adjust based on real-time volume
        num_slices = 5
        time_delta = (end_time - start_time) / num_slices
        
        for i in range(num_slices):
            slice_time = start_time + time_delta * (i + 0.5)
            
            if real_time_volume:
                slice_quantity = real_time_volume * self.participation_rate
            else:
                slice_quantity = order.quantity / num_slices
            
            slices.append(ExecutionSlice(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_quantity,
                target_time=slice_time,
                algorithm=ExecutionAlgorithm.POV
            ))
        
        return slices


class ISSlicer:
    """Implementation Shortfall execution."""
    
    def __init__(self, urgency: float = 0.8):
        self.urgency = urgency
    
    def generate_slices(
        self,
        order: Order,
        start_time: datetime,
        end_time: datetime
    ) -> List[ExecutionSlice]:
        """Generate IS slices (front-loaded for urgency)."""
        slices = []
        
        # Front-loaded execution for urgency
        num_slices = 10
        time_delta = (end_time - start_time) / num_slices
        
        # Exponential decay for slice sizes
        decay_factor = 0.9
        base_quantity = order.quantity * (1 - decay_factor) / (1 - decay_factor ** num_slices)
        
        cumulative_quantity = 0
        for i in range(num_slices):
            slice_time = start_time + time_delta * (i + 0.5)
            slice_quantity = base_quantity * (decay_factor ** i)
            
            # Ensure last slice gets remaining
            if i == num_slices - 1:
                slice_quantity = order.quantity - cumulative_quantity
            
            slices.append(ExecutionSlice(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_quantity,
                target_time=slice_time,
                algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
            ))
            
            cumulative_quantity += slice_quantity
        
        return slices


class AdaptiveExecutionEngine:
    """
    Adaptive Execution Engine with regime-based algorithm selection.
    
    Selects execution algorithm based on:
    - Market regime (volatility, trend)
    - Order urgency
    - Order size relative to volume
    - Fill quality feedback
    """
    
    def __init__(self):
        self.algorithms = {
            RegimeType.LOW_VOL: TWAPSlicer(num_slices=10),
            RegimeType.HIGH_VOL: VWAPSlicer(participation_rate=0.08),
            RegimeType.TRENDING: POVSlicer(participation_rate=0.1),
            RegimeType.PANIC: ISSlicer(urgency=0.9),
            RegimeType.NORMAL: VWAPSlicer(participation_rate=0.1)
        }
        
        self.execution_history: List[ExecutionResult] = []
        self.slippage_threshold = 5.0  # 5 bps threshold for adjustment
        
    def detect_execution_regime(
        self,
        volatility: float,
        trend_strength: float,
        volume_ratio: float
    ) -> RegimeType:
        """
        Detect execution regime based on market conditions.
        
        Args:
            volatility: Current volatility
            trend_strength: Trend strength (-1 to 1)
            volume_ratio: Volume ratio vs average
            
        Returns:
            Detected regime
        """
        if volatility > 0.3:
            return RegimeType.HIGH_VOL
        elif volatility < 0.12:
            return RegimeType.LOW_VOL
        elif abs(trend_strength) > 0.5:
            return RegimeType.TRENDING
        elif volume_ratio < 0.5:
            return RegimeType.PANIC
        else:
            return RegimeType.NORMAL
    
    def select_algorithm(
        self,
        order: Order,
        regime: RegimeType,
        volume_profile: Optional[pd.Series] = None
    ) -> List[ExecutionSlice]:
        """
        Select execution algorithm based on regime and generate slices.
        
        Args:
            order: Order to execute
            regime: Current market regime
            volume_profile: Optional volume profile for VWAP
            
        Returns:
            List of execution slices
        """
        algorithm = self.algorithms.get(regime, self.algorithms[RegimeType.NORMAL])
        
        start_time = order.start_time or datetime.now()
        end_time = order.end_time or (start_time + timedelta(minutes=30))
        
        if isinstance(algorithm, VWAPSlicer):
            return algorithm.generate_slices(order, start_time, end_time, volume_profile)
        else:
            return algorithm.generate_slices(order, start_time, end_time)
    
    def execute_order(
        self,
        order: Order,
        market_data: Dict,
        regime: Optional[RegimeType] = None
    ) -> ExecutionResult:
        """
        Execute an order using adaptive algorithm selection.
        
        Args:
            order: Order to execute
            market_data: Market data including volatility, volume, etc.
            regime: Optional pre-detected regime
            
        Returns:
            ExecutionResult with fill information
        """
        # Detect regime if not provided
        if regime is None:
            volatility = market_data.get('volatility', 0.15)
            trend_strength = market_data.get('trend_strength', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            regime = self.detect_execution_regime(volatility, trend_strength, volume_ratio)
        
        # Select algorithm and generate slices
        volume_profile = market_data.get('volume_profile')
        slices = self.select_algorithm(order, regime, volume_profile)
        
        # Simulate execution (in production, this would interact with broker API)
        fills = self._simulate_execution(order, slices, market_data)
        
        # Calculate metrics
        total_filled = sum(f.fill_quantity for f in fills)
        avg_fill_price = sum(f.fill_price * f.fill_quantity for f in fills) / total_filled if total_filled > 0 else 0
        total_slippage = sum(f.slippage_bps for f in fills) / len(fills) if fills else 0
        total_impact = sum(f.market_impact_bps for f in fills) / len(fills) if fills else 0
        
        result = ExecutionResult(
            order_id=str(id(order)),
            symbol=order.symbol,
            total_filled=total_filled,
            avg_fill_price=avg_fill_price,
            total_slippage_bps=total_slippage,
            total_market_impact_bps=total_impact,
            execution_time_seconds=(slices[-1].target_time - slices[0].target_time).total_seconds() if slices else 0,
            algorithm_used=slices[0].algorithm if slices else ExecutionAlgorithm.TWAP,
            slices=slices,
            fills=fills
        )
        
        self.execution_history.append(result)
        
        # Adjust algorithm if slippage too high
        if total_slippage > self.slippage_threshold:
            self._adjust_algorithm_speed(regime, 0.5)  # Slow down
        
        return result
    
    def _simulate_execution(
        self,
        order: Order,
        slices: List[ExecutionSlice],
        market_data: Dict
    ) -> List[Fill]:
        """Simulate order execution (placeholder for production)."""
        fills = []
        base_price = market_data.get('current_price', 100.0)
        
        for slice in slices:
            # Simulate fill with some randomness
            fill_price = base_price * (1 + np.random.normal(0, 0.001))
            slippage_bps = abs(fill_price - base_price) / base_price * 10000
            market_impact_bps = slippage_bps * 0.5  # Simplified
            
            fills.append(Fill(
                symbol=slice.symbol,
                side=slice.side,
                fill_quantity=slice.quantity,
                fill_price=fill_price,
                fill_time=slice.target_time,
                slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps
            ))
        
        return fills
    
    def _adjust_algorithm_speed(self, regime: RegimeType, speed_factor: float) -> None:
        """Adjust algorithm speed based on fill quality."""
        algorithm = self.algorithms.get(regime)
        
        if isinstance(algorithm, TWAPSlicer):
            algorithm.num_slices = max(5, int(algorithm.num_slices * speed_factor))
        elif isinstance(algorithm, VWAPSlicer):
            algorithm.participation_rate *= speed_factor
        elif isinstance(algorithm, POVSlicer):
            algorithm.participation_rate *= speed_factor
    
    def get_execution_statistics(self) -> Dict:
        """Get statistics on recent executions."""
        if not self.execution_history:
            return {}
        
        recent = self.execution_history[-100:]
        
        avg_slippage = np.mean([r.total_slippage_bps for r in recent])
        avg_impact = np.mean([r.total_market_impact_bps for r in recent])
        avg_time = np.mean([r.execution_time_seconds for r in recent])
        
        algorithm_counts = {}
        for result in recent:
            algo = result.algorithm_used.value
            algorithm_counts[algo] = algorithm_counts.get(algo, 0) + 1
        
        return {
            'avg_slippage_bps': avg_slippage,
            'avg_market_impact_bps': avg_impact,
            'avg_execution_time_seconds': avg_time,
            'algorithm_distribution': algorithm_counts,
            'total_executions': len(recent)
        }


class SmartOrderRouter:
    """Smart Order Router for multi-venue execution."""
    
    def __init__(self):
        self.venues = {
            'NSE': {'latency_ms': 5, 'fee_bps': 1.0, 'liquidity_score': 0.9},
            'BSE': {'latency_ms': 8, 'fee_bps': 1.2, 'liquidity_score': 0.7},
            'MFIF': {'latency_ms': 3, 'fee_bps': 0.5, 'liquidity_score': 0.6}
        }
    
    def route_order(
        self,
        order: Order,
        urgency: float = 0.5
    ) -> str:
        """
        Select best venue for order execution.
        
        Args:
            order: Order to route
            urgency: Order urgency (0 to 1)
            
        Returns:
            Selected venue name
        """
        scores = {}
        
        for venue, attrs in self.venues.items():
            # Score based on latency, fees, and liquidity
            latency_score = 1.0 / (attrs['latency_ms'] / 10 + 1)
            fee_score = 1.0 / (attrs['fee_bps'] / 2 + 1)
            liquidity_score = attrs['liquidity_score']
            
            # Weight based on urgency
            if urgency > 0.7:
                # Prioritize latency for urgent orders
                weights = {'latency': 0.6, 'fee': 0.2, 'liquidity': 0.2}
            else:
                # Prioritize cost and liquidity for non-urgent
                weights = {'latency': 0.2, 'fee': 0.4, 'liquidity': 0.4}
            
            scores[venue] = (
                weights['latency'] * latency_score +
                weights['fee'] * fee_score +
                weights['liquidity'] * liquidity_score
            )
        
        # Return venue with highest score
        return max(scores, key=scores.get)


if __name__ == "__main__":
    # Test the adaptive execution engine
    print("Testing Adaptive Execution Engine...")
    
    engine = AdaptiveExecutionEngine()
    
    # Create sample order
    order = Order(
        symbol="RELIANCE",
        side="buy",
        quantity=1000,
        urgency=0.5
    )
    
    # Sample market data
    market_data = {
        'volatility': 0.18,
        'trend_strength': 0.3,
        'volume_ratio': 1.2,
        'current_price': 2500.0
    }
    
    # Execute order
    result = engine.execute_order(order, market_data)
    
    print(f"Execution Result:")
    print(f"  Symbol: {result.symbol}")
    print(f"  Filled: {result.total_filled}")
    print(f"  Avg Price: {result.avg_fill_price:.2f}")
    print(f"  Slippage: {result.total_slippage_bps:.2f} bps")
    print(f"  Algorithm: {result.algorithm_used.value}")
    
    # Get statistics
    stats = engine.get_execution_statistics()
    print(f"\nExecution Statistics: {stats}")
