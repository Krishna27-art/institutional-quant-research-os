"""
Limits to Arbitrage: Arbitrage Constraints, Panic Detector, Liquidity Vacuum
Based on the critique: Build Limits to Arbitrage - markets stay irrational longer than expected

Find:
- Forced liquidations
- Margin calls
- Panic selling
- Retail FOMO

Many profitable trades come from these.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats


class ArbitrageConstraintType(Enum):
    """Types of arbitrage constraints."""
    CAPITAL = "capital"
    RISK_LIMITS = "risk_limits"
    SHORT_SALE_CONSTRAINTS = "short_sale_constraints"
    FUNDING_COSTS = "funding_costs"
    REGULATORY = "regulatory"
    LIQUIDITY = "liquidity"


@dataclass
class ArbitrageConstraint:
    """Arbitrage constraint data."""
    symbol: str
    timestamp: datetime
    constraint_type: ArbitrageConstraintType
    severity: float  # 0 to 1
    description: str
    impact_on_arbitrage: float  # -1 to 1


@dataclass
class PanicEvent:
    """Panic event data."""
    symbol: str
    timestamp: datetime
    event_type: str  # "forced_liquidation", "margin_call", "panic_selling", "fomo"
    severity: float  # 0 to 1
    price_impact: float
    volume_spike: float
    signal: float  # -1 to 1


@dataclass
class LiquidityVacuum:
    """Liquidity vacuum data."""
    symbol: str
    timestamp: datetime
    bid_ask_spread: float
    volume: float
    avg_volume: float
    vacuum_severity: float  # 0 to 1
    is_vacuum: bool


class LimitsToArbitrageEngine:
    """
    Limits to Arbitrage Engine for detecting market inefficiencies.
    
    Features:
    - Arbitrage constraint detection
    - Panic event detection
    - Liquidity vacuum detection
    - Forced liquidation detection
    - Margin call detection
    - FOMO detection
    """
    
    def __init__(self):
        self.constraints: Dict[str, List[ArbitrageConstraint]] = {}
        self.panic_events: Dict[str, List[PanicEvent]] = {}
        self.liquidity_vacuums: Dict[str, List[LiquidityVacuum]] = {}
        
        # Thresholds
        self.panic_price_drop_threshold = -0.05  # 5% drop
        self.volume_spike_threshold = 3.0  # 3x normal volume
        self.spread_vacuum_threshold = 0.01  # 1% spread
        self.volume_vacuum_threshold = 0.1  # 10% of average volume
        
        # Connect to prediction registry for IC validation
        try:
            import sys
            from pathlib import Path
            # Resolve root directory of repository and production path
            repo_root = Path(__file__).resolve().parent.parent
            prod_path = str(repo_root / "production")
            if prod_path not in sys.path:
                sys.path.append(prod_path)
            from src.alpha.prediction_registry import get_prediction_registry
            self.registry = get_prediction_registry()
        except ImportError:
            self.registry = None

    def compute_engine_ic(self, alpha_id: str = "LimitsToArbitrage_Panic") -> Dict[str, float]:
        """
        Compute and log Spearman rank IC and Sharpe for predictions registered by this engine.
        """
        if self.registry is None:
            import logging
            logging.getLogger(__name__).warning("Prediction registry not connected. Cannot compute IC.")
            return {"mean_ic": 0.0, "rolling_ic": 0.0, "sharpe": 0.0}
            
        try:
            report = self.registry.get_strategy_report(alpha_id)
            import logging
            logging.getLogger(__name__).info(
                f"[{alpha_id}] Performance Report: "
                f"Total Preds: {report.total_predictions}, "
                f"Mean IC: {report.mean_ic:.4f}, "
                f"Rolling IC: {report.rolling_ic:.4f}, "
                f"Realized Sharpe: {report.sharpe:.4f}, "
                f"Lifecycle: {report.lifecycle_stage}"
            )
            return {
                "mean_ic": report.mean_ic,
                "rolling_ic": report.rolling_ic,
                "sharpe": report.sharpe,
                "total_predictions": report.total_predictions,
                "lifecycle_stage": report.lifecycle_stage
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error computing IC for {alpha_id}: {e}")
            return {"mean_ic": 0.0, "rolling_ic": 0.0, "sharpe": 0.0}
    
    def detect_arbitrage_constraints(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float,
        short_interest: float,
        borrowing_cost: float
    ) -> List[ArbitrageConstraint]:
        """
        Detect arbitrage constraints for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            price: Current price
            volume: Current volume
            short_interest: Short interest as percentage
            borrowing_cost: Cost to borrow stock
            
        Returns:
            List of detected constraints
        """
        constraints = []
        
        # Short sale constraints
        if short_interest > 0.1:  # > 10% short interest
            severity = min(short_interest / 0.3, 1.0)
            constraints.append(ArbitrageConstraint(
                symbol=symbol,
                timestamp=timestamp,
                constraint_type=ArbitrageConstraintType.SHORT_SALE_CONSTRAINTS,
                severity=severity,
                description=f"High short interest: {short_interest:.1%}",
                impact_on_arbitrage=-severity * 0.5
            ))
        
        # Funding costs
        if borrowing_cost > 0.05:  # > 5% annual cost
            severity = min(borrowing_cost / 0.2, 1.0)
            constraints.append(ArbitrageConstraint(
                symbol=symbol,
                timestamp=timestamp,
                constraint_type=ArbitrageConstraintType.FUNDING_COSTS,
                severity=severity,
                description=f"High borrowing cost: {borrowing_cost:.1%}",
                impact_on_arbitrage=-severity * 0.3
            ))
        
        # Liquidity constraints
        if volume < 1000000:  # Low volume
            severity = 1.0 - min(volume / 10000000, 1.0)
            constraints.append(ArbitrageConstraint(
                symbol=symbol,
                timestamp=timestamp,
                constraint_type=ArbitrageConstraintType.LIQUIDITY,
                severity=severity,
                description=f"Low volume: {volume:.0f}",
                impact_on_arbitrage=-severity * 0.4
            ))
        
        # Store constraints
        if symbol not in self.constraints:
            self.constraints[symbol] = []
        self.constraints[symbol].extend(constraints)
        
        return constraints
    
    def detect_panic_selling(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float,
        historical_prices: pd.Series,
        historical_volumes: pd.Series
    ) -> Optional[PanicEvent]:
        """
        Detect panic selling.
        
        Panic selling indicators:
        - Large price drop
        - Volume spike
        - Selling pressure
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            price: Current price
            volume: Current volume
            historical_prices: Historical prices
            historical_volumes: Historical volumes
            
        Returns:
            PanicEvent if detected, None otherwise
        """
        if len(historical_prices) < 20:
            return None
        
        # Calculate price change
        price_change = (price - historical_prices.iloc[-1]) / historical_prices.iloc[-1]
        
        # Calculate volume ratio
        avg_volume = historical_volumes.iloc[-20:].mean()
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check for panic conditions
        is_panic = (
            price_change < self.panic_price_drop_threshold and
            volume_ratio > self.volume_spike_threshold
        )
        
        if is_panic:
            severity = min(abs(price_change) / 0.1, 1.0)
            
            panic_event = PanicEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="panic_selling",
                severity=severity,
                price_impact=price_change,
                volume_spike=volume_ratio,
                signal=-1.0  # Panic selling = negative signal
            )
            
            # Store event
            if symbol not in self.panic_events:
                self.panic_events[symbol] = []
            self.panic_events[symbol].append(panic_event)
            
            # Register with prediction registry
            if self.registry is not None:
                self.registry.register_prediction(
                    strategy="LimitsToArbitrage_Panic",
                    symbol=symbol,
                    timestamp=timestamp,
                    predicted_return=-panic_event.signal * 0.02,
                    confidence=panic_event.severity,
                    horizon_minutes=60,
                    current_price=price
                )
            
            return panic_event
        
        return None
    
    def detect_forced_liquidation(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float,
        historical_prices: pd.Series,
        historical_volumes: pd.Series
    ) -> Optional[PanicEvent]:
        """
        Detect forced liquidation.
        
        Forced liquidation indicators:
        - Rapid price decline
        - Very high volume
        - Selling at any price
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            price: Current price
            volume: Current volume
            historical_prices: Historical prices
            historical_volumes: Historical volumes
            
        Returns:
            PanicEvent if detected, None otherwise
        """
        if len(historical_prices) < 10:
            return None
        
        # Calculate price change over short period
        price_change = (price - historical_prices.iloc[-5]) / historical_prices.iloc[-5]
        
        # Calculate volume ratio
        avg_volume = historical_volumes.iloc[-10:].mean()
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check for forced liquidation conditions
        is_forced = (
            price_change < -0.03 and  # > 3% drop in 5 periods
            volume_ratio > 5.0  # > 5x normal volume
        )
        
        if is_forced:
            severity = min(abs(price_change) / 0.05, 1.0)
            
            forced_event = PanicEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="forced_liquidation",
                severity=severity,
                price_impact=price_change,
                volume_spike=volume_ratio,
                signal=-0.8  # Forced liquidation = strong negative signal
            )
            
            # Store event
            if symbol not in self.panic_events:
                self.panic_events[symbol] = []
            self.panic_events[symbol].append(forced_event)
            
            # Register with prediction registry
            if self.registry is not None:
                self.registry.register_prediction(
                    strategy="LimitsToArbitrage_ForcedLiq",
                    symbol=symbol,
                    timestamp=timestamp,
                    predicted_return=-forced_event.signal * 0.03,
                    confidence=forced_event.severity,
                    horizon_minutes=60,
                    current_price=price
                )
            
            return forced_event
        
        return None
    
    def detect_fomo(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        volume: float,
        historical_prices: pd.Series,
        historical_volumes: pd.Series
    ) -> Optional[PanicEvent]:
        """
        Detect FOMO (Fear Of Missing Out).
        
        FOMO indicators:
        - Rapid price increase
        - High volume
        - Buying pressure
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            price: Current price
            volume: Current volume
            historical_prices: Historical prices
            historical_volumes: Historical volumes
            
        Returns:
            PanicEvent if detected, None otherwise
        """
        if len(historical_prices) < 20:
            return None
        
        # Calculate price change
        price_change = (price - historical_prices.iloc[-1]) / historical_prices.iloc[-1]
        
        # Calculate volume ratio
        avg_volume = historical_volumes.iloc[-20:].mean()
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check for FOMO conditions
        is_fomo = (
            price_change > 0.05 and  # > 5% increase
            volume_ratio > 2.0  # > 2x normal volume
        )
        
        if is_fomo:
            severity = min(price_change / 0.1, 1.0)
            
            fomo_event = PanicEvent(
                symbol=symbol,
                timestamp=timestamp,
                event_type="fomo",
                severity=severity,
                price_impact=price_change,
                volume_spike=volume_ratio,
                signal=0.5  # FOMO = positive signal but risky
            )
            
            # Store event
            if symbol not in self.panic_events:
                self.panic_events[symbol] = []
            self.panic_events[symbol].append(fomo_event)
            
            # Register with prediction registry
            if self.registry is not None:
                self.registry.register_prediction(
                    strategy="LimitsToArbitrage_FOMO",
                    symbol=symbol,
                    timestamp=timestamp,
                    predicted_return=-fomo_event.signal * 0.02,
                    confidence=fomo_event.severity,
                    horizon_minutes=60,
                    current_price=price
                )
            
            return fomo_event
        
        return None
    
    def detect_liquidity_vacuum(
        self,
        symbol: str,
        timestamp: datetime,
        bid_price: float,
        ask_price: float,
        volume: float,
        historical_volumes: pd.Series
    ) -> LiquidityVacuum:
        """
        Detect liquidity vacuum.
        
        Liquidity vacuum indicators:
        - Wide bid-ask spread
        - Low volume relative to average
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            bid_price: Bid price
            ask_price: Ask price
            volume: Current volume
            historical_volumes: Historical volumes
            
        Returns:
            LiquidityVacuum
        """
        # Calculate spread
        spread = (ask_price - bid_price) / bid_price
        
        # Calculate average volume
        avg_volume = historical_volumes.iloc[-20:].mean() if len(historical_volumes) >= 20 else volume
        
        # Calculate volume ratio
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Calculate vacuum severity
        spread_severity = min(spread / self.spread_vacuum_threshold, 1.0)
        volume_severity = 1.0 - min(volume_ratio / self.volume_vacuum_threshold, 1.0)
        vacuum_severity = (spread_severity + volume_severity) / 2
        
        # Check if vacuum
        is_vacuum = vacuum_severity > 0.5
        
        liquidity_vacuum = LiquidityVacuum(
            symbol=symbol,
            timestamp=timestamp,
            bid_ask_spread=spread,
            volume=volume,
            avg_volume=avg_volume,
            vacuum_severity=vacuum_severity,
            is_vacuum=is_vacuum
        )
        
        # Store vacuum
        if symbol not in self.liquidity_vacuums:
            self.liquidity_vacuums[symbol] = []
        self.liquidity_vacuums[symbol].append(liquidity_vacuum)
        
        return liquidity_vacuum
    
    def get_arbitrage_opportunities(
        self,
        symbol: str,
        window_days: int = 5
    ) -> List[Dict]:
        """
        Get arbitrage opportunities based on limits to arbitrage.
        
        Opportunities arise when:
        - Panic selling (buy opportunity)
        - Forced liquidation (buy opportunity)
        - Liquidity vacuum (provides edge)
        
        Args:
            symbol: Trading symbol
            window_days: Lookback window
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        # Check panic events
        if symbol in self.panic_events:
            cutoff_date = datetime.now() - timedelta(days=window_days)
            recent_panics = [
                p for p in self.panic_events[symbol]
                if p.timestamp >= cutoff_date
            ]
            
            for panic in recent_panics:
                if panic.event_type in ["panic_selling", "forced_liquidation"]:
                    opportunities.append({
                        'type': 'panic_buy',
                        'timestamp': panic.timestamp,
                        'severity': panic.severity,
                        'signal': panic.signal,
                        'description': f"{panic.event_type} detected"
                    })
        
        # Check liquidity vacuums
        if symbol in self.liquidity_vacuums:
            cutoff_date = datetime.now() - timedelta(days=window_days)
            recent_vacuums = [
                v for v in self.liquidity_vacuums[symbol]
                if v.timestamp >= cutoff_date and v.is_vacuum
            ]
            
            for vacuum in recent_vacuums:
                opportunities.append({
                    'type': 'liquidity_vacuum',
                    'timestamp': vacuum.timestamp,
                    'severity': vacuum.vacuum_severity,
                    'signal': 0.3,  # Moderate positive signal
                    'description': f"Liquidity vacuum detected, spread: {vacuum.bid_ask_spread:.2%}"
                })
        
        return opportunities


if __name__ == "__main__":
    # Test the Limits to Arbitrage Engine
    print("Testing Limits to Arbitrage: Arbitrage Constraints, Panic Detector, Liquidity Vacuum...")
    
    engine = LimitsToArbitrageEngine()
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    
    historical_prices = pd.Series(np.random.normal(100, 10, n).cumsum(), index=dates)
    historical_volumes = pd.Series(np.random.normal(1000000, 200000, n), index=dates)
    
    # Detect arbitrage constraints
    print("\nDetecting Arbitrage Constraints...")
    constraints = engine.detect_arbitrage_constraints(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        price=2500,
        volume=5000000,
        short_interest=0.15,
        borrowing_cost=0.08
    )
    
    print(f"Detected {len(constraints)} constraints:")
    for constraint in constraints:
        print(f"  {constraint.constraint_type.value}: {constraint.description}")
        print(f"    Severity: {constraint.severity:.2f}")
        print(f"    Impact: {constraint.impact_on_arbitrage:.2f}")
    
    # Detect panic selling
    print("\nDetecting Panic Selling...")
    panic_event = engine.detect_panic_selling(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        price=2375,  # 5% drop
        volume=5000000,  # 5x normal volume
        historical_prices=historical_prices,
        historical_volumes=historical_volumes
    )
    
    if panic_event:
        print(f"Panic Selling Detected:")
        print(f"  Severity: {panic_event.severity:.2f}")
        print(f"  Price Impact: {panic_event.price_impact:.2%}")
        print(f"  Volume Spike: {panic_event.volume_spike:.1f}x")
        print(f"  Signal: {panic_event.signal:.2f}")
    
    # Detect forced liquidation
    print("\nDetecting Forced Liquidation...")
    forced_event = engine.detect_forced_liquidation(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        price=2425,  # 3% drop in 5 periods
        volume=10000000,  # 10x normal volume
        historical_prices=historical_prices,
        historical_volumes=historical_volumes
    )
    
    if forced_event:
        print(f"Forced Liquidation Detected:")
        print(f"  Severity: {forced_event.severity:.2f}")
        print(f"  Price Impact: {forced_event.price_impact:.2%}")
        print(f"  Volume Spike: {forced_event.volume_spike:.1f}x")
        print(f"  Signal: {forced_event.signal:.2f}")
    
    # Detect FOMO
    print("\nDetecting FOMO...")
    fomo_event = engine.detect_fomo(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        price=2625,  # 5% increase
        volume=3000000,  # 3x normal volume
        historical_prices=historical_prices,
        historical_volumes=historical_volumes
    )
    
    if fomo_event:
        print(f"FOMO Detected:")
        print(f"  Severity: {fomo_event.severity:.2f}")
        print(f"  Price Impact: {fomo_event.price_impact:.2%}")
        print(f"  Volume Spike: {fomo_event.volume_spike:.1f}x")
        print(f"  Signal: {fomo_event.signal:.2f}")
    
    # Detect liquidity vacuum
    print("\nDetecting Liquidity Vacuum...")
    vacuum = engine.detect_liquidity_vacuum(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        bid_price=2490,
        ask_price=2510,
        volume=100000,  # Low volume
        historical_volumes=historical_volumes
    )
    
    print(f"Liquidity Vacuum:")
    print(f"  Spread: {vacuum.bid_ask_spread:.2%}")
    print(f"  Volume: {vacuum.volume:.0f}")
    print(f"  Avg Volume: {vacuum.avg_volume:.0f}")
    print(f"  Severity: {vacuum.vacuum_severity:.2f}")
    print(f"  Is Vacuum: {vacuum.is_vacuum}")
    
    # Get arbitrage opportunities
    print("\nArbitrage Opportunities:")
    opportunities = engine.get_arbitrage_opportunities("RELIANCE", window_days=5)
    print(f"Found {len(opportunities)} opportunities:")
    for opp in opportunities:
        print(f"  {opp['type']}: {opp['description']}")
        print(f"    Signal: {opp['signal']:.2f}")
        print(f"    Severity: {opp['severity']:.2f}")
