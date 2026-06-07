"""
Market Microstructure Analysis
Based on the critique: Jane Street's edge comes heavily from microstructure

Need:
- Microprice
- Queue Position
- Order Flow Imbalance
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Kyle Lambda (Kyle's lambda - price impact parameter)
- Amihud Illiquidity
- Market Impact
- LOB (Limit Order Book) Imbalance

This is foundational for institutional trading.
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


@dataclass
class Microprice:
    """Microprice - weighted average of bid and ask."""
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    microprice: float
    mid_price: float
    spread: float


@dataclass
class QueuePosition:
    """Queue position in order book."""
    timestamp: datetime
    symbol: str
    side: str  # "bid" or "ask"
    price: float
    position: int  # Position in queue
    queue_size: int  # Total queue size
    probability_of_execution: float


@dataclass
class OrderFlowImbalance:
    """Order flow imbalance metrics."""
    timestamp: datetime
    symbol: str
    ofi: float  # Order flow imbalance
    vpin: float  # Volume-synchronized probability of informed trading
    buy_volume: float
    sell_volume: float
    imbalance_direction: str  # "buy" or "sell"


@dataclass
class KyleLambda:
    """Kyle's lambda - price impact parameter."""
    timestamp: datetime
    symbol: str
    lambda_param: float  # Price impact parameter
    informed_trading_intensity: float
    noise_trading_intensity: float


@dataclass
class AmihudIlliquidity:
    """Amihud illiquidity measure."""
    timestamp: datetime
    symbol: str
    illiquidity: float  # |return| / volume
    avg_daily_illiquidity: float
    liquidity_score: float  # 1 / illiquidity


@dataclass
class MarketImpact:
    """Market impact analysis."""
    timestamp: datetime
    symbol: str
    trade_size: float
    price_impact: float
    temporary_impact: float
    permanent_impact: float
    impact_cost_bps: float


@dataclass
class LOBImbalance:
    """Limit Order Book imbalance."""
    timestamp: datetime
    symbol: str
    bid_imbalance: float
    ask_imbalance: float
    total_imbalance: float
    depth_ratio: float
    is_skewed: bool


class MarketMicrostructureEngine:
    """
    Market Microstructure Engine for institutional trading.
    
    Features:
    - Microprice calculation
    - Queue position analysis
    - Order flow imbalance
    - VPIN calculation
    - Kyle's lambda estimation
    - Amihud illiquidity
    - Market impact modeling
    - LOB imbalance analysis
    """
    
    def __init__(self):
        self.microprice_history: Dict[str, List[Microprice]] = {}
        self.queue_positions: Dict[str, List[QueuePosition]] = {}
        self.ofi_history: Dict[str, List[OrderFlowImbalance]] = {}
        self.kyle_lambda_history: Dict[str, List[KyleLambda]] = {}
        self.illiquidity_history: Dict[str, List[AmihudIlliquidity]] = {}
        self.market_impact_history: Dict[str, List[MarketImpact]] = {}
        self.lob_imbalance_history: Dict[str, List[LOBImbalance]] = {}
    
    def calculate_microprice(
        self,
        timestamp: datetime,
        symbol: str,
        bid_price: float,
        ask_price: float,
        bid_size: float,
        ask_size: float
    ) -> Microprice:
        """
        Calculate microprice.
        
        Microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        This is the volume-weighted mid price.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            bid_price: Best bid price
            ask_price: Best ask price
            bid_size: Bid size
            ask_size: Ask size
            
        Returns:
            Microprice
        """
        # Calculate microprice
        if bid_size + ask_size > 0:
            microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        else:
            microprice = (bid_price + ask_price) / 2
        
        # Calculate mid price
        mid_price = (bid_price + ask_price) / 2
        
        # Calculate spread
        spread = ask_price - bid_price
        
        microprice_obj = Microprice(
            timestamp=timestamp,
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            microprice=microprice,
            mid_price=mid_price,
            spread=spread
        )
        
        # Store in history
        if symbol not in self.microprice_history:
            self.microprice_history[symbol] = []
        self.microprice_history[symbol].append(microprice_obj)
        
        return microprice_obj
    
    def calculate_queue_position(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        price: float,
        position: int,
        queue_size: int
    ) -> QueuePosition:
        """
        Calculate queue position and execution probability.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            side: "bid" or "ask"
            price: Price level
            position: Position in queue
            queue_size: Total queue size
            
        Returns:
            QueuePosition
        """
        # Probability of execution
        if queue_size > 0:
            probability_of_execution = 1.0 - (position / queue_size)
        else:
            probability_of_execution = 0.0
        
        queue_pos = QueuePosition(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            price=price,
            position=position,
            queue_size=queue_size,
            probability_of_execution=probability_of_execution
        )
        
        # Store in history
        if symbol not in self.queue_positions:
            self.queue_positions[symbol] = []
        self.queue_positions[symbol].append(queue_pos)
        
        return queue_pos
    
    def calculate_order_flow_imbalance(
        self,
        timestamp: datetime,
        symbol: str,
        buy_volume: float,
        sell_volume: float,
        trade_prices: pd.Series,
        window_seconds: int = 300
    ) -> OrderFlowImbalance:
        """
        Calculate order flow imbalance and VPIN.
        
        OFI = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        VPIN = Volume-synchronized Probability of Informed Trading
        Measures the probability that a trade is informed.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            buy_volume: Buy volume in window
            sell_volume: Sell volume in window
            trade_prices: Trade prices in window
            window_seconds: Window size in seconds
            
        Returns:
            OrderFlowImbalance
        """
        # Calculate OFI
        total_volume = buy_volume + sell_volume
        ofi = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate VPIN
        # VPIN = |price change| / volume
        if len(trade_prices) > 1:
            price_changes = trade_prices.diff().abs()
            avg_price_change = price_changes.mean()
            avg_volume = total_volume / len(trade_prices) if len(trade_prices) > 0 else 1
            vpin = avg_price_change / avg_volume if avg_volume > 0 else 0
        else:
            vpin = 0
        
        # Determine imbalance direction
        imbalance_direction = "buy" if ofi > 0 else "sell"
        
        ofi_obj = OrderFlowImbalance(
            timestamp=timestamp,
            symbol=symbol,
            ofi=ofi,
            vpin=vpin,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            imbalance_direction=imbalance_direction
        )
        
        # Store in history
        if symbol not in self.ofi_history:
            self.ofi_history[symbol] = []
        self.ofi_history[symbol].append(ofi_obj)
        
        return ofi_obj
    
    def estimate_kyle_lambda(
        self,
        timestamp: datetime,
        symbol: str,
        trade_prices: pd.Series,
        trade_volumes: pd.Series,
        window_seconds: int = 300
    ) -> KyleLambda:
        """
        Estimate Kyle's lambda (price impact parameter).
        
        Kyle's model: price change = lambda * order flow + noise
        Lambda measures price impact per unit of order flow.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            trade_prices: Trade prices
            trade_volumes: Trade volumes
            window_seconds: Window size in seconds
            
        Returns:
            KyleLambda
        """
        if len(trade_prices) < 2:
            return KyleLambda(
                timestamp=timestamp,
                symbol=symbol,
                lambda_param=0.0,
                informed_trading_intensity=0.0,
                noise_trading_intensity=0.0
            )
        
        # Calculate price changes
        price_changes = trade_prices.diff().values[1:]
        
        # Calculate order flow (signed volume)
        # Assume buys are positive, sells are negative
        # Simplified: use volume as proxy
        order_flow = trade_volumes.values[1:]
        
        # Estimate lambda using regression
        # price_change = lambda * order_flow + noise
        try:
            slope, intercept = np.polyfit(order_flow, price_changes, 1)
            lambda_param = abs(slope)
            
            # Calculate R^2 to estimate informed vs noise trading
            predictions = slope * order_flow + intercept
            ss_res = np.sum((price_changes - predictions) ** 2)
            ss_tot = np.sum((price_changes - np.mean(price_changes)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # Informed trading intensity ~ R^2
            informed_trading_intensity = r_squared
            noise_trading_intensity = 1 - r_squared
            
        except:
            lambda_param = 0.0
            informed_trading_intensity = 0.0
            noise_trading_intensity = 1.0
        
        kyle_lambda = KyleLambda(
            timestamp=timestamp,
            symbol=symbol,
            lambda_param=lambda_param,
            informed_trading_intensity=informed_trading_intensity,
            noise_trading_intensity=noise_trading_intensity
        )
        
        # Store in history
        if symbol not in self.kyle_lambda_history:
            self.kyle_lambda_history[symbol] = []
        self.kyle_lambda_history[symbol].append(kyle_lambda)
        
        return kyle_lambda
    
    def calculate_amihud_illiquidity(
        self,
        timestamp: datetime,
        symbol: str,
        returns: pd.Series,
        volume: pd.Series,
        window_days: int = 20
    ) -> AmihudIlliquidity:
        """
        Calculate Amihud illiquidity measure.
        
        Illiquidity = |return| / volume (in currency units)
        Higher illiquidity = less liquid
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            returns: Returns
            volume: Volume
            window_days: Window for averaging
            
        Returns:
            AmihudIlliquidity
        """
        if len(returns) < window_days:
            return AmihudIlliquidity(
                timestamp=timestamp,
                symbol=symbol,
                illiquidity=0.0,
                avg_daily_illiquidity=0.0,
                liquidity_score=0.0
            )
        
        # Calculate daily illiquidity
        daily_illiquidity = returns.abs() / volume
        
        # Average over window
        avg_illiquidity = daily_illiquidity.rolling(window_days).mean().iloc[-1]
        
        # Liquidity score (inverse of illiquidity)
        liquidity_score = 1.0 / avg_illiquidity if avg_illiquidity > 0 else 0
        
        illiquidity_obj = AmihudIlliquidity(
            timestamp=timestamp,
            symbol=symbol,
            illiquidity=avg_illiquidity,
            avg_daily_illiquidity=avg_illiquidity,
            liquidity_score=liquidity_score
        )
        
        # Store in history
        if symbol not in self.illiquidity_history:
            self.illiquidity_history[symbol] = []
        self.illiquidity_history[symbol].append(illiquidity_obj)
        
        return illiquidity_obj
    
    def calculate_market_impact(
        self,
        timestamp: datetime,
        symbol: str,
        trade_size: float,
        avg_daily_volume: float,
        pre_trade_price: float,
        post_trade_price: float,
        recovery_price: float
    ) -> MarketImpact:
        """
        Calculate market impact.
        
        Market impact = post_trade_price - pre_trade_price
        Temporary impact = recovery_price - post_trade_price
        Permanent impact = post_trade_price - pre_trade_price - temporary impact
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            trade_size: Trade size
            avg_daily_volume: Average daily volume
            pre_trade_price: Price before trade
            post_trade_price: Price after trade
            recovery_price: Price after recovery
            
        Returns:
            MarketImpact
        """
        # Calculate price impact
        price_impact = post_trade_price - pre_trade_price
        temporary_impact = recovery_price - post_trade_price
        permanent_impact = pre_trade_price - recovery_price
        
        # Convert to basis points
        impact_cost_bps = abs(price_impact / pre_trade_price) * 10000
        
        impact_obj = MarketImpact(
            timestamp=timestamp,
            symbol=symbol,
            trade_size=trade_size,
            price_impact=price_impact,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            impact_cost_bps=impact_cost_bps
        )
        
        # Store in history
        if symbol not in self.market_impact_history:
            self.market_impact_history[symbol] = []
        self.market_impact_history[symbol].append(impact_obj)
        
        return impact_obj
    
    def calculate_lob_imbalance(
        self,
        timestamp: datetime,
        symbol: str,
        bid_levels: List[Tuple[float, float]],  # (price, size)
        ask_levels: List[Tuple[float, float]]  # (price, size)
    ) -> LOBImbalance:
        """
        Calculate Limit Order Book imbalance.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            bid_levels: List of (price, size) for bids
            ask_levels: List of (price, size) for asks
            
        Returns:
            LOBImbalance
        """
        # Calculate total bid and ask volume
        bid_volume = sum(size for _, size in bid_levels)
        ask_volume = sum(size for _, size in ask_levels)
        
        # Calculate imbalances
        total_volume = bid_volume + ask_volume
        bid_imbalance = bid_volume / total_volume if total_volume > 0 else 0.5
        ask_imbalance = ask_volume / total_volume if total_volume > 0 else 0.5
        total_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate depth ratio
        depth_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
        
        # Check if skewed
        is_skewed = abs(total_imbalance) > 0.3
        
        lob_imbalance = LOBImbalance(
            timestamp=timestamp,
            symbol=symbol,
            bid_imbalance=bid_imbalance,
            ask_imbalance=ask_imbalance,
            total_imbalance=total_imbalance,
            depth_ratio=depth_ratio,
            is_skewed=is_skewed
        )
        
        # Store in history
        if symbol not in self.lob_imbalance_history:
            self.lob_imbalance_history[symbol] = []
        self.lob_imbalance_history[symbol].append(lob_imbalance)
        
        return lob_imbalance
    
    def get_microstructure_summary(self, symbol: str) -> Dict:
        """Get summary of microstructure metrics for a symbol."""
        summary = {}
        
        # Microprice
        if symbol in self.microprice_history and self.microprice_history[symbol]:
            latest = self.microprice_history[symbol][-1]
            summary['microprice'] = latest.microprice
            summary['mid_price'] = latest.mid_price
            summary['spread'] = latest.spread
        
        # OFI
        if symbol in self.ofi_history and self.ofi_history[symbol]:
            latest = self.ofi_history[symbol][-1]
            summary['ofi'] = latest.ofi
            summary['vpin'] = latest.vpin
            summary['imbalance_direction'] = latest.imbalance_direction
        
        # Kyle Lambda
        if symbol in self.kyle_lambda_history and self.kyle_lambda_history[symbol]:
            latest = self.kyle_lambda_history[symbol][-1]
            summary['kyle_lambda'] = latest.lambda_param
            summary['informed_intensity'] = latest.informed_trading_intensity
        
        # Illiquidity
        if symbol in self.illiquidity_history and self.illiquidity_history[symbol]:
            latest = self.illiquidity_history[symbol][-1]
            summary['amihud_illiquidity'] = latest.illiquidity
            summary['liquidity_score'] = latest.liquidity_score
        
        # LOB Imbalance
        if symbol in self.lob_imbalance_history and self.lob_imbalance_history[symbol]:
            latest = self.lob_imbalance_history[symbol][-1]
            summary['lob_imbalance'] = latest.total_imbalance
            summary['depth_ratio'] = latest.depth_ratio
            summary['is_skewed'] = latest.is_skewed
        
        return summary


if __name__ == "__main__":
    # Test the Market Microstructure Engine
    print("Testing Market Microstructure Engine...")
    
    engine = MarketMicrostructureEngine()
    
    # Calculate microprice
    print("\nCalculating Microprice...")
    microprice = engine.calculate_microprice(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        bid_price=2499.5,
        ask_price=2500.5,
        bid_size=10000,
        ask_size=8000
    )
    print(f"Microprice: {microprice.microprice:.2f}")
    print(f"Mid Price: {microprice.mid_price:.2f}")
    print(f"Spread: {microprice.spread:.2f}")
    
    # Calculate queue position
    print("\nCalculating Queue Position...")
    queue_pos = engine.calculate_queue_position(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        side="bid",
        price=2499.5,
        position=5,
        queue_size=20
    )
    print(f"Position: {queue_pos.position}/{queue_pos.queue_size}")
    print(f"Execution Probability: {queue_pos.probability_of_execution:.2%}")
    
    # Calculate order flow imbalance
    print("\nCalculating Order Flow Imbalance...")
    trade_prices = pd.Series(np.random.normal(2500, 10, 100))
    ofi = engine.calculate_order_flow_imbalance(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        buy_volume=5000000,
        sell_volume=3000000,
        trade_prices=trade_prices
    )
    print(f"OFI: {ofi.ofi:.4f}")
    print(f"VPIN: {ofi.vpin:.6f}")
    print(f"Direction: {ofi.imbalance_direction}")
    
    # Estimate Kyle Lambda
    print("\nEstimating Kyle Lambda...")
    trade_volumes = pd.Series(np.random.normal(100000, 20000, 100))
    kyle_lambda = engine.estimate_kyle_lambda(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        trade_prices=trade_prices,
        trade_volumes=trade_volumes
    )
    print(f"Lambda: {kyle_lambda.lambda_param:.6f}")
    print(f"Informed Intensity: {kyle_lambda.informed_trading_intensity:.2%}")
    
    # Calculate Amihud Illiquidity
    print("\nCalculating Amihud Illiquidity...")
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    volume = pd.Series(np.random.normal(1000000, 200000, 100))
    illiquidity = engine.calculate_amihud_illiquidity(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        returns=returns,
        volume=volume
    )
    print(f"Illiquidity: {illiquidity.illiquidity:.8f}")
    print(f"Liquidity Score: {illiquidity.liquidity_score:.2f}")
    
    # Calculate LOB Imbalance
    print("\nCalculating LOB Imbalance...")
    bid_levels = [(2499.5, 10000), (2499.0, 8000), (2498.5, 5000)]
    ask_levels = [(2500.5, 8000), (2501.0, 6000), (2501.5, 4000)]
    lob_imbalance = engine.calculate_lob_imbalance(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        bid_levels=bid_levels,
        ask_levels=ask_levels
    )
    print(f"Total Imbalance: {lob_imbalance.total_imbalance:.4f}")
    print(f"Depth Ratio: {lob_imbalance.depth_ratio:.2f}")
    print(f"Is Skewed: {lob_imbalance.is_skewed}")
    
    # Get summary
    print("\nMicrostructure Summary:")
    summary = engine.get_microstructure_summary("RELIANCE")
    for key, value in summary.items():
        print(f"  {key}: {value}")
