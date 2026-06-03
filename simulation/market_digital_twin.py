"""
Market Digital Twin - Agent-Based Market Simulator

This module implements an agent-based market simulator that models different
market participants (retail, institutional, HFT) to provide realistic
backtesting with market impact and liquidity effects.

Key Features:
- Agent-based market simulation
- Multiple participant types (retail, institutional, HFT)
- Order book dynamics
- Market impact modeling
- Liquidity provision/consumption
- Realistic price formation
- Calibrated to Indian markets

Based on V4 Blueprint - Institutional Architecture
Priority: Medium (Phase 2.3)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of market participants."""
    RETAIL = "retail"
    INSTITUTIONAL = "institutional"
    HFT = "hft"
    MARKET_MAKER = "market_maker"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class Order:
    """Market order."""
    order_id: str
    agent_id: str
    agent_type: AgentType
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    timestamp: datetime
    is_filled: bool = False
    fill_price: Optional[float] = None
    fill_quantity: int = 0
    
    def __hash__(self):
        return hash(self.order_id)


@dataclass
class Agent:
    """Market participant agent."""
    agent_id: str
    agent_type: AgentType
    capital: float
    risk_aversion: float
    trading_frequency: float  # trades per day
    order_size_distribution: Tuple[float, float]  # (mean, std)
    price_sensitivity: float
    
    def generate_order(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        signal: float = 0.0
    ) -> Optional[Order]:
        """
        Generate order based on agent behavior.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            timestamp: Order timestamp
            signal: Trading signal (-1 to 1)
            
        Returns:
            Order or None
        """
        # Determine if agent should trade
        if random.random() > self.trading_frequency:
            return None
        
        # Determine side based on signal and agent type
        side = self._determine_side(signal)
        
        # Determine order type
        order_type = self._determine_order_type()
        
        # Determine price
        price = self._determine_price(current_price, side, order_type)
        
        # Determine quantity
        quantity = int(np.random.normal(*self.order_size_distribution))
        quantity = max(1, quantity)
        
        order_id = f"{self.agent_id}_{timestamp.strftime('%Y%m%d%H%M%S%f')}_{random.randint(1000, 9999)}"
        
        order = Order(
            order_id=order_id,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            timestamp=timestamp
        )
        
        return order
    
    def _determine_side(self, signal: float) -> OrderSide:
        """Determine order side."""
        if self.agent_type == AgentType.MARKET_MAKER:
            # Market makers provide liquidity
            return OrderSide.BUY if random.random() < 0.5 else OrderSide.SELL
        
        # Other agents follow signal
        if signal > 0.1:
            return OrderSide.BUY
        elif signal < -0.1:
            return OrderSide.SELL
        else:
            return OrderSide.BUY if random.random() < 0.5 else OrderSide.SELL
    
    def _determine_order_type(self) -> OrderType:
        """Determine order type based on agent type."""
        if self.agent_type == AgentType.HFT:
            return OrderType.IOC
        elif self.agent_type == AgentType.RETAIL:
            return OrderType.MARKET
        elif self.agent_type == AgentType.MARKET_MAKER:
            return OrderType.LIMIT
        else:
            return OrderType.LIMIT
    
    def _determine_price(self, current_price: float, side: OrderSide, order_type: OrderType) -> float:
        """Determine order price."""
        if order_type == OrderType.MARKET:
            return current_price
        
        # Limit orders have price offset
        offset = np.random.normal(0, self.price_sensitivity * current_price)
        
        if side == OrderSide.BUY:
            price = current_price - abs(offset)
        else:
            price = current_price + abs(offset)
        
        return price


@dataclass
class OrderBook:
    """Order book for a symbol."""
    symbol: str
    bids: Dict[float, int]  # price -> quantity
    asks: Dict[float, int]  # price -> quantity
    last_price: float
    last_update: datetime
    
    def get_best_bid(self) -> Tuple[float, int]:
        """Get best bid (price, quantity)."""
        if not self.bids:
            return 0.0, 0
        best_price = max(self.bids.keys())
        return best_price, self.bids[best_price]
    
    def get_best_ask(self) -> Tuple[float, int]:
        """Get best ask (price, quantity)."""
        if not self.asks:
            return 0.0, 0
        best_price = min(self.asks.keys())
        return best_price, self.asks[best_price]
    
    def get_mid_price(self) -> float:
        """Get mid price."""
        bid_price, _ = self.get_best_bid()
        ask_price, _ = self.get_best_ask()
        if bid_price == 0 or ask_price == 0:
            return self.last_price
        return (bid_price + ask_price) / 2
    
    def get_spread(self) -> float:
        """Get bid-ask spread."""
        bid_price, _ = self.get_best_bid()
        ask_price, _ = self.get_best_ask()
        if bid_price == 0 or ask_price == 0:
            return 0.0
        return ask_price - bid_price
    
    def add_order(self, order: Order) -> None:
        """Add order to book."""
        if order.side == OrderSide.BUY:
            self.bids[order.price] = self.bids.get(order.price, 0) + order.quantity
        else:
            self.asks[order.price] = self.asks.get(order.price, 0) + order.quantity
        self.last_update = order.timestamp
    
    def match_order(self, order: Order) -> Tuple[float, int]:
        """
        Match order against book.
        
        Returns:
            (fill_price, fill_quantity)
        """
        fill_price = 0.0
        fill_quantity = 0
        remaining_qty = order.quantity
        
        if order.side == OrderSide.BUY:
            # Match against asks
            ask_prices = sorted(self.asks.keys())
            for price in ask_prices:
                if remaining_qty <= 0 or price > order.price:
                    break
                
                available_qty = self.asks[price]
                matched_qty = min(remaining_qty, available_qty)
                
                fill_price = (fill_price * fill_quantity + price * matched_qty) / (fill_quantity + matched_qty) if fill_quantity > 0 else price
                fill_quantity += matched_qty
                remaining_qty -= matched_qty
                
                # Update book
                self.asks[price] -= matched_qty
                if self.asks[price] <= 0:
                    del self.asks[price]
        else:
            # Match against bids
            bid_prices = sorted(self.bids.keys(), reverse=True)
            for price in bid_prices:
                if remaining_qty <= 0 or price < order.price:
                    break
                
                available_qty = self.bids[price]
                matched_qty = min(remaining_qty, available_qty)
                
                fill_price = (fill_price * fill_quantity + price * matched_qty) / (fill_quantity + matched_qty) if fill_quantity > 0 else price
                fill_quantity += matched_qty
                remaining_qty -= matched_qty
                
                # Update book
                self.bids[price] -= matched_qty
                if self.bids[price] <= 0:
                    del self.bids[price]
        
        # Update last price if filled
        if fill_quantity > 0:
            self.last_price = fill_price
            self.last_update = order.timestamp
        
        return fill_price, fill_quantity


class MarketDigitalTwin:
    """
    Market digital twin with agent-based simulation.
    
    This class simulates market dynamics using multiple agent types
    to provide realistic backtesting with market impact and liquidity effects.
    """
    
    def __init__(
        self,
        num_retail_agents: int = 1000,
        num_institutional_agents: int = 50,
        num_hft_agents: int = 20,
        num_market_makers: int = 10
    ):
        """
        Initialize market digital twin.
        
        Args:
            num_retail_agents: Number of retail agents
            num_institutional_agents: Number of institutional agents
            num_hft_agents: Number of HFT agents
            num_market_makers: Number of market makers
        """
        self.agents: List[Agent] = []
        self.order_books: Dict[str, OrderBook] = {}
        self.trade_history: List[Dict] = []
        
        # Initialize agents
        self._initialize_agents(
            num_retail_agents,
            num_institutional_agents,
            num_hft_agents,
            num_market_makers
        )
        
        logger.info(f"MarketDigitalTwin initialized with {len(self.agents)} agents")
    
    def _initialize_agents(
        self,
        num_retail: int,
        num_institutional: int,
        num_hft: int,
        num_market_makers: int
    ) -> None:
        """Initialize market agents."""
        
        # Retail agents: low capital, high frequency, small orders
        for i in range(num_retail):
            agent = Agent(
                agent_id=f"retail_{i}",
                agent_type=AgentType.RETAIL,
                capital=np.random.uniform(100000, 1000000),
                risk_aversion=np.random.uniform(0.5, 0.9),
                trading_frequency=np.random.uniform(0.01, 0.1),
                order_size_distribution=(100, 50),
                price_sensitivity=0.001
            )
            self.agents.append(agent)
        
        # Institutional agents: high capital, low frequency, large orders
        for i in range(num_institutional):
            agent = Agent(
                agent_id=f"institutional_{i}",
                agent_type=AgentType.INSTITUTIONAL,
                capital=np.random.uniform(10000000, 100000000),
                risk_aversion=np.random.uniform(0.3, 0.7),
                trading_frequency=np.random.uniform(0.001, 0.01),
                order_size_distribution=(10000, 5000),
                price_sensitivity=0.005
            )
            self.agents.append(agent)
        
        # HFT agents: medium capital, very high frequency, small orders
        for i in range(num_hft):
            agent = Agent(
                agent_id=f"hft_{i}",
                agent_type=AgentType.HFT,
                capital=np.random.uniform(5000000, 20000000),
                risk_aversion=np.random.uniform(0.1, 0.3),
                trading_frequency=np.random.uniform(0.5, 1.0),
                order_size_distribution=(500, 200),
                price_sensitivity=0.0001
            )
            self.agents.append(agent)
        
        # Market makers: provide liquidity
        for i in range(num_market_makers):
            agent = Agent(
                agent_id=f"mm_{i}",
                agent_type=AgentType.MARKET_MAKER,
                capital=np.random.uniform(50000000, 200000000),
                risk_aversion=np.random.uniform(0.2, 0.4),
                trading_frequency=np.random.uniform(0.8, 1.0),
                order_size_distribution=(5000, 2000),
                price_sensitivity=0.002
            )
            self.agents.append(agent)
    
    def initialize_order_book(self, symbol: str, initial_price: float) -> None:
        """
        Initialize order book for a symbol.
        
        Args:
            symbol: Stock symbol
            initial_price: Initial price
        """
        # Initialize with some liquidity
        bids = {}
        asks = {}
        
        # Add initial bid/ask levels
        for i in range(5):
            bid_price = initial_price * (1 - (i + 1) * 0.001)
            ask_price = initial_price * (1 + (i + 1) * 0.001)
            bids[bid_price] = int(np.random.uniform(1000, 10000))
            asks[ask_price] = int(np.random.uniform(1000, 10000))
        
        self.order_books[symbol] = OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            last_price=initial_price,
            last_update=datetime.now()
        )
        
        logger.info(f"Initialized order book for {symbol} at {initial_price}")
    
    def simulate_day(
        self,
        symbol: str,
        signal: float = 0.0,
        num_steps: int = 390  # 1 trading day in minutes
    ) -> pd.DataFrame:
        """
        Simulate one trading day.
        
        Args:
            symbol: Stock symbol
            signal: Trading signal (-1 to 1)
            num_steps: Number of simulation steps
            
        Returns:
            DataFrame with simulation results
        """
        if symbol not in self.order_books:
            initial_price = 1000.0  # Default
            self.initialize_order_book(symbol, initial_price)
        
        order_book = self.order_books[symbol]
        
        results = []
        current_time = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
        
        for step in range(num_steps):
            # Generate orders from agents
            orders = []
            for agent in self.agents:
                order = agent.generate_order(
                    symbol=symbol,
                    current_price=order_book.last_price,
                    timestamp=current_time,
                    signal=signal
                )
                if order:
                    orders.append(order)
            
            # Process orders
            for order in orders:
                # Add to book first
                if order.order_type == OrderType.LIMIT:
                    order_book.add_order(order)
                
                # Try to match
                fill_price, fill_qty = order_book.match_order(order)
                
                if fill_qty > 0:
                    order.is_filled = True
                    order.fill_price = fill_price
                    order.fill_quantity = fill_qty
                    
                    # Record trade
                    self.trade_history.append({
                        'timestamp': current_time,
                        'symbol': symbol,
                        'price': fill_price,
                        'quantity': fill_qty,
                        'agent_type': order.agent_type.value,
                        'side': order.side.value
                    })
            
            # Record market state
            bid_price, bid_qty = order_book.get_best_bid()
            ask_price, ask_qty = order_book.get_best_ask()
            mid_price = order_book.get_mid_price()
            spread = order_book.get_spread()
            
            results.append({
                'timestamp': current_time,
                'symbol': symbol,
                'mid_price': mid_price,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'spread': spread,
                'bid_qty': bid_qty,
                'ask_qty': ask_qty,
                'num_orders': len(orders),
                'num_trades': len([t for t in self.trade_history if t['timestamp'] == current_time])
            })
            
            # Advance time
            current_time += timedelta(minutes=1)
        
        return pd.DataFrame(results)
    
    def calculate_market_impact(
        self,
        symbol: str,
        order_size: int,
        side: OrderSide
    ) -> Tuple[float, float]:
        """
        Calculate market impact for an order.
        
        Args:
            symbol: Stock symbol
            order_size: Order size
            side: Order side
            
        Returns:
            (price_impact, quantity_impact)
        """
        if symbol not in self.order_books:
            return 0.0, 0.0
        
        order_book = self.order_books[symbol]
        
        # Simplified market impact model
        # Impact = k * (order_size / avg_daily_volume)^0.5
        avg_daily_volume = sum(order_book.bids.values()) + sum(order_book.asks.values())
        
        if avg_daily_volume == 0:
            return 0.0, 0.0
        
        participation_rate = order_size / avg_daily_volume
        price_impact = 0.01 * np.sqrt(participation_rate)  # 1% impact per sqrt(participation)
        quantity_impact = min(order_size, avg_daily_volume * 0.1)  # Max 10% of available liquidity
        
        return price_impact, quantity_impact
    
    def print_simulation_report(self, results: pd.DataFrame) -> None:
        """Print simulation report."""
        print("\n" + "="*60)
        print("MARKET DIGITAL TWIN SIMULATION REPORT")
        print("="*60)
        
        print(f"\nSimulation Steps: {len(results)}")
        print(f"Total Trades: {len(self.trade_history)}")
        
        if not results.empty:
            print(f"\nPrice Statistics:")
            print(f"  Start Price: {results['mid_price'].iloc[0]:.2f}")
            print(f"  End Price: {results['mid_price'].iloc[-1]:.2f}")
            print(f"  Price Change: {(results['mid_price'].iloc[-1] - results['mid_price'].iloc[0]):.2f}")
            print(f"  Volatility: {results['mid_price'].std():.4f}")
            
            print(f"\nSpread Statistics:")
            print(f"  Average Spread: {results['spread'].mean():.4f}")
            print(f"  Max Spread: {results['spread'].max():.4f}")
            
            print(f"\nOrder Flow:")
            print(f"  Total Orders: {results['num_orders'].sum()}")
            print(f"  Total Trades: {results['num_trades'].sum()}")
            print(f"  Trade Rate: {results['num_trades'].mean():.2f} per step")
        
        print(f"\nAgent Distribution:")
        agent_counts = {}
        for agent in self.agents:
            agent_counts[agent.agent_type.value] = agent_counts.get(agent.agent_type.value, 0) + 1
        
        for agent_type, count in agent_counts.items():
            print(f"  {agent_type}: {count}")
        
        print("\n" + "="*60)


def sample_market_digital_twin():
    """Demonstrate market digital twin."""
    print("=== Market Digital Twin Demo ===\n")
    
    # Initialize digital twin
    twin = MarketDigitalTwin(
        num_retail_agents=100,
        num_institutional_agents=10,
        num_hft_agents=5,
        num_market_makers=5
    )
    
    # Simulate a day
    print("Simulating trading day...")
    results = twin.simulate_day(
        symbol='RELIANCE',
        signal=0.2,  # Positive signal
        num_steps=100  # 100 minutes for demo
    )
    
    # Print report
    twin.print_simulation_report(results)
    
    # Calculate market impact
    print("\nCalculating market impact...")
    price_impact, qty_impact = twin.calculate_market_impact('RELIANCE', 10000, OrderSide.BUY)
    print(f"  Price Impact: {price_impact:.4f} ({price_impact*100:.2f}%)")
    print(f"  Quantity Impact: {qty_impact}")
    
    print("\n=== Market Digital Twin Demo Complete ===")
    print("Key capabilities:")
    print("- Agent-based market simulation")
    print("- Multiple participant types (retail, institutional, HFT, market makers)")
    print("- Order book dynamics")
    print("- Market impact modeling")
    print("- Realistic price formation")


if __name__ == "__main__":
    sample_market_digital_twin()
