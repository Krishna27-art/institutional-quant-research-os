"""
Event Replay Engine
Re-simulate specific historical days exactly as they happened
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json

from time_machine_simulator import TimeMachineSimulator, DataType, TimeSnapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayGranularity(Enum):
    """Granularity of replay"""
    TICK = "tick"
    MINUTE = "minute"
    FIVE_MINUTE = "five_minute"
    HOUR = "hour"
    DAY = "day"


@dataclass
class ReplayEvent:
    """Single replay event"""
    timestamp: datetime
    event_type: str  # "tick", "order", "fill", "signal"
    symbol: str
    data: Dict[str, Any]
    processed: bool = False


@dataclass
class ReplayState:
    """State during replay"""
    current_time: datetime
    portfolio: Dict[str, float]  # symbol -> quantity
    cash: float
    total_pnl: float
    fills: List[Dict]
    signals: List[Dict]
    orders: List[Dict]


@dataclass
class ReplayResult:
    """Result of event replay"""
    replay_date: datetime
    granularity: ReplayGranularity
    start_time: datetime
    end_time: datetime
    total_events: int
    processed_events: int
    final_pnl: float
    final_portfolio_value: float
    fill_ratio: float
    slippage: float
    events: List[ReplayEvent]
    state_history: List[ReplayState]
    timestamp: datetime


class OrderBook:
    """Simulated order book for replay"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: List[Tuple[float, float]] = []  # (price, quantity)
        self.asks: List[Tuple[float, float]] = []  # (price, quantity)
        self.last_price: float = 0.0
        self.last_volume: float = 0.0
    
    def update_from_tick(self, tick_data: Dict[str, Any]) -> None:
        """Update order book from tick data"""
        self.last_price = tick_data.get('price', self.last_price)
        self.last_volume = tick_data.get('volume', self.last_volume)
        
        # Simulate bid/ask spread
        spread = self.last_price * 0.001  # 0.1% spread
        self.bids = [(self.last_price - spread/2, 1000)]
        self.asks = [(self.last_price + spread/2, 1000)]
    
    def get_best_bid(self) -> float:
        """Get best bid price"""
        return self.bids[0][0] if self.bids else 0.0
    
    def get_best_ask(self) -> float:
        """Get best ask price"""
        return self.asks[0][0] if self.asks else 0.0
    
    def execute_order(
        self,
        side: str,
        quantity: float
    ) -> Tuple[float, float, float]:
        """
        Execute order on simulated order book
        
        Args:
            side: "BUY" or "SELL"
            quantity: Order quantity
            
        Returns:
            Tuple of (fill_price, fill_quantity, slippage)
        """
        if side == "BUY":
            execution_price = self.get_best_ask()
        else:
            execution_price = self.get_best_bid()
        
        # Simulate partial fills and slippage
        fill_quantity = min(quantity, 1000)  # Simulate liquidity constraint
        slippage = abs(execution_price - self.last_price) / self.last_price if self.last_price > 0 else 0.0
        
        return execution_price, fill_quantity, slippage


class EventReplayEngine:
    """
    Event Replay Engine for re-simulating historical days
    """
    
    def __init__(
        self,
        time_machine: TimeMachineSimulator,
        initial_capital: float = 100_000_000
    ):
        self.time_machine = time_machine
        self.initial_capital = initial_capital
        self.order_books: Dict[str, OrderBook] = {}
        
        logger.info("Event Replay Engine initialized")
    
    def replay_day(
        self,
        date: datetime,
        symbols: List[str],
        strategy_func: callable,
        granularity: ReplayGranularity = ReplayGranularity.MINUTE,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ReplayResult:
        """
        Replay a specific historical day
        
        Args:
            date: Date to replay
            symbols: Symbols to include
            strategy_func: Strategy function that generates signals
            granularity: Replay granularity
            start_time: Start time (default 9:15)
            end_time: End time (default 15:30)
            
        Returns:
            ReplayResult
        """
        if start_time is None:
            start_time = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=9, minutes=15)
        
        if end_time is None:
            end_time = datetime.combine(date.date(), datetime.min.time()) + timedelta(hours=15, minutes=30)
        
        logger.info(f"Replaying day {date.date()} from {start_time.time()} to {end_time.time()}")
        
        # Initialize order books
        for symbol in symbols:
            self.order_books[symbol] = OrderBook(symbol)
        
        # Get time series snapshots for the day
        if granularity == ReplayGranularity.MINUTE:
            frequency = '1min'
        elif granularity == ReplayGranularity.FIVE_MINUTE:
            frequency = '5min'
        elif granularity == ReplayGranularity.HOUR:
            frequency = '1H'
        else:
            frequency = '1D'
        
        snapshots = self.time_machine.get_snapshot_range(
            start_date=start_time,
            end_date=end_time,
            frequency=frequency,
            symbols=symbols,
            data_types=[DataType.OHLCV],
            lookback_days=5
        )
        
        # Initialize replay state
        replay_state = ReplayState(
            current_time=start_time,
            portfolio={symbol: 0.0 for symbol in symbols},
            cash=self.initial_capital,
            total_pnl=0.0,
            fills=[],
            signals=[],
            orders=[]
        )
        
        state_history = []
        events = []
        
        # Process each snapshot
        for snapshot in snapshots:
            # Update order books
            for symbol in symbols:
                if symbol in snapshot.features.index:
                    tick_data = {
                        'price': snapshot.features.loc[symbol, 'close'],
                        'volume': snapshot.features.loc[symbol, 'volume'],
                    }
                    self.order_books[symbol].update_from_tick(tick_data)
            
            # Generate signals using strategy
            signals = strategy_func(snapshot, replay_state)
            
            # Execute orders based on signals
            for signal in signals:
                symbol = signal['symbol']
                side = signal['side']
                quantity = signal['quantity']
                
                # Execute order
                order_book = self.order_books[symbol]
                fill_price, fill_quantity, slippage = order_book.execute_order(side, quantity)
                
                # Update portfolio
                if side == "BUY":
                    cost = fill_price * fill_quantity
                    replay_state.cash -= cost
                    replay_state.portfolio[symbol] += fill_quantity
                else:  # SELL
                    proceeds = fill_price * fill_quantity
                    replay_state.cash += proceeds
                    replay_state.portfolio[symbol] -= fill_quantity
                
                # Record fill
                fill = {
                    'timestamp': snapshot.timestamp,
                    'symbol': symbol,
                    'side': side,
                    'quantity': fill_quantity,
                    'price': fill_price,
                    'slippage': slippage,
                }
                replay_state.fills.append(fill)
                
                # Create event
                event = ReplayEvent(
                    timestamp=snapshot.timestamp,
                    event_type="fill",
                    symbol=symbol,
                    data=fill
                )
                events.append(event)
            
            # Calculate portfolio value
            portfolio_value = replay_state.cash
            for symbol, qty in replay_state.portfolio.items():
                if qty != 0:
                    current_price = self.order_books[symbol].last_price
                    portfolio_value += qty * current_price
            
            # Calculate PnL
            replay_state.total_pnl = portfolio_value - self.initial_capital
            
            # Update state
            replay_state.current_time = snapshot.timestamp
            state_history.append(replay_state)
        
        # Calculate final metrics
        final_portfolio_value = replay_state.cash
        for symbol, qty in replay_state.portfolio.items():
            if qty != 0:
                current_price = self.order_books[symbol].last_price
                final_portfolio_value += qty * current_price
        
        total_fills = len(replay_state.fills)
        total_orders = total_fills  # Assume 1:1 for simplicity
        fill_ratio = total_fills / total_orders if total_orders > 0 else 0.0
        
        # Calculate average slippage
        avg_slippage = np.mean([f['slippage'] for f in replay_state.fills]) if replay_state.fills else 0.0
        
        result = ReplayResult(
            replay_date=date,
            granularity=granularity,
            start_time=start_time,
            end_time=end_time,
            total_events=len(events),
            processed_events=len(events),
            final_pnl=replay_state.total_pnl,
            final_portfolio_value=final_portfolio_value,
            fill_ratio=fill_ratio,
            slippage=avg_slippage,
            events=events,
            state_history=state_history,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Replay complete: PnL={result.final_pnl:,.2f}, "
            f"Fill ratio={result.fill_ratio:.2%}, Slippage={result.slippage:.4%}"
        )
        
        return result
    
    def replay_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        strategy_func: callable,
        granularity: ReplayGranularity = ReplayGranularity.MINUTE
    ) -> List[ReplayResult]:
        """
        Replay a range of days
        
        Args:
            start_date: Start date
            end_date: End date
            symbols: Symbols to include
            strategy_func: Strategy function
            granularity: Replay granularity
            
        Returns:
            List of ReplayResults
        """
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        results = []
        
        for date in dates:
            try:
                result = self.replay_day(
                    date=date,
                    symbols=symbols,
                    strategy_func=strategy_func,
                    granularity=granularity
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to replay {date.date()}: {e}")
        
        logger.info(f"Replayed {len(results)} days from {start_date.date()} to {end_date.date()}")
        
        return results
    
    def compare_replays(
        self,
        replay1: ReplayResult,
        replay2: ReplayResult
    ) -> Dict[str, Any]:
        """
        Compare two replay results
        
        Args:
            replay1: First replay result
            replay2: Second replay result
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'replay1_date': replay1.replay_date.date(),
            'replay2_date': replay2.replay_date.date(),
            'pnl_difference': replay2.final_pnl - replay1.final_pnl,
            'pnl_percentage': (replay2.final_pnl - replay1.final_pnl) / abs(replay1.final_pnl) if replay1.final_pnl != 0 else 0,
            'fill_ratio_difference': replay2.fill_ratio - replay1.fill_ratio,
            'slippage_difference': replay2.slippage - replay1.slippage,
            'events_difference': replay2.total_events - replay1.total_events,
        }
        
        return comparison
    
    def save_replay(self, result: ReplayResult, save_path: str) -> None:
        """Save replay result to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        result_dict = {
            'replay_date': result.replay_date.isoformat(),
            'granularity': result.granularity.value,
            'start_time': result.start_time.isoformat(),
            'end_time': result.end_time.isoformat(),
            'total_events': result.total_events,
            'processed_events': result.processed_events,
            'final_pnl': result.final_pnl,
            'final_portfolio_value': result.final_portfolio_value,
            'fill_ratio': result.fill_ratio,
            'slippation': result.slippage,
            'timestamp': result.timestamp.isoformat(),
            'events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'event_type': e.event_type,
                    'symbol': e.symbol,
                    'data': e.data,
                }
                for e in result.events
            ],
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved replay to {save_path}")
    
    def load_replay(self, load_path: str) -> ReplayResult:
        """Load replay result from file"""
        with open(load_path, 'r') as f:
            result_dict = json.load(f)
        
        events = [
            ReplayEvent(
                timestamp=datetime.fromisoformat(e['timestamp']),
                event_type=e['event_type'],
                symbol=e['symbol'],
                data=e['data']
            )
            for e in result_dict['events']
        ]
        
        result = ReplayResult(
            replay_date=datetime.fromisoformat(result_dict['replay_date']),
            granularity=ReplayGranularity(result_dict['granularity']),
            start_time=datetime.fromisoformat(result_dict['start_time']),
            end_time=datetime.fromisoformat(result_dict['end_time']),
            total_events=result_dict['total_events'],
            processed_events=result_dict['processed_events'],
            final_pnl=result_dict['final_pnl'],
            final_portfolio_value=result_dict['final_portfolio_value'],
            fill_ratio=result_dict['fill_ratio'],
            slippage=result_dict['slippage'],
            events=events,
            state_history=[],
            timestamp=datetime.fromisoformat(result_dict['timestamp'])
        )
        
        logger.info(f"Loaded replay from {load_path}")
        
        return result


def sample_strategy(snapshot: TimeSnapshot, state: ReplayState) -> List[Dict]:
    """Sample strategy for testing"""
    signals = []
    
    for symbol, row in snapshot.features.iterrows():
        # Simple momentum strategy
        if row['returns_5d'] > 0.02:
            signals.append({
                'symbol': symbol,
                'side': 'BUY',
                'quantity': 100,
            })
        elif row['returns_5d'] < -0.02:
            signals.append({
                'symbol': symbol,
                'side': 'SELL',
                'quantity': 100,
            })
    
    return signals


def simulate_event_replay():
    """Simulate event replay"""
    
    print("="*60)
    print("EVENT REPLAY ENGINE SIMULATION")
    print("="*60)
    
    # Initialize time machine
    time_machine = TimeMachineSimulator()
    
    # Initialize replay engine
    replay_engine = EventReplayEngine(time_machine, initial_capital=10_000_000)
    
    # Replay a single day
    print("\n1. Replaying single day...")
    result = replay_engine.replay_day(
        date=datetime(2022, 1, 15),
        symbols=['NIFTY', 'BANKNIFTY'],
        strategy_func=sample_strategy,
        granularity=ReplayGranularity.MINUTE
    )
    
    print(f"  Date: {result.replay_date.date()}")
    print(f"  Final PnL: ₹{result.final_pnl:,.2f}")
    print(f"  Final Portfolio Value: ₹{result.final_portfolio_value:,.2f}")
    print(f"  Fill Ratio: {result.fill_ratio:.2%}")
    print(f"  Slippage: {result.slippage:.4%}")
    print(f"  Total Events: {result.total_events}")
    
    # Show sample events
    print("\n2. Sample events:")
    for event in result.events[:5]:
        print(f"  {event.timestamp.time()} - {event.event_type} - {event.symbol}")
    
    # Replay range
    print("\n3. Replaying date range...")
    results = replay_engine.replay_range(
        start_date=datetime(2022, 1, 10),
        end_date=datetime(2022, 1, 14),
        symbols=['NIFTY'],
        strategy_func=sample_strategy,
        granularity=ReplayGranularity.MINUTE
    )
    
    print(f"  Replayed {len(results)} days")
    
    # Calculate aggregate metrics
    total_pnl = sum(r.final_pnl for r in results)
    avg_fill_ratio = np.mean([r.fill_ratio for r in results])
    avg_slippage = np.mean([r.slippage for r in results])
    
    print(f"  Total PnL: ₹{total_pnl:,.2f}")
    print(f"  Average Fill Ratio: {avg_fill_ratio:.2%}")
    print(f"  Average Slippage: {avg_slippage:.4%}")
    
    # Compare replays
    print("\n4. Comparing replays...")
    if len(results) >= 2:
        comparison = replay_engine.compare_replays(results[0], results[1])
        print(f"  PnL difference: ₹{comparison['pnl_difference']:,.2f}")
        print(f"  PnL percentage: {comparison['pnl_percentage']:.2%}")
    
    # Save replay
    print("\n5. Saving replay...")
    replay_engine.save_replay(result, "data/event_replay_result.json")
    print("  Saved to data/event_replay_result.json")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_event_replay()
