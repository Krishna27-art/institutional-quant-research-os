"""
Opening Range Breakout (ORB) Strategy with Relative Volume

Implements the ORB strategy with relative volume filtering for enhanced
signal quality. The strategy trades breakouts from the opening range
(usually first 15-30 minutes) with volume confirmation.

Key Features:
- Opening range breakout detection
- Relative volume filtering (RV > threshold)
- Time-based exit (end of day or time stop)
- Volatility-adjusted position sizing
- Multiple timeframe support
- Risk management with stop-loss

Based on Blueprint Week 5-6: Alpha Models (Classical)
Reference: Crabel (1990) - Day Trading with Short Term Price Patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import time, datetime
import logging

logger = logging.getLogger(__name__)


class ORBSignal(Enum):
    """ORB signal classification."""
    LONG_BREAKOUT = 1
    SHORT_BREAKOUT = -1
    NO_BREAKOUT = 0
    INSUFFICIENT_VOLUME = -2


class ORBStrategy:
    """
    Opening Range Breakout Strategy with Relative Volume.
    
    The ORB strategy identifies breakouts from the opening range and
    filters signals using relative volume to ensure liquidity and
    conviction behind the move.
    """
    
    def __init__(
        self,
        open_range_minutes: int = 15,
        volume_window: int = 20,
        relative_volume_threshold: float = 1.5,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.02,
        max_hold_minutes: int = 180
    ):
        """
        Initialize ORB strategy.
        
        Args:
            open_range_minutes: Duration of opening range in minutes
            volume_window: Window for average volume calculation
            relative_volume_threshold: Minimum RV for signal
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_hold_minutes: Maximum holding time in minutes
        """
        self.open_range_minutes = open_range_minutes
        self.volume_window = volume_window
        self.relative_volume_threshold = relative_volume_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_minutes = max_hold_minutes
    
    def calculate_opening_range(
        self,
        data: pd.DataFrame,
        timestamp: datetime
    ) -> Dict:
        """
        Calculate opening range high and low.
        
        Args:
            data: OHLCV data
            timestamp: Current timestamp
            
        Returns:
            Dictionary with OR high, low, and volume
        """
        # Get data for opening range
        open_range_start = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
        open_range_end = open_range_start.replace(
            minute=open_range_start.minute + self.open_range_minutes
        )
        
        # Filter data for opening range
        or_data = data[
            (data.index >= open_range_start) & (data.index < open_range_end)
        ]
        
        if len(or_data) == 0:
            return {
                'or_high': None,
                'or_low': None,
                'or_volume': 0,
                'or_range': 0
            }
        
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_volume = or_data['volume'].sum()
        or_range = or_high - or_low
        
        return {
            'or_high': or_high,
            'or_low': or_low,
            'or_volume': or_volume,
            'or_range': or_range
        }
    
    def calculate_relative_volume(
        self,
        data: pd.DataFrame,
        current_volume: float
    ) -> float:
        """
        Calculate relative volume.
        
        Args:
            data: Historical data
            current_volume: Current volume
            
        Returns:
            Relative volume ratio
        """
        if len(data) < self.volume_window:
            return 0.0
        
        # Calculate average volume over window
        avg_volume = data['volume'].iloc[-self.volume_window:].mean()
        
        if avg_volume == 0:
            return 0.0
        
        relative_volume = current_volume / avg_volume
        
        return relative_volume
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        timestamp: datetime,
        current_price: float,
        current_volume: float
    ) -> Dict:
        """
        Generate ORB signal.
        
        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp
            current_price: Current price
            current_volume: Current volume
            
        Returns:
            Dictionary with signal and trade parameters
        """
        # Calculate opening range
        or_info = self.calculate_opening_range(data, timestamp)
        
        if or_info['or_high'] is None:
            return {
                'signal': ORBSignal.NO_BREAKOUT,
                'reason': 'Opening range not available'
            }
        
        # Check if we're past opening range
        open_range_end = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
        open_range_end = open_range_end.replace(
            minute=open_range_end.minute + self.open_range_minutes
        )
        
        if timestamp < open_range_end:
            return {
                'signal': ORBSignal.NO_BREAKOUT,
                'reason': 'Still in opening range'
            }
        
        # Calculate relative volume
        relative_volume = self.calculate_relative_volume(data, current_volume)
        
        # Check volume filter
        if relative_volume < self.relative_volume_threshold:
            return {
                'signal': ORBSignal.INSUFFICIENT_VOLUME,
                'reason': f'RV {relative_volume:.2f} below threshold {self.relative_volume_threshold}',
                'relative_volume': relative_volume
            }
        
        # Check for breakout
        if current_price > or_info['or_high']:
            # Long breakout
            signal = ORBSignal.LONG_BREAKOUT
            entry_price = current_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
            
        elif current_price < or_info['or_low']:
            # Short breakout
            signal = ORBSignal.SHORT_BREAKOUT
            entry_price = current_price
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
            
        else:
            signal = ORBSignal.NO_BREAKOUT
            entry_price = None
            stop_loss = None
            take_profit = None
        
        return {
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'or_high': or_info['or_high'],
            'or_low': or_info['or_low'],
            'or_range': or_info['or_range'],
            'relative_volume': relative_volume
        }
    
    def check_exit_conditions(
        self,
        position: Dict,
        current_price: float,
        entry_time: datetime,
        current_time: datetime
    ) -> Dict:
        """
        Check exit conditions for open position.
        
        Args:
            position: Position dictionary
            current_price: Current price
            entry_time: Entry time
            current_time: Current time
            
        Returns:
            Dictionary with exit signal and reason
        """
        signal = position['signal']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Check stop loss
        if signal == ORBSignal.LONG_BREAKOUT:
            if current_price <= stop_loss:
                return {
                    'exit': True,
                    'reason': 'STOP_LOSS',
                    'exit_price': current_price,
                    'pnl': (current_price - entry_price) / entry_price
                }
            elif current_price >= take_profit:
                return {
                    'exit': True,
                    'reason': 'TAKE_PROFIT',
                    'exit_price': current_price,
                    'pnl': (current_price - entry_price) / entry_price
                }
        elif signal == ORBSignal.SHORT_BREAKOUT:
            if current_price >= stop_loss:
                return {
                    'exit': True,
                    'reason': 'STOP_LOSS',
                    'exit_price': current_price,
                    'pnl': (entry_price - current_price) / entry_price
                }
            elif current_price <= take_profit:
                return {
                    'exit': True,
                    'reason': 'TAKE_PROFIT',
                    'exit_price': current_price,
                    'pnl': (entry_price - current_price) / entry_price
                }
        
        # Check time stop
        hold_time = (current_time - entry_time).total_seconds() / 60
        if hold_time >= self.max_hold_minutes:
            return {
                'exit': True,
                'reason': 'TIME_STOP',
                'exit_price': current_price,
                'pnl': (current_price - entry_price) / entry_price if signal == ORBSignal.LONG_BREAKOUT 
                      else (entry_price - current_price) / entry_price
            }
        
        return {
            'exit': False,
            'reason': None
        }
    
    def backtest(
        self,
        data: pd.DataFrame,
        symbol: str = 'TEST'
    ) -> pd.DataFrame:
        """
        Backtest ORB strategy.
        
        Args:
            data: OHLCV data with datetime index
            symbol: Symbol name
            
        Returns:
            DataFrame with trade results
        """
        trades = []
        current_position = None
        
        for i in range(len(data)):
            timestamp = data.index[i]
            row = data.iloc[i]
            
            # Check exit conditions if position exists
            if current_position is not None:
                exit_check = self.check_exit_conditions(
                    current_position,
                    row['close'],
                    current_position['entry_time'],
                    timestamp
                )
                
                if exit_check['exit']:
                    # Close position
                    trade = {
                        'symbol': symbol,
                        'entry_time': current_position['entry_time'],
                        'exit_time': timestamp,
                        'signal': current_position['signal'].value,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_check['exit_price'],
                        'pnl': exit_check['pnl'],
                        'exit_reason': exit_check['reason'],
                        'or_high': current_position['or_high'],
                        'or_low': current_position['or_low'],
                        'relative_volume': current_position['relative_volume']
                    }
                    trades.append(trade)
                    current_position = None
                    continue
            
            # Generate new signal
            signal = self.generate_signal(
                data.iloc[:i+1],
                timestamp,
                row['close'],
                row['volume']
            )
            
            if signal['signal'] in [ORBSignal.LONG_BREAKOUT, ORBSignal.SHORT_BREAKOUT]:
                current_position = {
                    'signal': signal['signal'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit'],
                    'entry_time': timestamp,
                    'or_high': signal['or_high'],
                    'or_low': signal['or_low'],
                    'relative_volume': signal['relative_volume']
                }
        
        return pd.DataFrame(trades)


class ORBPortfolio:
    """
    Portfolio manager for ORB strategy across multiple assets.
    """
    
    def __init__(
        self,
        max_positions: int = 5,
        position_size_pct: float = 0.2,
        max_drawdown_pct: float = 0.05
    ):
        """
        Initialize ORB portfolio.
        
        Args:
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as percentage of capital
            max_drawdown_pct: Maximum drawdown before stopping
        """
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        self.positions: Dict[str, Dict] = {}
        self.daily_pnl = 0.0
        self.peak_capital = 1.0
        self.current_capital = 1.0
    
    def add_position(
        self,
        symbol: str,
        signal: ORBSignal,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime
    ) -> bool:
        """
        Add position to portfolio.
        
        Args:
            symbol: Asset symbol
            signal: ORB signal
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            timestamp: Entry timestamp
            
        Returns:
            True if position added, False otherwise
        """
        # Check if max positions reached
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check if symbol already in portfolio
        if symbol in self.positions:
            return False
        
        # Add position
        self.positions[symbol] = {
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': timestamp,
            'size': self.position_size_pct
        }
        
        return True
    
    def remove_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[float]:
        """
        Remove position from portfolio.
        
        Args:
            symbol: Asset symbol
            exit_price: Exit price
            exit_reason: Reason for exit
            
        Returns:
            PnL from position
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        signal = position['signal']
        
        # Calculate PnL
        if signal == ORBSignal.LONG_BREAKOUT:
            pnl = (exit_price - entry_price) / entry_price * position['size']
        else:
            pnl = (entry_price - exit_price) / entry_price * position['size']
        
        # Update capital
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Remove position
        del self.positions[symbol]
        
        return pnl
    
    def check_drawdown(self) -> bool:
        """
        Check if drawdown exceeds limit.
        
        Returns:
            True if drawdown exceeds limit
        """
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        return drawdown > self.max_drawdown_pct


if __name__ == "__main__":
    # Test ORB strategy
    print("Testing ORB Strategy with Relative Volume...")
    
    # Create sample intraday data
    np.random.seed(42)
    n_minutes = 390  # 6.5 hours of trading
    
    timestamps = pd.date_range(
        start='2024-01-01 09:15:00',
        periods=n_minutes,
        freq='1min'
    )
    
    # Generate price data with opening range breakout
    prices = []
    volumes = []
    
    base_price = 100.0
    for i in range(n_minutes):
        if i < 15:
            # Opening range - random walk
            change = np.random.normal(0, 0.05)
            base_price += change
        else:
            # Breakout - trend
            if i == 15:
                # Breakout move
                change = 0.5
                base_price += change
            else:
                # Continue trend
                change = np.random.normal(0.01, 0.1)
                base_price += change
        
        prices.append(base_price)
        volumes.append(np.random.randint(10000, 50000))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.002)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.002)) for p in prices],
        'close': prices,
        'volume': volumes
    }, index=timestamps)
    
    # Test ORB strategy
    orb = ORBStrategy(
        open_range_minutes=15,
        relative_volume_threshold=1.5,
        stop_loss_pct=0.01,
        take_profit_pct=0.02
    )
    
    # Generate signal at a specific time
    test_timestamp = timestamps[30]  # 30 minutes after open
    signal = orb.generate_signal(
        data.iloc[:31],
        test_timestamp,
        data.loc[test_timestamp, 'close'],
        data.loc[test_timestamp, 'volume']
    )
    
    print(f"\nORB Signal at {test_timestamp}:")
    print(f"Signal: {signal['signal']}")
    print(f"Entry Price: {signal['entry_price']}")
    print(f"Stop Loss: {signal['stop_loss']}")
    print(f"Take Profit: {signal['take_profit']}")
    print(f"OR High: {signal['or_high']}")
    print(f"OR Low: {signal['or_low']}")
    print(f"Relative Volume: {signal.get('relative_volume', 0):.2f}")
    
    # Backtest
    print("\nRunning backtest...")
    trades = orb.backtest(data)
    print(f"Number of trades: {len(trades)}")
    
    if len(trades) > 0:
        print(f"Average PnL: {trades['pnl'].mean():.4f}")
        print(f"Win Rate: {(trades['pnl'] > 0).mean():.2%}")
    
    # Test portfolio
    print("\nTesting ORB Portfolio...")
    portfolio = ORBPortfolio(max_positions=3, position_size_pct=0.2)
    
    # Add position
    added = portfolio.add_position(
        'TEST',
        ORBSignal.LONG_BREAKOUT,
        100.0,
        99.0,
        102.0,
        timestamps[30]
    )
    print(f"Position added: {added}")
    print(f"Current positions: {len(portfolio.positions)}")
    
    # Remove position
    pnl = portfolio.remove_position('TEST', 101.0, 'TAKE_PROFIT')
    print(f"Position PnL: {pnl:.4f}")
    print(f"Current capital: {portfolio.current_capital:.4f}")
    
    print("\nORB Strategy with Relative Volume test completed.")
