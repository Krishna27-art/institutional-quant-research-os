"""
Trade Logger and Real Metrics Computation

This module provides comprehensive trade logging and real-time metrics computation
to replace hardcoded performance metrics with actual calculated values.

Key Features:
- Trade execution logging (every trade recorded)
- Real-time PnL tracking
- Win rate calculation from actual trades
- Sharpe ratio computation from returns
- Drawdown calculation from portfolio history
- Streak tracking (consecutive wins/losses)
- Accuracy calculation (prediction vs actual)

Based on Audit Report Priority 0: Critical - Week 1-2
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class TradeSide(Enum):
    """Trade side."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """Trade status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Trade:
    """Trade record."""
    trade_id: str
    symbol: str
    side: TradeSide
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = None
    exit_time: Optional[datetime] = None
    status: TradeStatus = TradeStatus.PENDING
    pnl: Optional[float] = None
    commission: float = 0.0
    slippage_bps: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def calculate_pnl(self) -> float:
        """Calculate PnL for the trade."""
        if self.exit_price is None:
            return 0.0
        
        if self.side == TradeSide.BUY:
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        net_pnl = gross_pnl - self.commission
        return net_pnl
    
    def is_win(self) -> Optional[bool]:
        """Check if trade is a win."""
        if self.exit_price is None:
            return None
        return self.calculate_pnl() > 0


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    current_streak: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    accuracy: float
    volatility: float
    last_update: datetime


class TradeLogger:
    """
    Trade logger and metrics calculator.
    
    This class logs all trades and computes real performance metrics
    instead of using hardcoded values.
    """
    
    def __init__(self, log_dir: str = None):
        """
        Initialize trade logger.
        
        Args:
            log_dir: Directory to store trade logs
        """
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.trades_file = self.log_dir / "trades.csv"
        self.pnl_file = self.log_dir / "pnl_history.csv"
        
        # In-memory trade storage
        self.trades: Dict[str, Trade] = {}
        self.pnl_history: List[Tuple[datetime, float]] = []
        
        # Load existing trades
        self._load_trades()
        self._load_pnl_history()
        
        logger.info(f"TradeLogger initialized with {len(self.trades)} trades")
    
    def log_trade(self, trade: Trade, alpha_id: Optional[str] = None, regime_id: Optional[str] = None) -> None:
        """
        Log a new trade execution.
        
        Args:
            trade: Trade record to log
            alpha_id: Which alpha generated this trade
            regime_id: The market regime at the time of execution
        """
        if trade.trade_id in self.trades:
            logger.warning(f"Trade {trade.trade_id} already exists, updating instead")
            update_kwargs = vars(trade).copy()
            update_kwargs.pop('trade_id', None)
            self.update_trade(trade.trade_id, **update_kwargs)
            return
            
        # Append attribution metadata
        if trade.metadata is None:
            trade.metadata = {}
        if alpha_id:
            trade.metadata['alpha_id'] = alpha_id
        if regime_id:
            trade.metadata['regime_id'] = regime_id

        self.trades[trade.trade_id] = trade
        
        # If trade is closed, calculate PnL and add to history
        if trade.exit_price is not None and trade.exit_time is not None:
            trade.pnl = trade.calculate_pnl()
            self.pnl_history.append((trade.exit_time, trade.pnl))
        
        # Persist to disk
        self._save_trades()
        self._save_pnl_history()
        self._save_positions_to_csv()
        
        # Persist to database
        self._save_trade_to_db(trade)
        
        logger.info(f"Logged trade {trade.trade_id}: {trade.side.value} {trade.quantity} {trade.symbol} @ {trade.entry_price}")
    
    def update_trade(self, trade_id: str, **kwargs) -> None:
        """
        Update an existing trade.
        
        Args:
            trade_id: Trade identifier
            **kwargs: Fields to update
        """
        if trade_id not in self.trades:
            logger.warning(f"Trade {trade_id} not found")
            return
        
        trade = self.trades[trade_id]
        
        for key, value in kwargs.items():
            if hasattr(trade, key):
                setattr(trade, key, value)
        
        # Recalculate PnL if trade is closed
        if trade.exit_price is not None and trade.exit_time is not None:
            trade.pnl = trade.calculate_pnl()
            if trade.exit_time not in [t[0] for t in self.pnl_history]:
                self.pnl_history.append((trade.exit_time, trade.pnl))
        
        self._save_trades()
        self._save_pnl_history()
        self._save_positions_to_csv()
        
        self._save_trade_to_db(trade)
        
        logger.info(f"Updated trade {trade_id}")
    
    def get_metrics(self, lookback_days: int = 30) -> PerformanceMetrics:
        """
        Calculate real performance metrics from trade history.
        
        Args:
            lookback_days: Number of days to look back for metrics
            
        Returns:
            PerformanceMetrics with calculated values
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Filter trades by date
        recent_trades = [
            t for t in self.trades.values()
            if t.entry_time >= cutoff_date and t.exit_price is not None
        ]
        
        if not recent_trades:
            return self._get_empty_metrics()
        
        # Calculate basic metrics
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t.is_win())
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate PnL metrics
        pnls = [t.calculate_pnl() for t in recent_trades]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe(pnls)
        
        # Calculate drawdown
        max_dd, current_dd = self._calculate_drawdown()
        
        # Calculate streaks
        current_streak, max_wins, max_losses = self._calculate_streaks(recent_trades)
        
        # Calculate accuracy (prediction vs actual)
        accuracy = self._calculate_accuracy(recent_trades)
        
        # Calculate volatility
        volatility = np.std(pnls) if len(pnls) > 1 else 0.0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            current_streak=current_streak,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            accuracy=accuracy,
            volatility=volatility,
            last_update=datetime.now()
        )
    
    def _calculate_sharpe(self, pnls: List[float]) -> float:
        """Calculate Sharpe ratio from PnL series."""
        if len(pnls) < 2:
            return 0.0
        
        returns = np.array(pnls)
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily trades)
        sharpe = mean_return / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_drawdown(self) -> Tuple[float, float]:
        """Calculate max drawdown and current drawdown from PnL history."""
        if not self.pnl_history:
            return 0.0, 0.0
        
        df = pd.DataFrame(self.pnl_history, columns=['timestamp', 'pnl'])
        df = df.sort_values('timestamp')
        
        # Calculate cumulative PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Calculate running maximum
        df['running_max'] = df['cumulative_pnl'].cummax()
        
        # Calculate drawdown
        df['drawdown'] = (df['cumulative_pnl'] - df['running_max']) / df['running_max']
        
        max_dd = df['drawdown'].min()
        current_dd = df['drawdown'].iloc[-1]
        
        return max_dd, current_dd
    
    def _calculate_streaks(self, trades: List[Trade]) -> Tuple[int, int, int]:
        """Calculate current streak, max consecutive wins, max consecutive losses."""
        if not trades:
            return 0, 0, 0
        
        # Sort by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time or t.entry_time)
        
        current_streak = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        temp_win_streak = 0
        temp_loss_streak = 0
        
        for trade in sorted_trades:
            is_win = trade.is_win()
            
            if is_win is None:
                continue
            
            if is_win:
                temp_win_streak += 1
                temp_loss_streak = 0
                max_consecutive_wins = max(max_consecutive_wins, temp_win_streak)
            else:
                temp_loss_streak += 1
                temp_win_streak = 0
                max_consecutive_losses = max(max_consecutive_losses, temp_loss_streak)
        
        # Current streak
        if sorted_trades:
            last_trade = sorted_trades[-1]
            if last_trade.is_win():
                current_streak = temp_win_streak
            else:
                current_streak = -temp_loss_streak
        
        return current_streak, max_consecutive_wins, max_consecutive_losses
    
    def _calculate_accuracy(self, trades: List[Trade]) -> float:
        """
        Calculate prediction accuracy.
        
        This compares the predicted direction (from metadata) with actual outcome.
        """
        if not trades:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for trade in trades:
            if 'predicted_direction' in trade.metadata:
                predicted = trade.metadata['predicted_direction']
                actual = 1 if trade.is_win() else 0
                
                if predicted == actual:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades exist."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl_per_trade=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            current_streak=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            accuracy=0.0,
            volatility=0.0,
            last_update=datetime.now()
        )
    
    def _get_db_connection(self):
        try:
            import os
            from src.shared.db.connection_manager import connection_manager, DatabaseConfig
            
            db_cfg = DatabaseConfig(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", 5432)),
                database=os.getenv("POSTGRES_DB", "quant_research"),
                username=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            return connection_manager.get_postgres_connection(db_cfg)
        except Exception as e:
            logger.warning(f"Could not get DB connection: {e}")
            return None

    def _save_trade_to_db(self, trade: Trade) -> None:
        """Save/update a single trade in the database, and update open positions."""
        conn = self._get_db_connection()
        if conn is None:
            return
        
        try:
            with conn.cursor() as cur:
                # Check if trade already exists
                cur.execute("SELECT 1 FROM trades WHERE id = %s", (trade.trade_id,))
                exists = cur.fetchone()
                
                if exists:
                    cur.execute("""
                        UPDATE trades 
                        SET timestamp = %s, symbol = %s, direction = %s, quantity = %s, price = %s, 
                            commission = %s, exit_price = %s, exit_time = %s, pnl = %s, status = %s
                        WHERE id = %s
                    """, (
                        trade.entry_time, trade.symbol, trade.side.value.upper(), trade.quantity, trade.entry_price,
                        trade.commission, trade.exit_price, trade.exit_time, trade.pnl, trade.status.value.upper(),
                        trade.trade_id
                    ))
                else:
                    cur.execute("""
                        INSERT INTO trades (id, timestamp, symbol, direction, quantity, price, commission, exit_price, exit_time, pnl, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        trade.trade_id, trade.entry_time, trade.symbol, trade.side.value.upper(), trade.quantity, trade.entry_price,
                        trade.commission, trade.exit_price, trade.exit_time, trade.pnl, trade.status.value.upper()
                    ))
                
                # Also update positions table
                active_qty = 0
                total_entry_val = 0.0
                for t in self.trades.values():
                    if t.symbol == trade.symbol and t.exit_price is None:
                        qty = t.quantity if t.side == TradeSide.BUY else -t.quantity
                        active_qty += qty
                        total_entry_val += qty * t.entry_price
                
                avg_price = total_entry_val / active_qty if active_qty != 0 else 0.0
                
                # Delete old position for symbol
                cur.execute("DELETE FROM positions WHERE symbol = %s", (trade.symbol,))
                
                # Insert new position if not flat
                if active_qty != 0:
                    cur.execute("""
                        INSERT INTO positions (symbol, quantity, entry_price, current_price, pnl) 
                        VALUES (%s, %s, %s, %s, %s)
                    """, (trade.symbol, int(active_qty), float(avg_price), float(trade.entry_price), 0.0))
                    
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist trade/position to PostgreSQL: {e}")

    def _save_positions_to_csv(self) -> None:
        """Save current active positions to CSV."""
        try:
            positions_file = self.log_dir / "positions.csv"
            pos_data = []
            
            # Group active trades to find positions
            for symbol in set(t.symbol for t in self.trades.values()):
                active_qty = 0
                total_entry_val = 0.0
                last_price = 0.0
                for t in self.trades.values():
                    if t.symbol == symbol and t.exit_price is None:
                        qty = t.quantity if t.side == TradeSide.BUY else -t.quantity
                        active_qty += qty
                        total_entry_val += qty * t.entry_price
                        last_price = t.entry_price
                
                if active_qty != 0:
                    avg_price = total_entry_val / active_qty
                    pos_data.append({
                        'symbol': symbol,
                        'quantity': active_qty,
                        'entry_price': avg_price,
                        'current_price': last_price,
                        'pnl': 0.0
                    })
            
            df = pd.DataFrame(pos_data)
            df.to_csv(positions_file, index=False)
        except Exception as e:
            logger.error(f"Failed to save positions to CSV: {e}")

    def _load_trades(self) -> None:
        """Load trades from database or disk fallback."""
        conn = self._get_db_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, symbol, side, quantity, entry_price, exit_price, entry_time, exit_time, status, pnl, commission, slippage_bps FROM trades")
                    rows = cur.fetchall()
                    for row in rows:
                        trade = Trade(
                            trade_id=str(row[0]),
                            symbol=row[1],
                            side=TradeSide(row[2].lower()),
                            quantity=int(row[3]),
                            entry_price=float(row[4]),
                            exit_price=float(row[5]) if row[5] is not None else None,
                            entry_time=row[6],
                            exit_time=row[7] if row[7] is not None else None,
                            status=TradeStatus(row[8].lower()) if row[8] is not None else TradeStatus.FILLED,
                            pnl=float(row[9]) if row[9] is not None else None,
                            commission=float(row[10] or 0.0),
                            slippage_bps=float(row[11] or 0.0)
                        )
                        self.trades[trade.trade_id] = trade
                logger.info(f"Loaded {len(self.trades)} trades from PostgreSQL")
                return
            except Exception as e:
                logger.warning(f"Failed to load trades from PostgreSQL: {e}. Falling back to CSV.")
        
        # CSV fallback
        if not self.trades_file.exists():
            return
        
        try:
            df = pd.read_csv(self.trades_file)
            for _, row in df.iterrows():
                trade = Trade(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    side=TradeSide(row['side']),
                    quantity=int(row['quantity']),
                    entry_price=float(row['entry_price']),
                    exit_price=float(row['exit_price']) if pd.notna(row['exit_price']) else None,
                    entry_time=pd.to_datetime(row['entry_time']),
                    exit_time=pd.to_datetime(row['exit_time']) if pd.notna(row['exit_time']) else None,
                    status=TradeStatus(row['status']),
                    pnl=float(row['pnl']) if pd.notna(row['pnl']) else None,
                    commission=float(row['commission']),
                    slippage_bps=float(row['slippage_bps'])
                )
                self.trades[trade.trade_id] = trade
            logger.info(f"Loaded {len(self.trades)} trades from disk")
        except Exception as e:
            logger.error(f"Failed to load trades: {e}")

    def _save_trades(self) -> None:
        """Save trades to disk."""
        try:
            trades_data = []
            for trade in self.trades.values():
                trades_data.append({
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'entry_time': trade.entry_time.isoformat() if isinstance(trade.entry_time, datetime) else str(trade.entry_time),
                    'exit_time': (trade.exit_time.isoformat() if isinstance(trade.exit_time, datetime) else str(trade.exit_time)) if trade.exit_time else None,
                    'status': trade.status.value,
                    'pnl': trade.pnl,
                    'commission': trade.commission,
                    'slippage_bps': trade.slippage_bps
                })
            
            df = pd.DataFrame(trades_data)
            df.to_csv(self.trades_file, index=False)
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")

    def _load_pnl_history(self) -> None:
        """Load PnL history from disk."""
        if not self.pnl_file.exists():
            return
        
        try:
            df = pd.read_csv(self.pnl_file)
            self.pnl_history = [
                (pd.to_datetime(row['timestamp']), float(row['pnl']))
                for _, row in df.iterrows()
            ]
            logger.info(f"Loaded {len(self.pnl_history)} PnL records from disk")
        except Exception as e:
            logger.error(f"Failed to load PnL history: {e}")

    def _save_pnl_history(self) -> None:
        """Save PnL history to disk."""
        try:
            df = pd.DataFrame(self.pnl_history, columns=['timestamp', 'pnl'])
            df.to_csv(self.pnl_file, index=False)
        except Exception as e:
            logger.error(f"Failed to save PnL history: {e}")
    
    def print_metrics(self, lookback_days: int = 30) -> None:
        """Print performance metrics."""
        metrics = self.get_metrics(lookback_days)
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS (Real-Time)")
        print("="*60)
        print(f"\nTotal Trades: {metrics.total_trades}")
        print(f"Winning Trades: {metrics.winning_trades}")
        print(f"Losing Trades: {metrics.losing_trades}")
        print(f"Win Rate: {metrics.win_rate:.2%}")
        print(f"Total PnL: ₹{metrics.total_pnl:,.2f}")
        print(f"Avg PnL/Trade: ₹{metrics.avg_pnl_per_trade:,.2f}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"Current Drawdown: {metrics.current_drawdown:.2%}")
        print(f"Current Streak: {metrics.current_streak:+d}")
        print(f"Max Consecutive Wins: {metrics.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {metrics.max_consecutive_losses}")
        print(f"Prediction Accuracy: {metrics.accuracy:.2%}")
        print(f"Volatility: {metrics.volatility:.2f}")
        print(f"\nLast Update: {metrics.last_update}")
        print("\n" + "="*60)


# Singleton instance
_trade_logger = None

def get_trade_logger() -> TradeLogger:
    """Get the singleton trade logger instance."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger


if __name__ == "__main__":
    # Test the trade logger
    print("Testing Trade Logger...")
    
    logger = TradeLogger()
    
    # Create sample trades
    for i in range(10):
        trade = Trade(
            trade_id=f"trade_{i}",
            symbol="RELIANCE",
            side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
            quantity=100,
            entry_price=1000 + i * 10,
            exit_price=1010 + i * 10 if i % 2 == 0 else 990 + i * 10,
            entry_time=datetime.now() - timedelta(days=i),
            exit_time=datetime.now() - timedelta(days=i) + timedelta(hours=1),
            status=TradeStatus.FILLED,
            commission=10.0,
            metadata={'predicted_direction': 1 if i % 2 == 0 else 0}
        )
        logger.log_trade(trade)
    
    # Print metrics
    logger.print_metrics()
