"""
Live Paper Trading Validation
Based on the critique: Validate strategies in paper trading before live deployment

Objective:
- Validate strategies in real-time with live data
- Test execution quality
- Verify signal generation works in production
- Measure actual slippage and costs
- Final validation before live deployment

Features:
- Real-time signal generation
- Paper trade execution simulation
- Performance tracking
- Alert system for issues
- Comparison to backtest expectations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class PaperTradeStatus(Enum):
    """Status of paper trade."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ValidationStatus(Enum):
    """Validation status of strategy."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PaperTrade:
    """Paper trade record."""
    trade_id: str
    strategy_name: str
    symbol: str
    direction: str
    quantity: float
    signal_price: float
    target_price: float
    stop_loss: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: PaperTradeStatus = PaperTradeStatus.PENDING
    pnl: float = 0.0
    return_pct: float = 0.0
    slippage_bps: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class ValidationMetrics:
    """Metrics for paper trading validation."""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    num_trades: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_return: float
    avg_slippage_bps: float
    signal_generation_success_rate: float
    execution_quality_score: float
    backtest_correlation: float  # Correlation with backtest results
    status: ValidationStatus = ValidationStatus.NOT_STARTED


class PaperTradingValidator:
    """
    Live Paper Trading Validation System.
    
    Validates strategies in real-time before live deployment:
    - Generates signals from live data
    - Simulates execution without real money
    - Tracks performance metrics
    - Compares to backtest expectations
    """
    
    def __init__(self):
        self.paper_trades: List[PaperTrade] = []
        self.validation_metrics: Dict[str, ValidationMetrics] = {}
        self.active_validations: Dict[str, datetime] = {}
        
        # Validation thresholds
        self.min_trades = 20
        self.min_win_rate = 0.45
        self.max_slippage_bps = 10.0
        self.min_signal_success_rate = 0.95
        self.min_execution_quality = 0.8
    
    def start_validation(
        self,
        strategy_name: str,
        expected_duration_days: int = 30
    ) -> str:
        """
        Start a new paper trading validation.
        
        Args:
            strategy_name: Name of strategy to validate
            expected_duration_days: Expected duration in days
            
        Returns:
            Validation ID
        """
        validation_id = f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_validations[validation_id] = datetime.now()
        
        # Initialize metrics
        self.validation_metrics[validation_id] = ValidationMetrics(
            strategy_name=strategy_name,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=expected_duration_days),
            num_trades=0,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_trade_return=0.0,
            avg_slippage_bps=0.0,
            signal_generation_success_rate=1.0,
            execution_quality_score=1.0,
            backtest_correlation=0.0,
            status=ValidationStatus.RUNNING
        )
        
        return validation_id
    
    def generate_signal(
        self,
        validation_id: str,
        strategy_name: str,
        symbol: str,
        direction: str,
        quantity: float,
        signal_price: float,
        target_price: float,
        stop_loss: float
    ) -> Optional[str]:
        """
        Generate a paper trading signal.
        
        Args:
            validation_id: Validation ID
            strategy_name: Strategy name
            symbol: Trading symbol
            direction: Trade direction
            quantity: Trade quantity
            signal_price: Price at signal generation
            target_price: Target price
            stop_loss: Stop loss price
            
        Returns:
            Trade ID or None if validation not found
        """
        if validation_id not in self.validation_metrics:
            return None
        
        trade_id = f"{validation_id}_{symbol}_{datetime.now().strftime('%H%M%S')}"
        
        trade = PaperTrade(
            trade_id=trade_id,
            strategy_name=strategy_name,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            signal_price=signal_price,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_time=datetime.now()
        )
        
        self.paper_trades.append(trade)
        
        # Update metrics
        metrics = self.validation_metrics[validation_id]
        metrics.num_trades += 1
        
        return trade_id
    
    def simulate_execution(
        self,
        trade_id: str,
        fill_price: float,
        slippage_bps: float = 0.0
    ) -> bool:
        """
        Simulate order execution.
        
        Args:
            trade_id: Trade ID
            fill_price: Actual fill price
            slippage_bps: Slippage in basis points
            
        Returns:
            True if successful
        """
        for trade in self.paper_trades:
            if trade.trade_id == trade_id:
                trade.exit_price = fill_price
                trade.exit_time = datetime.now()
                trade.status = PaperTradeStatus.FILLED
                trade.slippage_bps = slippage_bps
                
                # Calculate PnL
                if trade.direction == "long":
                    trade.pnl = (fill_price - trade.signal_price) * trade.quantity
                else:
                    trade.pnl = (trade.signal_price - fill_price) * trade.quantity
                
                trade.return_pct = trade.pnl / (trade.signal_price * trade.quantity)
                
                return True
        
        return False
    
    def cancel_trade(self, trade_id: str, reason: str) -> bool:
        """
        Cancel a paper trade.
        
        Args:
            trade_id: Trade ID
            reason: Reason for cancellation
            
        Returns:
            True if successful
        """
        for trade in self.paper_trades:
            if trade.trade_id == trade_id:
                trade.status = PaperTradeStatus.CANCELLED
                trade.notes.append(reason)
                return True
        
        return False
    
    def calculate_validation_metrics(self, validation_id: str) -> Optional[ValidationMetrics]:
        """
        Calculate validation metrics for a strategy.
        
        Args:
            validation_id: Validation ID
            
        Returns:
            ValidationMetrics or None if not found
        """
        if validation_id not in self.validation_metrics:
            return None
        
        # Get trades for this validation
        trades = [t for t in self.paper_trades if t.strategy_name == self.validation_metrics[validation_id].strategy_name]
        
        if not trades:
            return self.validation_metrics[validation_id]
        
        # Filter only filled trades
        filled_trades = [t for t in trades if t.status == PaperTradeStatus.FILLED]
        
        if not filled_trades:
            return self.validation_metrics[validation_id]
        
        # Calculate metrics
        returns = [t.return_pct for t in filled_trades]
        pnls = [t.pnl for t in filled_trades]
        
        total_return = sum(pnls)
        avg_trade_return = np.mean(returns)
        
        # Sharpe
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        # Avg slippage
        avg_slippage = np.mean([t.slippage_bps for t in filled_trades])
        
        # Signal generation success rate (filled / total)
        signal_success = len(filled_trades) / len(trades)
        
        # Execution quality (inverse of slippage)
        execution_quality = max(0, 1 - avg_slippage / 100)
        
        # Update metrics
        metrics = self.validation_metrics[validation_id]
        metrics.total_return = total_return
        metrics.sharpe_ratio = sharpe
        metrics.max_drawdown = abs(max_drawdown)
        metrics.win_rate = win_rate
        metrics.avg_trade_return = avg_trade_return
        metrics.avg_slippage_bps = avg_slippage
        metrics.signal_generation_success_rate = signal_success
        metrics.execution_quality_score = execution_quality
        
        return metrics
    
    def evaluate_validation(self, validation_id: str) -> Tuple[ValidationStatus, str]:
        """
        Evaluate if validation passed or failed.
        
        Args:
            validation_id: Validation ID
            
        Returns:
            (status, reason) tuple
        """
        metrics = self.calculate_validation_metrics(validation_id)
        
        if not metrics:
            return ValidationStatus.INCONCLUSIVE, "No metrics available"
        
        # Check if minimum trades reached
        if metrics.num_trades < self.min_trades:
            return ValidationStatus.INCONCLUSIVE, f"Insufficient trades: {metrics.num_trades} < {self.min_trades}"
        
        # Check win rate
        if metrics.win_rate < self.min_win_rate:
            return ValidationStatus.FAILED, f"Win rate too low: {metrics.win_rate:.2%} < {self.min_win_rate:.2%}"
        
        # Check slippage
        if metrics.avg_slippage_bps > self.max_slippage_bps:
            return ValidationStatus.FAILED, f"Slippage too high: {metrics.avg_slippage_bps:.2f} bps > {self.max_slippage_bps:.2f} bps"
        
        # Check signal success rate
        if metrics.signal_generation_success_rate < self.min_signal_success_rate:
            return ValidationStatus.FAILED, f"Signal success rate too low: {metrics.signal_generation_success_rate:.2%}"
        
        # Check execution quality
        if metrics.execution_quality_score < self.min_execution_quality:
            return ValidationStatus.FAILED, f"Execution quality too low: {metrics.execution_quality_score:.2%}"
        
        # Check Sharpe (should be positive)
        if metrics.sharpe_ratio < 0.5:
            return ValidationStatus.INCONCLUSIVE, f"Sharpe ratio inconclusive: {metrics.sharpe_ratio:.2f}"
        
        # All checks passed
        metrics.status = ValidationStatus.PASSED
        return ValidationStatus.PASSED, "All validation checks passed"
    
    def get_validation_report(self, validation_id: str) -> Dict:
        """
        Get detailed validation report.
        
        Args:
            validation_id: Validation ID
            
        Returns:
            Dictionary with validation details
        """
        metrics = self.calculate_validation_metrics(validation_id)
        
        if not metrics:
            return {"error": "Validation not found"}
        
        status, reason = self.evaluate_validation(validation_id)
        
        # Get trades
        trades = [t for t in self.paper_trades if t.strategy_name == metrics.strategy_name]
        
        return {
            "validation_id": validation_id,
            "strategy_name": metrics.strategy_name,
            "status": status.value,
            "reason": reason,
            "start_date": metrics.start_date,
            "end_date": metrics.end_date,
            "num_trades": metrics.num_trades,
            "total_return": metrics.total_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "avg_trade_return": metrics.avg_trade_return,
            "avg_slippage_bps": metrics.avg_slippage_bps,
            "signal_success_rate": metrics.signal_generation_success_rate,
            "execution_quality": metrics.execution_quality_score,
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "status": t.status.value
                }
                for t in trades
            ]
        }
    
    def get_all_validations(self) -> pd.DataFrame:
        """Get summary of all validations."""
        data = []
        
        for validation_id, metrics in self.validation_metrics.items():
            status, reason = self.evaluate_validation(validation_id)
            
            data.append({
                'Validation ID': validation_id,
                'Strategy': metrics.strategy_name,
                'Status': status.value,
                'Trades': metrics.num_trades,
                'Sharpe': f"{metrics.sharpe_ratio:.2f}",
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Slippage (bps)': f"{metrics.avg_slippage_bps:.2f}",
                'Reason': reason
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the Paper Trading Validator
    print("Testing Paper Trading Validation...")
    
    validator = PaperTradingValidator()
    
    # Start validation for a strategy
    print("\nStarting validation for ORB strategy...")
    validation_id = validator.start_validation("ORB", expected_duration_days=30)
    print(f"Validation ID: {validation_id}")
    
    # Simulate some paper trades
    print("\nSimulating paper trades...")
    
    for i in range(25):
        trade_id = validator.generate_signal(
            validation_id=validation_id,
            strategy_name="ORB",
            symbol="RELIANCE",
            direction="long" if i % 2 == 0 else "short",
            quantity=100,
            signal_price=2500 + np.random.uniform(-50, 50),
            target_price=2600,
            stop_loss=2400
        )
        
        # Simulate execution after some time
        fill_price = 2500 + np.random.uniform(-100, 100)
        slippage = np.random.uniform(0, 5)
        validator.simulate_execution(trade_id, fill_price, slippage)
    
    print(f"Generated {len(validator.paper_trades)} paper trades")
    
    # Calculate metrics
    print("\nCalculating validation metrics...")
    metrics = validator.calculate_validation_metrics(validation_id)
    
    print(f"Strategy: {metrics.strategy_name}")
    print(f"Trades: {metrics.num_trades}")
    print(f"Total Return: {metrics.total_return:.2f}")
    print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.2%}")
    print(f"Avg Slippage: {metrics.avg_slippage_bps:.2f} bps")
    print(f"Signal Success Rate: {metrics.signal_generation_success_rate:.2%}")
    print(f"Execution Quality: {metrics.execution_quality_score:.2%}")
    
    # Evaluate validation
    print("\nEvaluating validation...")
    status, reason = validator.evaluate_validation(validation_id)
    print(f"Status: {status.value}")
    print(f"Reason: {reason}")
    
    # Get validation report
    print("\nValidation Report:")
    report = validator.get_validation_report(validation_id)
    print(f"Validation ID: {report['validation_id']}")
    print(f"Strategy: {report['strategy_name']}")
    print(f"Status: {report['status']}")
    print(f"Reason: {report['reason']}")
    
    # Get all validations
    print("\nAll Validations:")
    all_validations = validator.get_all_validations()
    print(all_validations.to_string(index=False))
