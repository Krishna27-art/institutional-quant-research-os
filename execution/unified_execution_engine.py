"""
Unified Execution Engine - Single execution path for all modes

This provides a single execution path with adapters for:
- Backtest mode
- Paper mode
- Live mode

Architecture:
Signal → Portfolio → Risk → Execution
                                ├── Backtest Adapter
                                ├── Paper Adapter
                                └── Live Adapter
"""

from enum import Enum
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import pandas as pd
import numpy as np

from execution.adapters.backtest_adapter import BacktestAdapter, BacktestConfig, BacktestResult
from execution.adapters.paper_adapter import PaperAdapter, PaperConfig, PaperResult
from execution.adapters.live_adapter import LiveAdapter, LiveConfig, LiveResult
from execution.cost_models import MarketImpactModel, TransactionCostModel
from execution.risk_checks import PreTradeRiskChecker, PostTradeRiskChecker


class ExecutionMode(Enum):
    """Execution mode enum"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class UnifiedExecutionEngine:
    """
    Unified execution engine with single execution path
    
    This consolidates:
    - backtest/
    - backtesting/
    - paper_trading/
    - live/
    - live_trading/
    - execution/
    
    Into a single execution path with mode-specific adapters.
    """
    
    def __init__(self, mode: ExecutionMode, config: Union[BacktestConfig, PaperConfig, LiveConfig]):
        """
        Initialize unified execution engine
        
        Args:
            mode: Execution mode (BACKTEST, PAPER, LIVE)
            config: Configuration for the selected mode
        """
        self.mode = mode
        self.adapter = self._create_adapter(mode, config)
        
    def _create_adapter(self, mode: ExecutionMode, config: Union[BacktestConfig, PaperConfig, LiveConfig]):
        """Create appropriate adapter based on mode"""
        if mode == ExecutionMode.BACKTEST:
            return BacktestAdapter(config)
        elif mode == ExecutionMode.PAPER:
            return PaperAdapter(config)
        elif mode == ExecutionMode.LIVE:
            return LiveAdapter(config)
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
    
    def execute_order(self, symbol: str, quantity: int, price: float, 
                     direction: str, timestamp: datetime, **kwargs) -> Dict[str, Any]:
        """
        Execute order through unified interface
        
        This is the single execution path - all orders go through this method
        regardless of mode.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Execution price
            direction: 'BUY' or 'SELL'
            timestamp: Order timestamp
            **kwargs: Additional mode-specific parameters
            
        Returns:
            Trade execution details
        """
        return self.adapter.execute_order(symbol, quantity, price, direction, timestamp, **kwargs)
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Get current portfolio value
        
        Args:
            prices: Dictionary of current prices
            
        Returns:
            Portfolio value
        """
        return self.adapter.get_portfolio_value(prices)
    
    def get_performance_metrics(self) -> Union[BacktestResult, PaperResult, LiveResult]:
        """
        Get performance metrics
        
        Returns:
            Performance metrics for current mode
        """
        return self.adapter.get_performance_metrics()
    
    def validate_order(self, symbol: str, quantity: int, price: float) -> bool:
        """
        Validate order before execution
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Execution price
            
        Returns:
            True if order is valid
        """
        if hasattr(self.adapter, 'validate_order'):
            return self.adapter.validate_order(symbol, quantity, price)
        return True


class ExecutionPipeline:
    """
    Complete execution pipeline with risk checks and cost analysis
    
    This implements the single execution path:
    Signal → Portfolio → Risk → Cost Analysis → Execution
    """
    
    def __init__(self, execution_engine: UnifiedExecutionEngine, 
                 pre_trade_checker: Optional[PreTradeRiskChecker] = None,
                 cost_model: Optional[TransactionCostModel] = None,
                 portfolio_allocator: Optional[Any] = None,
                 risk_engine: Optional[Any] = None):
        """
        Initialize execution pipeline
        
        Args:
            execution_engine: Unified execution engine
            pre_trade_checker: Pre-trade risk checker
            cost_model: Transaction cost model
            portfolio_allocator: Portfolio allocator for sizing
            risk_engine: Institutional risk engine
        """
        self.execution_engine = execution_engine
        self.pre_trade_checker = pre_trade_checker or PreTradeRiskChecker()
        self.cost_model = cost_model or TransactionCostModel()
        self.post_trade_checker = PostTradeRiskChecker()
        
        # Connect PortfolioAllocator and InstitutionalRiskEngine
        from src.portfolio.engine import PortfolioAllocator
        from src.risk.institutional_risk_engine import InstitutionalRiskEngine
        
        self.portfolio_allocator = portfolio_allocator or PortfolioAllocator()
        self.risk_engine = risk_engine or InstitutionalRiskEngine()
        
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trading signal through the complete pipeline
        
        Args:
            signal: Trading signal with symbol, direction, quantity, etc.
            
        Returns:
            Execution result with cost analysis
        """
        import logging
        logger = logging.getLogger("execution_pipeline")
        
        # Connect capital and state dynamically to PreTradeRiskChecker
        if self.portfolio_allocator:
            current_cap = self.portfolio_allocator.get_current_capital()
            self.pre_trade_checker.current_capital = current_cap
            
            # Sync positions and sector exposures from allocator to checker
            for pos_symbol, spec in getattr(self.portfolio_allocator, 'current_positions', {}).items():
                self.pre_trade_checker.update_position(pos_symbol, spec.quantity)
        
        # 1. Sizer Gate (Portfolio Allocator)
        if self.portfolio_allocator:
            try:
                # Sizer check: allocate capital to the single signal
                allocations = self.portfolio_allocator.allocate([signal])
                if not allocations:
                    return {
                        'status': 'rejected',
                        'reason': f'Signal rejected by PortfolioAllocator filters/limits for {signal.get("symbol")}'
                    }
                
                allocation = allocations[0]
                if allocation.capital <= 0:
                    return {
                        'status': 'rejected',
                        'reason': f'PortfolioAllocator allocated zero capital for {signal.get("symbol")}'
                    }
                
                # Sizing calculation: scale quantity based on allocated capital
                price = signal.get('price') or signal.get('entry_price') or 100.0
                if price <= 0:
                    price = 100.0
                
                allocated_qty = int(allocation.capital / price)
                if 'quantity' in signal:
                    original_qty = signal['quantity']
                    if original_qty > allocated_qty:
                        signal['quantity'] = allocated_qty
                        logger.info(f"Scaled down signal quantity for {signal.get('symbol')} from {original_qty} to {allocated_qty} via PortfolioAllocator")
                else:
                    signal['quantity'] = allocated_qty
                
                if signal.get('quantity', 0) <= 0:
                    return {
                        'status': 'rejected',
                        'reason': 'Signal quantity scaled to 0 by PortfolioAllocator'
                    }
            except Exception as e:
                logger.error(f"Error in PortfolioAllocator sizing gate: {e}")
        
        # 2. Risk Gate (Institutional Risk Engine)
        if self.risk_engine:
            try:
                # Check circuit breaker
                cb_triggered, cb_reason = self.risk_engine.check_circuit_breaker(0.0)
                if cb_triggered:
                    return {
                        'status': 'rejected',
                        'reason': f'Signal rejected: InstitutionalRiskEngine circuit breaker active ({cb_reason})'
                    }
                
                # Check trailing drawdown
                current_cap = self.portfolio_allocator.get_current_capital() if self.portfolio_allocator else self.risk_engine.capital
                dd_triggered, dd_pct = self.risk_engine.check_trailing_drawdown_limit(current_cap)
                if dd_triggered:
                    return {
                        'status': 'rejected',
                        'reason': f'Signal rejected: InstitutionalRiskEngine trailing drawdown limit breached ({dd_pct:.2%})'
                    }
            except Exception as e:
                logger.error(f"Error in InstitutionalRiskEngine risk gate: {e}")
                
        # 3. Pre-trade risk check (existing Checker checks sector exposure, leverage, concentration, limits to arbitrage)
        risk_check = self.pre_trade_checker.check(signal)
        if not risk_check['approved']:
            return {'status': 'rejected', 'reason': risk_check['reason']}
        
        # Step 2: Transaction cost estimation
        cost_estimate = self.cost_model.estimate_cost(
            symbol=signal['symbol'],
            quantity=signal['quantity'],
            price=signal.get('price'),
            direction=signal['direction'],
            market_data=signal.get('market_data', {})
        )
        
        # Step 3: Check if cost is acceptable
        if cost_estimate['total_cost_bps'] > self.cost_model.max_cost_bps:
            return {
                'status': 'rejected',
                'reason': f'Cost too high: {cost_estimate["total_cost_bps"]:.2f} bps'
            }
        
        # Step 4: Execution
        execution_result = self.execution_engine.execute_order(
            symbol=signal['symbol'],
            quantity=signal['quantity'],
            price=signal.get('price'),
            direction=signal['direction'],
            timestamp=signal.get('timestamp', datetime.now()),
            cost_estimate=cost_estimate
        )
        
        # Step 5: Post-trade check
        post_trade_check = self.post_trade_checker.check(execution_result)
        
        return {
            **execution_result,
            'cost_estimate': cost_estimate,
            'post_trade_check': post_trade_check
        }
    
    def run_batch(self, signals: pd.DataFrame) -> Union[BacktestResult, PaperResult, LiveResult]:
        """
        Run batch of signals through pipeline
        
        Args:
            signals: DataFrame of trading signals
            
        Returns:
            Performance metrics
        """
        for _, signal in signals.iterrows():
            self.process_signal(signal.to_dict())
        
        return self.execution_engine.get_performance_metrics()


# Factory function for easy instantiation
def create_execution_engine(mode: str, config: Dict[str, Any]) -> UnifiedExecutionEngine:
    """
    Factory function to create execution engine
    
    Args:
        mode: Execution mode ('backtest', 'paper', 'live')
        config: Configuration dictionary
        
    Returns:
        UnifiedExecutionEngine instance
    """
    mode_enum = ExecutionMode(mode.lower())
    
    if mode_enum == ExecutionMode.BACKTEST:
        config_obj = BacktestConfig(**config)
    elif mode_enum == ExecutionMode.PAPER:
        config_obj = PaperConfig(**config)
    elif mode_enum == ExecutionMode.LIVE:
        config_obj = LiveConfig(**config)
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
    
    return UnifiedExecutionEngine(mode_enum, config_obj)
