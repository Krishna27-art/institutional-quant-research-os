import pytest
from datetime import datetime
from typing import Dict, Any
from execution.unified_execution_engine import ExecutionPipeline, UnifiedExecutionEngine, ExecutionMode
from execution.adapters.backtest_adapter import BacktestConfig
from portfolio.allocator import PortfolioAllocator
from risk.institutional_risk_engine import InstitutionalRiskEngine


def test_execution_pipeline_sizer_and_risk_gates():
    # 1. Setup mock execution engine
    config = {
        'start_date': datetime(2026, 1, 1),
        'end_date': datetime(2026, 12, 31),
        'initial_capital': 250_000_000.0,
        'commission_rate': 0.0005,
        'slippage_bps': 2.0
    }
    # Create unified execution engine in backtest mode
    engine = UnifiedExecutionEngine(ExecutionMode.BACKTEST, BacktestConfig(**config))
    
    # 2. Setup portfolio allocator and risk engine
    allocator = PortfolioAllocator(total_capital=250_000_000.0)
    risk_engine = InstitutionalRiskEngine(capital=250_000_000.0)
    
    # Create pipeline
    pipeline = ExecutionPipeline(
        execution_engine=engine,
        portfolio_allocator=allocator,
        risk_engine=risk_engine
    )
    
    # 3. Test sizer gate - allocation should scale quantity
    signal = {
        'symbol': 'RELIANCE',
        'direction': 1,
        'strength': 1.0,
        'confidence': 1.0,
        'price': 3000.0,
        'quantity': 1000000 # Very large quantity
    }
    
    # Allocator will cap allocation at max single stock pct (5% of 250M = 12.5M)
    # At 3000 price, 12.5M / 3000 = ~4166 shares
    result = pipeline.process_signal(signal)
    
    # Verify signal quantity was scaled down
    assert signal['quantity'] < 1000000
    assert signal['quantity'] <= 4167
    
    # 4. Test risk gate - active circuit breaker should reject signal
    risk_engine.circuit_breaker_active = True
    
    cb_signal = {
        'symbol': 'INFY',
        'direction': 1,
        'strength': 0.8,
        'confidence': 0.8,
        'price': 1500.0,
        'quantity': 100
    }
    
    cb_result = pipeline.process_signal(cb_signal)
    assert cb_result['status'] == 'rejected'
    assert 'circuit breaker' in cb_result['reason'].lower()
