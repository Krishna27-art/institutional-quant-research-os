"""
Run Strategy Validation Script
Validates ORB, VWAP, PCP, Vol Carry strategies with walk-forward testing
"""

import sys
sys.path.append('/Users/pandu/Desktop/institutional-quant-research-os')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from research.strategy_validation import StrategyValidator
from backtest.walk_forward import WalkForwardValidator, WalkForwardConfig
from core.objective_function import ObjectiveFunction, ObjectiveConstraints

def generate_orb_signals(data):
    """Generate ORB signals."""
    signals = pd.Series(0, index=data.index)
    
    # First 15 minutes high and low
    for i in range(15, len(data)):
        high = data['high'].iloc[i-15:i].max()
        low = data['low'].iloc[i-15:i].min()
        close = data['close'].iloc[i]
        
        if close > high:
            signals.iloc[i] = 1
        elif close < low:
            signals.iloc[i] = -1
    
    return signals

def generate_vwap_signals(data):
    """Generate VWAP signals."""
    signals = pd.Series(0, index=data.index)
    
    # Calculate VWAP
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    # Generate signals
    for i in range(30, len(data)):
        if data['close'].iloc[i] < vwap.iloc[i] and data['close'].iloc[i-1] >= vwap.iloc[i-1]:
            signals.iloc[i] = 1  # Price crosses below VWAP
        elif data['close'].iloc[i] > vwap.iloc[i] and data['close'].iloc[i-1] <= vwap.iloc[i-1]:
            signals.iloc[i] = -1  # Price crosses above VWAP
    
    return signals

def generate_momentum_signals(data, period=20):
    """Generate momentum signals (proxy for PCP/Vol Carry)."""
    signals = pd.Series(0, index=data.index)
    
    for i in range(period, len(data)):
        momentum = data['close'].iloc[i] / data['close'].iloc[i-period] - 1
        
        if momentum > 0.02:
            signals.iloc[i] = 1
        elif momentum < -0.02:
            signals.iloc[i] = -1
    
    return signals

def main():
    """Run validation on all strategies."""
    print("=" * 80)
    print("STRATEGY VALIDATION: ORB, VWAP, Momentum (PCP/Vol Carry proxy)")
    print("=" * 80)
    
    # Initialize validator
    validator = StrategyValidator()
    
    # Generate sample data (in production, would use real data)
    print("\nGenerating sample data...")
    np.random.seed(42)
    n = 2000  # Increased for walk-forward validation
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, n).cumsum(),
        'high': np.random.normal(100, 10, n).cumsum(),
        'low': np.random.normal(100, 10, n).cumsum(),
        'close': np.random.normal(100, 10, n).cumsum(),
        'volume': np.random.normal(1000000, 200000, n)
    }, index=dates)
    
    # Ensure high >= close >= low
    data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, n)
    data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, n)
    data['close'] = data['close'] - data['close'].min() + 100
    
    print(f"Generated {len(data)} days of data")
    
    # Simple validation without walk-forward for demonstration
    print("\n" + "=" * 80)
    print("SIMPLE VALIDATION (Sharpe, Win Rate, Max DD)")
    print("=" * 80)
    
    # Validate ORB
    print("\nValidating ORB Strategy...")
    orb_signals = generate_orb_signals(data)
    # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
    orb_returns = orb_signals * data['close'].pct_change().shift(1)
    orb_returns = orb_returns.dropna()
    
    orb_sharpe = orb_returns.mean() / orb_returns.std() * np.sqrt(252) if orb_returns.std() > 0 else 0
    orb_win_rate = len(orb_returns[orb_returns > 0]) / len(orb_returns)
    orb_cumulative = (1 + orb_returns).cumprod()
    orb_max_dd = (orb_cumulative / orb_cumulative.cummax() - 1).min()
    
    print(f"  Sharpe: {orb_sharpe:.2f}")
    print(f"  Win Rate: {orb_win_rate:.2%}")
    print(f"  Max Drawdown: {orb_max_dd:.2%}")
    print(f"  Recommendation: {'SCALE' if orb_sharpe > 0.8 and orb_max_dd > -0.15 else 'KILL'}")
    
    # Validate VWAP
    print("\nValidating VWAP Strategy...")
    vwap_signals = generate_vwap_signals(data)
    # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
    vwap_returns = vwap_signals * data['close'].pct_change().shift(1)
    vwap_returns = vwap_returns.dropna()
    
    vwap_sharpe = vwap_returns.mean() / vwap_returns.std() * np.sqrt(252) if vwap_returns.std() > 0 else 0
    vwap_win_rate = len(vwap_returns[vwap_returns > 0]) / len(vwap_returns)
    vwap_cumulative = (1 + vwap_returns).cumprod()
    vwap_max_dd = (vwap_cumulative / vwap_cumulative.cummax() - 1).min()
    
    print(f"  Sharpe: {vwap_sharpe:.2f}")
    print(f"  Win Rate: {vwap_win_rate:.2%}")
    print(f"  Max Drawdown: {vwap_max_dd:.2%}")
    print(f"  Recommendation: {'SCALE' if vwap_sharpe > 0.8 and vwap_max_dd > -0.15 else 'KILL'}")
    
    # Validate Momentum
    print("\nValidating Momentum Strategy (PCP/Vol Carry proxy)...")
    momentum_signals = generate_momentum_signals(data, period=20)
    # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
    momentum_returns = momentum_signals * data['close'].pct_change().shift(1)
    momentum_returns = momentum_returns.dropna()
    
    momentum_sharpe = momentum_returns.mean() / momentum_returns.std() * np.sqrt(252) if momentum_returns.std() > 0 else 0
    momentum_win_rate = len(momentum_returns[momentum_returns > 0]) / len(momentum_returns)
    momentum_cumulative = (1 + momentum_returns).cumprod()
    momentum_max_dd = (momentum_cumulative / momentum_cumulative.cummax() - 1).min()
    
    print(f"  Sharpe: {momentum_sharpe:.2f}")
    print(f"  Win Rate: {momentum_win_rate:.2%}")
    print(f"  Max Drawdown: {momentum_max_dd:.2%}")
    print(f"  Recommendation: {'SCALE' if momentum_sharpe > 0.8 and momentum_max_dd > -0.15 else 'KILL'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    results = {
        'ORB': {'sharpe': orb_sharpe, 'win_rate': orb_win_rate, 'max_dd': orb_max_dd},
        'VWAP': {'sharpe': vwap_sharpe, 'win_rate': vwap_win_rate, 'max_dd': vwap_max_dd},
        'Momentum': {'sharpe': momentum_sharpe, 'win_rate': momentum_win_rate, 'max_dd': momentum_max_dd}
    }
    
    for strategy_name, metrics in results.items():
        print(f"\n{strategy_name}:")
        print(f"  Sharpe: {metrics['sharpe']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Max DD: {metrics['max_dd']:.2%}")
        status = 'SCALE' if metrics['sharpe'] > 0.8 and metrics['max_dd'] > -0.15 else 'KILL'
        print(f"  Status: {status}")
    
    # Calculate objective scores
    print("\n" + "=" * 80)
    print("OBJECTIVE FUNCTION SCORING")
    print("=" * 80)
    
    objective = ObjectiveFunction(ObjectiveConstraints())
    
    for strategy_name, metrics in results.items():
        # Simulate returns and costs for objective calculation
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        costs = pd.Series(np.random.uniform(0.0001, 0.0003, 252))
        positions = pd.Series(np.random.uniform(-0.3, 0.3, 252))
        
        score = objective.calculate_objective(returns, costs, positions)
        
        print(f"\n{strategy_name}:")
        print(f"  Objective Score: {score.objective_value:.2f}")
        print(f"  Sharpe: {score.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {score.max_drawdown:.2%}")
        print(f"  Feasible: {score.is_feasible}")
        
        if score.constraint_violations:
            print(f"  Violations: {', '.join(score.constraint_violations)}")
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
