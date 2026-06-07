"""
Strategy Validation - Walk-forward Testing for Existing Strategies
Based on the critique: Validate ORB, VWAP, PCP, Vol Carry before adding new models

Objective:
- Determine which strategies actually make money
- Use walk-forward validation for robust out-of-sample testing
- Kill strategies that don't perform
- Focus on proven edges before scaling

Strategies to Validate:
1. ORB (Opening Range Breakout)
2. VWAP (Volume-Weighted Average Price)
3. PCP (Pair Trading)
4. Vol Carry (Volatility Carry)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from backtest.walk_forward import WalkForwardValidator, WalkForwardConfig, FoldResult


@dataclass
class StrategyValidationResult:
    """Result of strategy validation."""
    strategy_name: str
    num_folds: int
    train_sharpe_mean: float
    train_sharpe_std: float
    test_sharpe_mean: float
    test_sharpe_std: float
    train_return_mean: float
    test_return_mean: float
    train_max_dd_mean: float
    test_max_dd_mean: float
    decay_rate: float
    overfitting_ratio: float
    is_overfitted: bool
    is_viable: bool
    recommendation: str


class StrategyValidator:
    """
    Validate existing strategies using walk-forward testing.
    
    Process:
    1. Load historical data for each strategy
    2. Generate signals using strategy logic
    3. Run walk-forward validation
    4. Evaluate performance metrics
    5. Make kill/scale recommendation
    """
    
    def __init__(self):
        self.validator = WalkForwardValidator(WalkForwardConfig())
        self.validation_results: Dict[str, StrategyValidationResult] = {}
        
        # Viability thresholds
        self.min_test_sharpe = 0.8
        self.max_overfitting = 0.4
        self.max_decay_rate = 0.3
    
    def validate_orb_strategy(self, data: Dict[str, pd.DataFrame]) -> StrategyValidationResult:
        """Validate ORB strategy with walk-forward testing."""
        print("\n" + "="*60)
        print("VALIDATING ORB STRATEGY")
        print("="*60)
        
        # Generate ORB signals for all symbols
        all_returns = []
        all_dates = []
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            # ORB signal: break out of first 15-minute range
            # Simplified for validation
            df = df.copy()
            df['orb_signal'] = 0
            
            # Calculate opening range (first 15 minutes = 15 bars for 1-min data)
            for i in range(15, len(df)):
                or_high = df['high'].iloc[i-15:i].max()
                or_low = df['low'].iloc[i-15:i].min()
                current_price = df['close'].iloc[i]
                
                if current_price > or_high:
                    df.loc[df.index[i], 'orb_signal'] = 1
                elif current_price < or_low:
                    df.loc[df.index[i], 'orb_signal'] = -1
            
            # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
            # Calculate returns
            returns = df['close'].pct_change().shift(1)
            strategy_returns = df['orb_signal'] * returns
            
            all_returns.extend(strategy_returns.dropna().tolist())
            all_dates.extend(strategy_returns.dropna().index.tolist())
        
        # Create combined dataset
        combined_df = pd.DataFrame({
            'returns': all_returns
        }, index=all_dates)
        
        # Run walk-forward validation
        def train_func(X_train, y_train):
            # ORB is rule-based, no training needed
            return None
        
        def predict_func(model, X_test):
            # For ORB, returns are the signals
            return X_test['returns'] if 'returns' in X_test else pd.Series(0, index=X_test.index)
        
        # Create feature column (dummy for walk-forward)
        combined_df['feature'] = 0
        
        results = self.validator.validate(
            data=combined_df,
            train_func=train_func,
            predict_func=predict_func,
            target_col="returns"
        )
        
        # Evaluate viability
        is_viable = (
            results['test_sharpe_mean'] >= self.min_test_sharpe and
            not results['is_overfitted'] and
            results['decay_rate'] <= self.max_decay_rate
        )
        
        recommendation = "SCALE" if is_viable else "KILL"
        
        result = StrategyValidationResult(
            strategy_name="ORB",
            num_folds=results['num_folds'],
            train_sharpe_mean=results['train_sharpe_mean'],
            train_sharpe_std=results['train_sharpe_std'],
            test_sharpe_mean=results['test_sharpe_mean'],
            test_sharpe_std=results['test_sharpe_std'],
            train_return_mean=results['train_return_mean'],
            test_return_mean=results['test_return_mean'],
            train_max_dd_mean=results['train_max_dd_mean'],
            test_max_dd_mean=results['test_max_dd_mean'],
            decay_rate=results['decay_rate'],
            overfitting_ratio=results['overfitting_ratio'],
            is_overfitted=results['is_overfitted'],
            is_viable=is_viable,
            recommendation=recommendation
        )
        
        self.validation_results["ORB"] = result
        return result
    
    def validate_vwap_strategy(self, data: Dict[str, pd.DataFrame]) -> StrategyValidationResult:
        """Validate VWAP strategy with walk-forward testing."""
        print("\n" + "="*60)
        print("VALIDATING VWAP STRATEGY")
        print("="*60)
        
        # Generate VWAP signals for all symbols
        all_returns = []
        all_dates = []
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            df = df.copy()
            
            # Calculate VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # VWAP reversion signal
            df['vwap_signal'] = 0
            for i in range(20, len(df)):
                current_price = df['close'].iloc[i]
                vwap = df['vwap'].iloc[i]
                
                # Reversion: buy when price below VWAP, sell when above
                if current_price < vwap * 0.99:
                    df.loc[df.index[i], 'vwap_signal'] = 1
                elif current_price > vwap * 1.01:
                    df.loc[df.index[i], 'vwap_signal'] = -1
            
            # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
            # Calculate returns
            returns = df['close'].pct_change().shift(1)
            strategy_returns = df['vwap_signal'] * returns
            
            all_returns.extend(strategy_returns.dropna().tolist())
            all_dates.extend(strategy_returns.dropna().index.tolist())
        
        # Create combined dataset
        combined_df = pd.DataFrame({
            'returns': all_returns
        }, index=all_dates)
        
        # Run walk-forward validation
        def train_func(X_train, y_train):
            return None
        
        def predict_func(model, X_test):
            return X_test['returns'] if 'returns' in X_test else pd.Series(0, index=X_test.index)
        
        combined_df['feature'] = 0
        
        results = self.validator.validate(
            data=combined_df,
            train_func=train_func,
            predict_func=predict_func,
            target_col="returns"
        )
        
        # Evaluate viability
        is_viable = (
            results['test_sharpe_mean'] >= self.min_test_sharpe and
            not results['is_overfitted'] and
            results['decay_rate'] <= self.max_decay_rate
        )
        
        recommendation = "SCALE" if is_viable else "KILL"
        
        result = StrategyValidationResult(
            strategy_name="VWAP",
            num_folds=results['num_folds'],
            train_sharpe_mean=results['train_sharpe_mean'],
            train_sharpe_std=results['train_sharpe_std'],
            test_sharpe_mean=results['test_sharpe_mean'],
            test_sharpe_std=results['test_sharpe_std'],
            train_return_mean=results['train_return_mean'],
            test_return_mean=results['test_return_mean'],
            train_max_dd_mean=results['train_max_dd_mean'],
            test_max_dd_mean=results['test_max_dd_mean'],
            decay_rate=results['decay_rate'],
            overfitting_ratio=results['overfitting_ratio'],
            is_overfitted=results['is_overfitted'],
            is_viable=is_viable,
            recommendation=recommendation
        )
        
        self.validation_results["VWAP"] = result
        return result
    
    def validate_momentum_strategy(self, data: Dict[str, pd.DataFrame]) -> StrategyValidationResult:
        """Validate simple momentum strategy as proxy for PCP/Vol Carry."""
        print("\n" + "="*60)
        print("VALIDATING MOMENTUM STRATEGY (Proxy for PCP/Vol Carry)")
        print("="*60)
        
        # Generate momentum signals
        all_returns = []
        all_dates = []
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            df = df.copy()
            
            # 20-day momentum
            df['momentum'] = df['close'].pct_change(20)
            df['momentum_signal'] = 0
            
            for i in range(20, len(df)):
                momentum = df['momentum'].iloc[i]
                
                if momentum > 0.02:
                    df.loc[df.index[i], 'momentum_signal'] = 1
                elif momentum < -0.02:
                    df.loc[df.index[i], 'momentum_signal'] = -1
            
            # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
            # Calculate returns
            returns = df['close'].pct_change().shift(1)
            strategy_returns = df['momentum_signal'] * returns
            
            all_returns.extend(strategy_returns.dropna().tolist())
            all_dates.extend(strategy_returns.dropna().index.tolist())
        
        # Create combined dataset
        combined_df = pd.DataFrame({
            'returns': all_returns
        }, index=all_dates)
        
        # Run walk-forward validation
        def train_func(X_train, y_train):
            return None
        
        def predict_func(model, X_test):
            return X_test['returns'] if 'returns' in X_test else pd.Series(0, index=X_test.index)
        
        combined_df['feature'] = 0
        
        results = self.validator.validate(
            data=combined_df,
            train_func=train_func,
            predict_func=predict_func,
            target_col="returns"
        )
        
        # Evaluate viability
        is_viable = (
            results['test_sharpe_mean'] >= self.min_test_sharpe and
            not results['is_overfitted'] and
            results['decay_rate'] <= self.max_decay_rate
        )
        
        recommendation = "SCALE" if is_viable else "KILL"
        
        result = StrategyValidationResult(
            strategy_name="Momentum",
            num_folds=results['num_folds'],
            train_sharpe_mean=results['train_sharpe_mean'],
            train_sharpe_std=results['train_sharpe_std'],
            test_sharpe_mean=results['test_sharpe_mean'],
            test_sharpe_std=results['test_sharpe_std'],
            train_return_mean=results['train_return_mean'],
            test_return_mean=results['test_return_mean'],
            train_max_dd_mean=results['train_max_dd_mean'],
            test_max_dd_mean=results['test_max_dd_mean'],
            decay_rate=results['decay_rate'],
            overfitting_ratio=results['overfitting_ratio'],
            is_overfitted=results['is_overfitted'],
            is_viable=is_viable,
            recommendation=recommendation
        )
        
        self.validation_results["Momentum"] = result
        return result
    
    def validate_all_strategies(self, data: Dict[str, pd.DataFrame]) -> Dict[str, StrategyValidationResult]:
        """Validate all existing strategies."""
        print("\n" + "="*60)
        print("STRATEGY VALIDATION SUITE")
        print("="*60)
        
        # Validate ORB
        orb_result = self.validate_orb_strategy(data)
        
        # Validate VWAP
        vwap_result = self.validate_vwap_strategy(data)
        
        # Validate Momentum (proxy for PCP/Vol Carry)
        momentum_result = self.validate_momentum_strategy(data)
        
        return self.validation_results
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validation results."""
        data = []
        
        for strategy_name, result in self.validation_results.items():
            data.append({
                'Strategy': result.strategy_name,
                'Test Sharpe': f"{result.test_sharpe_mean:.2f} ± {result.test_sharpe_std:.2f}",
                'Test Return': f"{result.test_return_mean:.4f}",
                'Test Max DD': f"{result.test_max_dd_mean:.4f}",
                'Decay Rate': f"{result.decay_rate:.2%}",
                'Overfitting': f"{result.overfitting_ratio:.2%}",
                'Viable': result.is_viable,
                'Recommendation': result.recommendation
            })
        
        return pd.DataFrame(data)
    
    def get_model_graveyard(self) -> List[str]:
        """Get list of strategies to kill."""
        return [
            name for name, result in self.validation_results.items()
            if result.recommendation == "KILL"
        ]
    
    def get_production_strategies(self) -> List[str]:
        """Get list of strategies to scale to production."""
        return [
            name for name, result in self.validation_results.items()
            if result.recommendation == "SCALE"
        ]


if __name__ == "__main__":
    # Test the Strategy Validator
    print("Testing Strategy Validation...")
    
    validator = StrategyValidator()
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    data = {}
    
    for symbol in symbols:
        n = 2520  # 10 years of daily data
        dates = pd.date_range("2014-01-01", periods=n, freq="D")
        prices = np.random.normal(100, 10, n).cumsum()
        prices = prices - prices.min() + 100
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
            'close': prices,
            'volume': np.random.normal(1000000, 200000, n)
        }, index=dates)
        
        data[symbol] = df
    
    # Validate all strategies
    results = validator.validate_all_strategies(data)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    summary = validator.get_validation_summary()
    print(summary.to_string(index=False))
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    production = validator.get_production_strategies()
    graveyard = validator.get_model_graveyard()
    
    print(f"\nScale to Production: {production}")
    print(f"Kill (Move to Graveyard): {graveyard}")
    
    if not production:
        print("\n⚠️  WARNING: No strategies passed validation!")
        print("   Focus on finding a single proven edge before scaling.")
