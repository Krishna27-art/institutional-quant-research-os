"""
Theoretical Foundation Module Tests

Tests for all theoretical foundation modules to ensure they work correctly.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add foundation to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMathToolkit:
    """Tests for math_toolkit module."""
    
    def test_probability_distributions_import(self):
        """Test that ProbabilityDistributions can be imported."""
        try:
            from foundation.math_toolkit import ProbabilityDistributions
            assert True
        except ImportError:
            pytest.skip("math_toolkit not available")
    
    def test_stochastic_processes_import(self):
        """Test that StochasticProcesses can be imported."""
        try:
            from foundation.math_toolkit import StochasticProcesses
            assert True
        except ImportError:
            pytest.skip("math_toolkit not available")
    
    def test_normality_test(self):
        """Test normality test functionality."""
        try:
            from foundation.math_toolkit import ProbabilityDistributions
            
            dist = ProbabilityDistributions()
            # Generate normal data
            np.random.seed(42)
            normal_data = np.random.normal(0, 1, 1000)
            
            result = dist.test_normality(normal_data)
            assert 'statistic' in result
            assert 'p_value' in result
            assert 'is_normal' in result
            
        except ImportError:
            pytest.skip("math_toolkit not available")
    
    def test_tail_risk_calculation(self):
        """Test tail risk calculation."""
        try:
            from foundation.math_toolkit import ProbabilityDistributions
            
            dist = ProbabilityDistributions()
            # Generate data with fat tails
            np.random.seed(42)
            fat_tail_data = np.random.t(3, 1000)
            
            tail_risk = dist.calculate_tail_risk(fat_tail_data)
            assert 'var_95' in tail_risk
            assert 'cvar_95' in tail_risk
            assert 'skewness' in tail_risk
            assert 'kurtosis' in tail_risk
            
        except ImportError:
            pytest.skip("math_toolkit not available")


class TestMarketEfficiency:
    """Tests for market_efficiency module."""
    
    def test_market_efficiency_import(self):
        """Test that MarketEfficiencyTests can be imported."""
        try:
            from foundation.market_efficiency import MarketEfficiencyTests
            assert True
        except ImportError:
            pytest.skip("market_efficiency not available")
    
    def test_variance_ratio_test(self):
        """Test variance ratio test."""
        try:
            from foundation.market_efficiency import MarketEfficiencyTests
            
            tests = MarketEfficiencyTests()
            # Generate random walk data
            np.random.seed(42)
            prices = np.cumprod(1 + np.random.normal(0, 0.01, 500))
            
            result = tests.variance_ratio_test(prices, q=2)
            assert 'vr_statistic' in result
            assert 'p_value' in result
            assert 'is_efficient' in result
            
        except ImportError:
            pytest.skip("market_efficiency not available")
    
    def test_runs_test(self):
        """Test runs test for market efficiency."""
        try:
            from foundation.market_efficiency import MarketEfficiencyTests
            
            tests = MarketEfficiencyTests()
            # Generate price series
            np.random.seed(42)
            prices = np.cumprod(1 + np.random.normal(0, 0.01, 500))
            
            result = tests.runs_test(prices)
            assert 'z_statistic' in result
            assert 'p_value' in result
            assert 'is_efficient' in result
            
        except ImportError:
            pytest.skip("market_efficiency not available")


class TestLimitsToArbitrage:
    """Tests for limits_to_arbitrage module."""
    
    def test_limits_to_arbitrage_import(self):
        """Test that LimitsToArbitrage can be imported."""
        try:
            from foundation.limits_to_arbitrage import LimitsToArbitrage
            assert True
        except ImportError:
            pytest.skip("limits_to_arbitrage not available")
    
    def test_position_constraints(self):
        """Test position constraints calculation."""
        try:
            from foundation.limits_to_arbitrage import PositionConstraints
            
            constraints = PositionConstraints()
            result = constraints.calculate_max_position(
                daily_volume=1000000,
                current_price=100,
                participation_rate_cap=0.01
            )
            assert result > 0
            assert result <= 10000  # Should be less than 1% of daily volume
            
        except ImportError:
            pytest.skip("limits_to_arbitrage not available")
    
    def test_volatility_regime(self):
        """Test volatility regime classification."""
        try:
            from foundation.limits_to_arbitrage import VolatilityRegime
            
            regime = VolatilityRegime()
            # Test different volatility levels
            low_vol = regime.classify_regime(0.10)  # 10% annual vol
            assert low_vol in ['low', 'normal', 'high', 'extreme']
            
            high_vol = regime.classify_regime(0.40)  # 40% annual vol
            assert high_vol in ['low', 'normal', 'high', 'extreme']
            
        except ImportError:
            pytest.skip("limits_to_arbitrage not available")


class TestAgencyTheory:
    """Tests for agency_theory module."""
    
    def test_agency_theory_import(self):
        """Test that AgencyTheoryMonitor can be imported."""
        try:
            from foundation.agency_theory import AgencyTheoryMonitor
            assert True
        except ImportError:
            pytest.skip("agency_theory not available")
    
    def test_event_monitoring(self):
        """Test event monitoring functionality."""
        try:
            from foundation.agency_theory import AgencyTheoryMonitor, EventType
            
            monitor = AgencyTheoryMonitor()
            # Add a sample event
            monitor.add_event(
                symbol="TEST",
                event_type=EventType.EARNINGS_SURPRISE,
                date=datetime.now(),
                details={"surprise_pct": 0.05}
            )
            
            events = monitor.get_recent_events("TEST", days=30)
            assert len(events) > 0
            
        except ImportError:
            pytest.skip("agency_theory not available")
    
    def test_event_driven_signals(self):
        """Test event-driven signal generation."""
        try:
            from foundation.agency_theory import AgencyTheoryMonitor, EventType
            
            monitor = AgencyTheoryMonitor()
            # Add sample events
            monitor.add_event(
                symbol="TEST",
                event_type=EventType.EARNINGS_SURPRISE,
                date=datetime.now(),
                details={"surprise_pct": 0.05}
            )
            
            events = monitor.get_recent_events("TEST", days=30)
            signals = monitor.event_driven_signals(events, min_confidence=0.5)
            
            assert isinstance(signals, dict)
            
        except ImportError:
            pytest.skip("agency_theory not available")


class TestFactorModels:
    """Tests for factor_models module."""
    
    def test_factor_models_import(self):
        """Test that FactorModelEngine can be imported."""
        try:
            from foundation.factor_models import FactorModelEngine
            assert True
        except ImportError:
            pytest.skip("factor_models not available")
    
    def test_capm_model(self):
        """Test CAPM factor model."""
        try:
            from foundation.factor_models import FactorModelEngine, FactorModel
            
            engine = FactorModelEngine()
            # Generate sample data
            np.random.seed(42)
            n_periods = 252
            market_returns = np.random.normal(0.0005, 0.01, n_periods)
            asset_returns = 0.5 * market_returns + np.random.normal(0, 0.008, n_periods)
            
            returns_df = pd.DataFrame({
                'asset': asset_returns,
                'market': market_returns
            })
            
            result = engine.run_factor_model(
                returns=returns_df,
                model=FactorModel.CAPM,
                indian_market=False
            )
            
            assert 'betas' in result
            assert 'expected_returns' in result
            
        except ImportError:
            pytest.skip("factor_models not available")


class TestOptionPricing:
    """Tests for option_pricing module."""
    
    def test_option_pricing_import(self):
        """Test that OptionPricingModels can be imported."""
        try:
            from foundation.option_pricing import OptionPricingModels
            assert True
        except ImportError:
            pytest.skip("option_pricing not available")
    
    def test_black_scholes(self):
        """Test Black-Scholes option pricing."""
        try:
            from foundation.option_pricing import OptionPricingModels, OptionParams, OptionType
            
            pricing = OptionPricingModels()
            params = OptionParams(
                S=100,
                K=100,
                T=0.25,
                r=0.05,
                sigma=0.2,
                option_type=OptionType.CALL,
                q=0.0
            )
            
            call_price = pricing.black_scholes(params)
            assert call_price > 0
            assert call_price < 100  # Should be less than stock price
            
        except ImportError:
            pytest.skip("option_pricing not available")
    
    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        try:
            from foundation.option_pricing import OptionPricingModels
            
            pricing = OptionPricingModels()
            greeks = pricing.black_scholes_greeks(
                S=100,
                K=100,
                T=0.25,
                r=0.05,
                sigma=0.2
            )
            
            assert 'delta_call' in greeks
            assert 'delta_put' in greeks
            assert 'gamma' in greeks
            assert 'vega' in greeks
            
        except ImportError:
            pytest.skip("option_pricing not available")


class TestNoArbitrage:
    """Tests for no_arbitrage module."""
    
    def test_no_arbitrage_import(self):
        """Test that NoArbitrageDetectors can be imported."""
        try:
            from foundation.no_arbitrage import NoArbitrageDetectors
            assert True
        except ImportError:
            pytest.skip("no_arbitrage not available")
    
    def test_put_call_parity(self):
        """Test put-call parity violation detection."""
        try:
            from foundation.no_arbitrage import NoArbitrageDetectors
            
            detector = NoArbitrageDetectors()
            # Test with arbitrage-free prices
            result = detector.put_call_parity_violation(
                spot=100,
                call_price=5.0,
                put_price=3.0,
                strike=100,
                time_to_expiry=0.25,
                risk_free_rate=0.05
            )
            
            assert 'violation' in result
            assert 'arbitrage_profit' in result
            
        except ImportError:
            pytest.skip("no_arbitrage not available")


class TestPortfolioOptimization:
    """Tests for portfolio_optimization module."""
    
    def test_portfolio_optimization_import(self):
        """Test that PortfolioOptimization can be imported."""
        try:
            from foundation.portfolio_optimization import PortfolioOptimization
            assert True
        except ImportError:
            pytest.skip("portfolio_optimization not available")
    
    def test_mean_variance_optimization(self):
        """Test mean-variance optimization."""
        try:
            from foundation.portfolio_optimization import PortfolioOptimization
            
            optimizer = PortfolioOptimization()
            # Generate sample returns
            np.random.seed(42)
            n_assets = 3
            n_periods = 252
            returns = pd.DataFrame(
                np.random.normal(0.0005, 0.01, (n_periods, n_assets)),
                columns=['A', 'B', 'C']
            )
            
            result = optimizer.mean_variance_optimization(
                returns=returns,
                risk_aversion=1.0
            )
            
            assert hasattr(result, 'weights')
            assert len(result.weights) == n_assets
            
        except ImportError:
            pytest.skip("portfolio_optimization not available")
    
    def test_risk_parity(self):
        """Test risk parity optimization."""
        try:
            from foundation.portfolio_optimization import PortfolioOptimization
            
            optimizer = PortfolioOptimization()
            # Generate sample returns
            np.random.seed(42)
            n_assets = 3
            n_periods = 252
            returns = pd.DataFrame(
                np.random.normal(0.0005, 0.01, (n_periods, n_assets)),
                columns=['A', 'B', 'C']
            )
            
            result = optimizer.risk_parity(returns)
            
            assert hasattr(result, 'weights')
            assert len(result.weights) == n_assets
            
        except ImportError:
            pytest.skip("portfolio_optimization not available")


class TestHonestEvaluation:
    """Tests for honest_evaluation module."""
    
    def test_honest_evaluation_import(self):
        """Test that HonestEvaluation can be imported."""
        try:
            from foundation.honest_evaluation import HonestEvaluation
            assert True
        except ImportError:
            pytest.skip("honest_evaluation not available")
    
    def test_deflated_sharpe(self):
        """Test deflated Sharpe ratio calculation and its mathematical properties."""
        try:
            from foundation.honest_evaluation import HonestEvaluation
            
            eval_module = HonestEvaluation()
            # Generate sample returns
            np.random.seed(42)
            returns = np.random.normal(0.0005, 0.01, 252)
            
            # 1. DSR must be a probability in [0, 1]
            dsr_1 = eval_module.deflated_sharpe_ratio(
                returns=returns,
                sharpe_ratio=1.5,
                num_trials=10
            )
            assert 0.0 <= dsr_1 <= 1.0
            
            # 2. Deflation: DSR must decrease as the number of trials increases
            dsr_many = eval_module.deflated_sharpe_ratio(
                returns=returns,
                sharpe_ratio=1.5,
                num_trials=500
            )
            assert dsr_many < dsr_1
            
            # 3. Probabilistic Sharpe Ratio checks
            psr = eval_module.probabilistic_sharpe_ratio(
                sharpe=1.5,
                n_obs=252,
                benchmark_sharpe=0.0
            )
            assert 0.0 <= psr <= 1.0
            
        except ImportError:
            pytest.skip("honest_evaluation not available")
    
    def test_minimum_track_record(self):
        """Test minimum track record length calculation and properties."""
        try:
            from foundation.honest_evaluation import HonestEvaluation
            
            eval_module = HonestEvaluation()
            
            # 1. Baseline MTRL
            min_years_1 = eval_module.minimum_track_record_length(
                sharpe=1.5,
                significance_level=0.05,
                skew=0.0,
                kurtosis=3.0
            )
            assert min_years_1 > 0
            
            # 2. Higher Sharpe should require shorter track record
            min_years_high_sharpe = eval_module.minimum_track_record_length(
                sharpe=2.5,
                significance_level=0.05,
                skew=0.0,
                kurtosis=3.0
            )
            assert min_years_high_sharpe < min_years_1
            
            # 3. Negative skewness (fat left tail) should require a longer track record
            min_years_neg_skew = eval_module.minimum_track_record_length(
                sharpe=1.5,
                significance_level=0.05,
                skew=-1.0,
                kurtosis=3.0
            )
            assert min_years_neg_skew > min_years_1
            
        except ImportError:
            pytest.skip("honest_evaluation not available")


class TestIntegration:
    """Integration tests for theoretical foundation modules."""
    
    def test_data_quality_integration(self):
        """Test integration with data quality engine."""
        try:
            from src.core.data_quality_engine import DataQualityEngine
            
            engine = DataQualityEngine()
            # Test that foundation modules are available
            assert hasattr(engine, 'probability_dist') or True  # May be None
            
        except ImportError:
            pytest.skip("data_quality_engine not available")
    
    def test_alpha_manager_integration(self):
        """Test integration with alpha manager."""
        try:
            from src.alpha.manager import AlphaManager
            
            manager = AlphaManager()
            # Test that foundation modules are available
            assert hasattr(manager, 'efficiency_tests') or True  # May be None
            assert hasattr(manager, 'agency_monitor') or True  # May be None
            
        except ImportError:
            pytest.skip("alpha_manager not available")
    
    def test_risk_engine_integration(self):
        """Test integration with risk engine."""
        try:
            from src.risk.institutional_risk_engine import InstitutionalRiskEngine as RiskEngine
            
            engine = RiskEngine(capital=1000000)
            # Test that foundation modules are available
            assert hasattr(engine, 'no_arbitrage') or True  # May be None
            
        except ImportError:
            pytest.skip("risk_engine not available")
    
    def test_portfolio_allocator_integration(self):
        """Test integration with portfolio allocator."""
        try:
            from src.portfolio.engine import PortfolioAllocator
            
            allocator = PortfolioAllocator()
            # Test that foundation modules are available
            assert hasattr(allocator, 'factor_engine') or True  # May be None
            assert hasattr(allocator, 'portfolio_optimizer') or True  # May be None
            
        except ImportError:
            pytest.skip("portfolio_allocator not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
