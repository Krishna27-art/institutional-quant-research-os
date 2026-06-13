"""
Modern Portfolio Theory Extensions: Factor Risk Model, Liquidity Risk, Stress Testing
Based on the critique: Build MPT extensions with Factor Risk, Liquidity Risk, Tail Risk, Regime Risk

Institutional level:
- Risk parity
- Factor risk
- Tail risk
- Liquidity risk
- Regime risk

Build:
- Risk decomposition
- Scenario engine (Covid crash, 2008 crisis, 2022 inflation, Election shock)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize


class RiskType(Enum):
    """Types of risk."""
    MARKET = "market"
    FACTOR = "factor"
    SPECIFIC = "specific"
    LIQUIDITY = "liquidity"
    TAIL = "tail"
    REGIME = "regime"


@dataclass
class FactorRisk:
    """Factor risk for a portfolio."""
    timestamp: datetime
    portfolio_id: str
    factor_name: str
    exposure: float
    contribution_to_risk: float
    marginal_risk: float


@dataclass
class LiquidityRisk:
    """Liquidity risk for a portfolio."""
    timestamp: datetime
    portfolio_id: str
    liquidity_score: float  # 0 to 1, higher = more liquid
    bid_ask_spread: float
    volume_ratio: float
    liquidity_at_risk: float  # Potential loss due to liquidity
    is_liquidity_constrained: bool


@dataclass
class StressTestResult:
    """Stress test result."""
    scenario_name: str
    portfolio_value_before: float
    portfolio_value_after: float
    loss: float
    loss_percentage: float
    worst_drawdown: float
    recovery_time_days: float


class MPTExtensionsEngine:
    """
    Modern Portfolio Theory Extensions Engine.
    
    Features:
    - Factor risk model
    - Liquidity risk assessment
    - Stress testing (Covid, 2008, 2022 inflation, Election)
    - Risk decomposition
    - Scenario analysis
    """
    
    def __init__(self):
        self.factor_risks: Dict[str, List[FactorRisk]] = {}
        self.liquidity_risks: Dict[str, List[LiquidityRisk]] = {}
        self.stress_test_results: Dict[str, List[StressTestResult]] = {}
        
        # Risk thresholds
        self.liquidity_threshold = 0.3  # 30% liquidity score threshold
        self.max_drawdown_threshold = 0.2  # 20% max drawdown threshold
    
    def calculate_factor_risk(
        self,
        portfolio_id: str,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        weights: pd.Series
    ) -> Dict[str, FactorRisk]:
        """
        Calculate factor risk decomposition.
        
        Args:
            portfolio_id: Portfolio ID
            returns: Portfolio returns
            factor_returns: Factor returns DataFrame
            weights: Portfolio weights
            
        Returns:
            Dictionary of factor risks
        """
        # Calculate portfolio returns
        portfolio_returns = returns
        
        # Align data
        aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
        
        if len(aligned) < 30:
            return {}
        
        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        
        # Fit factor model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate factor exposures (betas)
        factor_exposures = dict(zip(X.columns, model.coef_))
        
        # Calculate portfolio variance
        portfolio_variance = portfolio_returns.var()
        
        # Calculate factor contributions to risk
        factor_risks = {}
        total_contribution = 0
        
        for factor_name, exposure in factor_exposures.items():
            factor_variance = X[factor_name].var()
            contribution = (exposure ** 2) * factor_variance
            total_contribution += contribution
        
        # Normalize contributions
        for factor_name, exposure in factor_exposures.items():
            factor_variance = X[factor_name].var()
            contribution = (exposure ** 2) * factor_variance
            marginal_risk = exposure * factor_variance
            
            factor_risk = FactorRisk(
                timestamp=datetime.now(),
                portfolio_id=portfolio_id,
                factor_name=factor_name,
                exposure=exposure,
                contribution_to_risk=contribution / total_contribution if total_contribution > 0 else 0,
                marginal_risk=marginal_risk
            )
            
            # Store
            key = f"{portfolio_id}_{factor_name}"
            if key not in self.factor_risks:
                self.factor_risks[key] = []
            self.factor_risks[key].append(factor_risk)
        
        return factor_risks
    
    def calculate_liquidity_risk(
        self,
        portfolio_id: str,
        holdings: Dict[str, float],  # symbol -> weight
        bid_ask_spreads: Dict[str, float],
        volumes: Dict[str, float],
        avg_volumes: Dict[str, float]
    ) -> LiquidityRisk:
        """
        Calculate liquidity risk for portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            holdings: Portfolio holdings (symbol -> weight)
            bid_ask_spreads: Bid-ask spreads
            volumes: Current volumes
            avg_volumes: Average volumes
            
        Returns:
            LiquidityRisk
        """
        # Calculate portfolio-weighted spread
        weighted_spread = sum(
            holdings.get(symbol, 0) * bid_ask_spreads.get(symbol, 0.01)
            for symbol in holdings
        )
        
        # Calculate portfolio liquidity score
        # Higher score = more liquid
        liquidity_scores = []
        for symbol in holdings:
            spread = bid_ask_spreads.get(symbol, 0.01)
            volume_ratio = volumes.get(symbol, 0) / avg_volumes.get(symbol, 1) if avg_volumes.get(symbol, 1) > 0 else 0
            
            # Liquidity score = 1 - spread - (1 / volume_ratio)
            score = 1 - spread - (1 / (volume_ratio + 1))
            liquidity_scores.append(score)
        
        portfolio_liquidity_score = np.mean(liquidity_scores) if liquidity_scores else 0.5
        
        # Calculate volume ratio
        portfolio_volume = sum(volumes.get(symbol, 0) for symbol in holdings)
        portfolio_avg_volume = sum(avg_volumes.get(symbol, 0) for symbol in holdings)
        volume_ratio = portfolio_volume / portfolio_avg_volume if portfolio_avg_volume > 0 else 1.0
        
        # Calculate liquidity at risk
        # Potential loss if liquidity dries up
        liquidity_at_risk = weighted_spread * 0.5  # Assume 50% of spread as loss
        
        # Check if liquidity constrained
        is_liquidity_constrained = portfolio_liquidity_score < self.liquidity_threshold
        
        liquidity_risk = LiquidityRisk(
            timestamp=datetime.now(),
            portfolio_id=portfolio_id,
            liquidity_score=portfolio_liquidity_score,
            bid_ask_spread=weighted_spread,
            volume_ratio=volume_ratio,
            liquidity_at_risk=liquidity_at_risk,
            is_liquidity_constrained=is_liquidity_constrained
        )
        
        # Store
        if portfolio_id not in self.liquidity_risks:
            self.liquidity_risks[portfolio_id] = []
        self.liquidity_risks[portfolio_id].append(liquidity_risk)
        
        return liquidity_risk
    
    def run_stress_test(
        self,
        portfolio_id: str,
        portfolio_value: float,
        holdings: Dict[str, float],
        scenario_shocks: Dict[str, float]
    ) -> Dict[str, StressTestResult]:
        """
        Run stress test on portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            portfolio_value: Current portfolio value
            holdings: Portfolio holdings (symbol -> weight)
            scenario_shocks: Dictionary of scenario -> shock percentage
            
        Returns:
            Dictionary of stress test results
        """
        results = {}
        
        for scenario_name, shock in scenario_shocks.items():
            # Apply shock to portfolio
            # Simplified: assume all holdings are affected by shock
            portfolio_value_after = portfolio_value * (1 + shock)
            
            loss = portfolio_value - portfolio_value_after
            loss_percentage = shock
            
            # Estimate worst drawdown during stress
            worst_drawdown = abs(shock) * 1.5  # Assume 1.5x shock as worst drawdown
            
            # Estimate recovery time (days)
            recovery_time_days = abs(shock) * 252 / 0.1  # Assume 10% annual recovery rate
            
            result = StressTestResult(
                scenario_name=scenario_name,
                portfolio_value_before=portfolio_value,
                portfolio_value_after=portfolio_value_after,
                loss=loss,
                loss_percentage=loss_percentage,
                worst_drawdown=worst_drawdown,
                recovery_time_days=recovery_time_days
            )
            
            results[scenario_name] = result
            
            # Store
            if portfolio_id not in self.stress_test_results:
                self.stress_test_results[portfolio_id] = []
            self.stress_test_results[portfolio_id].append(result)
        
        return results
    
    def get_standard_scenarios(self) -> Dict[str, float]:
        """
        Get standard stress test scenarios.
        
        Returns:
            Dictionary of scenario -> shock percentage
        """
        return {
            'Covid Crash': -0.30,  # 30% drop
            '2008 Crisis': -0.40,  # 40% drop
            '2022 Inflation': -0.15,  # 15% drop
            'Election Shock': -0.10,  # 10% drop
            'Rate Hike': -0.08,  # 8% drop
            'Flash Crash': -0.05,  # 5% drop
            'Black Swan': -0.50,  # 50% drop
            'Bull Market': 0.20,  # 20% gain
        }
    
    def decompose_risk(
        self,
        portfolio_id: str,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        weights: pd.Series
    ) -> Dict:
        """
        Decompose portfolio risk into components.
        
        Args:
            portfolio_id: Portfolio ID
            returns: Portfolio returns
            factor_returns: Factor returns
            weights: Portfolio weights
            
        Returns:
            Dictionary with risk decomposition
        """
        # Calculate total risk (volatility)
        total_risk = returns.std() * np.sqrt(252)
        
        # Calculate factor risk
        factor_risks = self.calculate_factor_risk(portfolio_id, returns, factor_returns, weights)
        
        # Sum factor contributions
        factor_risk_contribution = sum(r.contribution_to_risk for r in factor_risks.values())
        
        # Specific risk (idiosyncratic)
        specific_risk_contribution = 1 - factor_risk_contribution
        
        return {
            'total_risk': total_risk,
            'factor_risk': total_risk * factor_risk_contribution,
            'specific_risk': total_risk * specific_risk_contribution,
            'factor_risk_pct': factor_risk_contribution,
            'specific_risk_pct': specific_risk_contribution,
            'factor_breakdown': {f.factor_name: f.contribution_to_risk for f in factor_risks.values()}
        }
    
    def get_risk_summary(self, portfolio_id: str) -> Dict:
        """Get comprehensive risk summary for a portfolio."""
        summary = {}
        
        # Factor risk
        if portfolio_id in self.factor_risks:
            latest_factor_risks = [r for key, risks in self.factor_risks.items() if portfolio_id in key for r in risks]
            if latest_factor_risks:
                summary['factor_risks'] = {
                    r.factor_name: {
                        'exposure': r.exposure,
                        'contribution': r.contribution_to_risk,
                        'marginal_risk': r.marginal_risk
                    }
                    for r in latest_factor_risks[-10:]  # Last 10
                }
        
        # Liquidity risk
        if portfolio_id in self.liquidity_risks:
            latest_liquidity = self.liquidity_risks[portfolio_id][-1]
            summary['liquidity_risk'] = {
                'liquidity_score': latest_liquidity.liquidity_score,
                'bid_ask_spread': latest_liquidity.bid_ask_spread,
                'volume_ratio': latest_liquidity.volume_ratio,
                'liquidity_at_risk': latest_liquidity.liquidity_at_risk,
                'is_constrained': latest_liquidity.is_liquidity_constrained
            }
        
        # Stress test results
        if portfolio_id in self.stress_test_results:
            latest_stress = self.stress_test_results[portfolio_id][-8:]  # Last 8 scenarios
            summary['stress_tests'] = {
                r.scenario_name: {
                    'loss_pct': r.loss_percentage,
                    'worst_drawdown': r.worst_drawdown,
                    'recovery_days': r.recovery_time_days
                }
                for r in latest_stress
            }
        
        return summary


if __name__ == "__main__":
    # Test the MPT Extensions Engine
    print("Testing Modern Portfolio Theory Extensions: Factor Risk Model, Liquidity Risk, Stress Testing...")
    
    engine = MPTExtensionsEngine()
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    portfolio_returns = pd.Series(np.random.normal(0.0005, 0.015, n), index=dates)
    
    factor_returns = pd.DataFrame({
        'Market': np.random.normal(0.0003, 0.012, n),
        'Size': np.random.normal(0.0001, 0.008, n),
        'Value': np.random.normal(0.0002, 0.01, n),
        'Momentum': np.random.normal(0.0004, 0.011, n)
    }, index=dates)
    
    weights = pd.Series({'RELIANCE': 0.3, 'TCS': 0.25, 'HDFCBANK': 0.25, 'INFY': 0.2})
    
    # Calculate factor risk
    print("\nCalculating Factor Risk...")
    factor_risks = engine.calculate_factor_risk("portfolio_001", portfolio_returns, factor_returns, weights)
    
    print(f"Factor Risks:")
    for factor_name, risk in factor_risks.items():
        print(f"  {factor_name}:")
        print(f"    Exposure: {risk.exposure:.4f}")
        print(f"    Contribution: {risk.contribution_to_risk:.2%}")
        print(f"    Marginal Risk: {risk.marginal_risk:.4f}")
    
    # Calculate liquidity risk
    print("\nCalculating Liquidity Risk...")
    holdings = {'RELIANCE': 0.3, 'TCS': 0.25, 'HDFCBANK': 0.25, 'INFY': 0.2}
    bid_ask_spreads = {'RELIANCE': 0.002, 'TCS': 0.001, 'HDFCBANK': 0.003, 'INFY': 0.0015}
    volumes = {'RELIANCE': 5000000, 'TCS': 3000000, 'HDFCBANK': 4000000, 'INFY': 3500000}
    avg_volumes = {'RELIANCE': 10000000, 'TCS': 8000000, 'HDFCBANK': 9000000, 'INFY': 7000000}
    
    liquidity_risk = engine.calculate_liquidity_risk("portfolio_001", holdings, bid_ask_spreads, volumes, avg_volumes)
    
    print(f"Liquidity Risk:")
    print(f"  Liquidity Score: {liquidity_risk.liquidity_score:.2f}")
    print(f"  Bid-Ask Spread: {liquidity_risk.bid_ask_spread:.2%}")
    print(f"  Volume Ratio: {liquidity_risk.volume_ratio:.2f}")
    print(f"  Liquidity at Risk: {liquidity_risk.liquidity_at_risk:.2%}")
    print(f"  Is Constrained: {liquidity_risk.is_liquidity_constrained}")
    
    # Run stress tests
    print("\nRunning Stress Tests...")
    portfolio_value = 100000000  # 100 crore INR
    scenarios = engine.get_standard_scenarios()
    
    stress_results = engine.run_stress_test("portfolio_001", portfolio_value, holdings, scenarios)
    
    print(f"Stress Test Results:")
    for scenario_name, result in stress_results.items():
        print(f"  {scenario_name}:")
        print(f"    Loss: {result.loss_percentage:.2%}")
        print(f"    Worst Drawdown: {result.worst_drawdown:.2%}")
        print(f"    Recovery Time: {result.recovery_time_days:.0f} days")
    
    # Decompose risk
    print("\nDecomposing Risk...")
    risk_decomposition = engine.decompose_risk("portfolio_001", portfolio_returns, factor_returns, weights)
    
    print(f"Risk Decomposition:")
    print(f"  Total Risk: {risk_decomposition['total_risk']:.2%}")
    print(f"  Factor Risk: {risk_decomposition['factor_risk']:.2%}")
    print(f"  Specific Risk: {risk_decomposition['specific_risk']:.2%}")
    print(f"  Factor Risk %: {risk_decomposition['factor_risk_pct']:.2%}")
    print(f"  Specific Risk %: {risk_decomposition['specific_risk_pct']:.2%}")
    print(f"  Factor Breakdown: {risk_decomposition['factor_breakdown']}")
    
    # Get risk summary
    print("\nRisk Summary:")
    risk_summary = engine.get_risk_summary("portfolio_001")
    
    if 'factor_risks' in risk_summary:
        print(f"  Factor Risks: {list(risk_summary['factor_risks'].keys())}")
    
    if 'liquidity_risk' in risk_summary:
        print(f"  Liquidity Score: {risk_summary['liquidity_risk']['liquidity_score']:.2f}")
    
    if 'stress_tests' in risk_summary:
        print(f"  Stress Tests: {list(risk_summary['stress_tests'].keys())}")
