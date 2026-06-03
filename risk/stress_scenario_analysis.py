"""
Stress Testing and Scenario Analysis

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#43)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Historical stress scenarios
- Monte Carlo stress testing
- Forward-looking scenarios
- Regulatory stress testing (Basel III)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class StressTestConfig:
    """Configuration for Stress Testing"""
    # Scenario parameters
    num_scenarios: int = 1000  # Number of Monte Carlo scenarios
    time_horizon: int = 10  # Days
    
    # Historical scenarios
    historical_scenarios: List[str] = None
    
    # Stress parameters
    equity_shock: float = -0.30  # 30% equity shock
    rate_shock: float = 0.02  # 2% rate shock
    fx_shock: float = 0.15  # 15% FX shock
    commodity_shock: float = -0.25  # 25% commodity shock
    
    # Confidence levels
    confidence_levels: List[float] = None
    
    # Regulatory parameters
    regulatory_capital: float = 0.08  # 8% regulatory capital


class HistoricalScenario:
    """Historical stress scenario"""
    
    def __init__(self, name: str, date: str, shocks: Dict[str, float]):
        self.name = name
        self.date = date
        self.shocks = shocks


class StressTestEngine:
    """
    Stress Testing Engine
    
    Performs stress testing and scenario analysis
    to assess portfolio resilience.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        
        # Historical scenarios
        self.historical_scenarios: List[HistoricalScenario] = []
        self._initialize_historical_scenarios()
        
        # Scenario results
        self.scenario_results: Dict[str, Dict] = {}
    
    def _initialize_historical_scenarios(self) -> None:
        """Initialize historical stress scenarios"""
        scenarios = [
            HistoricalScenario(
                "2008 Financial Crisis",
                "2008-09-15",
                {"Equity": -0.40, "Bonds": 0.05, "Gold": 0.20, "Commodities": -0.50}
            ),
            HistoricalScenario(
                "COVID-19 Crash",
                "2020-03-23",
                {"Equity": -0.35, "Bonds": 0.10, "Gold": 0.15, "Commodities": -0.30}
            ),
            HistoricalScenario(
                "2013 Taper Tantrum",
                "2013-06-19",
                {"Equity": -0.05, "Bonds": -0.10, "Gold": -0.15, "Commodities": -0.10}
            ),
            HistoricalScenario(
                "Black Monday 1987",
                "1987-10-19",
                {"Equity": -0.22, "Bonds": 0.03, "Gold": 0.05, "Commodities": -0.15}
            )
        ]
        
        self.historical_scenarios = scenarios
    
    def apply_historical_scenario(self, portfolio_value: float, 
                                 positions: Dict[str, float],
                                 scenario: HistoricalScenario) -> Dict:
        """
        Apply historical scenario to portfolio
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            scenario: Historical scenario
            
        Returns:
            Scenario results
        """
        portfolio_loss = 0.0
        
        for asset, position in positions.items():
            shock = scenario.shocks.get(asset, 0.0)
            asset_loss = position * shock
            portfolio_loss += asset_loss
        
        portfolio_value_after = portfolio_value + portfolio_loss
        loss_pct = portfolio_loss / portfolio_value
        
        return {
            "scenario_name": scenario.name,
            "scenario_date": scenario.date,
            "portfolio_loss": portfolio_loss,
            "portfolio_value_after": portfolio_value_after,
            "loss_pct": loss_pct
        }
    
    def run_historical_scenarios(self, portfolio_value: float,
                                 positions: Dict[str, float]) -> List[Dict]:
        """
        Run all historical scenarios
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            
        Returns:
            List of scenario results
        """
        results = []
        
        for scenario in self.historical_scenarios:
            result = self.apply_historical_scenario(portfolio_value, positions, scenario)
            results.append(result)
            self.scenario_results[scenario.name] = result
        
        return results
    
    def monte_carlo_stress_test(self, portfolio_value: float,
                                 positions: Dict[str, float],
                                 returns: pd.DataFrame) -> Dict:
        """
        Monte Carlo stress test
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            returns: Historical returns
            
        Returns:
            Monte Carlo results
        """
        # Calculate covariance matrix
        cov = returns.cov()
        
        # Generate scenarios
        n_scenarios = self.config.num_scenarios
        n_assets = len(positions)
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov.values)
        
        # Generate random shocks
        random_shocks = np.random.randn(n_scenarios, n_assets)
        correlated_shocks = random_shocks @ L.T
        
        # Calculate portfolio losses
        portfolio_losses = []
        
        for i in range(n_scenarios):
            scenario_loss = 0.0
            
            for j, asset in enumerate(positions.keys()):
                shock = correlated_shocks[i, j]
                position = positions[asset]
                asset_loss = position * shock
                scenario_loss += asset_loss
            
            portfolio_losses.append(scenario_loss)
        
        portfolio_losses = np.array(portfolio_losses)
        
        # Calculate statistics
        mean_loss = portfolio_losses.mean()
        std_loss = portfolio_losses.std()
        
        # Calculate VaR and CVaR at different confidence levels
        confidence_levels = self.config.confidence_levels or [0.95, 0.99]
        
        var_results = {}
        cvar_results = {}
        
        for conf in confidence_levels:
            var = np.percentile(portfolio_losses, (1 - conf) * 100)
            
            # CVaR (average of losses beyond VaR)
            losses_beyond_var = portfolio_losses[portfolio_losses <= var]
            cvar = losses_beyond_var.mean() if len(losses_beyond_var) > 0 else var
            
            var_results[f"VaR_{int(conf*100)}%"] = var
            cvar_results[f"CVaR_{int(conf*100)}%"] = cvar
        
        return {
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "min_loss": portfolio_losses.min(),
            "max_loss": portfolio_losses.max(),
            "var_results": var_results,
            "cvar_results": cvar_results
        }
    
    def apply_forward_scenario(self, portfolio_value: float,
                             positions: Dict[str, float],
                             scenario_shocks: Dict[str, float]) -> Dict:
        """
        Apply forward-looking scenario
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            scenario_shocks: Custom scenario shocks
            
        Returns:
            Scenario results
        """
        portfolio_loss = 0.0
        
        for asset, position in positions.items():
            shock = scenario_shocks.get(asset, 0.0)
            asset_loss = position * shock
            portfolio_loss += asset_loss
        
        portfolio_value_after = portfolio_value + portfolio_loss
        loss_pct = portfolio_loss / portfolio_value
        
        return {
            "portfolio_loss": portfolio_loss,
            "portfolio_value_after": portfolio_value_after,
            "loss_pct": loss_pct
        }
    
    def run_regulatory_stress_test(self, portfolio_value: float,
                                    positions: Dict[str, float]) -> Dict:
        """
        Run regulatory stress test (Basel III)
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            
        Returns:
            Regulatory stress test results
        """
        # Apply regulatory shocks
        regulatory_shocks = {
            "Equity": self.config.equity_shock,
            "Bonds": self.config.rate_shock,
            "FX": self.config.fx_shock,
            "Commodities": self.config.commodity_shock
        }
        
        result = self.apply_forward_scenario(portfolio_value, positions, regulatory_shocks)
        
        # Calculate required capital
        required_capital = abs(result["portfolio_loss"]) * self.config.regulatory_capital
        
        return {
            **result,
            "required_capital": required_capital,
            "capital_ratio": required_capital / portfolio_value
        }
    
    def get_stress_test_summary(self) -> Dict:
        """Get stress test summary"""
        if not self.scenario_results:
            return {}
        
        worst_scenario = max(self.scenario_results.values(), key=lambda x: x["loss_pct"])
        
        return {
            "num_scenarios": len(self.scenario_results),
            "worst_scenario": worst_scenario["scenario_name"],
            "worst_loss_pct": worst_scenario["loss_pct"],
            "scenario_results": self.scenario_results
        }


def simulate_portfolio_positions(n_assets: int = 4) -> Dict[str, float]:
    """Simulate portfolio positions for testing"""
    np.random.seed(42)
    
    asset_classes = ["Equity", "Bonds", "Gold", "Commodities"]
    positions = {}
    
    for asset in asset_classes:
        positions[asset] = np.random.uniform(100000, 500000)
    
    return positions


def simulate_returns(n_assets: int = 4, n_days: int = 252) -> pd.DataFrame:
    """Simulate returns for testing"""
    np.random.seed(42)
    
    asset_names = ["Equity", "Bonds", "Gold", "Commodities"]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.3, 0.1, 0.2],
        [0.3, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.3],
        [0.2, 0.1, 0.3, 1.0]
    ])
    
    L = np.linalg.cholesky(correlation_matrix)
    
    independent_returns = np.random.randn(n_days, n_assets)
    correlated_returns = independent_returns @ L.T
    
    drifts = np.array([0.0003, 0.0001, 0.0002, 0.0001])
    scales = np.array([0.015, 0.005, 0.01, 0.02])
    
    returns = pd.DataFrame(
        correlated_returns * scales + drifts,
        index=dates,
        columns=asset_names
    )
    
    return returns


if __name__ == "__main__":
    # Example usage
    config = StressTestConfig(
        num_scenarios=1000,
        confidence_levels=[0.95, 0.99]
    )
    
    stress_test = StressTestEngine(config)
    
    # Simulate portfolio
    print("Simulating portfolio...")
    portfolio_value = 1000000  # $1M
    positions = simulate_portfolio_positions(4)
    returns = simulate_returns(4, 252)
    
    print(f"  Portfolio Value: ${portfolio_value:,.0f}")
    print(f"  Positions: {positions}")
    
    # Run historical scenarios
    print("\nRunning historical stress scenarios...")
    historical_results = stress_test.run_historical_scenarios(portfolio_value, positions)
    
    print(f"\nHistorical Scenario Results:")
    for result in historical_results:
        print(f"  {result['scenario_name']}: {result['loss_pct']:.2%} loss")
    
    # Monte Carlo stress test
    print("\nRunning Monte Carlo stress test...")
    mc_results = stress_test.monte_carlo_stress_test(portfolio_value, positions, returns)
    
    print(f"\nMonte Carlo Results:")
    print(f"  Mean Loss: ${mc_results['mean_loss']:,.0f}")
    print(f"  Std Loss: ${mc_results['std_loss']:,.0f}")
    print(f"  Min Loss: ${mc_results['min_loss']:,.0f}")
    print(f"  Max Loss: ${mc_results['max_loss']:,.0f}")
    
    for var_name, var_value in mc_results["var_results"].items():
        print(f"  {var_name}: ${var_value:,.0f}")
    
    for cvar_name, cvar_value in mc_results["cvar_results"].items():
        print(f"  {cvar_name}: ${cvar_value:,.0f}")
    
    # Regulatory stress test
    print("\nRunning regulatory stress test...")
    regulatory_results = stress_test.run_regulatory_stress_test(portfolio_value, positions)
    
    print(f"\nRegulatory Stress Test Results:")
    print(f"  Portfolio Loss: ${regulatory_results['portfolio_loss']:,.0f}")
    print(f"  Loss %: {regulatory_results['loss_pct']:.2%}")
    print(f"  Required Capital: ${regulatory_results['required_capital']:,.0f}")
    print(f"  Capital Ratio: {regulatory_results['capital_ratio']:.2%}")
    
    # Summary
    print("\nStress Test Summary:")
    summary = stress_test.get_stress_test_summary()
    for key, value in summary.items():
        if key != "scenario_results":
            print(f"  {key}: {value}")
