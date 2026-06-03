"""
VaR/CVaR Risk Limits

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#15)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Value at Risk (VaR) at 99% confidence, 1-day horizon
- Conditional Value at Risk (CVaR) / Expected Shortfall
- Stress tests (2008, COVID, 2022, custom scenarios)
- Circuit breakers and real-time limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats


@dataclass
class RiskLimitConfig:
    """Configuration for VaR/CVaR Limits"""
    # VaR parameters
    var_confidence: float = 0.99  # 99% confidence
    var_horizon_days: int = 1  # 1-day horizon
    var_method: str = "historical"  # "historical", "parametric", "monte_carlo"
    var_window: int = 252  # 1 year for historical VaR
    
    # CVaR parameters
    cvar_confidence: float = 0.95  # 95% confidence
    cvar_window: int = 252
    
    # Limits
    max_var_pct: float = 0.02  # Maximum 2% daily VaR
    max_cvar_pct: float = 0.03  # Maximum 3% daily CVaR
    max_portfolio_var_pct: float = 0.05  # Maximum 5% portfolio VaR
    
    # Circuit breakers
    enable_circuit_breakers: bool = True
    consecutive_loss_days: int = 5  # Stop after 5 consecutive losing days
    daily_loss_limit_pct: float = 0.05  # Stop if daily loss > 5%
    rolling_loss_limit_pct: float = 0.15  # Reduce size if 20d loss > 15%
    
    # Stress tests
    enable_stress_tests: bool = True
    stress_scenarios: List[str] = None  # "2008", "COVID", "2022", "custom"


class VaRCVaRLimits:
    """
    VaR/CVaR Risk Limits Manager
    
    Implements Value at Risk and Conditional Value at Risk limits
    with circuit breakers and stress testing.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: RiskLimitConfig):
        self.config = config
        
        # Risk history
        self.var_history: List[float] = []
        self.cvar_history: List[float] = []
        self.loss_history: List[float] = []
        
        # Current state
        self.current_var: float = 0.0
        self.current_cvar: float = 0.0
        self.current_drawdown: float = 0.0
        self.consecutive_losses: int = 0
        
        # Circuit breaker state
        self.circuit_breaker_triggered: bool = False
        self.circuit_breaker_reason: str = ""
    
    def calculate_var(self, returns: pd.Series, portfolio_value: float = 1_000_000) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            VaR in currency units
        """
        if len(returns) < self.config.var_window:
            # Use available data
            returns = returns[-len(returns):]
        else:
            returns = returns[-self.config.var_window:]
        
        if self.config.var_method == "historical":
            var = self._historical_var(returns)
        elif self.config.var_method == "parametric":
            var = self._parametric_var(returns)
        elif self.config.var_method == "monte_carlo":
            var = self._monte_carlo_var(returns)
        else:
            var = self._historical_var(returns)
        
        # Convert to currency
        var_currency = var * portfolio_value
        self.current_var = var_currency
        self.var_history.append(var_currency)
        
        return var_currency
    
    def _historical_var(self, returns: pd.Series) -> float:
        """Historical VaR (non-parametric)"""
        alpha = 1 - self.config.var_confidence
        var = np.percentile(returns, alpha * 100)
        return abs(var)
    
    def _parametric_var(self, returns: pd.Series) -> float:
        """Parametric VaR (assuming normal distribution)"""
        mu = returns.mean()
        sigma = returns.std()
        
        z_score = stats.norm.ppf(1 - self.config.var_confidence)
        var = mu - z_score * sigma
        
        return abs(var)
    
    def _monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000) -> float:
        """Monte Carlo VaR"""
        mu = returns.mean()
        sigma = returns.std()
        
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        var = np.percentile(simulated_returns, (1 - self.config.var_confidence) * 100)
        
        return abs(var)
    
    def calculate_cvar(self, returns: pd.Series, portfolio_value: float = 1_000_000) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            
        Returns:
            CVaR in currency units
        """
        if len(returns) < self.config.cvar_window:
            returns = returns[-len(returns):]
        else:
            returns = returns[-self.config.cvar_window:]
        
        # Get VaR threshold
        alpha = 1 - self.config.cvar_confidence
        var_threshold = np.percentile(returns, alpha * 100)
        
        # CVaR is mean of returns below VaR threshold
        tail_losses = returns[returns <= var_threshold]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var_threshold
        
        # Convert to currency
        cvar_currency = abs(cvar) * portfolio_value
        self.current_cvar = cvar_currency
        self.cvar_history.append(cvar_currency)
        
        return cvar_currency
    
    def check_var_limit(self, portfolio_value: float = 1_000_000) -> Tuple[bool, float]:
        """
        Check if VaR exceeds limit
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (exceeds_limit, var_pct)
        """
        var_pct = self.current_var / portfolio_value
        exceeds = var_pct > self.config.max_var_pct
        
        return exceeds, var_pct
    
    def check_cvar_limit(self, portfolio_value: float = 1_000_000) -> Tuple[bool, float]:
        """
        Check if CVaR exceeds limit
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (exceeds_limit, cvar_pct)
        """
        cvar_pct = self.current_cvar / portfolio_value
        exceeds = cvar_pct > self.config.max_cvar_pct
        
        return exceeds, cvar_pct
    
    def check_circuit_breakers(self, daily_return: float, rolling_returns: pd.Series) -> List[str]:
        """
        Check circuit breaker conditions
        
        Args:
            daily_return: Today's return
            rolling_returns: Rolling returns (last 20 days)
            
        Returns:
            List of triggered circuit breakers
        """
        triggers = []
        
        if not self.config.enable_circuit_breakers:
            return triggers
        
        # Check daily loss limit
        if daily_return < -self.config.daily_loss_limit_pct:
            triggers.append(f"Daily loss limit exceeded: {daily_return:.2%} < -{self.config.daily_loss_limit_pct:.0%}")
            self.circuit_breaker_triggered = True
            self.circuit_breaker_reason = "daily_loss_limit"
        
        # Check consecutive losses
        if daily_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.consecutive_losses >= self.config.consecutive_loss_days:
            triggers.append(f"Consecutive loss days: {self.consecutive_losses} >= {self.config.consecutive_loss_days}")
            self.circuit_breaker_triggered = True
            self.circuit_breaker_reason = "consecutive_losses"
        
        # Check rolling loss limit
        if len(rolling_returns) >= 20:
            rolling_loss = rolling_returns.sum()
            if rolling_loss < -self.config.rolling_loss_limit_pct:
                triggers.append(f"Rolling loss limit exceeded: {rolling_loss:.2%} < -{self.config.rolling_loss_limit_pct:.0%}")
                self.circuit_breaker_triggered = True
                self.circuit_breaker_reason = "rolling_loss_limit"
        
        return triggers
    
    def run_stress_test(self, positions: Dict[str, float], 
                       historical_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Run stress tests on current positions
        
        Args:
            positions: Current positions (symbol -> value)
            historical_returns: Historical returns DataFrame
            
        Returns:
            Dictionary of stress scenario -> loss
        """
        if not self.config.enable_stress_tests:
            return {}
        
        stress_results = {}
        
        # Default scenarios
        scenarios = self.config.stress_scenarios or ["2008", "COVID", "2022"]
        
        for scenario in scenarios:
            if scenario == "2008":
                # 2008 crisis: -40% market drop
                shock = -0.40
            elif scenario == "COVID":
                # COVID crash: -30% market drop
                shock = -0.30
            elif scenario == "2022":
                # 2022 volatility spike: -20% market drop
                shock = -0.20
            else:
                # Custom: -25% market drop
                shock = -0.25
            
            # Calculate portfolio loss under scenario
            portfolio_loss = 0.0
            for symbol, position_value in positions.items():
                if symbol in historical_returns.columns:
                    # Add some correlation to the shock
                    asset_beta = historical_returns[symbol].std() / historical_returns.values.std()
                    asset_shock = shock * asset_beta
                    portfolio_loss += position_value * asset_shock
                else:
                    portfolio_loss += position_value * shock
            
            stress_results[scenario] = portfolio_loss
        
        return stress_results
    
    def get_risk_summary(self, portfolio_value: float = 1_000_000) -> Dict:
        """Get comprehensive risk summary"""
        var_exceeds, var_pct = self.check_var_limit(portfolio_value)
        cvar_exceeds, cvar_pct = self.check_cvar_limit(portfolio_value)
        
        return {
            "current_var": self.current_var,
            "current_var_pct": var_pct,
            "var_limit_pct": self.config.max_var_pct,
            "var_exceeds_limit": var_exceeds,
            "current_cvar": self.current_cvar,
            "current_cvar_pct": cvar_pct,
            "cvar_limit_pct": self.config.max_cvar_pct,
            "cvar_exceeds_limit": cvar_exceeds,
            "current_drawdown": self.current_drawdown,
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_reason": self.circuit_breaker_reason
        }


def simulate_returns(n_days: int = 252) -> pd.Series:
    """Simulate portfolio returns for testing"""
    np.random.seed(42)
    
    # Generate returns with fat tails
    returns = np.random.standard_t(df=3, size=n_days) * 0.02
    returns = pd.Series(returns)
    
    return returns


if __name__ == "__main__":
    # Example usage
    config = RiskLimitConfig(
        var_confidence=0.99,
        var_method="historical",
        max_var_pct=0.02,
        max_cvar_pct=0.03,
        enable_circuit_breakers=True,
        enable_stress_tests=True
    )
    
    risk_manager = VaRCVaRLimits(config)
    
    # Simulate returns
    print("Simulating portfolio returns...")
    returns = simulate_returns(500)
    
    # Calculate VaR
    print("\nCalculating VaR...")
    var = risk_manager.calculate_var(returns, portfolio_value=1_000_000)
    print(f"  VaR (99%, 1d): ₹{var:,.0f}")
    
    # Calculate CVaR
    print("\nCalculating CVaR...")
    cvar = risk_manager.calculate_cvar(returns, portfolio_value=1_000_000)
    print(f"  CVaR (95%): ₹{cvar:,.0f}")
    
    # Check limits
    print("\nChecking risk limits...")
    var_exceeds, var_pct = risk_manager.check_var_limit()
    cvar_exceeds, cvar_pct = risk_manager.check_cvar_limit()
    
    print(f"  VaR: {var_pct:.2%} (limit: {config.max_var_pct:.0%}) - {'EXCEEDS' if var_exceeds else 'OK'}")
    print(f"  CVaR: {cvar_pct:.2%} (limit: {config.max_cvar_pct:.0%}) - {'EXCEEDS' if cvar_exceeds else 'OK'}")
    
    # Circuit breakers
    print("\nTesting circuit breakers...")
    daily_return = -0.06  # 6% loss
    rolling_returns = returns.tail(20)
    triggers = risk_manager.check_circuit_breakers(daily_return, rolling_returns)
    
    if triggers:
        print(f"  Circuit breakers triggered:")
        for trigger in triggers:
            print(f"    - {trigger}")
    else:
        print(f"  No circuit breakers triggered")
    
    # Stress tests
    print("\nRunning stress tests...")
    positions = {"RELIANCE": 200000, "TCS": 300000, "HDFCBANK": 500000}
    historical_returns = pd.DataFrame(np.random.randn(500, 3) * 0.02, 
                                     columns=["RELIANCE", "TCS", "HDFCBANK"])
    
    stress_results = risk_manager.run_stress_test(positions, historical_returns)
    print(f"  Stress test results:")
    for scenario, loss in stress_results.items():
        print(f"    {scenario}: -₹{abs(loss):,.0f}")
    
    # Risk summary
    print("\nRisk Summary:")
    summary = risk_manager.get_risk_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
