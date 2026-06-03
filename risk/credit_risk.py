"""
Credit Risk Modeling

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#45)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Probability of Default (PD) modeling
- Loss Given Default (LGD) estimation
- Exposure at Default (EAD) calculation
- Credit VaR (CVaR)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Install with: pip install scipy")


@dataclass
class CreditRiskConfig:
    """Configuration for Credit Risk Modeling"""
    # PD parameters
    pd_window: int = 252  # 1 year lookback for PD
    pd_threshold: float = 0.05  # 5% PD threshold
    
    # LGD parameters
    recovery_rate: float = 0.4  # 40% recovery rate
    lgd_std: float = 0.2  # LGD standard deviation
    
    # EAD parameters
    ead_multiplier: float = 1.0  # EAD multiplier
    
    # CVaR parameters
    cvar_confidence: float = 0.99  # 99% confidence for CVaR
    correlation: float = 0.3  # Asset correlation
    
    # Regulatory parameters
    regulatory_capital: float = 0.08  # 8% regulatory capital


class CreditRiskModel:
    """
    Credit Risk Model
    
    Models credit risk for counterparties and issuers.
    Calculates PD, LGD, EAD, and Credit VaR.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: CreditRiskConfig):
        self.config = config
        
        # Credit metrics
        self.pd_values: Dict[str, float] = {}
        self.lgd_values: Dict[str, float] = {}
        self.ead_values: Dict[str, float] = {}
    
    def calculate_merton_pd(self, asset_value: float, debt: float, 
                           volatility: float, risk_free_rate: float,
                           time_horizon: float = 1.0) -> float:
        """
        Calculate Probability of Default using Merton model
        
        Args:
            asset_value: Current asset value
            debt: Debt value
            volatility: Asset volatility
            risk_free_rate: Risk-free rate
            time_horizon: Time horizon in years
            
        Returns:
            Probability of Default
        """
        if asset_value <= 0 or debt <= 0:
            return 1.0
        
        # Distance to default
        d2 = (np.log(asset_value / debt) + (risk_free_rate - 0.5 * volatility ** 2) * time_horizon) / (volatility * np.sqrt(time_horizon))
        
        if SCIPY_AVAILABLE:
            pd = norm.cdf(-d2)
        else:
            # Fallback: simple approximation
            pd = max(0, min(1, -d2))
        
        return pd
    
    def calculate_structural_pd(self, equity: float, debt: float,
                              volatility: float, risk_free_rate: float) -> float:
        """
        Calculate PD using structural model
        
        Args:
            equity: Equity value
            debt: Debt value
            volatility: Equity volatility
            risk_free_rate: Risk-free rate
            
        Returns:
            Probability of Default
        """
        # Simplified structural model
        leverage = debt / (equity + debt) if (equity + debt) > 0 else 1.0
        
        # PD increases with leverage
        pd = min(1.0, leverage * 2.0)
        
        return pd
    
    def calculate_lgd(self, recovery_rate: float = None) -> float:
        """
        Calculate Loss Given Default
        
        Args:
            recovery_rate: Recovery rate (uses config if None)
            
        Returns:
            Loss Given Default
        """
        if recovery_rate is None:
            recovery_rate = self.config.recovery_rate
        
        lgd = 1.0 - recovery_rate
        return lgd
    
    def calculate_ead(self, exposure: float, multiplier: float = None) -> float:
        """
        Calculate Exposure at Default
        
        Args:
            exposure: Current exposure
            multiplier: EAD multiplier (uses config if None)
            
        Returns:
            Exposure at Default
        """
        if multiplier is None:
            multiplier = self.config.ead_multiplier
        
        ead = exposure * multiplier
        return ead
    
    def calculate_expected_loss(self, pd: float, lgd: float, ead: float) -> float:
        """
        Calculate Expected Loss (EL)
        
        Args:
            pd: Probability of Default
            lgd: Loss Given Default
            ead: Exposure at Default
            
        Returns:
            Expected Loss
        """
        el = pd * lgd * ead
        return el
    
    def calculate_credit_var(self, exposures: pd.Series, pds: pd.Series,
                            lgds: pd.Series, correlation: float = None) -> float:
        """
        Calculate Credit VaR using Vasicek model
        
        Args:
            exposures: Exposure at Default for each exposure
            pds: Probability of Default for each exposure
            lgds: Loss Given Default for each exposure
            correlation: Asset correlation (uses config if None)
            
        Returns:
            Credit VaR
        """
        if correlation is None:
            correlation = self.config.correlation
        
        # Calculate individual ELs
        els = pds * lgds * exposures
        total_el = els.sum()
        
        # Calculate unexpected loss (UL)
        # Vasicek model
        if SCIPY_AVAILABLE:
            # Inverse normal of PD
            inverse_pds = norm.ppf(pds)
            
            # Conditional PD at confidence level
            confidence = self.config.cvar_confidence
            inverse_confidence = norm.ppf(confidence)
            
            # Conditional PD
            conditional_pds = norm.cdf((inverse_pds + np.sqrt(correlation) * inverse_confidence) / np.sqrt(1 - correlation))
            
            # Conditional EL
            conditional_els = conditional_pds * lgds * exposures
            total_ul = conditional_els.sum() - total_el
            
            # CVaR = EL + UL
            cvar = total_el + total_ul
        else:
            # Fallback: simple VaR
            cvar = total_el * 2.0
        
        return cvar
    
    def calculate_regulatory_capital(self, exposures: pd.Series, pds: pd.Series,
                                    lgds: pd.Series) -> Dict:
        """
        Calculate regulatory capital requirements
        
        Args:
            exposures: Exposure at Default
            pds: Probability of Default
            lgds: Loss Given Default
            
        Returns:
            Regulatory capital requirements
        """
        # Calculate RWA (Risk-Weighted Assets)
        rwas = exposures * 1.0  # Simplified: 100% risk weight
        
        # Calculate capital requirement
        capital_requirement = rwas.sum() * self.config.regulatory_capital
        
        # Calculate EL and UL
        el = (pds * lgds * exposures).sum()
        cvar = self.calculate_credit_var(exposures, pds, lgds)
        ul = cvar - el
        
        return {
            "rwa": rwas.sum(),
            "capital_requirement": capital_requirement,
            "expected_loss": el,
            "unexpected_loss": ul,
            "credit_var": cvar
        }
    
    def run_credit_analysis(self, counterparties: List[Dict]) -> Dict:
        """
        Run comprehensive credit analysis
        
        Args:
            counterparties: List of counterparty data
            
        Returns:
            Credit analysis results
        """
        results = {}
        
        for cp in counterparties:
            name = cp["name"]
            
            # Calculate PD
            if "equity" in cp and "debt" in cp:
                pd = self.calculate_structural_pd(cp["equity"], cp["debt"], 
                                                   cp.get("volatility", 0.2), 0.05)
            else:
                pd = cp.get("pd", 0.05)
            
            # Calculate LGD
            lgd = self.calculate_lgd(cp.get("recovery_rate"))
            
            # Calculate EAD
            ead = self.calculate_ead(cp["exposure"])
            
            # Calculate EL
            el = self.calculate_expected_loss(pd, lgd, ead)
            
            # Store metrics
            self.pd_values[name] = pd
            self.lgd_values[name] = lgd
            self.ead_values[name] = ead
            
            results[name] = {
                "pd": pd,
                "lgd": lgd,
                "ead": ead,
                "expected_loss": el
            }
        
        return results
    
    def get_credit_summary(self) -> Dict:
        """Get credit risk summary"""
        if not self.pd_values:
            return {}
        
        total_ead = sum(self.ead_values.values())
        total_el = sum(self.pd_values[k] * self.lgd_values[k] * self.ead_values[k] 
                        for k in self.pd_values.keys())
        
        return {
            "num_counterparties": len(self.pd_values),
            "total_ead": total_ead,
            "total_expected_loss": total_el,
            "avg_pd": np.mean(list(self.pd_values.values())),
            "avg_lgd": np.mean(list(self.lgd_values.values()))
        }


def simulate_counterparties(n_counterparties: int = 20) -> List[Dict]:
    """Simulate counterparty data for testing"""
    np.random.seed(42)
    
    counterparties = []
    
    for i in range(n_counterparties):
        equity = np.random.uniform(1000000, 10000000)
        debt = np.random.uniform(500000, 5000000)
        exposure = np.random.uniform(100000, 1000000)
        volatility = np.random.uniform(0.1, 0.4)
        
        cp = {
            "name": f"COUNTERPARTY_{i}",
            "equity": equity,
            "debt": debt,
            "exposure": exposure,
            "volatility": volatility
        }
        
        counterparties.append(cp)
    
    return counterparties


if __name__ == "__main__":
    # Example usage
    config = CreditRiskConfig(
        pd_window=252,
        pd_threshold=0.05,
        recovery_rate=0.4,
        cvar_confidence=0.99
    )
    
    credit_model = CreditRiskModel(config)
    
    # Simulate counterparties
    print("Simulating counterparties...")
    counterparties = simulate_counterparties(20)
    
    # Run credit analysis
    print("\nRunning credit analysis...")
    results = credit_model.run_credit_analysis(counterparties)
    
    print(f"\nCredit Analysis Results (first 5):")
    for name, result in list(results.items())[:5]:
        print(f"  {name}:")
        print(f"    PD: {result['pd']:.4f}")
        print(f"    LGD: {result['lgd']:.4f}")
        print(f"    EAD: ${result['ead']:,.0f}")
        print(f"    Expected Loss: ${result['expected_loss']:,.0f}")
    
    # Calculate CVaR
    print("\nCalculating Credit VaR...")
    exposures = pd.Series([cp["exposure"] for cp in counterparties], 
                          index=[cp["name"] for cp in counterparties])
    pds = pd.Series([results[cp["name"]]["pd"] for cp in counterparties],
                    index=[cp["name"] for cp in counterparties])
    lgds = pd.Series([results[cp["name"]]["lgd"] for cp in counterparties],
                    index=[cp["name"] for cp in counterparties])
    
    cvar = credit_model.calculate_credit_var(exposures, pds, lgds)
    print(f"  Credit VaR (99%): ${cvar:,.0f}")
    
    # Regulatory capital
    print("\nCalculating regulatory capital...")
    reg_cap = credit_model.calculate_regulatory_capital(exposures, pds, lgds)
    print(f"  RWA: ${reg_cap['rwa']:,.0f}")
    print(f"  Capital Requirement: ${reg_cap['capital_requirement']:,.0f}")
    print(f"  Expected Loss: ${reg_cap['expected_loss']:,.0f}")
    print(f"  Unexpected Loss: ${reg_cap['unexpected_loss']:,.0f}")
    
    # Summary
    print("\nCredit Summary:")
    summary = credit_model.get_credit_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
