"""
Counterparty Risk

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#46)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Counterparty credit risk
- Netting agreements
- Collateral management
- Exposure profiling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CounterpartyRiskConfig:
    """Configuration for Counterparty Risk"""
    # Exposure parameters
    exposure_window: int = 20  # Days for exposure calculation
    confidence_level: float = 0.95  # Confidence level for PFE
    
    # Netting parameters
    netting_agreement: bool = True
    netting_efficiency: float = 0.6  # 60% netting efficiency
    
    # Collateral parameters
    collateral_threshold: float = 0.5  # 50% collateral threshold
    haircut: float = 0.1  # 10% haircut
    
    # Limits
    max_exposure_per_counterparty: float = 10000000  # $10M max exposure
    max_total_exposure: float = 100000000  # $100M total exposure


class CounterpartyRiskManager:
    """
    Counterparty Risk Manager
    
    Measures and manages counterparty risk in derivatives
    and other bilateral transactions.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: CounterpartyRiskConfig):
        self.config = config
        
        # Counterparty exposures
        self.current_exposures: Dict[str, float] = {}
        
        # Collateral
        self.collateral: Dict[str, float] = {}
        
        # Netting sets
        self.netting_sets: Dict[str, List[str]] = {}
    
    def calculate_current_exposure(self, mtm_values: pd.Series) -> float:
        """
        Calculate current exposure (positive MTM)
        
        Args:
            mtm_values: Mark-to-market values
            
        Returns:
            Current exposure
        """
        current_exposure = mtm_values[mtm_values > 0].sum()
        return current_exposure
    
    def calculate_potential_future_exposure(self, mtm_values: pd.Series,
                                           volatilities: pd.Series,
                                           time_horizon: float = 1.0) -> float:
        """
        Calculate Potential Future Exposure (PFE)
        
        Args:
            mtm_values: Mark-to-market values
            volatilities: Volatilities
            time_horizon: Time horizon in years
            
        Returns:
            PFE at confidence level
        """
        # Calculate exposure volatility
        exposure_vol = np.sqrt((volatilities ** 2).sum())
        
        # Calculate PFE
        if self.config.confidence_level > 0.5:
            from scipy.stats import norm
            multiplier = norm.ppf(self.config.confidence_level)
        else:
            multiplier = 1.0
        
        current_exposure = self.calculate_current_exposure(mtm_values)
        pfe = current_exposure + multiplier * exposure_vol * np.sqrt(time_horizon)
        
        return max(pfe, 0)
    
    def calculate_net_exposure(self, mtm_values: pd.Series, netting_set: List[str]) -> float:
        """
        Calculate net exposure after netting
        
        Args:
            mtm_values: Mark-to-market values
            netting_set: List of trades in netting set
            
        Returns:
            Net exposure
        """
        if not self.config.netting_agreement:
            return self.calculate_current_exposure(mtm_values)
        
        # Net positive and negative exposures
        net_exposure = mtm_values.sum()
        
        # Apply netting efficiency
        gross_exposure = mtm_values[mtm_values > 0].sum()
        netted_exposure = gross_exposure * (1 - self.config.netting_efficiency)
        
        return max(netted_exposure, 0)
    
    def calculate_collateral_requirement(self, exposure: float, 
                                      collateral_value: float = 0) -> float:
        """
        Calculate collateral requirement
        
        Args:
            exposure: Current exposure
            collateral_value: Value of posted collateral
            
        Returns:
            Additional collateral required
        """
        # Apply haircut to collateral
        collateral_adjusted = collateral_value * (1 - self.config.haircut)
        
        # Calculate required collateral
        required_collateral = exposure * self.config.collateral_threshold
        
        # Additional collateral needed
        additional_collateral = max(0, required_collateral - collateral_adjusted)
        
        return additional_collateral
    
    def update_exposure(self, counterparty: str, exposure: float) -> None:
        """
        Update counterparty exposure
        
        Args:
            counterparty: Counterparty name
            exposure: Exposure amount
        """
        self.current_exposures[counterparty] = exposure
    
    def update_collateral(self, counterparty: str, collateral: float) -> None:
        """
        Update collateral for counterparty
        
        Args:
            counterparty: Counterparty name
            collateral: Collateral amount
        """
        self.collateral[counterparty] = collateral
    
    def check_exposure_limits(self) -> Dict:
        """
        Check if exposures exceed limits
        
        Returns:
            Limit check results
        """
        violations = []
        
        for counterparty, exposure in self.current_exposures.items():
            if exposure > self.config.max_exposure_per_counterparty:
                violations.append({
                    "counterparty": counterparty,
                    "exposure": exposure,
                    "limit": self.config.max_exposure_per_counterparty,
                    "excess": exposure - self.config.max_exposure_per_counterparty
                })
        
        total_exposure = sum(self.current_exposures.values())
        total_violation = total_exposure > self.config.max_total_exposure
        
        return {
            "individual_violations": violations,
            "num_violations": len(violations),
            "total_exposure": total_exposure,
            "total_limit": self.config.max_total_exposure,
            "total_violation": total_violation,
            "compliant": len(violations) == 0 and not total_violation
        }
    
    def calculate_cva(self, exposures: Dict[str, float], pds: Dict[str, float],
                     lgds: Dict[str, float]) -> float:
        """
        Calculate Credit Valuation Adjustment (CVA)
        
        Args:
            exposures: Counterparty exposures
            pds: Probability of Default
            lgds: Loss Given Default
            
        Returns:
            CVA
        """
        cva = 0.0
        
        for counterparty, exposure in exposures.items():
            pd = pds.get(counterparty, 0.05)
            lgd = lgds.get(counterparty, 0.6)
            
            cva += exposure * pd * lgd
        
        return cva
    
    def run_counterparty_analysis(self, counterparties: List[Dict]) -> Dict:
        """
        Run comprehensive counterparty analysis
        
        Args:
            counterparties: List of counterparty data
            
        Returns:
            Counterparty analysis results
        """
        results = {}
        
        for cp in counterparties:
            name = cp["name"]
            exposure = cp["exposure"]
            collateral = cp.get("collateral", 0)
            
            # Update exposure and collateral
            self.update_exposure(name, exposure)
            self.update_collateral(name, collateral)
            
            # Calculate collateral requirement
            additional_collateral = self.calculate_collateral_requirement(exposure, collateral)
            
            results[name] = {
                "exposure": exposure,
                "collateral": collateral,
                "additional_collateral": additional_collateral,
                "net_exposure": exposure - collateral * (1 - self.config.haircut)
            }
        
        # Check limits
        limit_check = self.check_exposure_limits()
        
        return {
            "counterparty_results": results,
            "limit_check": limit_check
        }
    
    def get_counterparty_summary(self) -> Dict:
        """Get counterparty risk summary"""
        total_exposure = sum(self.current_exposures.values())
        total_collateral = sum(self.collateral.values())
        
        return {
            "num_counterparties": len(self.current_exposures),
            "total_exposure": total_exposure,
            "total_collateral": total_collateral,
            "net_exposure": total_exposure - total_collateral * (1 - self.config.haircut)
        }


def simulate_counterparties(n_counterparties: int = 15) -> List[Dict]:
    """Simulate counterparty data for testing"""
    np.random.seed(42)
    
    counterparties = []
    
    for i in range(n_counterparties):
        exposure = np.random.uniform(100000, 5000000)
        collateral = np.random.uniform(0, exposure * 0.5)
        
        cp = {
            "name": f"COUNTERPARTY_{i}",
            "exposure": exposure,
            "collateral": collateral
        }
        
        counterparties.append(cp)
    
    return counterparties


if __name__ == "__main__":
    # Example usage
    config = CounterpartyRiskConfig(
        exposure_window=20,
        confidence_level=0.95,
        netting_agreement=True,
        max_exposure_per_counterparty=5000000,
        max_total_exposure=50000000
    )
    
    cp_manager = CounterpartyRiskManager(config)
    
    # Simulate counterparties
    print("Simulating counterparties...")
    counterparties = simulate_counterparties(15)
    
    # Run counterparty analysis
    print("\nRunning counterparty analysis...")
    results = cp_manager.run_counterparty_analysis(counterparties)
    
    print(f"\nCounterparty Results (first 5):")
    for name, result in list(results["counterparty_results"].items())[:5]:
        print(f"  {name}:")
        print(f"    Exposure: ${result['exposure']:,.0f}")
        print(f"    Collateral: ${result['collateral']:,.0f}")
        print(f"    Additional Collateral: ${result['additional_collateral']:,.0f}")
        print(f"    Net Exposure: ${result['net_exposure']:,.0f}")
    
    # Limit check
    print("\nLimit Check:")
    limit_check = results["limit_check"]
    print(f"  Total Exposure: ${limit_check['total_exposure']:,.0f}")
    print(f"  Total Limit: ${limit_check['total_limit']:,.0f}")
    print(f"  Compliant: {limit_check['compliant']}")
    print(f"  Violations: {limit_check['num_violations']}")
    
    # Summary
    print("\nCounterparty Summary:")
    summary = cp_manager.get_counterparty_summary()
    for key, value in summary.items():
        print(f"  {key}: ${value:,.0f}")
