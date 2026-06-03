"""
Regulatory Compliance

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#48)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- SEBI compliance for Indian markets
- Position limits monitoring
- Reporting requirements
- Regulatory capital adequacy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ComplianceConfig:
    """Configuration for Regulatory Compliance"""
    # SEBI parameters
    max_position_pct: float = 0.10  # 10% max position per stock
    max_derivative_exposure: float = 0.20  # 20% max derivative exposure
    
    # Reporting parameters
    reporting_frequency: str = "daily"
    report_retention_days: int = 365
    
    # Capital parameters
    min_capital_ratio: float = 0.15  # 15% minimum capital ratio
    capital_buffer: float = 0.025  # 2.5% capital buffer
    
    # Risk parameters
    var_limit: float = 0.05  # 5% VaR limit
    leverage_limit: float = 3.0  # 3x leverage limit


class RegulatoryComplianceManager:
    """
    Regulatory Compliance Manager
    
    Ensures compliance with SEBI and other regulatory
    requirements for trading operations.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: ComplianceConfig):
        self.config = config
        
        # Compliance records
        self.compliance_records: List[Dict] = []
        
        # Violations
        self.violations: List[Dict] = []
        
        # Capital
        self.regulatory_capital: float = 0.0
        self.risk_weighted_assets: float = 0.0
    
    def check_position_limits(self, positions: Dict[str, float],
                            market_caps: Dict[str, float]) -> Dict:
        """
        Check position limits compliance
        
        Args:
            positions: Current positions
            market_caps: Market capitalizations
            
        Returns:
            Compliance check results
        """
        violations = []
        
        for asset, position in positions.items():
            if asset in market_caps:
                market_cap = market_caps[asset]
                position_pct = abs(position) / market_cap
                
                if position_pct > self.config.max_position_pct:
                    violations.append({
                        "asset": asset,
                        "position": position,
                        "market_cap": market_cap,
                        "position_pct": position_pct,
                        "limit": self.config.max_position_pct,
                        "excess": position_pct - self.config.max_position_pct
                    })
        
        return {
            "violations": violations,
            "num_violations": len(violations),
            "compliant": len(violations) == 0
        }
    
    def check_derivative_exposure(self, derivative_positions: Dict[str, float],
                                 total_capital: float) -> Dict:
        """
        Check derivative exposure limits
        
        Args:
            derivative_positions: Derivative positions
            total_capital: Total capital
            
        Returns:
            Compliance check results
        """
        total_derivative_exposure = sum(abs(pos) for pos in derivative_positions.values())
        derivative_exposure_pct = total_derivative_exposure / total_capital if total_capital > 0 else 0
        
        violation = derivative_exposure_pct > self.config.max_derivative_exposure
        
        return {
            "total_derivative_exposure": total_derivative_exposure,
            "derivative_exposure_pct": derivative_exposure_pct,
            "limit": self.config.max_derivative_exposure,
            "compliant": not violation,
            "excess": derivative_exposure_pct - self.config.max_derivative_exposure if violation else 0
        }
    
    def check_leverage_limit(self, total_exposure: float, total_capital: float) -> Dict:
        """
        Check leverage limit compliance
        
        Args:
            total_exposure: Total exposure
            total_capital: Total capital
            
        Returns:
            Compliance check results
        """
        leverage = total_exposure / total_capital if total_capital > 0 else 0
        
        violation = leverage > self.config.leverage_limit
        
        return {
            "leverage": leverage,
            "limit": self.config.leverage_limit,
            "compliant": not violation,
            "excess": leverage - self.config.leverage_limit if violation else 0
        }
    
    def check_var_limit(self, current_var: float, total_capital: float) -> Dict:
        """
        Check VaR limit compliance
        
        Args:
            current_var: Current VaR
            total_capital: Total capital
            
        Returns:
            Compliance check results
        """
        var_pct = current_var / total_capital if total_capital > 0 else 0
        
        violation = var_pct > self.config.var_limit
        
        return {
            "var": current_var,
            "var_pct": var_pct,
            "limit": self.config.var_limit,
            "compliant": not violation,
            "excess": var_pct - self.config.var_limit if violation else 0
        }
    
    def check_capital_adequacy(self, regulatory_capital: float,
                              risk_weighted_assets: float) -> Dict:
        """
        Check capital adequacy ratio
        
        Args:
            regulatory_capital: Regulatory capital
            risk_weighted_assets: Risk-weighted assets
            
        Returns:
            Capital adequacy results
        """
        capital_ratio = regulatory_capital / risk_weighted_assets if risk_weighted_assets > 0 else 0
        
        minimum_ratio = self.config.min_capital_ratio + self.config.capital_buffer
        
        violation = capital_ratio < minimum_ratio
        
        self.regulatory_capital = regulatory_capital
        self.risk_weighted_assets = risk_weighted_assets
        
        return {
            "capital_ratio": capital_ratio,
            "minimum_ratio": minimum_ratio,
            "compliant": not violation,
            "deficit": minimum_ratio - capital_ratio if violation else 0
        }
    
    def generate_compliance_report(self) -> Dict:
        """
        Generate compliance report
        
        Returns:
            Compliance report
        """
        report = {
            "timestamp": datetime.now(),
            "total_violations": len(self.violations),
            "capital_ratio": self.regulatory_capital / self.risk_weighted_assets 
                           if self.risk_weighted_assets > 0 else 0,
            "compliant": len(self.violations) == 0
        }
        
        self.compliance_records.append(report)
        
        return report
    
    def record_violation(self, violation_type: str, details: Dict) -> None:
        """
        Record a compliance violation
        
        Args:
            violation_type: Type of violation
            details: Violation details
        """
        self.violations.append({
            "timestamp": datetime.now(),
            "type": violation_type,
            "details": details
        })
    
    def run_compliance_check(self, positions: Dict[str, float],
                           market_caps: Dict[str, float],
                           derivative_positions: Dict[str, float],
                           total_capital: float,
                           total_exposure: float,
                           current_var: float,
                           risk_weighted_assets: float) -> Dict:
        """
        Run comprehensive compliance check
        
        Args:
            positions: Current positions
            market_caps: Market capitalizations
            derivative_positions: Derivative positions
            total_capital: Total capital
            total_exposure: Total exposure
            current_var: Current VaR
            risk_weighted_assets: Risk-weighted assets
            
        Returns:
            Compliance check results
        """
        results = {}
        
        # Check position limits
        position_check = self.check_position_limits(positions, market_caps)
        results["position_limits"] = position_check
        
        if not position_check["compliant"]:
            for violation in position_check["violations"]:
                self.record_violation("position_limit", violation)
        
        # Check derivative exposure
        derivative_check = self.check_derivative_exposure(derivative_positions, total_capital)
        results["derivative_exposure"] = derivative_check
        
        if not derivative_check["compliant"]:
            self.record_violation("derivative_exposure", derivative_check)
        
        # Check leverage limit
        leverage_check = self.check_leverage_limit(total_exposure, total_capital)
        results["leverage_limit"] = leverage_check
        
        if not leverage_check["compliant"]:
            self.record_violation("leverage_limit", leverage_check)
        
        # Check VaR limit
        var_check = self.check_var_limit(current_var, total_capital)
        results["var_limit"] = var_check
        
        if not var_check["compliant"]:
            self.record_violation("var_limit", var_check)
        
        # Check capital adequacy
        capital_check = self.check_capital_adequacy(total_capital * 0.15, risk_weighted_assets)
        results["capital_adequacy"] = capital_check
        
        if not capital_check["compliant"]:
            self.record_violation("capital_adequacy", capital_check)
        
        # Generate report
        report = self.generate_compliance_report()
        results["report"] = report
        
        return results
    
    def get_compliance_summary(self) -> Dict:
        """Get compliance summary"""
        return {
            "total_violations": len(self.violations),
            "capital_ratio": self.regulatory_capital / self.risk_weighted_assets 
                           if self.risk_weighted_assets > 0 else 0,
            "compliant": len(self.violations) == 0,
            "violation_types": list(set(v["type"] for v in self.violations))
        }


def simulate_compliance_data() -> Tuple[Dict, Dict, Dict, float, float, float, float]:
    """Simulate compliance data for testing"""
    np.random.seed(42)
    
    # Positions
    positions = {f"STOCK_{i}": np.random.uniform(100000, 500000) for i in range(10)}
    
    # Market caps
    market_caps = {f"STOCK_{i}": np.random.uniform(10000000, 100000000) for i in range(10)}
    
    # Derivative positions
    derivative_positions = {f"DERIVATIVE_{i}": np.random.uniform(50000, 200000) for i in range(5)}
    
    total_capital = 10000000  # $10M
    total_exposure = sum(abs(pos) for pos in positions.values()) + sum(abs(pos) for pos in derivative_positions.values())
    current_var = total_exposure * 0.02  # 2% VaR
    risk_weighted_assets = total_exposure * 0.8  # 80% risk weighting
    
    return positions, market_caps, derivative_positions, total_capital, total_exposure, current_var, risk_weighted_assets


if __name__ == "__main__":
    # Example usage
    config = ComplianceConfig(
        max_position_pct=0.10,
        max_derivative_exposure=0.20,
        min_capital_ratio=0.15,
        var_limit=0.05,
        leverage_limit=3.0
    )
    
    compliance_manager = RegulatoryComplianceManager(config)
    
    # Simulate data
    print("Simulating compliance data...")
    positions, market_caps, derivative_positions, total_capital, total_exposure, current_var, rwa = simulate_compliance_data()
    
    print(f"  Total Capital: ${total_capital:,.0f}")
    print(f"  Total Exposure: ${total_exposure:,.0f}")
    print(f"  Current VaR: ${current_var:,.0f}")
    
    # Run compliance check
    print("\nRunning compliance check...")
    results = compliance_manager.run_compliance_check(
        positions, market_caps, derivative_positions,
        total_capital, total_exposure, current_var, rwa
    )
    
    print(f"\nCompliance Results:")
    print(f"  Position Limits: {'COMPLIANT' if results['position_limits']['compliant'] else 'VIOLATION'}")
    print(f"    Violations: {results['position_limits']['num_violations']}")
    
    print(f"  Derivative Exposure: {'COMPLIANT' if results['derivative_exposure']['compliant'] else 'VIOLATION'}")
    print(f"    Exposure %: {results['derivative_exposure']['derivative_exposure_pct']:.2%}")
    
    print(f"  Leverage Limit: {'COMPLIANT' if results['leverage_limit']['compliant'] else 'VIOLATION'}")
    print(f"    Leverage: {results['leverage_limit']['leverage']:.2f}x")
    
    print(f"  VaR Limit: {'COMPLIANT' if results['var_limit']['compliant'] else 'VIOLATION'}")
    print(f"    VaR %: {results['var_limit']['var_pct']:.2%}")
    
    print(f"  Capital Adequacy: {'COMPLIANT' if results['capital_adequacy']['compliant'] else 'VIOLATION'}")
    print(f"    Capital Ratio: {results['capital_adequacy']['capital_ratio']:.2%}")
    
    print(f"\nOverall Compliance: {'COMPLIANT' if results['report']['compliant'] else 'VIOLATION'}")
    
    # Summary
    print("\nCompliance Summary:")
    summary = compliance_manager.get_compliance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
