"""
Operational Risk

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#47)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Loss event data analysis
- Operational VaR (OpVaR)
- Key risk indicators (KRIs)
- Business impact analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OperationalRiskConfig:
    """Configuration for Operational Risk"""
    # Loss data parameters
    loss_window: int = 252  # 1 year lookback
    loss_threshold: float = 10000  # $10K loss threshold
    
    # OpVaR parameters
    opvar_confidence: float = 0.99  # 99% confidence for OpVaR
    
    # KRI parameters
    kri_threshold_high: float = 0.8  # High risk threshold
    kri_threshold_medium: float = 0.5  # Medium risk threshold
    
    # Business impact parameters
    revenue_impact_threshold: float = 0.05  # 5% revenue impact threshold
    
    # Regulatory parameters
    basel_ii_alpha: float = 0.12  # 12% alpha for operational risk


class OperationalRiskManager:
    """
    Operational Risk Manager
    
    Measures and manages operational risk from internal
    processes, people, and systems.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: OperationalRiskConfig):
        self.config = config
        
        # Loss events
        self.loss_events: List[Dict] = []
        
        # KRIs
        self.kris: Dict[str, float] = {}
        
        # Risk categories
        self.risk_categories = [
            "Internal Fraud",
            "External Fraud",
            "Employment Practices",
            "Clients & Business Practices",
            "Damage to Physical Assets",
            "Business Disruption",
            "Execution & Process Management"
        ]
    
    def add_loss_event(self, date: datetime, category: str, amount: float,
                      description: str = "") -> None:
        """
        Add a loss event
        
        Args:
            date: Event date
            category: Risk category
            amount: Loss amount
            description: Event description
        """
        self.loss_events.append({
            "date": date,
            "category": category,
            "amount": amount,
            "description": description
        })
    
    def calculate_opvar(self, confidence: float = None) -> float:
        """
        Calculate Operational VaR (OpVaR)
        
        Args:
            confidence: Confidence level (uses config if None)
            
        Returns:
            OpVaR
        """
        if not self.loss_events:
            return 0.0
        
        if confidence is None:
            confidence = self.config.opvar_confidence
        
        # Extract loss amounts
        losses = [event["amount"] for event in self.loss_events]
        
        # Calculate OpVaR
        opvar = np.percentile(losses, (1 - confidence) * 100)
        
        return opvar
    
    def calculate_expected_loss(self) -> float:
        """
        Calculate expected operational loss
        
        Returns:
            Expected loss
        """
        if not self.loss_events:
            return 0.0
        
        losses = [event["amount"] for event in self.loss_events]
        expected_loss = np.mean(losses)
        
        return expected_loss
    
    def calculate_loss_by_category(self) -> Dict[str, Dict]:
        """
        Calculate loss statistics by category
        
        Returns:
            Dictionary of category -> statistics
        """
        category_stats = {}
        
        for category in self.risk_categories:
            category_losses = [event["amount"] for event in self.loss_events 
                              if event["category"] == category]
            
            if category_losses:
                category_stats[category] = {
                    "count": len(category_losses),
                    "total_loss": sum(category_losses),
                    "avg_loss": np.mean(category_losses),
                    "max_loss": max(category_losses)
                }
            else:
                category_stats[category] = {
                    "count": 0,
                    "total_loss": 0,
                    "avg_loss": 0,
                    "max_loss": 0
                }
        
        return category_stats
    
    def calculate_kri(self, metric_name: str, current_value: float,
                     target_value: float, weight: float = 1.0) -> float:
        """
        Calculate Key Risk Indicator (KRI)
        
        Args:
            metric_name: KRI name
            current_value: Current metric value
            target_value: Target metric value
            weight: Weight for KRI
            
        Returns:
            KRI score (0-1)
        """
        # Calculate deviation from target
        deviation = abs(current_value - target_value) / (target_value + 1e-8)
        
        # Normalize to 0-1
        kri = min(deviation * weight, 1.0)
        
        self.kris[metric_name] = kri
        return kri
    
    def assess_kri_risk(self, kri: float) -> str:
        """
        Assess risk level from KRI
        
        Args:
            kri: KRI score
            
        Returns:
            Risk level: "high", "medium", or "low"
        """
        if kri > self.config.kri_threshold_high:
            return "high"
        elif kri > self.config.kri_threshold_medium:
            return "medium"
        else:
            return "low"
    
    def calculate_business_impact(self, loss_amount: float, revenue: float) -> Dict:
        """
        Calculate business impact of operational loss
        
        Args:
            loss_amount: Loss amount
            revenue: Annual revenue
            
        Returns:
            Business impact metrics
        """
        revenue_impact = loss_amount / revenue if revenue > 0 else 0
        
        return {
            "loss_amount": loss_amount,
            "revenue_impact": revenue_impact,
            "significant": revenue_impact > self.config.revenue_impact_threshold
        }
    
    def calculate_regulatory_capital(self, revenue: float) -> float:
        """
        Calculate regulatory capital for operational risk (Basel II)
        
        Args:
            revenue: Annual revenue
            
        Returns:
            Regulatory capital requirement
        """
        # Basic Indicator Approach (BIA)
        capital = revenue * self.config.basel_ii_alpha
        
        return capital
    
    def run_operational_risk_analysis(self, revenue: float) -> Dict:
        """
        Run comprehensive operational risk analysis
        
        Args:
            revenue: Annual revenue
            
        Returns:
            Operational risk analysis results
        """
        # Calculate OpVaR
        opvar = self.calculate_opvar()
        
        # Calculate expected loss
        expected_loss = self.calculate_expected_loss()
        
        # Calculate loss by category
        category_stats = self.calculate_loss_by_category()
        
        # Calculate regulatory capital
        reg_capital = self.calculate_regulatory_capital(revenue)
        
        # KRI risk assessment
        kri_risks = {}
        for kri_name, kri_value in self.kris.items():
            kri_risks[kri_name] = self.assess_kri_risk(kri_value)
        
        return {
            "opvar": opvar,
            "expected_loss": expected_loss,
            "category_stats": category_stats,
            "regulatory_capital": reg_capital,
            "kri_risks": kri_risks,
            "num_loss_events": len(self.loss_events)
        }
    
    def get_operational_risk_summary(self) -> Dict:
        """Get operational risk summary"""
        if not self.loss_events:
            return {}
        
        total_loss = sum(event["amount"] for event in self.loss_events)
        
        return {
            "num_loss_events": len(self.loss_events),
            "total_loss": total_loss,
            "avg_loss": total_loss / len(self.loss_events),
            "max_loss": max(event["amount"] for event in self.loss_events),
            "num_kris": len(self.kris),
            "high_risk_kris": sum(1 for kri in self.kris.values() 
                                 if self.assess_kri_risk(kri) == "high")
        }


def simulate_loss_events(n_events: int = 100) -> List[Dict]:
    """Simulate operational loss events for testing"""
    np.random.seed(42)
    
    categories = [
        "Internal Fraud",
        "External Fraud",
        "Employment Practices",
        "Clients & Business Practices",
        "Damage to Physical Assets",
        "Business Disruption",
        "Execution & Process Management"
    ]
    
    events = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_events):
        date = start_date + timedelta(days=np.random.randint(0, 365))
        category = np.random.choice(categories)
        
        # Simulate loss amount (log-normal distribution)
        amount = np.random.lognormal(10, 1)
        
        events.append({
            "date": date,
            "category": category,
            "amount": amount,
            "description": f"Simulated loss event {i}"
        })
    
    return events


if __name__ == "__main__":
    from datetime import timedelta
    
    # Example usage
    config = OperationalRiskConfig(
        loss_window=252,
        opvar_confidence=0.99
    )
    
    op_risk_manager = OperationalRiskManager(config)
    
    # Simulate loss events
    print("Simulating operational loss events...")
    loss_events = simulate_loss_events(100)
    
    for event in loss_events:
        op_risk_manager.add_loss_event(
            event["date"],
            event["category"],
            event["amount"],
            event["description"]
        )
    
    # Calculate KRIs
    print("\nCalculating KRIs...")
    op_risk_manager.calculate_kri("System Availability", 0.95, 0.99, weight=2.0)
    op_risk_manager.calculate_kri("Error Rate", 0.05, 0.01, weight=1.5)
    op_risk_manager.calculate_kri("Staff Turnover", 0.15, 0.10, weight=1.0)
    
    # Run analysis
    print("\nRunning operational risk analysis...")
    revenue = 100000000  # $100M revenue
    results = op_risk_manager.run_operational_risk_analysis(revenue)
    
    print(f"\nOperational Risk Results:")
    print(f"  OpVaR (99%): ${results['opvar']:,.0f}")
    print(f"  Expected Loss: ${results['expected_loss']:,.0f}")
    print(f"  Regulatory Capital: ${results['regulatory_capital']:,.0f}")
    print(f"  Number of Loss Events: {results['num_loss_events']}")
    
    print(f"\nLoss by Category:")
    for category, stats in results["category_stats"].items():
        if stats["count"] > 0:
            print(f"  {category}:")
            print(f"    Count: {stats['count']}")
            print(f"    Total Loss: ${stats['total_loss']:,.0f}")
            print(f"    Avg Loss: ${stats['avg_loss']:,.0f}")
    
    print(f"\nKRI Risk Assessment:")
    for kri_name, risk_level in results["kri_risks"].items():
        print(f"  {kri_name}: {risk_level}")
    
    # Summary
    print("\nOperational Risk Summary:")
    summary = op_risk_manager.get_operational_risk_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
