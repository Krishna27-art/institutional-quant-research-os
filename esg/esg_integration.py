"""
ESG Integration

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#49)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- ESG scoring and rating
- Sustainability metrics
- ESG risk assessment
- ESG-adjusted portfolio construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ESGConfig:
    """Configuration for ESG Integration"""
    # ESG weights
    e_weight: float = 0.4  # Environmental weight
    s_weight: float = 0.3  # Social weight
    g_weight: float = 0.3  # Governance weight
    
    # ESG thresholds
    min_esg_score: float = 50  # Minimum ESG score
    esg_exclusion_threshold: float = 30  # ESG exclusion threshold
    
    # Portfolio parameters
    esg_adjustment_factor: float = 0.1  # ESG adjustment factor
    
    # Risk parameters
    esg_risk_premium: float = 0.02  # 2% ESG risk premium


class ESGScorer:
    """
    ESG Scorer
    
    Calculates ESG scores for companies based on
    environmental, social, and governance metrics.
    """
    
    def __init__(self, config: ESGConfig):
        self.config = config
    
    def calculate_e_score(self, carbon_emissions: float, energy_efficiency: float,
                        renewable_energy: float, waste_management: float) -> float:
        """
        Calculate Environmental score
        
        Args:
            carbon_emissions: Carbon emissions (lower is better)
            energy_efficiency: Energy efficiency (higher is better)
            renewable_energy: Renewable energy usage (higher is better)
            waste_management: Waste management score (higher is better)
            
        Returns:
            E score (0-100)
        """
        # Normalize metrics to 0-100 scale
        carbon_score = max(0, 100 - carbon_emissions * 10)
        energy_score = min(100, energy_efficiency * 100)
        renewable_score = min(100, renewable_energy * 100)
        waste_score = min(100, waste_management * 100)
        
        # Weighted average
        e_score = (carbon_score * 0.3 + energy_score * 0.25 + 
                  renewable_score * 0.25 + waste_score * 0.2)
        
        return e_score
    
    def calculate_s_score(self, labor_practices: float, diversity: float,
                        community_relations: float, human_rights: float) -> float:
        """
        Calculate Social score
        
        Args:
            labor_practices: Labor practices score
            diversity: Diversity score
            community_relations: Community relations score
            human_rights: Human rights score
            
        Returns:
            S score (0-100)
        """
        s_score = (labor_practices * 0.3 + diversity * 0.3 + 
                  community_relations * 0.2 + human_rights * 0.2)
        
        return s_score
    
    def calculate_g_score(self, board_independence: float, executive_comp: float,
                        transparency: float, ethics: float) -> float:
        """
        Calculate Governance score
        
        Args:
            board_independence: Board independence score
            executive_comp: Executive compensation score
            transparency: Transparency score
            ethics: Ethics score
            
        Returns:
            G score (0-100)
        """
        g_score = (board_independence * 0.3 + executive_comp * 0.25 + 
                  transparency * 0.25 + ethics * 0.2)
        
        return g_score
    
    def calculate_esg_score(self, e_score: float, s_score: float, g_score: float) -> float:
        """
        Calculate overall ESG score
        
        Args:
            e_score: Environmental score
            s_score: Social score
            g_score: Governance score
            
        Returns:
            ESG score (0-100)
        """
        esg_score = (e_score * self.config.e_weight + 
                    s_score * self.config.s_weight + 
                    g_score * self.config.g_weight)
        
        return esg_score


class ESGPortfolioManager:
    """
    ESG Portfolio Manager
    
    Integrates ESG considerations into portfolio construction
    and risk management.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: ESGConfig):
        self.config = config
        
        self.esg_scorer = ESGScorer(config)
        
        # ESG scores
        self.esg_scores: Dict[str, Dict] = {}
    
    def score_company(self, company: str, e_metrics: Dict, s_metrics: Dict, 
                     g_metrics: Dict) -> Dict:
        """
        Score a company on ESG metrics
        
        Args:
            company: Company name
            e_metrics: Environmental metrics
            s_metrics: Social metrics
            g_metrics: Governance metrics
            
        Returns:
            ESG scores
        """
        # Calculate individual scores
        e_score = self.esg_scorer.calculate_e_score(
            e_metrics.get("carbon_emissions", 5.0),
            e_metrics.get("energy_efficiency", 0.5),
            e_metrics.get("renewable_energy", 0.3),
            e_metrics.get("waste_management", 0.6)
        )
        
        s_score = self.esg_scorer.calculate_s_score(
            s_metrics.get("labor_practices", 0.6),
            s_metrics.get("diversity", 0.5),
            s_metrics.get("community_relations", 0.6),
            s_metrics.get("human_rights", 0.7)
        )
        
        g_score = self.esg_scorer.calculate_g_score(
            g_metrics.get("board_independence", 0.6),
            g_metrics.get("executive_comp", 0.5),
            g_metrics.get("transparency", 0.6),
            g_metrics.get("ethics", 0.7)
        )
        
        # Calculate overall ESG score
        esg_score = self.esg_scorer.calculate_esg_score(e_score, s_score, g_score)
        
        scores = {
            "e_score": e_score,
            "s_score": s_score,
            "g_score": g_score,
            "esg_score": esg_score
        }
        
        self.esg_scores[company] = scores
        
        return scores
    
    def filter_by_esg(self, companies: List[str], min_score: float = None) -> List[str]:
        """
        Filter companies by ESG score
        
        Args:
            companies: List of companies
            min_score: Minimum ESG score (uses config if None)
            
        Returns:
            Filtered companies
        """
        if min_score is None:
            min_score = self.config.min_esg_score
        
        filtered = [c for c in companies if c in self.esg_scores 
                   and self.esg_scores[c]["esg_score"] >= min_score]
        
        return filtered
    
    def adjust_weights_by_esg(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust portfolio weights by ESG scores
        
        Args:
            weights: Original weights
            
        Returns:
            ESG-adjusted weights
        """
        adjusted_weights = {}
        
        for company, weight in weights.items():
            if company in self.esg_scores:
                esg_score = self.esg_scores[company]["esg_score"]
                
                # Adjust weight based on ESG score
                esg_adjustment = (esg_score / 100) * self.config.esg_adjustment_factor
                adjusted_weight = weight * (1 + esg_adjustment)
                
                adjusted_weights[company] = adjusted_weight
            else:
                adjusted_weights[company] = weight
        
        # Normalize
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def calculate_esg_risk(self, companies: List[str]) -> Dict:
        """
        Calculate ESG risk for portfolio
        
        Args:
            companies: List of companies
            
        Returns:
            ESG risk metrics
        """
        low_esg_companies = [c for c in companies if c in self.esg_scores 
                            and self.esg_scores[c]["esg_score"] < self.config.esg_exclusion_threshold]
        
        avg_esg_score = np.mean([self.esg_scores[c]["esg_score"] 
                               for c in companies if c in self.esg_scores])
        
        return {
            "low_esg_count": len(low_esg_companies),
            "low_esg_companies": low_esg_companies,
            "avg_esg_score": avg_esg_score,
            "esg_risk_premium": self.config.esg_risk_premium * (1 - avg_esg_score / 100)
        }
    
    def run_esg_analysis(self, companies: List[Dict]) -> Dict:
        """
        Run comprehensive ESG analysis
        
        Args:
            companies: List of company data
            
        Returns:
            ESG analysis results
        """
        results = {}
        
        for company_data in companies:
            name = company_data["name"]
            e_metrics = company_data.get("e_metrics", {})
            s_metrics = company_data.get("s_metrics", {})
            g_metrics = company_data.get("g_metrics", {})
            
            scores = self.score_company(name, e_metrics, s_metrics, g_metrics)
            results[name] = scores
        
        # Calculate portfolio ESG risk
        company_names = [c["name"] for c in companies]
        esg_risk = self.calculate_esg_risk(company_names)
        
        return {
            "company_scores": results,
            "esg_risk": esg_risk
        }
    
    def get_esg_summary(self) -> Dict:
        """Get ESG summary"""
        if not self.esg_scores:
            return {}
        
        avg_e_score = np.mean([s["e_score"] for s in self.esg_scores.values()])
        avg_s_score = np.mean([s["s_score"] for s in self.esg_scores.values()])
        avg_g_score = np.mean([s["g_score"] for s in self.esg_scores.values()])
        avg_esg_score = np.mean([s["esg_score"] for s in self.esg_scores.values()])
        
        return {
            "num_companies": len(self.esg_scores),
            "avg_e_score": avg_e_score,
            "avg_s_score": avg_s_score,
            "avg_g_score": avg_g_score,
            "avg_esg_score": avg_esg_score
        }


def simulate_company_data(n_companies: int = 20) -> List[Dict]:
    """Simulate company ESG data for testing"""
    np.random.seed(42)
    
    companies = []
    
    for i in range(n_companies):
        company = {
            "name": f"COMPANY_{i}",
            "e_metrics": {
                "carbon_emissions": np.random.uniform(1, 10),
                "energy_efficiency": np.random.uniform(0.3, 0.8),
                "renewable_energy": np.random.uniform(0.1, 0.6),
                "waste_management": np.random.uniform(0.4, 0.9)
            },
            "s_metrics": {
                "labor_practices": np.random.uniform(0.4, 0.9),
                "diversity": np.random.uniform(0.3, 0.8),
                "community_relations": np.random.uniform(0.4, 0.9),
                "human_rights": np.random.uniform(0.5, 0.9)
            },
            "g_metrics": {
                "board_independence": np.random.uniform(0.4, 0.9),
                "executive_comp": np.random.uniform(0.3, 0.8),
                "transparency": np.random.uniform(0.4, 0.9),
                "ethics": np.random.uniform(0.5, 0.9)
            }
        }
        
        companies.append(company)
    
    return companies


if __name__ == "__main__":
    # Example usage
    config = ESGConfig(
        e_weight=0.4,
        s_weight=0.3,
        g_weight=0.3,
        min_esg_score=50,
        esg_adjustment_factor=0.1
    )
    
    esg_manager = ESGPortfolioManager(config)
    
    # Simulate company data
    print("Simulating company ESG data...")
    companies = simulate_company_data(20)
    
    # Run ESG analysis
    print("\nRunning ESG analysis...")
    results = esg_manager.run_esg_analysis(companies)
    
    print(f"\nESG Scores (first 5):")
    for name, scores in list(results["company_scores"].items())[:5]:
        print(f"  {name}:")
        print(f"    E Score: {scores['e_score']:.2f}")
        print(f"    S Score: {scores['s_score']:.2f}")
        print(f"    G Score: {scores['g_score']:.2f}")
        print(f"    ESG Score: {scores['esg_score']:.2f}")
    
    # ESG risk
    print(f"\nESG Risk:")
    esg_risk = results["esg_risk"]
    print(f"  Low ESG Count: {esg_risk['low_esg_count']}")
    print(f"  Avg ESG Score: {esg_risk['avg_esg_score']:.2f}")
    print(f"  ESG Risk Premium: {esg_risk['esg_risk_premium']:.2%}")
    
    # Filter by ESG
    print("\nFiltering by ESG score...")
    company_names = [c["name"] for c in companies]
    filtered = esg_manager.filter_by_esg(company_names)
    print(f"  Original: {len(company_names)} companies")
    print(f"  Filtered: {len(filtered)} companies")
    
    # Adjust weights
    print("\nAdjusting weights by ESG...")
    weights = {c: 1.0 / len(company_names) for c in company_names}
    adjusted_weights = esg_manager.adjust_weights_by_esg(weights)
    
    print(f"  Adjusted weights (first 5):")
    for company, weight in list(adjusted_weights.items())[:5]:
        print(f"    {company}: {weight:.4f}")
    
    # Summary
    print("\nESG Summary:")
    summary = esg_manager.get_esg_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
