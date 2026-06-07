"""
Confidence-Based Scaling
Weight strategies by P(alpha works) instead of just volatility.

Critical for institutional capital allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from scipy import stats


class ConfidenceSource(Enum):
    """Sources of confidence estimates"""
    HISTORICAL_WIN_RATE = "historical_win_rate"
    WALK_FORWARD_SHARPE = "walk_forward_sharpe"
    ADVERSARIAL_ROBUSTNESS = "adversarial_robustness"
    ENSEMBLE_AGREEMENT = "ensemble_agreement"
    BAYESIAN_POSTERIOR = "bayesian_posterior"


@dataclass
class AlphaConfidence:
    """Confidence estimate for an alpha"""
    alpha_id: str
    confidence: float  # 0-1, probability that alpha works
    confidence_source: ConfidenceSource
    sample_size: int  # Number of observations
    variance: float  # Variance of confidence estimate
    last_updated: datetime


class ConfidenceEstimator:
    """
    Confidence Estimator
    
    Estimates P(alpha works) using multiple methods:
    1. Historical win rate
    2. Walk-forward Sharpe consistency
    3. Adversarial robustness
    4. Ensemble agreement
    5. Bayesian posterior
    """
    
    def __init__(self):
        self.confidences: Dict[str, AlphaConfidence] = {}
    
    def estimate_from_win_rate(self, alpha_id: str, wins: int, total: int) -> AlphaConfidence:
        """Estimate confidence from historical win rate"""
        if total == 0:
            confidence = 0.5
        else:
            confidence = wins / total
        
        # Beta distribution variance
        variance = (confidence * (1 - confidence)) / (total + 1)
        
        alpha_confidence = AlphaConfidence(
            alpha_id=alpha_id,
            confidence=confidence,
            confidence_source=ConfidenceSource.HISTORICAL_WIN_RATE,
            sample_size=total,
            variance=variance,
            last_updated=datetime.now()
        )
        
        self.confidences[alpha_id] = alpha_confidence
        return alpha_confidence
    
    def estimate_from_walk_forward(self, alpha_id: str, sharpe_values: List[float]) -> AlphaConfidence:
        """Estimate confidence from walk-forward Sharpe consistency"""
        if not sharpe_values:
            confidence = 0.5
            variance = 0.25
        else:
            # Confidence = proportion of folds with Sharpe > 0.5
            positive_sharpe = sum(1 for s in sharpe_values if s > 0.5)
            confidence = positive_sharpe / len(sharpe_values)
            
            # Variance from binomial
            variance = (confidence * (1 - confidence)) / len(sharpe_values)
        
        alpha_confidence = AlphaConfidence(
            alpha_id=alpha_id,
            confidence=confidence,
            confidence_source=ConfidenceSource.WALK_FORWARD_SHARPE,
            sample_size=len(sharpe_values),
            variance=variance,
            last_updated=datetime.now()
        )
        
        self.confidences[alpha_id] = alpha_confidence
        return alpha_confidence
    
    def estimate_from_adversarial(self, alpha_id: str, passed_scenarios: int,
                                  total_scenarios: int) -> AlphaConfidence:
        """Estimate confidence from adversarial robustness"""
        if total_scenarios == 0:
            confidence = 0.5
            variance = 0.25
        else:
            confidence = passed_scenarios / total_scenarios
            variance = (confidence * (1 - confidence)) / total_scenarios
        
        alpha_confidence = AlphaConfidence(
            alpha_id=alpha_id,
            confidence=confidence,
            confidence_source=ConfidenceSource.ADVERSARIAL_ROBUSTNESS,
            sample_size=total_scenarios,
            variance=variance,
            last_updated=datetime.now()
        )
        
        self.confidences[alpha_id] = alpha_confidence
        return alpha_confidence
    
    def estimate_from_ensemble(self, alpha_id: str, model_agreements: List[float]) -> AlphaConfidence:
        """Estimate confidence from ensemble agreement"""
        if not model_agreements:
            confidence = 0.5
            variance = 0.25
        else:
            # Confidence = mean agreement
            confidence = np.mean(model_agreements)
            variance = np.var(model_agreements)
        
        alpha_confidence = AlphaConfidence(
            alpha_id=alpha_id,
            confidence=confidence,
            confidence_source=ConfidenceSource.ENSEMBLE_AGREEMENT,
            sample_size=len(model_agreements),
            variance=variance,
            last_updated=datetime.now()
        )
        
        self.confidences[alpha_id] = alpha_confidence
        return alpha_confidence
    
    def bayesian_update(self, alpha_id: str, prior_confidence: float,
                       prior_variance: float, new_evidence: float,
                       evidence_variance: float) -> AlphaConfidence:
        """Bayesian update of confidence"""
        # Posterior = (Prior * Likelihood) / Evidence
        posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / evidence_variance)
        posterior_mean = posterior_variance * (prior_confidence / prior_variance + new_evidence / evidence_variance)
        
        alpha_confidence = AlphaConfidence(
            alpha_id=alpha_id,
            confidence=posterior_mean,
            confidence_source=ConfidenceSource.BAYESIAN_POSTERIOR,
            sample_size=1,
            variance=posterior_variance,
            last_updated=datetime.now()
        )
        
        self.confidences[alpha_id] = alpha_confidence
        return alpha_confidence
    
    def get_confidence(self, alpha_id: str) -> Optional[float]:
        """Get confidence for alpha"""
        if alpha_id not in self.confidences:
            return None
        return self.confidences[alpha_id].confidence
    
    def get_all_confidences(self) -> Dict[str, float]:
        """Get all confidences"""
        return {alpha_id: conf.confidence for alpha_id, conf in self.confidences.items()}


class ConfidenceScaler:
    """
    Confidence-Based Scaler
    
    Scales strategy weights by confidence instead of just volatility.
    
    Formula:
    weight = confidence * expected_return / volatility
    
    This ensures that:
    - High confidence alphas get more capital
    - Low confidence alphas get less capital
    - Risk is still controlled by volatility
    """
    
    def __init__(self, min_confidence: float = 0.3, max_confidence: float = 0.9):
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.estimator = ConfidenceEstimator()
    
    def calculate_scaled_weight(self, expected_return: float, volatility: float,
                               confidence: float) -> float:
        """
        Calculate confidence-scaled weight.
        
        Args:
            expected_return: Expected return of alpha
            volatility: Volatility of alpha
            confidence: Confidence that alpha works (0-1)
        
        Returns:
            Scaled weight
        """
        if volatility == 0:
            return 0.0
        
        # Clip confidence to bounds
        confidence = np.clip(confidence, self.min_confidence, self.max_confidence)
        
        # Confidence-adjusted Sharpe
        confidence_sharpe = (expected_return / volatility) * confidence
        
        # Scale weight by confidence-adjusted Sharpe
        weight = confidence_sharpe / (1.0 + confidence_sharpe)
        
        return weight
    
    def scale_portfolio_weights(self, alpha_metrics: Dict[str, Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Scale portfolio weights by confidence.
        
        Args:
            alpha_metrics: Dictionary of alpha_id -> (expected_return, volatility, confidence)
        
        Returns:
            Dictionary of alpha_id -> scaled_weight
        """
        scaled_weights = {}
        
        for alpha_id, (exp_ret, vol, conf) in alpha_metrics.items():
            weight = self.calculate_scaled_weight(exp_ret, vol, conf)
            scaled_weights[alpha_id] = weight
        
        # Normalize to sum to 1
        total_weight = sum(scaled_weights.values())
        if total_weight > 0:
            for alpha_id in scaled_weights:
                scaled_weights[alpha_id] /= total_weight
        
        return scaled_weights
    
    def compare_with_risk_parity(self, alpha_metrics: Dict[str, Tuple[float, float, float]]) -> Dict:
        """
        Compare confidence scaling with risk parity.
        
        Args:
            alpha_metrics: Dictionary of alpha_id -> (expected_return, volatility, confidence)
        
        Returns:
            Comparison dictionary
        """
        # Confidence scaling
        confidence_weights = self.scale_portfolio_weights(alpha_metrics)
        
        # Risk parity (inverse volatility)
        inv_vol_weights = {}
        total_inv_vol = 0.0
        for alpha_id, (exp_ret, vol, conf) in alpha_metrics.items():
            inv_vol = 1.0 / (vol + 1e-6)
            inv_vol_weights[alpha_id] = inv_vol
            total_inv_vol += inv_vol
        
        for alpha_id in inv_vol_weights:
            inv_vol_weights[alpha_id] /= total_inv_vol
        
        # Compare
        comparison = {}
        for alpha_id in alpha_metrics:
            comparison[alpha_id] = {
                "confidence_weight": confidence_weights.get(alpha_id, 0),
                "risk_parity_weight": inv_vol_weights.get(alpha_id, 0),
                "difference": confidence_weights.get(alpha_id, 0) - inv_vol_weights.get(alpha_id, 0)
            }
        
        return comparison
    
    def generate_report(self, alpha_metrics: Dict[str, Tuple[float, float, float]]) -> str:
        """Generate confidence scaling report"""
        comparison = self.compare_with_risk_parity(alpha_metrics)
        
        report = f"""
Confidence-Based Scaling Report
{'=' * 50}
Min Confidence: {self.min_confidence:.2f}
Max Confidence: {self.max_confidence:.2f}

Weight Comparison:
{'-' * 50}
"""
        
        for alpha_id, comp in comparison.items():
            report += f"{alpha_id}:\n"
            report += f"  Confidence Weight: {comp['confidence_weight']:.2%}\n"
            report += f"  Risk Parity Weight: {comp['risk_parity_weight']:.2%}\n"
            report += f"  Difference: {comp['difference']:+.2%}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    scaler = ConfidenceScaler(min_confidence=0.3, max_confidence=0.9)
    
    # Define alpha metrics
    alpha_metrics = {
        "momentum": (0.15, 0.20, 0.8),
        "mean_reversion": (0.12, 0.15, 0.7),
        "stat_arb": (0.10, 0.10, 0.9),
        "pairs_trading": (0.08, 0.08, 0.6)
    }
    
    # Scale weights
    scaled_weights = scaler.scale_portfolio_weights(alpha_metrics)
    
    print("Scaled Weights:")
    for alpha_id, weight in scaled_weights.items():
        print(f"  {alpha_id}: {weight:.2%}")
    
    print(scaler.generate_report(alpha_metrics))
