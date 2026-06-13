"""
Online Learning - FTRL for Alpha Weights

Based on Profit-Centric Audit - High ROI Addition (#3)
Expected ΔSharpe: +0.20
Capacity: 1x
Difficulty: High

Methodology:
- Use FTRL (Follow-the-Regularized-Leader) algorithm
- Adapt alpha weights every 5 minutes based on recent performance
- Replace static regime-based weights with dynamic learning
- Enables rapid adaptation to regime changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class FTRLConfig:
    """Configuration for FTRL Online Learning"""
    alpha: float = 0.1  # Learning rate
    beta: float = 1.0  # L2 regularization parameter
    lambda1: float = 0.01  # L1 regularization parameter
    lambda2: float = 0.01  # L2 regularization parameter
    update_frequency_minutes: int = 5  # Update every 5 minutes
    lookback_minutes: int = 60  # Use last 60 minutes for performance estimation
    min_weight: float = 0.05  # Minimum weight per alpha (5%)
    max_weight: float = 0.5  # Maximum weight per alpha (50%)
    initial_weights: Optional[Dict[str, float]] = None  # Initial weights


class FTRLAlphaWeights:
    """
    FTRL (Follow-the-Regularized-Leader) for Online Alpha Weight Learning
    
    Methodology:
    1. Maintain weights for each alpha strategy
    2. Update weights based on recent performance (last 60 minutes)
    3. Use FTRL algorithm with L1/L2 regularization
    4. Adapt weights every 5 minutes
    5. Enforce weight constraints (min 5%, max 50%)
    """
    
    def __init__(self, config: FTRLConfig, alpha_names: List[str]):
        self.config = config
        self.alpha_names = alpha_names
        self.n_alphas = len(alpha_names)
        
        # Initialize weights
        if config.initial_weights:
            self.weights = np.array([config.initial_weights.get(name, 1.0/self.n_alphas) 
                                      for name in alpha_names])
        else:
            self.weights = np.ones(self.n_alphas) / self.n_alphas
        
        # FTRL state
        self.z = np.zeros(self.n_alphas)  # Accumulated gradients
        self.n = np.zeros(self.n_alphas)  # Accumulated squared gradients
        
        # Performance tracking
        self.alpha_returns: Dict[str, List[float]] = defaultdict(list)
        self.last_update_time: Optional[datetime] = None
        
        # Performance metrics
        self.sharpe_history: List[float] = []
    
    def update_weights(self, alpha_returns: Dict[str, float]) -> None:
        """
        Update weights using FTRL algorithm based on recent alpha returns
        
        Args:
            alpha_returns: Dictionary of alpha_name -> return for last period
        """
        # Store returns
        for alpha_name, ret in alpha_returns.items():
            if alpha_name in self.alpha_names:
                self.alpha_returns[alpha_name].append(ret)
        
        # Keep only recent returns
        for alpha_name in self.alpha_names:
            if len(self.alpha_returns[alpha_name]) > 100:  # Keep last 100 periods
                self.alpha_returns[alpha_name] = self.alpha_returns[alpha_name][-100:]
        
        # Compute recent performance (Sharpe-like metric)
        performance = np.zeros(self.n_alphas)
        for i, alpha_name in enumerate(self.alpha_names):
            returns = self.alpha_returns[alpha_name]
            if len(returns) > 10:
                performance[i] = np.mean(returns) / (np.std(returns) + 1e-8)
            else:
                performance[i] = 0.0
        
        # Normalize performance to [0, 1]
        if np.std(performance) > 0:
            performance = (performance - performance.min()) / (performance.max() - performance.min())
        
        # Compute gradient (negative because we want to maximize)
        gradient = -performance
        
        # Update FTRL state
        sigma = (np.sqrt(self.n + gradient**2) - np.sqrt(self.n)) / self.config.alpha
        self.z += gradient - sigma * self.weights
        self.n += gradient**2
        
        # Compute new weights
        new_weights = np.zeros(self.n_alphas)
        for i in range(self.n_alphas):
            if abs(self.z[i]) <= self.config.lambda1:
                new_weights[i] = 0.0
            else:
                new_weights[i] = -(self.z[i] - np.sign(self.z[i]) * self.config.lambda1) / \
                               ((self.config.beta + np.sqrt(self.n[i])) / self.config.alpha + self.config.lambda2)
        
        # Ensure non-negative
        new_weights = np.maximum(new_weights, 0.0)
        
        # Normalize to sum to 1
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
        else:
            new_weights = np.ones(self.n_alphas) / self.n_alphas
        
        # Enforce weight constraints
        new_weights = np.clip(new_weights, self.config.min_weight, self.config.max_weight)
        
        # Renormalize after clipping
        new_weights = new_weights / new_weights.sum()
        
        self.weights = new_weights
        self.last_update_time = datetime.now()
        
        # Track portfolio Sharpe
        portfolio_return = np.sum(self.weights * performance)
        self.sharpe_history.append(portfolio_return)
        
        if len(self.sharpe_history) > 100:
            self.sharpe_history = self.sharpe_history[-100:]
    
    def get_weights(self) -> Dict[str, float]:
        """Get current alpha weights"""
        return {name: weight for name, weight in zip(self.alpha_names, self.weights)}
    
    def should_update(self, current_time: datetime) -> bool:
        """Check if weights should be updated"""
        if self.last_update_time is None:
            return True
        
        minutes_since_update = (current_time - self.last_update_time).total_seconds() / 60
        return minutes_since_update >= self.config.update_frequency_minutes
    
    def get_portfolio_return(self, alpha_returns: Dict[str, float]) -> float:
        """
        Compute portfolio return given individual alpha returns
        
        Args:
            alpha_returns: Dictionary of alpha_name -> return
            
        Returns:
            Portfolio return
        """
        total_return = 0.0
        for i, alpha_name in enumerate(self.alpha_names):
            if alpha_name in alpha_returns:
                total_return += self.weights[i] * alpha_returns[alpha_name]
        
        return total_return
    
    def get_portfolio_sharpe(self) -> float:
        """Get estimated portfolio Sharpe from recent history"""
        if len(self.sharpe_history) < 10:
            return 0.0
        
        returns = np.array(self.sharpe_history)
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)


class AdaptiveAlphaCombination:
    """
    Adaptive Alpha Combination using Online Learning
    
    Combines multiple alpha strategies with dynamically learned weights.
    Replaces static regime-based weights with FTRL online learning.
    """
    
    def __init__(self, config: FTRLConfig, alpha_names: List[str]):
        self.config = config
        self.ftrl = FTRLAlphaWeights(config, alpha_names)
        
        # Performance tracking
        self.portfolio_returns: List[float] = []
        self.alpha_returns_history: Dict[str, List[float]] = defaultdict(list)
    
    def update(self, alpha_returns: Dict[str, float]) -> None:
        """
        Update alpha weights based on recent performance
        
        Args:
            alpha_returns: Dictionary of alpha_name -> return for last period
        """
        # Store returns
        for alpha_name, ret in alpha_returns.items():
            self.alpha_returns_history[alpha_name].append(ret)
        
        # Update FTRL weights
        self.ftrl.update_weights(alpha_returns)
        
        # Compute portfolio return
        portfolio_return = self.ftrl.get_portfolio_return(alpha_returns)
        self.portfolio_returns.append(portfolio_return)
        
        # Keep only recent history
        if len(self.portfolio_returns) > 1000:
            self.portfolio_returns = self.portfolio_returns[-1000:]
        
        for alpha_name in self.alpha_returns_history:
            if len(self.alpha_returns_history[alpha_name]) > 1000:
                self.alpha_returns_history[alpha_name] = self.alpha_returns_history[alpha_name][-1000:]
    
    def get_weights(self) -> Dict[str, float]:
        """Get current alpha weights"""
        return self.ftrl.get_weights()
    
    def get_portfolio_return(self, alpha_returns: Dict[str, float]) -> float:
        """Compute portfolio return given individual alpha returns"""
        return self.ftrl.get_portfolio_return(alpha_returns)
    
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio performance metrics"""
        if len(self.portfolio_returns) < 10:
            return {
                "sharpe": 0.0,
                "total_return": 0.0,
                "num_periods": len(self.portfolio_returns)
            }
        
        returns = np.array(self.portfolio_returns)
        
        return {
            "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            "total_return": np.sum(returns),
            "num_periods": len(self.portfolio_returns),
            "current_weights": self.get_weights()
        }
    
    def get_alpha_metrics(self) -> Dict[str, Dict]:
        """Get individual alpha performance metrics"""
        metrics = {}
        
        for alpha_name in self.ftrl.alpha_names:
            returns = self.alpha_returns_history[alpha_name]
            if len(returns) > 10:
                metrics[alpha_name] = {
                    "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
                    "total_return": np.sum(returns),
                    "num_periods": len(returns),
                    "current_weight": self.ftrl.weights[self.ftrl.alpha_names.index(alpha_name)]
                }
            else:
                metrics[alpha_name] = {
                    "sharpe": 0.0,
                    "total_return": 0.0,
                    "num_periods": len(returns),
                    "current_weight": self.ftrl.weights[self.ftrl.alpha_names.index(alpha_name)]
                }
        
        return metrics


def simulate_online_learning(
    alpha_returns_data: pd.DataFrame,
    config: FTRLConfig
) -> Dict:
    """
    Simulate online learning on historical alpha returns
    
    Args:
        alpha_returns_data: DataFrame with alpha names as columns, dates as index
        config: Configuration for FTRL
        
    Returns:
        Dictionary with simulation results
    """
    alpha_names = alpha_returns_data.columns.tolist()
    adaptive_comb = AdaptiveAlphaCombination(config, alpha_names)
    
    # Simulate online updates
    for date, row in alpha_returns_data.iterrows():
        alpha_returns = row.to_dict()
        adaptive_comb.update(alpha_returns)
    
    # Get final metrics
    portfolio_metrics = adaptive_comb.get_portfolio_metrics()
    alpha_metrics = adaptive_comb.get_alpha_metrics()
    
    return {
        "portfolio_sharpe": portfolio_metrics["sharpe"],
        "portfolio_return": portfolio_metrics["total_return"],
        "final_weights": portfolio_metrics["current_weights"],
        "alpha_metrics": alpha_metrics
    }


if __name__ == "__main__":
    # Example usage
    config = FTRLConfig()
    
    # Generate synthetic alpha returns data for testing
    np.random.seed(42)
    n_alphas = 5
    n_days = 100
    
    alpha_returns_data = pd.DataFrame(
        np.random.randn(n_days, n_alphas) * 0.001,
        index=pd.date_range(start="2024-01-01", periods=n_days),
        columns=["ORB", "VWAP", "PCP", "VolCarry", "GapFade"]
    )
    
    # Add some structure: ORB and VWAP perform better
    alpha_returns_data["ORB"] += 0.0005
    alpha_returns_data["VWAP"] += 0.0003
    
    print("Simulating Online Learning (FTRL)...")
    results = simulate_online_learning(alpha_returns_data, config)
    print(f"Portfolio Sharpe: {results['portfolio_sharpe']:.2f}")
    print(f"Portfolio Return: {results['portfolio_return']:.4f}")
    print(f"Final Weights: {results['final_weights']}")
    print(f"\nAlpha Metrics:")
    for alpha, metrics in results['alpha_metrics'].items():
        print(f"  {alpha}: Sharpe={metrics['sharpe']:.2f}, Weight={metrics['current_weight']:.2f}")
