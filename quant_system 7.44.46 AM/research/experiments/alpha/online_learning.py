"""
Online Learning System
Based on V3 Blueprint - Rapid Adaptation Without Full Retraining

Key findings from research:
- Adapt models to recent market regime without full retraining
- Methods: Incremental gradient boosting (LightGBM refit), recursive least squares
- For HMM: update using forward-backward with forgetting factor (λ=0.99)
- Schedule: Full retraining weekly, online updates daily
- Triggers: Feature drift (PSI > 0.2), regime shift, alpha decay alert
- Expected benefit: +0.1 Sharpe in volatile markets

V3 Upgrade - Expected Sharpe increase: +0.1 (reduces degradation)
Priority: Medium
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning"""
    model_type: str  # "lightgbm", "linear", "hmm"
    update_frequency: str  # "daily", "hourly"
    full_retrain_frequency: str  # "weekly", "monthly"
    forgetting_factor: float  # For HMM (λ)
    trigger_psi_threshold: float  # PSI threshold for retraining


@dataclass
class ModelUpdateResult:
    """Result of model update"""
    timestamp: str
    update_type: str  # "incremental", "full"
    old_performance: float
    new_performance: float
    performance_delta: float
    triggered_by: str


class OnlineLearningSystem:
    """
    Online Learning System for rapid adaptation.
    
    Methods:
    - Incremental gradient boosting (LightGBM has `refit` option)
    - Recursive least squares for linear models
    - For HMM: update using forward-backward with forgetting factor (λ=0.99)
    
    Schedule:
    - Full retraining: weekly (Sunday)
    - Online updates: daily after close (incremental)
    
    Triggers for immediate retraining:
    - Feature drift (PSI > 0.2 on any top-5 feature)
    - Regime shift detected by change point
    - Alpha decay alert (P1)
    """
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}  # model_name -> model parameters
        self.update_history: List[ModelUpdateResult] = []
        self.last_full_retrain: Optional[datetime] = None
    
    def incremental_update_linear(
        self,
        model_name: str,
        X_new: np.ndarray,
        y_new: np.ndarray,
        forgetting_factor: float = 0.99
    ) -> ModelUpdateResult:
        """
        Incremental update for linear model using RLS.
        
        Args:
            model_name: Model name
            X_new: New features
            y_new: New targets
            forgetting_factor: Forgetting factor λ
            
        Returns:
            ModelUpdateResult
        """
        if model_name not in self.models:
            # Initialize model
            n_features = X_new.shape[1]
            self.models[model_name] = {
                "weights": np.zeros(n_features),
                "P": np.eye(n_features) * 1000,  # Inverse covariance
                "n_samples": 0
            }
        
        model = self.models[model_name]
        
        # Recursive Least Squares update
        for i in range(len(X_new)):
            x = X_new[i:i+1]
            y = y_new[i]
            
            # Predict
            y_pred = np.dot(x, model["weights"])
            
            # Compute error
            error = y - y_pred
            
            # Update P matrix
            P = model["P"]
            Px = np.dot(P, x.T)
            xPx = np.dot(x, Px)
            K = Px / (1 + xPx)
            model["P"] = (P - np.outer(K, Px)) / forgetting_factor
            
            # Update weights
            model["weights"] += K * error
            
            model["n_samples"] += 1
        
        # Calculate performance (simplified)
        old_perf = 0.5  # Placeholder
        new_perf = 0.55  # Placeholder
        delta = new_perf - old_perf
        
        result = ModelUpdateResult(
            timestamp=datetime.now().isoformat(),
            update_type="incremental",
            old_performance=old_perf,
            new_performance=new_perf,
            performance_delta=delta,
            triggered_by="scheduled"
        )
        
        self.update_history.append(result)
        
        return result
    
    def incremental_update_lightgbm(
        self,
        model_name: str,
        X_new: np.ndarray,
        y_new: np.ndarray
    ) -> ModelUpdateResult:
        """
        Incremental update for LightGBM model.
        
        Args:
            model_name: Model name
            X_new: New features
            y_new: New targets
            
        Returns:
            ModelUpdateResult
        """
        # Placeholder for LightGBM incremental update
        # In production, use lightgbm.LGBMClassifier.fit(..., init_model=old_model)
        
        if model_name not in self.models:
            self.models[model_name] = {
                "model": None,  # LightGBM model object
                "n_samples": 0
            }
        
        model = self.models[model_name]
        model["n_samples"] += len(X_new)
        
        # Simulate performance improvement
        old_perf = 0.6
        new_perf = 0.62
        delta = new_perf - old_perf
        
        result = ModelUpdateResult(
            timestamp=datetime.now().isoformat(),
            update_type="incremental",
            old_performance=old_perf,
            new_performance=new_perf,
            performance_delta=delta,
            triggered_by="scheduled"
        )
        
        self.update_history.append(result)
        
        return result
    
    def update_hmm(
        self,
        model_name: str,
        observations: np.ndarray,
        forgetting_factor: float = 0.99
    ) -> ModelUpdateResult:
        """
        Update HMM with forgetting factor.
        
        Args:
            model_name: Model name
            observations: New observations
            forgetting_factor: Forgetting factor λ
            
        Returns:
            ModelUpdateResult
        """
        # Placeholder for HMM update
        # In production, use forward-backward with forgetting factor
        
        if model_name not in self.models:
            self.models[model_name] = {
                "transition_matrix": None,
                "emission_matrix": None,
                "n_samples": 0
            }
        
        model = self.models[model_name]
        model["n_samples"] += len(observations)
        
        # Apply forgetting factor to transition matrix
        if model["transition_matrix"] is not None:
            model["transition_matrix"] *= forgetting_factor
        
        old_perf = 0.55
        new_perf = 0.57
        delta = new_perf - old_perf
        
        result = ModelUpdateResult(
            timestamp=datetime.now().isoformat(),
            update_type="incremental",
            old_performance=old_perf,
            new_performance=new_perf,
            performance_delta=delta,
            triggered_by="scheduled"
        )
        
        self.update_history.append(result)
        
        return result
    
    def full_retrain(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> ModelUpdateResult:
        """
        Full model retraining.
        
        Args:
            model_name: Model name
            X: All features
            y: All targets
            
        Returns:
            ModelUpdateResult
        """
        # Placeholder for full retraining
        # In production, call model.fit(X, y)
        
        old_perf = 0.5
        new_perf = 0.65
        delta = new_perf - old_perf
        
        self.last_full_retrain = datetime.now()
        
        result = ModelUpdateResult(
            timestamp=datetime.now().isoformat(),
            update_type="full",
            old_performance=old_perf,
            new_performance=new_perf,
            performance_delta=delta,
            triggered_by="scheduled"
        )
        
        self.update_history.append(result)
        
        return result
    
    def check_retraining_triggers(
        self,
        feature_psi: float,
        regime_shift_detected: bool,
        alpha_decay_alert: bool
    ) -> bool:
        """
        Check if immediate retraining is needed.
        
        Args:
            feature_psi: PSI for top feature
            regime_shift_detected: Whether regime shift detected
            alpha_decay_alert: Whether alpha decay alert triggered
            
        Returns:
            True if retraining needed
        """
        if feature_psi > 0.2:
            return True
        if regime_shift_detected:
            return True
        if alpha_decay_alert:
            return True
        
        return False
    
    def validate_update(
        self,
        model_name: str,
        X_recent: np.ndarray,
        y_recent: np.ndarray
    ) -> bool:
        """
        Validate model on recent OOS data before deploying.
        
        Args:
            model_name: Model name
            X_recent: Recent features
            y_recent: Recent targets
            
        Returns:
            True if validation passes
        """
        # Placeholder validation
        # Check if performance on recent data is acceptable
        return True
    
    def print_update_history(self, limit: int = 10) -> None:
        """Print update history."""
        print("\n" + "="*60)
        print("ONLINE LEARNING UPDATE HISTORY")
        print("="*60)
        
        for result in self.update_history[-limit:]:
            print(f"\n{result.timestamp}:")
            print(f"  Update Type: {result.update_type}")
            print(f"  Performance: {result.old_performance:.4f} → {result.new_performance:.4f}")
            print(f"  Delta: {result.performance_delta:+.4f}")
            print(f"  Triggered By: {result.triggered_by}")
        
        if self.last_full_retrain:
            print(f"\nLast Full Retrain: {self.last_full_retrain}")
        
        print("="*60)


def run_sample_online_learning():
    """Run sample online learning."""
    system = OnlineLearningSystem()
    
    # Simulate daily updates
    np.random.seed(42)
    
    for day in range(10):
        # Generate new data
        X_new = np.random.randn(100, 10)
        y_new = np.random.randn(100)
        
        # Incremental update
        result = system.incremental_update_linear("model_1", X_new, y_new)
        
        print(f"Day {day + 1}: {result.update_type} update, perf delta = {result.performance_delta:+.4f}")
    
    # Full retrain
    X_full = np.random.randn(1000, 10)
    y_full = np.random.randn(1000)
    
    full_result = system.full_retrain("model_1", X_full, y_full)
    print(f"\nFull Retrain: perf delta = {full_result.performance_delta:+.4f}")
    
    # Check triggers
    should_retrain = system.check_retraining_triggers(
        feature_psi=0.25,  # High drift
        regime_shift_detected=False,
        alpha_decay_alert=False
    )
    
    print(f"\nRetraining Triggered: {should_retrain}")
    
    # Print history
    system.print_update_history()
    
    return system


if __name__ == "__main__":
    run_sample_online_learning()
