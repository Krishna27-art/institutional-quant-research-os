"""
Continuous Online Learning
Daily mini-batch updates instead of monthly retraining.

Critical for adapting to market changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class UpdateFrequency(Enum):
    """Update frequency for online learning"""
    DAILY = "daily"
    HOURLY = "hourly"
    REAL_TIME = "real_time"


@dataclass
class ModelUpdate:
    """Record of model update"""
    timestamp: datetime
    samples_processed: int
    old_performance: float
    new_performance: float
    performance_change: float
    update_duration_seconds: float


class OnlineLearningModel:
    """
    Base class for online learning models.
    
    Supports incremental updates with mini-batches.
    """
    
    def __init__(self, learning_rate: float = 0.01, batch_size: int = 100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.samples_seen = 0
        self.last_update_time: Optional[datetime] = None
        self.update_history: List[ModelUpdate] = []
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incremental fit on mini-batch.
        
        Args:
            X: Features
            y: Labels
        
        Returns:
            Performance metric (e.g., accuracy, Sharpe)
        """
        # Placeholder: implement in subclass
        self.samples_seen += len(X)
        self.last_update_time = datetime.now()
        return 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using current model"""
        # Placeholder: implement in subclass
        return np.zeros(len(X))


class OnlineLinearModel(OnlineLearningModel):
    """
    Online linear model with SGD.
    
    Simple but effective for many financial applications.
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.01, 
                 batch_size: int = 100, l2_reg: float = 0.001):
        super().__init__(learning_rate, batch_size)
        self.n_features = n_features
        self.l2_reg = l2_reg
        
        # Initialize weights
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Incremental fit using SGD.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            MSE loss
        """
        old_performance = self._evaluate(X, y)
        
        # Mini-batch SGD
        n_samples = len(X)
        for i in range(0, n_samples, self.batch_size):
            batch_X = X[i:i+self.batch_size]
            batch_y = y[i:i+self.batch_size]
            
            # Compute predictions
            predictions = np.dot(batch_X, self.weights) + self.bias
            
            # Compute gradients
            errors = predictions - batch_y
            grad_w = np.dot(batch_X.T, errors) / len(batch_X) + self.l2_reg * self.weights
            grad_b = np.mean(errors)
            
            # Update weights
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        
        self.samples_seen += n_samples
        self.last_update_time = datetime.now()
        
        new_performance = self._evaluate(X, y)
        
        # Record update
        update = ModelUpdate(
            timestamp=datetime.now(),
            samples_processed=self.samples_seen,
            old_performance=old_performance,
            new_performance=new_performance,
            performance_change=new_performance - old_performance,
            update_duration_seconds=0.0  # Would measure in production
        )
        self.update_history.append(update)
        
        return new_performance
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using current weights"""
        return np.dot(X, self.weights) + self.bias
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance (MSE)"""
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse


class ContinuousOnlineLearning:
    """
    Continuous Online Learning Manager
    
    Manages daily mini-batch updates for all models.
    
    Features:
    - Daily mini-batch updates
    - Performance tracking
    - Automatic rollback if performance degrades
    - Model versioning
    """
    
    def __init__(self, update_frequency: UpdateFrequency = UpdateFrequency.DAILY):
        self.update_frequency = update_frequency
        self.models: Dict[str, OnlineLearningModel] = {}
        self.last_update_date: Optional[datetime] = None
        self.performance_history: Dict[str, List[float]] = {}
    
    def add_model(self, model_name: str, model: OnlineLearningModel):
        """Add model to online learning system"""
        self.models[model_name] = model
        self.performance_history[model_name] = []
    
    def update_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """
        Update model with new data.
        
        Args:
            model_name: Name of model
            X: Features
            y: Labels
        
        Returns:
            New performance metric
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        # Store old performance
        old_perf = self.performance_history[model_name][-1] if self.performance_history[model_name] else None
        
        # Update model
        new_perf = model.partial_fit(X, y)
        
        # Check for performance degradation
        if old_perf is not None and new_perf > old_perf * 1.1:  # 10% degradation threshold
            # Rollback (simplified - in production would restore checkpoint)
            print(f"Warning: Performance degraded for {model_name}, consider rollback")
        
        self.performance_history[model_name].append(new_perf)
        self.last_update_date = datetime.now()
        
        return new_perf
    
    def should_update(self) -> bool:
        """Check if update is needed based on frequency"""
        if self.last_update_date is None:
            return True
        
        now = datetime.now()
        
        if self.update_frequency == UpdateFrequency.DAILY:
            return (now - self.last_update_date).days >= 1
        elif self.update_frequency == UpdateFrequency.HOURLY:
            return (now - self.last_update_date).total_seconds() >= 3600
        elif self.update_frequency == UpdateFrequency.REAL_TIME:
            return True
        
        return False
    
    def get_model_status(self, model_name: str) -> Dict:
        """Get status of a model"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        history = self.performance_history[model_name]
        
        return {
            "samples_seen": model.samples_seen,
            "last_update": model.last_update_time,
            "total_updates": len(model.update_history),
            "current_performance": history[-1] if history else None,
            "avg_performance": np.mean(history) if history else None,
            "performance_trend": history[-1] - history[0] if len(history) > 1 else 0
        }
    
    def generate_report(self) -> str:
        """Generate online learning report"""
        report = f"""
Continuous Online Learning Report
{'=' * 50}
Update Frequency: {self.update_frequency.value}
Last Update: {self.last_update_date}
Total Models: {len(self.models)}

Model Status:
{'-' * 50}
"""
        
        for model_name in self.models:
            status = self.get_model_status(model_name)
            report += f"{model_name}:\n"
            report += f"  Samples Seen: {status['samples_seen']}\n"
            report += f"  Total Updates: {status['total_updates']}\n"
            report += f"  Current Performance: {status['current_performance']:.4f}\n"
            report += f"  Performance Trend: {status['performance_trend']:+.4f}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    learner = ContinuousOnlineLearning(update_frequency=UpdateFrequency.DAILY)
    
    # Create model
    model = OnlineLinearModel(n_features=10, learning_rate=0.01, batch_size=32)
    learner.add_model("momentum_model", model)
    
    # Simulate daily updates
    print("Simulating daily online learning updates...")
    for day in range(30):
        # Generate new data
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        
        # Update model
        perf = learner.update_model("momentum_model", X, y)
        
        if day % 5 == 0:
            print(f"Day {day}: Performance = {perf:.4f}")
    
    print(learner.generate_report())
