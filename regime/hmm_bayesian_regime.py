"""
Market Regime Modeling (HMM + Bayesian)

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#30)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Combined HMM and Bayesian model averaging for regime detection
- Ensemble of regime models
- Improved regime identification
- Used for adaptive strategy allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not available. Install with: pip install hmmlearn")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class RegimeConfig:
    """Configuration for Regime Modeling"""
    # HMM parameters
    n_states: int = 5  # Number of regimes
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-6
    
    # Bayesian parameters
    prior_strength: float = 1.0  # Strength of prior
    update_frequency: int = 20  # Update frequency
    
    # Ensemble parameters
    ensemble_size: int = 5  # Number of HMM models in ensemble
    use_bayesian_averaging: bool = True
    
    # Features
    features: List[str] = None


class EnsembleHMMRegime:
    """
    Ensemble HMM Regime Model
    
    Combines multiple HMM models for robust regime detection.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        
        # Ensemble of HMM models
        self.models: List = []
        
        # Regime probabilities
        self.regime_probabilities: Optional[np.ndarray] = None
        
        # Feature scaler
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit ensemble of HMM models
        
        Args:
            data: Feature data
        """
        if not HMM_AVAILABLE:
            print("HMM not available")
            return
        
        # Prepare features
        features = self.config.features or data.columns.tolist()
        X = data[features].values
        
        # Scale features
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Fit ensemble
        self.models = []
        for i in range(self.config.ensemble_size):
            model = hmm.GaussianHMM(
                n_components=self.config.n_states,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                tol=self.config.tol,
                random_state=i
            )
            
            # Add noise to data for diversity
            X_noisy = X + np.random.randn(*X.shape) * 0.01
            model.fit(X_noisy)
            self.models.append(model)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities using ensemble
        
        Args:
            data: Feature data
            
        Returns:
            Regime probabilities
        """
        if not self.models:
            return np.zeros(self.config.n_states)
        
        # Prepare features
        features = self.config.features or data.columns.tolist()
        X = data[features].values
        
        # Scale features
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Get predictions from each model
        all_posteriors = []
        for model in self.models:
            posteriors = model.predict_proba(X)
            all_posteriors.append(posteriors)
        
        # Average posteriors
        avg_posteriors = np.mean(all_posteriors, axis=0)
        
        # Use last time step
        self.regime_probabilities = avg_posteriors[-1]
        
        return self.regime_probabilities
    
    def get_regime(self) -> int:
        """Get most likely regime"""
        if self.regime_probabilities is None:
            return 0
        
        return int(np.argmax(self.regime_probabilities))


class BayesianRegimeUpdate:
    """
    Bayesian Update for Regime Probabilities
    
    Updates regime probabilities using Bayesian inference.
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        
        # Prior probabilities
        self.prior: Optional[np.ndarray] = None
        
        # Posterior probabilities
        self.posterior: Optional[np.ndarray] = None
        
        # Likelihood history
        self.likelihood_history: List[np.ndarray] = []
    
    def set_prior(self, prior: np.ndarray) -> None:
        """Set prior regime probabilities"""
        self.prior = prior
        self.posterior = prior.copy()
    
    def update(self, likelihood: np.ndarray) -> np.ndarray:
        """
        Update posterior using Bayes' rule
        
        Args:
            likelihood: Likelihood of each regime
            
        Returns:
            Updated posterior
        """
        if self.prior is None:
            # Uniform prior
            n_regimes = len(likelihood)
            self.prior = np.ones(n_regimes) / n_regimes
            self.posterior = self.prior.copy()
        
        # Bayes' rule: posterior ∝ prior * likelihood
        unnormalized = self.posterior * likelihood
        
        # Normalize
        self.posterior = unnormalized / unnormalized.sum()
        
        self.likelihood_history.append(likelihood)
        
        return self.posterior
    
    def get_posterior(self) -> np.ndarray:
        """Get current posterior"""
        return self.posterior if self.posterior is not None else np.zeros(1)


class CombinedRegimeModel:
    """
    Combined HMM + Bayesian Regime Model
    
    Combines ensemble HMM with Bayesian updating for robust
    regime detection and adaptive strategy allocation.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: RegimeConfig):
        self.config = config
        
        self.hmm_ensemble = EnsembleHMMRegime(config)
        self.bayesian_update = BayesianRegimeUpdate(config)
        
        # Regime labels
        self.regime_labels = ["Bull", "Bear", "Sideways", "High Vol", "Low Vol"]
        
        # Regime history
        self.regime_history: List[int] = []
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit combined model
        
        Args:
            data: Feature data
        """
        # Fit HMM ensemble
        self.hmm_ensemble.fit(data)
        
        # Set uniform prior for Bayesian update
        n_regimes = self.config.n_states
        self.bayesian_update.set_prior(np.ones(n_regimes) / n_regimes)
    
    def update(self, data: pd.DataFrame) -> Dict:
        """
        Update regime probabilities
        
        Args:
            data: New feature data
            
        Returns:
            Dictionary with regime information
        """
        # Get HMM predictions
        hmm_probs = self.hmm_ensemble.predict(data)
        
        # Use HMM probabilities as likelihood for Bayesian update
        if self.config.use_bayesian_averaging:
            posterior = self.bayesian_update.update(hmm_probs)
        else:
            posterior = hmm_probs
        
        # Get regime
        regime = int(np.argmax(posterior))
        self.regime_history.append(regime)
        
        return {
            "regime": regime,
            "regime_label": self.regime_labels[regime] if regime < len(self.regime_labels) else f"Regime_{regime}",
            "probabilities": posterior,
            "hmm_probabilities": hmm_probs
        }
    
    def get_regime_allocation(self) -> Dict[str, float]:
        """
        Get strategy allocation based on regime
        
        Returns:
            Dictionary of strategy -> allocation
        """
        if not self.regime_history:
            return {"equity": 1.0, "bonds": 0.0}
        
        current_regime = self.regime_history[-1]
        
        # Simple allocation based on regime
        if current_regime == 0:  # Bull
            return {"equity": 0.8, "bonds": 0.2}
        elif current_regime == 1:  # Bear
            return {"equity": 0.3, "bonds": 0.7}
        elif current_regime == 2:  # Sideways
            return {"equity": 0.5, "bonds": 0.5}
        elif current_regime == 3:  # High Vol
            return {"equity": 0.4, "bonds": 0.6}
        else:  # Low Vol
            return {"equity": 0.6, "bonds": 0.4}
    
    def get_regime_summary(self) -> Dict:
        """Get regime summary statistics"""
        if not self.regime_history:
            return {}
        
        regime_counts = pd.Series(self.regime_history).value_counts()
        
        return {
            "current_regime": self.regime_history[-1],
            "regime_distribution": regime_counts.to_dict(),
            "num_regime_changes": sum(1 for i in range(1, len(self.regime_history)) 
                                     if self.regime_history[i] != self.regime_history[i-1])
        }


def simulate_regime_data(n_samples: int = 500, n_features: int = 5) -> pd.DataFrame:
    """Simulate data with regime switching"""
    np.random.seed(42)
    
    # Define regime parameters
    regimes = [
        {"mu": 0.001, "sigma": 0.01},  # Bull
        {"mu": -0.001, "sigma": 0.015},  # Bear
        {"mu": 0.0, "sigma": 0.008},  # Sideways
        {"mu": 0.0, "sigma": 0.02},  # High Vol
        {"mu": 0.0005, "sigma": 0.005}  # Low Vol
    ]
    
    # Simulate regime transitions
    regime_sequence = []
    current_regime = 0
    
    for _ in range(n_samples):
        regime_sequence.append(current_regime)
        # Random regime switch (10% probability)
        if np.random.random() < 0.1:
            current_regime = np.random.randint(0, 5)
    
    # Generate data based on regime
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        regime = regimes[regime_sequence[i]]
        data[i] = np.random.randn(n_features) * regime["sigma"] + regime["mu"]
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    dates = pd.date_range(start="2023-01-01", periods=n_samples)
    
    return pd.DataFrame(data, index=dates, columns=feature_names)


if __name__ == "__main__":
    # Example usage
    config = RegimeConfig(
        n_states=5,
        ensemble_size=5,
        use_bayesian_averaging=True
    )
    
    model = CombinedRegimeModel(config)
    
    # Simulate data
    print("Simulating regime data...")
    data = simulate_regime_data(500, 5)
    
    # Fit model
    print("\nFitting combined regime model...")
    model.fit(data)
    
    # Update with new data
    print("\nUpdating regime probabilities...")
    for i in range(20, len(data), 20):
        result = model.update(data.iloc[:i])
        if i % 100 == 0:
            print(f"  Step {i}: Regime={result['regime_label']} ({result['probabilities'].argmax()})")
    
    # Get regime summary
    print("\nRegime Summary:")
    summary = model.get_regime_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get allocation
    print("\nRegime-based Allocation:")
    allocation = model.get_regime_allocation()
    for strategy, weight in allocation.items():
        print(f"  {strategy}: {weight:.1%}")
