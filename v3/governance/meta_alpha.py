"""
Meta Alpha Layer
Learns alpha-specific weights based on regime, health, and drift.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AlphaWeightMultiplier:
    """Weight multiplier for a specific alpha"""
    alpha_id: str
    multiplier: float
    confidence: float
    active: bool
    reason: str
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "alpha_id": self.alpha_id,
            "multiplier": self.multiplier,
            "confidence": self.confidence,
            "active": self.active,
            "reason": self.reason,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class MetaModel:
    """Meta model for a specific alpha"""
    alpha_id: str
    model_type: str  # "classifier" or "regression"
    trained: bool = False
    last_trained: Optional[datetime] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "alpha_id": self.alpha_id,
            "model_type": self.model_type,
            "trained": self.trained,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "feature_importance": self.feature_importance,
        }


class MetaAlphaLayer:
    """
    Meta-learning layer that predicts alpha-specific weights based on current market state.
    Uses LightGBM classifiers (active/inactive) and regressions (weight multiplier).
    """
    
    def __init__(
        self,
        base_weights: Dict[str, float],
        confidence_threshold: float = 0.6,
        fallback_equal_weights: bool = True
    ):
        self.base_weights = base_weights
        self.confidence_threshold = confidence_threshold
        self.fallback_equal_weights = fallback_equal_weights
        
        # Meta models (one classifier and one regression per alpha)
        self.classifiers: Dict[str, MetaModel] = {}
        self.regressions: Dict[str, MetaModel] = {}
        
        # Current multipliers
        self.current_multipliers: Dict[str, AlphaWeightMultiplier] = {}
        
        # Training data
        self.training_data: List[Dict] = []
        
        # Initialize multipliers
        for alpha_id in base_weights:
            self.current_multipliers[alpha_id] = AlphaWeightMultiplier(
                alpha_id=alpha_id,
                multiplier=1.0,
                confidence=0.5,
                active=True,
                reason="Initial"
            )
    
    def prepare_features(
        self,
        alpha_returns: Dict[str, np.ndarray],
        regime: str,
        feature_drift_scores: Dict[str, float],
        alpha_health: Dict[str, Dict[str, float]],
        vix_change: float,
        turnover_change: float
    ) -> np.ndarray:
        """
        Prepare features for meta model inference.
        
        Args:
            alpha_returns: Dictionary of alpha_id -> daily returns array
            regime: Current market regime
            feature_drift_scores: Dictionary of feature_name -> PSI score
            alpha_health: Dictionary of alpha_id -> health metrics
            vix_change: Change in VIX
            turnover_change: Change in turnover
        
        Returns:
            Feature array for all alphas
        """
        features = []
        
        for alpha_id in self.base_weights.keys():
            alpha_features = []
            
            # Lagged returns (5d, 20d)
            returns = alpha_returns.get(alpha_id, np.array([]))
            if len(returns) >= 20:
                alpha_features.append(np.mean(returns[-5:]))  # 5d return
                alpha_features.append(np.mean(returns[-20:]))  # 20d return
            else:
                alpha_features.extend([0.0, 0.0])
            
            # Regime (one-hot)
            regimes = ["bull_trend", "bear_trend", "sideways", "high_vol"]
            for r in regimes:
                alpha_features.append(1.0 if regime == r else 0.0)
            
            # Feature drift (average for this alpha)
            # This would typically be alpha-specific feature drift
            avg_drift = np.mean(list(feature_drift_scores.values())) if feature_drift_scores else 0.0
            alpha_features.append(avg_drift)
            
            # Alpha health metrics
            health = alpha_health.get(alpha_id, {})
            alpha_features.append(health.get("rolling_sharpe_20d", 0.0))
            alpha_features.append(health.get("information_coefficient", 0.0))
            
            # Market features
            alpha_features.append(vix_change)
            alpha_features.append(turnover_change)
            
            features.append(alpha_features)
        
        return np.array(features)
    
    def train_classifier(
        self,
        alpha_id: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Train classifier for alpha activation (binary: active/inactive).
        
        Args:
            alpha_id: Alpha identifier
            X: Feature matrix
            y: Binary labels (1 = active, 0 = inactive)
        """
        # Simplified: use logistic regression
        # In production, use LightGBM
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Store feature importance
        feature_names = [
            "return_5d", "return_20d",
            "regime_bull", "regime_bear", "regime_sideways", "regime_high_vol",
            "feature_drift", "sharpe", "ic", "vix_change", "turnover_change"
        ]
        importance = dict(zip(feature_names, np.abs(model.coef_[0])))
        
        self.classifiers[alpha_id] = MetaModel(
            alpha_id=alpha_id,
            model_type="classifier",
            trained=True,
            last_trained=datetime.now(),
            feature_importance=importance
        )
    
    def train_regression(
        self,
        alpha_id: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Train regression for alpha weight multiplier.
        
        Args:
            alpha_id: Alpha identifier
            X: Feature matrix
            y: Weight multipliers
        """
        # Simplified: use linear regression
        # In production, use LightGBM
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Store feature importance
        feature_names = [
            "return_5d", "return_20d",
            "regime_bull", "regime_bear", "regime_sideways", "regime_high_vol",
            "feature_drift", "sharpe", "ic", "vix_change", "turnover_change"
        ]
        importance = dict(zip(feature_names, np.abs(model.coef_)))
        
        self.regressions[alpha_id] = MetaModel(
            alpha_id=alpha_id,
            model_type="regression",
            trained=True,
            last_trained=datetime.now(),
            feature_importance=importance
        )
    
    def predict(
        self,
        alpha_returns: Dict[str, np.ndarray],
        regime: str,
        feature_drift_scores: Dict[str, float],
        alpha_health: Dict[str, Dict[str, float]],
        vix_change: float = 0.0,
        turnover_change: float = 0.0
    ) -> Dict[str, AlphaWeightMultiplier]:
        """
        Predict alpha weights using meta models.
        
        Args:
            alpha_returns: Dictionary of alpha_id -> daily returns
            regime: Current market regime
            feature_drift_scores: Feature drift scores
            alpha_health: Alpha health metrics
            vix_change: VIX change
            turnover_change: Turnover change
        
        Returns:
            Dictionary of alpha_id -> AlphaWeightMultiplier
        """
        # Prepare features
        X = self.prepare_features(
            alpha_returns=alpha_returns,
            regime=regime,
            feature_drift_scores=feature_drift_scores,
            alpha_health=alpha_health,
            vix_change=vix_change,
            turnover_change=turnover_change
        )
        
        results = {}
        
        for i, alpha_id in enumerate(self.base_weights.keys()):
            features = X[i:i+1]
            
            # Check if models are trained
            classifier = self.classifiers.get(alpha_id)
            regression = self.regressions.get(alpha_id)
            
            if classifier and regression and classifier.trained and regression.trained:
                # Use trained models
                from sklearn.linear_model import LogisticRegression, LinearRegression
                
                # Predict active/inactive
                active_prob = classifier.model.predict_proba(features)[0, 1] if hasattr(classifier, 'model') else 0.5
                active = active_prob > 0.5
                
                # Predict multiplier
                multiplier = regression.model.predict(features)[0] if hasattr(regression, 'model') else 1.0
                multiplier = max(0.0, min(2.0, multiplier))  # Clip to [0, 2]
                
                confidence = active_prob
                reason = "Meta model prediction"
            else:
                # Fallback: use heuristics
                health = alpha_health.get(alpha_id, {})
                sharpe = health.get("rolling_sharpe_20d", 0.0)
                
                # Simple heuristic: active if Sharpe > 0.5
                active = sharpe > 0.5
                multiplier = 1.0 if active else 0.0
                confidence = 0.5
                reason = "Heuristic fallback"
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold and self.fallback_equal_weights:
                multiplier = 1.0
                confidence = 0.5
                reason = "Fallback to equal weights (low confidence)"
            
            results[alpha_id] = AlphaWeightMultiplier(
                alpha_id=alpha_id,
                multiplier=multiplier,
                confidence=confidence,
                active=active,
                reason=reason
            )
        
        # Update current multipliers
        self.current_multipliers = results
        
        return results
    
    def compute_final_weights(self) -> Dict[str, float]:
        """
        Compute final alpha weights: base_weights * meta_multiplier.
        
        Returns:
            Dictionary of alpha_id -> final weight
        """
        final_weights = {}
        
        for alpha_id, base_weight in self.base_weights.items():
            multiplier = self.current_multipliers.get(alpha_id)
            if multiplier and multiplier.active:
                final_weights[alpha_id] = base_weight * multiplier.multiplier
            else:
                final_weights[alpha_id] = 0.0
        
        # Normalize to sum to 1
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}
        
        return final_weights
    
    def get_current_multipliers(self) -> Dict[str, Dict]:
        """Get current weight multipliers"""
        return {k: v.to_dict() for k, v in self.current_multipliers.items()}
    
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all meta models"""
        status = {}
        
        for alpha_id in self.base_weights.keys():
            status[alpha_id] = {
                "classifier": self.classifiers.get(alpha_id).to_dict() if alpha_id in self.classifiers else None,
                "regression": self.regressions.get(alpha_id).to_dict() if alpha_id in self.regressions else None,
            }
        
        return status
    
    def add_training_sample(
        self,
        features: np.ndarray,
        active_labels: np.ndarray,
        multipliers: np.ndarray
    ) -> None:
        """
        Add training sample for meta models.
        
        Args:
            features: Feature matrix
            active_labels: Binary labels for activation
            multipliers: Weight multipliers
        """
        self.training_data.append({
            "features": features,
            "active_labels": active_labels,
            "multipliers": multipliers,
            "timestamp": datetime.now()
        })
    
    def retrain_all(self) -> None:
        """Retrain all meta models with accumulated training data"""
        if not self.training_data:
            return
        
        # Aggregate training data
        all_features = []
        all_active_labels = {}
        all_multipliers = {}
        
        for sample in self.training_data:
            all_features.append(sample["features"])
            
            for i, alpha_id in enumerate(self.base_weights.keys()):
                if alpha_id not in all_active_labels:
                    all_active_labels[alpha_id] = []
                    all_multipliers[alpha_id] = []
                
                all_active_labels[alpha_id].append(sample["active_labels"][i])
                all_multipliers[alpha_id].append(sample["multipliers"][i])
        
        X = np.vstack(all_features)
        
        # Train each alpha's models
        for alpha_id in self.base_weights.keys():
            if alpha_id in all_active_labels and len(all_active_labels[alpha_id]) > 10:
                y_active = np.array(all_active_labels[alpha_id])
                y_multiplier = np.array(all_multipliers[alpha_id])
                
                self.train_classifier(alpha_id, X, y_active)
                self.train_regression(alpha_id, X, y_multiplier)
    
    def reset_multipliers(self) -> None:
        """Reset all multipliers to 1.0"""
        for alpha_id in self.current_multipliers:
            self.current_multipliers[alpha_id] = AlphaWeightMultiplier(
                alpha_id=alpha_id,
                multiplier=1.0,
                confidence=0.5,
                active=True,
                reason="Manual reset"
            )
