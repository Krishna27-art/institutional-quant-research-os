"""
Probabilistic Forecasting
Enhanced signal output with uncertainty quantification, calibration, and confidence bands.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


@dataclass
class ProbabilisticPrediction:
    """Probabilistic prediction with uncertainty quantification"""
    strategy_id: str
    symbol: str
    timestamp: datetime
    
    # Core predictions
    expected_return: float  # Expected return
    prob_positive: float  # Probability of positive return
    confidence: float  # Overall confidence score
    uncertainty_band: Tuple[float, float]  # 80% prediction interval
    
    # Raw model outputs
    raw_logit: float = 0.0
    model_agreement: float = 0.0  # Agreement among ensemble models
    
    # Metadata
    prediction_type: str = "long"  # "long", "short", "neutral"
    calibrated: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "expected_return": self.expected_return,
            "prob_positive": self.prob_positive,
            "confidence": self.confidence,
            "uncertainty_band": list(self.uncertainty_band),
            "raw_logit": self.raw_logit,
            "model_agreement": self.model_agreement,
            "prediction_type": self.prediction_type,
            "calibrated": self.calibrated,
        }
    
    def get_position_size_multiplier(self) -> float:
        """
        Calculate position size multiplier based on probabilistic outputs.
        position_size ∝ expected_return * prob_positive * confidence
        """
        return self.expected_return * self.prob_positive * self.confidence


@dataclass
class CalibrationCurve:
    """Calibration curve for probability calibration"""
    strategy_id: str
    fitted: bool = False
    calibration_data: List[Tuple[float, float]] = field(default_factory=list)
    logistic_model: Optional[LogisticRegression] = None
    
    # Calibration metrics
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "fitted": self.fitted,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
        }


class ProbabilisticForecaster:
    """
    Converts binary signals to probabilistic predictions with uncertainty quantification.
    Includes calibration and confidence band estimation.
    """
    
    def __init__(self):
        self.calibration_curves: Dict[str, CalibrationCurve] = {}
        self.prediction_history: List[ProbabilisticPrediction] = []
    
    def calibrate(
        self,
        strategy_id: str,
        predicted_probs: np.ndarray,
        actual_returns: np.ndarray
    ) -> CalibrationCurve:
        """
        Calibrate predicted probabilities using logistic regression.
        
        Args:
            strategy_id: Strategy identifier
            predicted_probs: Predicted probabilities from model
            actual_returns: Actual returns (binary: positive/negative)
        
        Returns:
            CalibrationCurve with fitted model
        """
        # Fit logistic regression: predicted_prob -> actual_prob
        X = predicted_probs.reshape(-1, 1)
        y = (actual_returns > 0).astype(int)
        
        logistic_model = LogisticRegression()
        logistic_model.fit(X, y)
        
        # Calculate calibration metrics
        calibrated_probs = logistic_model.predict_proba(X)[:, 1]
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        prob_true, prob_pred = calibration_curve(y, calibrated_probs, n_bins=n_bins)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        # Brier score
        brier_score = np.mean((calibrated_probs - y) ** 2)
        
        # Create calibration curve
        curve = CalibrationCurve(
            strategy_id=strategy_id,
            fitted=True,
            calibration_data=list(zip(prob_pred, prob_true)),
            logistic_model=logistic_model,
            expected_calibration_error=ece,
            brier_score=brier_score
        )
        
        self.calibration_curves[strategy_id] = curve
        
        return curve
    
    def predict(
        self,
        strategy_id: str,
        symbol: str,
        raw_logit: float,
        features: Optional[np.ndarray] = None,
        model_predictions: Optional[List[float]] = None,
        uncertainty_estimate: Optional[float] = None
    ) -> ProbabilisticPrediction:
        """
        Generate probabilistic prediction from raw model output.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            raw_logit: Raw logit from model
            features: Feature vector (for uncertainty estimation)
            model_predictions: List of predictions from ensemble models
            uncertainty_estimate: Pre-computed uncertainty estimate
        
        Returns:
            ProbabilisticPrediction with calibrated probabilities
        """
        # Apply sigmoid to get probability
        prob_positive = 1.0 / (1.0 + np.exp(-raw_logit))
        
        # Calibrate if calibration curve exists
        calibrated = False
        if strategy_id in self.calibration_curves:
            curve = self.calibration_curves[strategy_id]
            if curve.fitted and curve.logistic_model:
                prob_positive = curve.logistic_model.predict_proba([[prob_positive]])[0, 1]
                calibrated = True
        
        # Calculate expected return (simplified: scale by probability)
        # In production, use model-specific return estimation
        expected_return = (prob_positive - 0.5) * 2.0  # Scale to [-1, 1]
        
        # Calculate model agreement
        if model_predictions:
            model_agreement = np.std(model_predictions)
            confidence = 1.0 - model_agreement  # Higher agreement = higher confidence
        else:
            model_agreement = 0.5
            confidence = 0.5
        
        # Estimate uncertainty band
        if uncertainty_estimate is None:
            # Simplified: use probability-based uncertainty
            uncertainty = np.sqrt(prob_positive * (1 - prob_positive)) * 2.0
        else:
            uncertainty = uncertainty_estimate
        
        uncertainty_band = (
            expected_return - uncertainty,
            expected_return + uncertainty
        )
        
        # Determine prediction type
        if expected_return > 0.1:
            prediction_type = "long"
        elif expected_return < -0.1:
            prediction_type = "short"
        else:
            prediction_type = "neutral"
        
        prediction = ProbabilisticPrediction(
            strategy_id=strategy_id,
            symbol=symbol,
            timestamp=datetime.now(),
            expected_return=expected_return,
            prob_positive=prob_positive,
            confidence=confidence,
            uncertainty_band=uncertainty_band,
            raw_logit=raw_logit,
            model_agreement=model_agreement,
            prediction_type=prediction_type,
            calibrated=calibrated
        )
        
        self.prediction_history.append(prediction)
        
        return prediction
    
    def predict_batch(
        self,
        strategy_id: str,
        symbols: List[str],
        raw_logits: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> List[ProbabilisticPrediction]:
        """
        Generate probabilistic predictions for multiple symbols.
        
        Args:
            strategy_id: Strategy identifier
            symbols: List of trading symbols
            raw_logits: Array of raw logits
            features: Feature matrix (optional)
        
        Returns:
            List of ProbabilisticPrediction objects
        """
        predictions = []
        
        for i, (symbol, logit) in enumerate(zip(symbols, raw_logits)):
            feature_vector = features[i:i+1] if features is not None else None
            prediction = self.predict(
                strategy_id=strategy_id,
                symbol=symbol,
                raw_logit=logit,
                features=feature_vector
            )
            predictions.append(prediction)
        
        return predictions
    
    def get_calibration_curve(self, strategy_id: str) -> Optional[CalibrationCurve]:
        """Get calibration curve for a strategy"""
        return self.calibration_curves.get(strategy_id)
    
    def get_prediction_history(
        self,
        strategy_id: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict]:
        """Get recent prediction history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        predictions = [
            p for p in self.prediction_history
            if p.timestamp >= cutoff_time
            and (strategy_id is None or p.strategy_id == strategy_id)
        ]
        
        return [p.to_dict() for p in predictions]
    
    def evaluate_calibration(
        self,
        strategy_id: str,
        recent_predictions: int = 100
    ) -> Dict:
        """
        Evaluate calibration quality for recent predictions.
        
        Args:
            strategy_id: Strategy identifier
            recent_predictions: Number of recent predictions to evaluate
        
        Returns:
            Calibration evaluation metrics
        """
        if strategy_id not in self.calibration_curves:
            return {
                "strategy_id": strategy_id,
                "status": "No calibration curve available"
            }
        
        curve = self.calibration_curves[strategy_id]
        
        if not curve.fitted:
            return {
                "strategy_id": strategy_id,
                "status": "Calibration not fitted"
            }
        
        return {
            "strategy_id": strategy_id,
            "expected_calibration_error": curve.expected_calibration_error,
            "brier_score": curve.brier_score,
            "calibration_quality": "good" if curve.expected_calibration_error < 0.1 else "poor",
            "recommendations": [
                "Calibration looks good" if curve.expected_calibration_error < 0.1
                else "Consider recalibrating with recent data"
            ]
        }
    
    def clear_history(self, strategy_id: Optional[str] = None) -> None:
        """Clear prediction history, optionally filtered by strategy"""
        if strategy_id is None:
            self.prediction_history.clear()
        else:
            self.prediction_history = [
                p for p in self.prediction_history
                if p.strategy_id != strategy_id
            ]


def calculate_uncertainty_band(
    expected_return: float,
    prob_positive: float,
    confidence: float,
    volatility: float = 0.02
) -> Tuple[float, float]:
    """
    Calculate uncertainty band for prediction.
    
    Args:
        expected_return: Expected return
        prob_positive: Probability of positive return
        confidence: Confidence score
        volatility: Volatility estimate
    
    Returns:
        Tuple of (lower_bound, upper_bound) for 80% prediction interval
    """
    # Uncertainty increases with volatility and decreases with confidence
    uncertainty = volatility * (1.0 - confidence) * 2.0
    
    # Adjust based on probability (more uncertainty near 0.5)
    probability_uncertainty = 4.0 * prob_positive * (1 - prob_positive)
    
    total_uncertainty = uncertainty + probability_uncertainty
    
    lower_bound = expected_return - total_uncertainty
    upper_bound = expected_return + total_uncertainty
    
    return (lower_bound, upper_bound)


def ensemble_model_agreement(predictions: List[float]) -> float:
    """
    Calculate agreement among ensemble models.
    
    Args:
        predictions: List of predictions from ensemble models
    
    Returns:
        Agreement score (0 = no agreement, 1 = perfect agreement)
    """
    if not predictions:
        return 0.5
    
    # Calculate standard deviation
    std = np.std(predictions)
    
    # Convert to agreement score (lower std = higher agreement)
    # Assume max reasonable std is 2.0
    agreement = max(0.0, 1.0 - std / 2.0)
    
    return agreement
