"""
Probabilistic Forecasting with Prediction Intervals
Based on V3 Blueprint - Prediction with Confidence

Key findings from research:
- Binary BUY/SELL signals ignore uncertainty
- Enhanced signal output with uncertainty quantification
- Output: expected_return, prob_positive, confidence, uncertainty_band
- Calibration: logistic regression on validation set
- Position size ∝ expected_return * prob_positive * confidence

V3 Upgrade - Expected Sharpe increase: +0.1–0.2 (improves sizing)
Priority: Medium
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


@dataclass
class ProbabilisticSignal:
    """Probabilistic signal with uncertainty"""
    expected_return: float
    prob_positive: float  # Probability of positive return
    confidence: float  # Model confidence (0-1)
    uncertainty_band: Tuple[float, float]  # (lower, upper) 80% interval
    calibrated_prob: float  # Calibrated probability


@dataclass
class CalibrationResult:
    """Calibration result"""
    calibration_method: str
    calibration_score: float  # Brier score or similar
    is_well_calibrated: bool


class ProbabilisticForecaster:
    """
    Probabilistic Forecaster with prediction intervals.
    
    Output Format:
    {
      "expected_return": 1.3,
      "prob_positive": 0.64,
      "confidence": 0.78,
      "uncertainty_band": [0.7, 2.1]
    }
    
    Calibration:
    - Use validation set to fit logistic regression: predicted_prob → actual_prob
    - Store calibration curve
    """
    
    def __init__(self):
        self.calibration_models: Dict[str, LogisticRegression] = {}
        self.isotonic_calibrators: Dict[str, IsotonicRegression] = {}
        self.calibration_history: List[CalibrationResult] = []
    
    def compute_prediction_interval(
        self,
        predictions: np.ndarray,
        confidence_level: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction interval using quantile regression or bootstrap.
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level (0-1)
            
        Returns:
            (lower_bound, upper_bound)
        """
        # Simple approach: use prediction std
        pred_std = np.std(predictions)
        
        # Assuming normal distribution
        z_score = 1.28  # For 80% interval
        lower = predictions - z_score * pred_std
        upper = predictions + z_score * pred_std
        
        return lower, upper
    
    def compute_confidence(
        self,
        predictions: np.ndarray,
        model_std: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute model confidence based on prediction variance.
        
        Args:
            predictions: Array of predictions
            model_std: Model standard deviation (optional)
            
        Returns:
            Confidence scores (0-1)
        """
        if model_std is None:
            model_std = np.std(predictions)
        
        # Confidence = 1 - (prediction_std / model_std)
        pred_std = np.std(predictions)
        confidence = 1 - (pred_std / (model_std + 1e-8))
        confidence = np.clip(confidence, 0, 1)
        
        return confidence
    
    def calibrate_probabilities(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        method: str = "isotonic"
    ) -> CalibrationResult:
        """
        Calibrate predicted probabilities.
        
        Args:
            predicted_probs: Predicted probabilities
            actual_outcomes: Actual binary outcomes
            method: Calibration method ("isotonic" or "logistic")
            
        Returns:
            CalibrationResult
        """
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(predicted_probs, actual_outcomes)
            self.isotonic_calibrators["default"] = calibrator
            calibrated_probs = calibrator.predict(predicted_probs)
        else:
            # Logistic regression calibration
            calibrator = LogisticRegression()
            calibrator.fit(predicted_probs.reshape(-1, 1), actual_outcomes)
            self.calibration_models["default"] = calibrator
            calibrated_probs = calibrator.predict_proba(predicted_probs.reshape(-1, 1))[:, 1]
        
        # Calculate Brier score
        brier_score = np.mean((calibrated_probs - actual_outcomes) ** 2)
        
        # Check if well calibrated (Brier score < 0.25)
        is_well_calibrated = brier_score < 0.25
        
        result = CalibrationResult(
            calibration_method=method,
            calibration_score=brier_score,
            is_well_calibrated=is_well_calibrated
        )
        
        self.calibration_history.append(result)
        
        return result
    
    def apply_calibration(self, predicted_prob: float, method: str = "isotonic") -> float:
        """
        Apply calibration to a predicted probability.
        
        Args:
            predicted_prob: Predicted probability
            method: Calibration method
            
        Returns:
            Calibrated probability
        """
        if method == "isotonic" and "default" in self.isotonic_calibrators:
            calibrator = self.isotonic_calibrators["default"]
            calibrated = calibrator.predict([predicted_prob])[0]
        elif "default" in self.calibration_models:
            calibrator = self.calibration_models["default"]
            calibrated = calibrator.predict_proba([[predicted_prob]])[0, 1]
        else:
            calibrated = predicted_prob
        
        return np.clip(calibrated, 0, 1)
    
    def generate_probabilistic_signal(
        self,
        raw_prediction: float,
        prediction_std: float,
        calibrated_prob: Optional[float] = None
    ) -> ProbabilisticSignal:
        """
        Generate probabilistic signal with uncertainty.
        
        Args:
            raw_prediction: Raw model prediction (expected return)
            prediction_std: Prediction standard deviation
            calibrated_prob: Calibrated probability (optional)
            
        Returns:
            ProbabilisticSignal
        """
        # Compute probability positive using sigmoid
        prob_positive = expit(raw_prediction / prediction_std)
        
        # Apply calibration if available
        if calibrated_prob is not None:
            prob_positive = calibrated_prob
        else:
            prob_positive = self.apply_calibration(prob_positive)
        
        # Compute confidence
        confidence = 1 - min(prediction_std / 0.02, 1.0)  # Normalize around 2% std
        
        # Compute uncertainty band
        lower = raw_prediction - 1.28 * prediction_std
        upper = raw_prediction + 1.28 * prediction_std
        
        signal = ProbabilisticSignal(
            expected_return=raw_prediction,
            prob_positive=prob_positive,
            confidence=confidence,
            uncertainty_band=(lower, upper),
            calibrated_prob=prob_positive
        )
        
        return signal
    
    def compute_position_size(
        self,
        signal: ProbabilisticSignal,
        base_kelly: float
    ) -> float:
        """
        Compute position size using probabilistic signal.
        
        Formula: position_size = Kelly × prob_positive × confidence
        
        Args:
            signal: Probabilistic signal
            base_kelly: Base Kelly fraction
            
        Returns:
            Adjusted position size
        """
        adjusted_size = base_kelly * signal.prob_positive * signal.confidence
        return adjusted_size
    
    def print_signal(self, signal: ProbabilisticSignal) -> None:
        """Print probabilistic signal."""
        print("\n" + "="*60)
        print("PROBABILISTIC SIGNAL")
        print("="*60)
        print(f"Expected Return: {signal.expected_return:.4f}")
        print(f"Probability Positive: {signal.prob_positive:.2%}")
        print(f"Confidence: {signal.confidence:.2%}")
        print(f"Uncertainty Band: [{signal.uncertainty_band[0]:.4f}, {signal.uncertainty_band[1]:.4f}]")
        print(f"Calibrated Probability: {signal.calibrated_prob:.2%}")
        print("="*60)


def run_sample_probabilistic_forecasting():
    """Run sample probabilistic forecasting."""
    forecaster = ProbabilisticForecaster()
    
    # Generate sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    raw_predictions = np.random.normal(0.001, 0.01, n_samples)
    prediction_stds = np.random.uniform(0.005, 0.02, n_samples)
    actual_outcomes = (raw_predictions > 0).astype(int)
    
    # Calibrate probabilities
    predicted_probs = expit(raw_predictions / prediction_stds)
    calibration_result = forecaster.calibrate_probabilities(predicted_probs, actual_outcomes)
    
    print(f"Calibration Result: {calibration_result.calibration_method}")
    print(f"Brier Score: {calibration_result.calibration_score:.4f}")
    print(f"Well Calibrated: {calibration_result.is_well_calibrated}")
    
    # Generate sample signals
    print("\nSample Signals:")
    for i in range(5):
        signal = forecaster.generate_probabilistic_signal(
            raw_predictions[i],
            prediction_stds[i]
        )
        forecaster.print_signal(signal)
        
        # Compute position size
        base_kelly = 0.25
        position_size = forecaster.compute_position_size(signal, base_kelly)
        print(f"Position Size: {position_size:.2%} of Kelly")
    
    return forecaster


if __name__ == "__main__":
    run_sample_probabilistic_forecasting()
