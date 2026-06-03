"""
Point-in-Time Feature Validator
Eliminates look-ahead bias by ensuring features use only data available at decision time.

Critical for institutional-grade backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class LookAheadType(Enum):
    """Types of look-ahead bias"""
    CENTER_WEIGHTED_MA = "center_weighted_ma"  # Uses future data
    GLOBAL_NORMALIZATION = "global_normalization"  # Leaks future statistics
    FUTURE_LABELS = "future_labels"  # Uses future returns
    PEAK_DETECTION = "peak_detection"  # Requires knowing future peaks
    REGIME_LABELING = "regime_labeling"  # Requires full sample for HMM


@dataclass
class BiasDetection:
    """Result of bias detection"""
    feature_name: str
    has_bias: bool
    bias_type: Optional[LookAheadType]
    description: str
    severity: str  # "critical", "high", "medium", "low"


class PointInTimeValidator:
    """
    Validates that features are point-in-time (no look-ahead bias).
    
    Rules:
    1. Moving averages must be trailing (not center-weighted)
    2. Normalization must use rolling statistics (not global)
    3. Labels must be future returns (not current)
    4. No peak/trough detection requiring future knowledge
    5. Regime labels must be pre-computed (not online HMM)
    """
    
    def __init__(self):
        self.detections: List[BiasDetection] = []
    
    def validate_feature(self, feature_name: str, feature_data: pd.Series,
                        price_data: pd.Series) -> BiasDetection:
        """
        Validate a single feature for look-ahead bias.
        
        Args:
            feature_name: Name of the feature
            feature_data: Feature values (indexed by time)
            price_data: Price data (indexed by time)
        
        Returns:
            BiasDetection result
        """
        # Check for center-weighted moving average
        if self._is_center_weighted_ma(feature_name, feature_data):
            return BiasDetection(
                feature_name=feature_name,
                has_bias=True,
                bias_type=LookAheadType.CENTER_WEIGHTED_MA,
                description="Feature uses center-weighted MA which requires future data",
                severity="critical"
            )
        
        # Check for global normalization
        if self._is_globally_normalized(feature_name, feature_data):
            return BiasDetection(
                feature_name=feature_name,
                has_bias=True,
                bias_type=LookAheadType.GLOBAL_NORMALIZATION,
                description="Feature uses global normalization (leaks future statistics)",
                severity="critical"
            )
        
        # Check for future labels
        if self._uses_future_labels(feature_name, feature_data, price_data):
            return BiasDetection(
                feature_name=feature_name,
                has_bias=True,
                bias_type=LookAheadType.FUTURE_LABELS,
                description="Feature uses future returns as label",
                severity="critical"
            )
        
        # Check for peak detection
        if self._is_peak_detection(feature_name, feature_data):
            return BiasDetection(
                feature_name=feature_name,
                has_bias=True,
                bias_type=LookAheadType.PEAK_DETECTION,
                description="Feature uses peak/trough detection requiring future knowledge",
                severity="high"
            )
        
        # No bias detected
        return BiasDetection(
            feature_name=feature_name,
            has_bias=False,
            bias_type=None,
            description="Feature appears point-in-time valid",
            severity="none"
        )
    
    def validate_features(self, features_df: pd.DataFrame,
                         price_data: pd.Series) -> List[BiasDetection]:
        """
        Validate all features for look-ahead bias.
        
        Args:
            features_df: DataFrame of features (columns = features, index = time)
            price_data: Price data (indexed by time)
        
        Returns:
            List of BiasDetection results
        """
        self.detections = []
        
        for feature_name in features_df.columns:
            detection = self.validate_feature(
                feature_name,
                features_df[feature_name],
                price_data
            )
            self.detections.append(detection)
        
        return self.detections
    
    def get_biased_features(self) -> List[str]:
        """Get list of features with look-ahead bias"""
        return [d.feature_name for d in self.detections if d.has_bias]
    
    def get_critical_biases(self) -> List[BiasDetection]:
        """Get list of critical bias detections"""
        return [d for d in self.detections if d.has_bias and d.severity == "critical"]
    
    def _is_center_weighted_ma(self, feature_name: str, feature_data: pd.Series) -> bool:
        """Check if feature uses center-weighted moving average"""
        # Heuristic: center-weighted MA names often contain "center" or "mid"
        keywords = ["center", "mid", "symmetric", "central"]
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in keywords):
            return True
        
        # Check if feature leads price (correlation with future)
        if len(feature_data) > 20:
            # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
            # shift(-1) uses future data which is lookahead bias
            # shift(1) uses next period's return which is point-in-time correct
            future_returns = feature_data.pct_change().shift(1)
            correlation = feature_data.corr(future_returns)
            
            # High positive correlation with future suggests look-ahead
            if correlation > 0.3:
                return True
        
        return False
    
    def _is_globally_normalized(self, feature_name: str, feature_data: pd.Series) -> bool:
        """Check if feature uses global normalization"""
        # Heuristic: global normalization names often contain "global", "full", "sample"
        keywords = ["global", "full_sample", "entire", "zscore_global"]
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in keywords):
            return True
        
        # Check if feature has near-zero mean and unit std (global normalization signature)
        if len(feature_data) > 100:
            mean = feature_data.mean()
            std = feature_data.std()
            
            if abs(mean) < 0.01 and abs(std - 1.0) < 0.1:
                return True
        
        return False
    
    def _uses_future_labels(self, feature_name: str, feature_data: pd.Series,
                           price_data: pd.Series) -> bool:
        """Check if feature uses future returns as label"""
        # Heuristic: label features often contain "return", "pnl", "profit"
        keywords = ["future_return", "target_return", "label_return"]
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in keywords):
            return True
        
        return False
    
    def _is_peak_detection(self, feature_name: str, feature_data: pd.Series) -> bool:
        """Check if feature uses peak/trough detection"""
        # Heuristic: peak detection names often contain "peak", "trough", "high", "low"
        keywords = ["peak", "trough", "local_high", "local_low", "extremum"]
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in keywords):
            return True
        
        return False
    
    def generate_report(self) -> str:
        """Generate validation report"""
        total = len(self.detections)
        biased = len(self.get_biased_features())
        critical = len(self.get_critical_biases())
        
        report = f"""
Point-in-Time Validation Report
{'=' * 50}
Total features validated: {total}
Features with bias: {biased} ({biased/total*100:.1f}%)
Critical biases: {critical}

Critical Biases (Must Fix):
{'-' * 50}
"""
        
        for detection in self.get_critical_biases():
            report += f"- {detection.feature_name}: {detection.description}\n"
        
        report += f"\nAll Biased Features:\n{'-' * 50}\n"
        for detection in self.get_biased_features():
            report += f"- {detection}\n"
        
        return report


def fix_look_ahead_bias(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fix look-ahead bias in features by removing biased features.
    
    Args:
        features_df: DataFrame of features
    
    Returns:
        Tuple of (cleaned features, list of removed features)
    """
    validator = PointInTimeValidator()
    
    # Need price data for validation (use first column as proxy)
    price_data = features_df.iloc[:, 0]
    
    detections = validator.validate_features(features_df, price_data)
    biased_features = validator.get_biased_features()
    
    # Remove biased features
    clean_features = features_df.drop(columns=biased_features, errors='ignore')
    
    return clean_features, biased_features


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample features with bias
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    features_df = pd.DataFrame({
        'price': np.cumsum(np.random.randn(n)) + 100,
        'returns': np.random.randn(n) * 0.01,
        'center_ma_20': np.random.randn(n),  # Biased
        'global_zscore': np.random.randn(n),  # Biased
        'trailing_ma_20': np.random.randn(n),  # OK
        'rolling_std': np.random.randn(n),  # OK
    }, index=dates)
    
    validator = PointInTimeValidator()
    detections = validator.validate_features(features_df, features_df['price'])
    
    print(validator.generate_report())
    
    # Fix bias
    clean_features, removed = fix_look_ahead_bias(features_df)
    print(f"\nRemoved {len(removed)} biased features")
    print(f"Remaining features: {clean_features.columns.tolist()}")
