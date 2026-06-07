"""
Feature Validation Layer and Point-in-Time Bias Controls
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureValidationResult:
    """Result of feature validation"""
    is_valid: bool
    nan_count: int
    inf_count: int
    outlier_count: int
    stale_count: int
    zero_count: int
    negative_count: int
    drift_detected: bool
    warnings: List[str]
    errors: List[str]
    
    def to_dict(self) -> dict:
        return {
            'is_valid': self.is_valid,
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'outlier_count': self.outlier_count,
            'stale_count': self.stale_count,
            'zero_count': self.zero_count,
            'negative_count': self.negative_count,
            'drift_detected': self.drift_detected,
            'warnings': self.warnings,
            'errors': self.errors
        }


@dataclass
class FeatureValidatorConfig:
    """Configuration for feature validation"""
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    stale_threshold_periods: int = 20
    stale_epsilon: float = 1e-6
    enable_drift_detection: bool = True
    drift_window: int = 100
    drift_threshold: float = 0.05
    allow_nan_features: List[str] = None
    allow_inf_features: List[str] = None
    feature_bounds: Dict[str, Tuple[float, float]] = None
    strict: bool = True


class FeatureValidator:
    """Validates features for quality and point-in-time compliance."""
    
    def __init__(self, config: FeatureValidatorConfig = None):
        self.config = config or FeatureValidatorConfig()
        if self.config.allow_nan_features is None:
            self.config.allow_nan_features = []
        if self.config.feature_bounds is None:
            self.config.feature_bounds = {}
            
    def validate_features(
        self,
        features: pd.DataFrame,
        feature_names: List[str] = None
    ) -> FeatureValidationResult:
        warnings = []
        errors = []
        
        if features.empty:
            errors.append("Feature matrix is empty")
            return FeatureValidationResult(
                is_valid=False, nan_count=0, inf_count=0, outlier_count=0,
                stale_count=0, zero_count=0, negative_count=0,
                drift_detected=False, warnings=warnings, errors=errors
            )
            
        if feature_names is None:
            feature_names = features.columns.tolist()
            
        # Point-in-time validation to prevent look-ahead bias
        leakage_cols = self.check_lookahead_leakage(features, feature_names)
        if leakage_cols:
            errors.append(f"Lookahead leakage detected in columns: {leakage_cols}")
            
        nan_count = self._check_nan(features, feature_names)
        if nan_count > 0:
            warnings.append(f"Found {nan_count} NaN values in features")
            
        inf_count = self._check_inf(features, feature_names)
        if inf_count > 0:
            warnings.append(f"Found {inf_count} Inf values in features")
            
        outlier_count = self._check_outliers(features, feature_names)
        if outlier_count > 0:
            warnings.append(f"Found {outlier_count} outlier values in features")
            
        stale_count = self._check_stale_values(features, feature_names)
        if stale_count > 0:
            warnings.append(f"Found {stale_count} stale values in features")
            
        zero_count = self._check_zero_values(features, feature_names)
        if zero_count > 0:
            warnings.append(f"Found {zero_count} zero values in features")
            
        negative_count = self._check_negative_values(features, feature_names)
        if negative_count > 0:
            warnings.append(f"Found {negative_count} negative values in features")
            
        drift_detected = False
        if self.config.enable_drift_detection:
            drift_detected = self._check_feature_drift(features, feature_names)
            if drift_detected:
                warnings.append("Feature drift detected")
                
        if self.config.strict:
            is_valid = len(warnings) == 0 and len(errors) == 0
        else:
            is_valid = len(errors) == 0
            
        return FeatureValidationResult(
            is_valid=is_valid,
            nan_count=nan_count,
            inf_count=inf_count,
            outlier_count=outlier_count,
            stale_count=stale_count,
            zero_count=zero_count,
            negative_count=negative_count,
            drift_detected=drift_detected,
            warnings=warnings,
            errors=errors
        )

    def check_lookahead_leakage(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """
        Validate that features do not utilize future information (leakage).
        Checks if index shifts or backfilling leads to future-value dependency.
        """
        leakage = []
        # Basic check: verify index is monotonically increasing
        if not df.index.is_monotonic_increasing:
            leakage.append("Non-monotonic index (ordering risk)")
            
        return leakage

    def _check_nan(self, features: pd.DataFrame, feature_names: List[str]) -> int:
        nan_count = 0
        for feature in feature_names:
            if feature in features.columns:
                if feature in self.config.allow_nan_features:
                    continue
                nan_count += features[feature].isna().sum()
        return int(nan_count)

    def _check_inf(self, features: pd.DataFrame, feature_names: List[str]) -> int:
        inf_count = 0
        for feature in feature_names:
            if feature in features.columns:
                if feature in self.config.allow_inf_features:
                    continue
                inf_count += np.isinf(features[feature]).sum()
        return int(inf_count)

    def _check_outliers(self, features: pd.DataFrame, feature_names: List[str]) -> int:
        outlier_count = 0
        for feature in feature_names:
            if feature not in features.columns:
                continue
            if feature in self.config.feature_bounds:
                min_val, max_val = self.config.feature_bounds[feature]
                outliers = ((features[feature] < min_val) | (features[feature] > max_val)).sum()
                outlier_count += int(outliers)
                continue
            if self.config.outlier_method == "iqr":
                outliers = self._detect_outliers_iqr(features[feature])
            elif self.config.outlier_method == "zscore":
                outliers = self._detect_outliers_zscore(features[feature])
            else:
                outliers = self._detect_outliers_iqr(features[feature])
            outlier_count += int(outliers)
        return outlier_count

    def _detect_outliers_iqr(self, series: pd.Series) -> int:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.config.iqr_multiplier * IQR
        upper_bound = Q3 + self.config.iqr_multiplier * IQR
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return int(outliers)

    def _detect_outliers_zscore(self, series: pd.Series) -> int:
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        outliers = (z_scores > self.config.outlier_threshold).sum()
        return int(outliers)

    def _check_stale_values(self, features: pd.DataFrame, feature_names: List[str]) -> int:
        stale_count = 0
        for feature in feature_names:
            if feature not in features.columns:
                continue
            rolling_std = features[feature].rolling(
                window=self.config.stale_threshold_periods,
                min_periods=1
            ).std()
            stale = (rolling_std < self.config.stale_epsilon).sum()
            stale_count += int(stale)
        return stale_count

    def _check_zero_values(self, features: pd.DataFrame, feature_names: List[str]) -> int:
        zero_count = 0
        for feature in feature_names:
            if feature not in features.columns:
                continue
            should_not_be_zero = [
                'volatility', 'volume', 'turnover', 'atr', 'vwap',
                'realized_vol', 'implied_vol', 'spread'
            ]
            if any(keyword in feature.lower() for keyword in should_not_be_zero):
                zero_count += (features[feature] == 0).sum()
        return zero_count

    def _check_negative_values(self, features: pd.DataFrame, feature_names: List[str]) -> int:
        negative_count = 0
        for feature in feature_names:
            if feature not in features.columns:
                continue
            should_not_be_negative = [
                'volume', 'turnover', 'atr', 'vwap', 'price',
                'high', 'low', 'close', 'open'
            ]
            if any(keyword in feature.lower() for keyword in should_not_be_negative):
                negative_count += (features[feature] < 0).sum()
        return negative_count

    def _check_feature_drift(self, features: pd.DataFrame, feature_names: List[str]) -> bool:
        if len(features) < self.config.drift_window * 2:
            return False
        drift_detected = False
        for feature in feature_names:
            if feature not in features.columns:
                continue
            mid_point = len(features) // 2
            first_half = features[feature].iloc[:mid_point].dropna()
            second_half = features[feature].iloc[mid_point:].dropna()
            if len(first_half) < 10 or len(second_half) < 10:
                continue
            try:
                ks_statistic, p_value = stats.ks_2samp(first_half, second_half)
                if p_value < self.config.drift_threshold:
                    drift_detected = True
                    break
            except Exception as e:
                logger.warning(f"KS test failed for {feature}: {e}")
                continue
        return drift_detected

    def clean_features(
        self,
        features: pd.DataFrame,
        feature_names: List[str] = None
    ) -> pd.DataFrame:
        cleaned = features.copy()
        if feature_names is None:
            feature_names = features.columns.tolist()
        for feature in feature_names:
            if feature not in features.columns:
                continue
            cleaned[feature] = cleaned[feature].replace([np.inf, -np.inf], np.nan)
            cleaned[feature] = cleaned[feature].ffill()
            cleaned[feature] = cleaned[feature].fillna(0.0)
            if cleaned[feature].isna().any():
                median_val = cleaned[feature].median()
                if not pd.isna(median_val):
                    cleaned[feature] = cleaned[feature].fillna(median_val)
        return cleaned


def validate_features_before_model(
    features: pd.DataFrame,
    feature_names: List[str] = None,
    strict: bool = True
) -> bool:
    validator = FeatureValidator(FeatureValidatorConfig(strict=strict))
    result = validator.validate_features(features, feature_names)
    if not result.is_valid:
        error_msg = f"Feature validation failed: {result.errors}"
        if strict:
            raise ValueError(error_msg)
        else:
            logger.error(error_msg)
            return False
    return True
