"""
Fractional Differencing Feature Store (López de Prado)
Advances in Financial Machine Learning (2018)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FractionalDiffResult:
    """Result of fractional differencing"""
    differenced_series: pd.Series
    weights: np.ndarray
    d: float
    threshold: float
    n_weights: int
    memory_preserved: float


class FractionalDifferencing:
    """Fractional Differencing (López de Prado)"""
    
    def __init__(self, d: float = 0.4, threshold: float = 1e-5, use_fft: bool = True):
        self.d = d
        self.threshold = threshold
        self.use_fft = use_fft
        
    def _compute_weights(self, n: int) -> np.ndarray:
        """Compute fractional differencing weights"""
        weights = [1.0]
        for k in range(1, n):
            w = -weights[-1] * (self.d - k + 1) / k
            if abs(w) < self.threshold:
                break
            weights.append(w)
        return np.array(weights)
    
    def frac_diff(self, series: pd.Series, d: Optional[float] = None, threshold: Optional[float] = None) -> FractionalDiffResult:
        """Compute fractionally differenced series"""
        if d is None:
            d = self.d
        if threshold is None:
            threshold = self.threshold
        
        series = series.fillna(method='ffill').fillna(method='bfill')
        weights = self._compute_weights(len(series))
        
        if self.use_fft and len(series) > 10000:
            differenced = self._frac_diff_fft(series.values, weights)
        else:
            differenced = self._frac_diff_convolution(series.values, weights)
        
        result_series = pd.Series(differenced, index=series.index[-len(differenced):])
        memory_preserved = self._calculate_memory_preserved(series.values, differenced)
        
        return FractionalDiffResult(
            differenced_series=result_series,
            weights=weights,
            d=d,
            threshold=threshold,
            n_weights=len(weights),
            memory_preserved=memory_preserved
        )
    
    def _frac_diff_convolution(self, series: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Fractional differencing using convolution"""
        return np.convolve(series, weights[::-1], mode='valid')
    
    def _frac_diff_fft(self, series: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Fractional differencing using FFT"""
        n = len(series)
        weights_padded = np.zeros(n)
        weights_padded[:len(weights)] = weights[::-1]
        series_fft = np.fft.fft(series)
        weights_fft = np.fft.fft(weights_padded)
        result_fft = series_fft * weights_fft
        result = np.fft.ifft(result_fft).real
        return result[len(weights)-1:]
    
    def _calculate_memory_preserved(self, original: np.ndarray, differenced: np.ndarray) -> float:
        """Calculate how much memory is preserved"""
        min_len = min(len(original), len(differenced))
        orig_aligned = original[-min_len:]
        diff_aligned = differenced[-min_len:]
        correlation = np.corrcoef(orig_aligned, diff_aligned)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0


def frac_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """Compute fractionally differenced series"""
    differencer = FractionalDifferencing(d=d, threshold=threshold)
    result = differencer.frac_diff(series)
    return result.differenced_series
