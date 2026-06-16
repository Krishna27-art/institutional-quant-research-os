"""
Random Matrix Theory (RMT) Covariance Estimator.
Implements Ledoit-Wolf shrinkage to produce well-conditioned covariance matrices
from historical returns, even when N (assets) > T (observations).
"""

import numpy as np
import logging
from typing import Optional
try:
    from sklearn.covariance import LedoitWolf
except ImportError:
    LedoitWolf = None

logger = logging.getLogger(__name__)

class CovarianceEstimator:
    """Estimates empirical covariance matrices utilizing Ledoit-Wolf shrinkage."""
    
    def __init__(self, assume_centered: bool = False):
        self.assume_centered = assume_centered
        if LedoitWolf is None:
            logger.warning("scikit-learn is not installed. LedoitWolf shrinkage will fallback to sample covariance.")
            
    def estimate(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Estimate the covariance matrix using Ledoit-Wolf shrinkage.
        
        Args:
            returns_matrix: A 2D array of historical returns.
                          Shape must be (T, N) where T is time observations and N is assets/alphas.
                          
        Returns:
            np.ndarray: Shrinkage-adjusted (N, N) covariance matrix.
        """
        if returns_matrix is None or returns_matrix.size == 0:
            raise ValueError("Returns matrix cannot be empty.")
            
        # Ensure 2D array
        if returns_matrix.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (T, N), got {returns_matrix.ndim}D array.")
            
        T, N = returns_matrix.shape
        
        # If we only have 1 observation, or completely flat data, fallback to identity
        if T < 2:
            logger.warning(f"Insufficient observations (T={T}). Falling back to identity matrix.")
            return np.eye(N)
            
        if LedoitWolf is not None:
            try:
                lw = LedoitWolf(assume_centered=self.assume_centered)
                lw.fit(returns_matrix)
                return lw.covariance_
            except Exception as e:
                logger.error(f"Ledoit-Wolf estimation failed: {e}. Falling back to sample covariance.")
        
        # Fallback: Sample Covariance with minor diagonal ridge
        cov = np.cov(returns_matrix, rowvar=False)
        if N == 1:
            cov = np.array([[cov]])
            
        # Add small ridge to ensure positive semi-definiteness
        cov += np.eye(N) * 1e-6
        return cov

def get_covariance_estimator() -> CovarianceEstimator:
    """Factory method to get a configured covariance estimator."""
    return CovarianceEstimator()
