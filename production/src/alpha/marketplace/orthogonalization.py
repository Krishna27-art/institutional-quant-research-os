"""
Factor Orthogonalization Engine.
Implements a PCA-based factor orthogonalization pipeline so new alpha candidates 
are neutralized against existing live factors before deployment.
"""

import numpy as np
import logging
try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
except ImportError:
    PCA = None
    LinearRegression = None

logger = logging.getLogger(__name__)

class FactorOrthogonalizer:
    """PCA-based factor orthogonalization engine."""
    
    def __init__(self, n_components: float = 0.95):
        """
        Args:
            n_components: If float between 0 and 1, retains components explaining this variance ratio.
                          If int, retains exactly this many components.
        """
        self.n_components = n_components
        self.pca = None
        self.principal_components_ = None
        
        if PCA is None or LinearRegression is None:
            logger.warning("scikit-learn is not installed. Orthogonalization will act as a pass-through.")
            
    def fit(self, live_alpha_returns: np.ndarray) -> None:
        """
        Fit PCA on the returns of currently live alphas to extract systemic risk factors.
        
        Args:
            live_alpha_returns: 2D array of shape (T, N) where T is time, N is live alphas.
        """
        if PCA is None:
            return
            
        if live_alpha_returns is None or live_alpha_returns.size == 0:
            logger.warning("Empty live alpha returns provided. Cannot fit PCA.")
            return
            
        T, N = live_alpha_returns.shape
        if N < 2:
            logger.warning("Fewer than 2 live alphas. PCA orthogonalization is bypassed.")
            return
            
        # We need more observations than components
        n_comps = self.n_components
        if isinstance(n_comps, int) and n_comps > min(T, N):
            n_comps = min(T, N)
            
        try:
            self.pca = PCA(n_components=n_comps)
            # Principal components in shape (T, k)
            self.principal_components_ = self.pca.fit_transform(live_alpha_returns)
            logger.info(f"Fitted PCA. Extracted {self.pca.n_components_} factors explaining {np.sum(self.pca.explained_variance_ratio_):.2%} of variance.")
        except Exception as e:
            logger.error(f"PCA fitting failed: {e}")
            self.pca = None
            self.principal_components_ = None
            
    def orthogonalize(self, candidate_returns: np.ndarray) -> np.ndarray:
        """
        Neutralize candidate alpha returns against the established principal components.
        
        Args:
            candidate_returns: 1D or 2D array of candidate returns. Shape (T,) or (T, 1).
            
        Returns:
            np.ndarray: Residual (orthogonalized) returns of the candidate alpha.
        """
        if LinearRegression is None or self.pca is None or self.principal_components_ is None:
            return candidate_returns
            
        # Ensure candidate_returns is 2D column
        orig_shape = candidate_returns.shape
        if candidate_returns.ndim == 1:
            candidate_returns = candidate_returns.reshape(-1, 1)
            
        T_cand = candidate_returns.shape[0]
        T_pca = self.principal_components_.shape[0]
        
        if T_cand != T_pca:
            logger.error(f"Time dimension mismatch. Candidate has {T_cand}, PCA has {T_pca}.")
            return candidate_returns.reshape(orig_shape)
            
        try:
            # Regress candidate returns on the principal components
            lr = LinearRegression(fit_intercept=True)
            lr.fit(self.principal_components_, candidate_returns)
            
            # Predict the systematic component
            systematic_returns = lr.predict(self.principal_components_)
            
            # The residual is the purely orthogonal alpha
            orthogonal_returns = candidate_returns - systematic_returns
            
            return orthogonal_returns.reshape(orig_shape)
        except Exception as e:
            logger.error(f"Orthogonalization regression failed: {e}")
            return candidate_returns.reshape(orig_shape)

def get_orthogonalizer(n_components: float = 0.95) -> FactorOrthogonalizer:
    return FactorOrthogonalizer(n_components)
