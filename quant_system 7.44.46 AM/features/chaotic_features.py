"""
Chaotic Feature Generation (BCF-GCN)
Logistic and Tent Map Transformations for Financial Time Series
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChaoticTransformResult:
    """Result of chaotic transformation"""
    original: np.ndarray
    logistic: np.ndarray
    tent: np.ndarray
    combined: np.ndarray
    r: float
    mu: float


class ChaoticTransform:
    """
    Chaotic Feature Generation using Logistic and Tent Maps
    
    Logistic map: x_{n+1} = r * x_n * (1 - x_n)
    Tent map: x_{n+1} = mu * x_n if x_n < 0.5, else mu * (1 - x_n)
    
    These transformations introduce controlled chaos to capture
    non-linear dynamics in financial time series.
    """
    
    def __init__(self, r: float = 3.8, mu: float = 1.8):
        """
        Args:
            r: Logistic map parameter (chaotic for r > 3.57)
            mu: Tent map parameter (chaotic for mu > 1)
        """
        self.r = r
        self.mu = mu
        
    def logistic(self, x: np.ndarray) -> np.ndarray:
        """
        Apply logistic map transformation.
        
        Args:
            x: Input array
            
        Returns:
            Transformed array
        """
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        return self.r * x_norm * (1 - x_norm)
    
    def tent(self, x: np.ndarray) -> np.ndarray:
        """
        Apply tent map transformation.
        
        Args:
            x: Input array
            
        Returns:
            Transformed array
        """
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        return np.where(x_norm < 0.5, self.mu * x_norm, self.mu * (1 - x_norm))
    
    def transform(self, X: np.ndarray) -> ChaoticTransformResult:
        """
        Apply both chaotic transformations.
        
        Args:
            X: Input array
            
        Returns:
            ChaoticTransformResult with all transformations
        """
        X_log = self.logistic(X)
        X_tent = self.tent(X)
        X_combined = np.column_stack([X, X_log, X_tent])
        
        return ChaoticTransformResult(
            original=X,
            logistic=X_log,
            tent=X_tent,
            combined=X_combined,
            r=self.r,
            mu=self.mu
        )
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply chaotic transformations to a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with chaotic features
        """
        result_df = pd.DataFrame(index=df.index)
        
        for column in df.columns:
            try:
                result = self.transform(df[column].values)
                result_df[f"{column}_logistic"] = result.logistic
                result_df[f"{column}_tent"] = result.tent
            except Exception as e:
                logger.warning(f"Failed to transform {column}: {e}")
        
        return result_df


class LearnableChaoticTransform(ChaoticTransform):
    """
    Learnable Chaotic Transform with parameter optimization.
    
    Uses gradient-based optimization to find optimal r and mu
    parameters for maximum feature separation.
    """
    
    def __init__(self, r: float = 3.8, mu: float = 1.8, learning_rate: float = 0.01):
        super().__init__(r, mu)
        self.learning_rate = learning_rate
        
    def optimize_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 100
    ) -> dict:
        """
        Optimize chaotic parameters for classification/regression.
        
        Args:
            X: Input features
            y: Target variable
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary with optimized parameters
        """
        # Simple grid search (in production, use gradient descent)
        r_values = np.linspace(3.5, 4.0, 10)
        mu_values = np.linspace(1.5, 2.0, 10)
        
        best_score = -np.inf
        best_params = {'r': self.r, 'mu': self.mu}
        
        for r in r_values:
            for mu in mu_values:
                self.r = r
                self.mu = mu
                result = self.transform(X)
                
                # Simple correlation score (in production, use actual model)
                score = np.mean([np.corrcoef(result.combined[:, i], y)[0, 1] for i in range(3)])
                
                if score > best_score:
                    best_score = score
                    best_params = {'r': r, 'mu': mu}
        
        self.r = best_params['r']
        self.mu = best_params['mu']
        
        return best_params
