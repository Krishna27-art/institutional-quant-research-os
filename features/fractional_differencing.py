"""
Fractional Differencing of Features
Based on V3 Blueprint - Stationarity while Preserving Long Memory

Key findings from research:
- Fractional differencing (d=0.4) for stationary but persistent series
- ADF p-value crosses 1% at d≈0.4
- MLP outperforms naive on fractionally differenced series
- Use López de Prado algorithm

V3 Upgrade - Expected Sharpe increase: +0.1–0.2
Priority: Medium
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from statsmodels.tsa.stattools import adfuller


@dataclass
class FractionalDifferencingResult:
    """Result of fractional differencing"""
    original_series: pd.Series
    differenced_series: pd.Series
    d_estimate: float
    is_stationary: bool
    adf_pvalue: float


class FractionalDifferencingEngine:
    """
    Fractional Differencing Engine for feature stationarity.
    
    Based on López de Prado algorithm for fractional differencing.
    
    The method finds the minimum d such that the series is stationary
    (ADF test p-value < 0.05) while preserving long memory.
    
    Default d ≈ 0.4 for financial time series.
    """
    
    def __init__(self, max_d: float = 0.5, precision: float = 0.01):
        self.max_d = max_d
        self.precision = precision
    
    def get_weights(self, d: float, threshold: float = 1e-5) -> np.ndarray:
        """
        Compute weights for fractional differencing.
        
        Args:
            d: Fractional differencing parameter
            threshold: Threshold for weight truncation
            
        Returns:
            Array of weights
        """
        weights = [1.0]
        k = 1
        
        while True:
            weight = -weights[-1] * ((k - 1 - d) / k)
            weights.append(weight)
            
            if abs(weight) < threshold:
                break
            
            k += 1
        
        return np.array(weights)
    
    def fractional_difference(
        self,
        series: pd.Series,
        d: float,
        threshold: float = 1e-5
    ) -> pd.Series:
        """
        Apply fractional differencing to a series.
        
        Args:
            series: Input series
            d: Fractional differencing parameter
            threshold: Weight truncation threshold
            
        Returns:
            Fractionally differenced series
        """
        weights = self.get_weights(d, threshold)
        width = len(weights)
        
        # Convolve series with weights
        differenced = np.convolve(series.values, weights, mode='valid')
        
        # Create output series with same index (shifted)
        if width > len(series):
            return pd.Series([], index=series.index[:0])
        result = pd.Series(differenced, index=series.index[width-1:])
        
        return result
    
    def find_optimal_d(
        self,
        series: pd.Series,
        min_d: float = 0.0,
        adf_significance: float = 0.05
    ) -> Tuple[float, float]:
        """
        Find optimal d that makes series stationary.
        
        Args:
            series: Input series
            min_d: Minimum d to try
            adf_significance: ADF test significance level
            
        Returns:
            (optimal_d, adf_pvalue)
        """
        d = min_d
        
        while d <= self.max_d:
            diff_series = self.fractional_difference(series, d)
            
            # Skip if too few points
            if len(diff_series) < 20:
                d += self.precision
                continue
            
            # ADF test
            try:
                adf_result = adfuller(diff_series, maxlag=1, regression='c')
                pvalue = adf_result[1]
                
                if pvalue < adf_significance:
                    return d, pvalue
            except:
                pass
            
            d += self.precision
        
        # If no d found, return max_d
        diff_series = self.fractional_difference(series, self.max_d)
        try:
            adf_result = adfuller(diff_series, maxlag=1, regression='c')
            pvalue = adf_result[1]
        except:
            pvalue = 1.0
        
        return self.max_d, pvalue
    
    def apply_fractional_differencing(
        self,
        series: pd.Series,
        fixed_d: Optional[float] = None
    ) -> FractionalDifferencingResult:
        """
        Apply fractional differencing with optimal or fixed d.
        
        Args:
            series: Input series
            fixed_d: Fixed d value (optional, otherwise find optimal)
            
        Returns:
            FractionalDifferencingResult
        """
        if fixed_d is not None:
            d = fixed_d
            diff_series = self.fractional_difference(series, d)
            
            # ADF test
            try:
                adf_result = adfuller(diff_series, maxlag=1, regression='c')
                pvalue = adf_result[1]
                is_stationary = pvalue < 0.05
            except:
                pvalue = 1.0
                is_stationary = False
        else:
            d, pvalue = self.find_optimal_d(series)
            diff_series = self.fractional_difference(series, d)
            is_stationary = pvalue < 0.05
        
        return FractionalDifferencingResult(
            original_series=series,
            differenced_series=diff_series,
            d_estimate=d,
            is_stationary=is_stationary,
            adf_pvalue=pvalue
        )
    
    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        fixed_d: Optional[float] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, FractionalDifferencingResult]:
        """
        Apply fractional differencing to DataFrame columns.
        
        Args:
            df: Input DataFrame
            fixed_d: Fixed d value (optional)
            columns: Columns to process (optional, otherwise all numeric)
            
        Returns:
            Dictionary of column -> FractionalDifferencingResult
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for col in columns:
            if col in df.columns:
                result = self.apply_fractional_differencing(df[col].dropna(), fixed_d)
                results[col] = result
        
        return results
    
    def print_result(self, result: FractionalDifferencingResult, column_name: str) -> None:
        """Print fractional differencing result."""
        print("\n" + "="*60)
        print(f"FRACTIONAL DIFFERENCING: {column_name}")
        print("="*60)
        print(f"d estimate: {result.d_estimate:.4f}")
        print(f"Stationary: {result.is_stationary}")
        print(f"ADF p-value: {result.adf_pvalue:.6f}")
        print(f"Original length: {len(result.original_series)}")
        print(f"Differenced length: {len(result.differenced_series)}")
        print("="*60)


def run_sample_fractional_differencing():
    """Run sample fractional differencing."""
    engine = FractionalDifferencingEngine()
    
    # Generate sample data with long memory
    np.random.seed(42)
    n = 500
    
    # Generate fractionally integrated series (simplified)
    # Start with random shocks
    shocks = np.random.normal(0, 0.01, n)
    
    # Cumulative sum with decay (simulating long memory)
    series = np.cumsum(shocks)
    series = pd.Series(series, index=pd.date_range("2022-01-01", periods=n, freq="D"))
    
    # Add trend
    series = series + np.linspace(0, 10, n)
    
    # Apply fractional differencing
    result = engine.apply_fractional_differencing(series)
    engine.print_result(result, "Sample Series")
    
    # Test with fixed d = 0.4
    result_fixed = engine.apply_fractional_differencing(series, fixed_d=0.4)
    engine.print_result(result_fixed, "Sample Series (fixed d=0.4)")
    
    # Apply to DataFrame
    df = pd.DataFrame({
        "price": series,
        "returns": series.pct_change().fillna(0),
        "volume": np.random.randint(100000, 200000, n)
    })
    
    results = engine.apply_to_dataframe(df, fixed_d=0.4)
    
    print("\nDataFrame Results:")
    for col, res in results.items():
        print(f"  {col}: d={res.d_estimate:.4f}, stationary={res.is_stationary}")
    
    return engine


if __name__ == "__main__":
    run_sample_fractional_differencing()
