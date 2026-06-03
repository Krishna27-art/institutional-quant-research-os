"""
Factor Research Platform: Discovery, Library, Validation, Decay, Capacity
Based on the critique: Build Factor Research Platform for systematic factor discovery

Need:
- Factor Discovery
- Factor Library
- Factor Validation
- Factor Decay
- Factor Capacity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.linear_model import LinearRegression


class FactorStatus(Enum):
    """Status of factor."""
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DECAYED = "decayed"
    RETIRED = "retired"


@dataclass
class FactorDefinition:
    """Factor definition."""
    factor_id: str
    name: str
    description: str
    category: str
    data_source: str
    calculation_method: str
    created_at: datetime


@dataclass
class FactorValidationResult:
    """Factor validation result."""
    factor_id: str
    timestamp: datetime
    sharpe_ratio: float
    t_statistic: float
    p_value: float
    information_ratio: float
    max_drawdown: float
    turnover: float
    is_significant: bool
    validation_status: FactorStatus


@dataclass
class FactorDecayMetrics:
    """Factor decay metrics."""
    factor_id: str
    timestamp: datetime
    initial_sharpe: float
    current_sharpe: float
    decay_rate: float
    half_life_days: float
    decay_stage: str


@dataclass
class FactorCapacityMetrics:
    """Factor capacity metrics."""
    factor_id: str
    timestamp: datetime
    max_capacity: float
    current_aum: float
    capacity_utilization: float
    slippage_at_current: float
    slippage_at_max: float
    is_at_capacity: bool


class FactorResearchPlatform:
    """
    Factor Research Platform for systematic factor discovery and validation.
    
    Features:
    - Factor Discovery
    - Factor Library
    - Factor Validation
    - Factor Decay
    - Factor Capacity
    """
    
    def __init__(self):
        self.factor_library: Dict[str, FactorDefinition] = {}
        self.validation_results: Dict[str, List[FactorValidationResult]] = {}
        self.decay_metrics: Dict[str, List[FactorDecayMetrics]] = {}
        self.capacity_metrics: Dict[str, List[FactorCapacityMetrics]] = {}
        
        # Validation thresholds
        self.min_sharpe = 0.5
        self.min_t_stat = 2.0
        self.max_turnover = 0.5
        self.max_drawdown = 0.2
    
    def discover_factor(
        self,
        name: str,
        description: str,
        category: str,
        data_source: str,
        calculation_method: str
    ) -> FactorDefinition:
        """
        Discover a new factor.
        
        Args:
            name: Factor name
            description: Factor description
            category: Factor category
            data_source: Data source
            calculation_method: Calculation method
            
        Returns:
            FactorDefinition
        """
        factor_id = f"factor_{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        
        factor = FactorDefinition(
            factor_id=factor_id,
            name=name,
            description=description,
            category=category,
            data_source=data_source,
            calculation_method=calculation_method,
            created_at=datetime.now()
        )
        
        self.factor_library[factor_id] = factor
        return factor
    
    def validate_factor(
        self,
        factor_id: str,
        factor_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> FactorValidationResult:
        """
        Validate a factor.
        
        Args:
            factor_id: Factor ID
            factor_returns: Factor returns
            benchmark_returns: Benchmark returns
            
        Returns:
            FactorValidationResult
        """
        # Align returns
        aligned = pd.concat([factor_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned) < 60:
            return FactorValidationResult(
                factor_id=factor_id,
                timestamp=datetime.now(),
                sharpe_ratio=0.0,
                t_statistic=0.0,
                p_value=1.0,
                information_ratio=0.0,
                max_drawdown=0.0,
                turnover=0.0,
                is_significant=False,
                validation_status=FactorStatus.DISCOVERED
            )
        
        factor_ret = aligned.iloc[:, 0]
        benchmark_ret = aligned.iloc[:, 1]
        
        # Calculate Sharpe ratio
        sharpe_ratio = factor_ret.mean() / factor_ret.std() * np.sqrt(252) if factor_ret.std() > 0 else 0
        
        # Calculate t-statistic
        t_statistic = factor_ret.mean() / (factor_ret.std() / np.sqrt(len(factor_ret))) if factor_ret.std() > 0 else 0
        
        # Calculate p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), len(factor_ret) - 1))
        
        # Calculate Information Ratio
        excess_returns = factor_ret - benchmark_ret
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + excess_returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        
        # Calculate turnover (simplified)
        turnover = 0.1  # Placeholder
        
        # Check if significant
        is_significant = (
            sharpe_ratio >= self.min_sharpe and
            abs(t_statistic) >= self.min_t_stat and
            p_value < 0.05 and
            abs(max_dd) <= self.max_drawdown
        )
        
        # Determine validation status
        if is_significant:
            validation_status = FactorStatus.VALIDATED
        else:
            validation_status = FactorStatus.DISCOVERED
        
        result = FactorValidationResult(
            factor_id=factor_id,
            timestamp=datetime.now(),
            sharpe_ratio=sharpe_ratio,
            t_statistic=t_statistic,
            p_value=p_value,
            information_ratio=information_ratio,
            max_drawdown=max_dd,
            turnover=turnover,
            is_significant=is_significant,
            validation_status=validation_status
        )
        
        # Store result
        if factor_id not in self.validation_results:
            self.validation_results[factor_id] = []
        self.validation_results[factor_id].append(result)
        
        return result
    
    def calculate_factor_decay(
        self,
        factor_id: str,
        returns: pd.Series,
        window_days: int = 60
    ) -> FactorDecayMetrics:
        """
        Calculate factor decay metrics.
        
        Args:
            factor_id: Factor ID
            returns: Factor returns
            window_days: Window for rolling Sharpe
            
        Returns:
            FactorDecayMetrics
        """
        if len(returns) < window_days * 2:
            return FactorDecayMetrics(
                factor_id=factor_id,
                timestamp=datetime.now(),
                initial_sharpe=0.0,
                current_sharpe=0.0,
                decay_rate=0.0,
                half_life_days=float('inf'),
                decay_stage="unknown"
            )
        
        # Calculate initial Sharpe (first window)
        initial_returns = returns.iloc[:window_days]
        initial_sharpe = initial_returns.mean() / initial_returns.std() * np.sqrt(252) if initial_returns.std() > 0 else 0
        
        # Calculate current Sharpe (last window)
        current_returns = returns.iloc[-window_days:]
        current_sharpe = current_returns.mean() / current_returns.std() * np.sqrt(252) if current_returns.std() > 0 else 0
        
        # Calculate decay rate
        rolling_sharpe = returns.rolling(window_days).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Fit exponential decay
        x = np.arange(len(rolling_sharpe))
        y = rolling_sharpe.dropna().values
        
        if len(y) < 10:
            decay_rate = 0.0
        else:
            y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
            
            def exp_decay(x, a, b):
                return a * np.exp(b * x)
            
            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(exp_decay, x, y_norm, p0=[1.0, -0.01], maxfev=10000)
                decay_rate = popt[1]
            except:
                decay_rate = 0.0
        
        # Calculate half-life
        if decay_rate < 0:
            half_life_days = np.log(0.5) / decay_rate
        else:
            half_life_days = float('inf')
        
        # Determine decay stage
        sharpe_ratio = current_sharpe / initial_sharpe if initial_sharpe > 0 else 0
        
        if decay_rate > 0:
            decay_stage = "growth"
        elif sharpe_ratio > 0.8:
            decay_stage = "maturity"
        elif sharpe_ratio > 0.5:
            decay_stage = "early_decay"
        elif sharpe_ratio > 0.2:
            decay_stage = "late_decay"
        else:
            decay_stage = "death"
        
        metrics = FactorDecayMetrics(
            factor_id=factor_id,
            timestamp=datetime.now(),
            initial_sharpe=initial_sharpe,
            current_sharpe=current_sharpe,
            decay_rate=decay_rate,
            half_life_days=half_life_days,
            decay_stage=decay_stage
        )
        
        # Store metrics
        if factor_id not in self.decay_metrics:
            self.decay_metrics[factor_id] = []
        self.decay_metrics[factor_id].append(metrics)
        
        return metrics
    
    def calculate_factor_capacity(
        self,
        factor_id: str,
        avg_daily_volume: float,
        avg_trade_size: float
    ) -> FactorCapacityMetrics:
        """
        Calculate factor capacity metrics.
        
        Args:
            factor_id: Factor ID
            avg_daily_volume: Average daily volume
            avg_trade_size: Average trade size
            
        Returns:
            FactorCapacityMetrics
        """
        # Estimate capacity based on volume participation
        # Assume max 1% of daily volume without significant impact
        max_capacity = avg_daily_volume * 0.01 * 252  # Annualized
        
        # Current AUM (simplified)
        current_aum = avg_trade_size * 252  # Annualized
        
        # Capacity utilization
        capacity_utilization = min(current_aum / max_capacity, 1.0)
        
        # Estimate slippage at current size
        slippage_current = 0.001 * np.sqrt(capacity_utilization)
        
        # Estimate slippage at max capacity
        slippage_max = 0.001 * np.sqrt(1.0)
        
        # Check if at capacity
        is_at_capacity = capacity_utilization > 0.8
        
        metrics = FactorCapacityMetrics(
            factor_id=factor_id,
            timestamp=datetime.now(),
            max_capacity=max_capacity,
            current_aum=current_aum,
            capacity_utilization=capacity_utilization,
            slippage_at_current=slippage_current,
            slippage_at_max=slippage_max,
            is_at_capacity=is_at_capacity
        )
        
        # Store metrics
        if factor_id not in self.capacity_metrics:
            self.capacity_metrics[factor_id] = []
        self.capacity_metrics[factor_id].append(metrics)
        
        return metrics
    
    def get_factor_library(self) -> pd.DataFrame:
        """Get factor library."""
        data = []
        
        for factor_id, factor in self.factor_library.items():
            data.append({
                'Factor ID': factor_id,
                'Name': factor.name,
                'Description': factor.description,
                'Category': factor.category,
                'Data Source': factor.data_source,
                'Created': factor.created_at.strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(data)
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get validation summary for all factors."""
        data = []
        
        for factor_id, results in self.validation_results.items():
            latest = results[-1]
            data.append({
                'Factor ID': factor_id,
                'Sharpe': latest.sharpe_ratio,
                'T-Stat': latest.t_statistic,
                'P-Value': latest.p_value,
                'IR': latest.information_ratio,
                'Max DD': latest.max_drawdown,
                'Significant': latest.is_significant,
                'Status': latest.validation_status.value
            })
        
        return pd.DataFrame(data)
    
    def get_decay_summary(self) -> pd.DataFrame:
        """Get decay summary for all factors."""
        data = []
        
        for factor_id, metrics_list in self.decay_metrics.items():
            latest = metrics_list[-1]
            data.append({
                'Factor ID': factor_id,
                'Initial Sharpe': latest.initial_sharpe,
                'Current Sharpe': latest.current_sharpe,
                'Decay Rate': latest.decay_rate,
                'Half-Life (days)': latest.half_life_days,
                'Stage': latest.decay_stage
            })
        
        return pd.DataFrame(data)
    
    def get_capacity_summary(self) -> pd.DataFrame:
        """Get capacity summary for all factors."""
        data = []
        
        for factor_id, metrics_list in self.capacity_metrics.items():
            latest = metrics_list[-1]
            data.append({
                'Factor ID': factor_id,
                'Max Capacity': latest.max_capacity,
                'Current AUM': latest.current_aum,
                'Utilization': latest.capacity_utilization,
                'Slippage Current': latest.slippage_at_current,
                'Slippage at Max': latest.slippage_at_max,
                'At Capacity': latest.is_at_capacity
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the Factor Research Platform
    print("Testing Factor Research Platform: Discovery, Library, Validation, Decay, Capacity...")
    
    platform = FactorResearchPlatform()
    
    # Discover factors
    print("\nDiscovering factors...")
    factor1 = platform.discover_factor(
        name="Momentum",
        description="12-month price momentum",
        category="momentum",
        data_source="price_data",
        calculation_method="12-month_return"
    )
    
    factor2 = platform.discover_factor(
        name="Value",
        description="Book-to-market ratio",
        category="value",
        data_source="fundamental_data",
        calculation_method="book_to_market"
    )
    
    factor3 = platform.discover_factor(
        name="Low Volatility",
        description="Low volatility stocks",
        category="volatility",
        data_source="price_data",
        calculation_method="rolling_volatility"
    )
    
    print(f"Discovered {len(platform.factor_library)} factors")
    
    # Validate factors
    print("\nValidating factors...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    for factor_id, factor in platform.factor_library.items():
        factor_returns = pd.Series(np.random.normal(0.0005, 0.015, n), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.012, n), index=dates)
        
        result = platform.validate_factor(factor_id, factor_returns, benchmark_returns)
        print(f"{factor.name}: Sharpe={result.sharpe_ratio:.2f}, Significant={result.is_significant}")
    
    # Calculate factor decay
    print("\nCalculating factor decay...")
    for factor_id in platform.factor_library.keys():
        returns = pd.Series(np.random.normal(0.0005, 0.015, 500))
        decay_metrics = platform.calculate_factor_decay(factor_id, returns)
        print(f"{factor_id}: Stage={decay_metrics.decay_stage}, Half-Life={decay_metrics.half_life_days:.0f} days")
    
    # Calculate factor capacity
    print("\nCalculating factor capacity...")
    for factor_id in platform.factor_library.keys():
        capacity_metrics = platform.calculate_factor_capacity(factor_id, avg_daily_volume=100000000, avg_trade_size=1000000)
        print(f"{factor_id}: Utilization={capacity_metrics.capacity_utilization:.2%}, At Capacity={capacity_metrics.is_at_capacity}")
    
    # Get summaries
    print("\nFactor Library:")
    library = platform.get_factor_library()
    print(library.to_string(index=False))
    
    print("\nValidation Summary:")
    validation_summary = platform.get_validation_summary()
    print(validation_summary.to_string(index=False))
    
    print("\nDecay Summary:")
    decay_summary = platform.get_decay_summary()
    print(decay_summary.to_string(index=False))
    
    print("\nCapacity Summary:")
    capacity_summary = platform.get_capacity_summary()
    print(capacity_summary.to_string(index=False))
