"""
Execution Alpha: TWAP/VWAP Optimization

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#26)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- TWAP (Time-Weighted Average Price) optimization
- VWAP (Volume-Weighted Average Price) optimization
- Minimizes market impact
- Optimizes execution schedule
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class ExecutionAlgorithm(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "is"


@dataclass
class ExecutionConfig:
    """Configuration for Execution Optimization"""
    # Order parameters
    total_quantity: int = 10000
    side: str = "buy"  # "buy" or "sell"
    
    # Time parameters
    start_time: datetime = None
    end_time: datetime = None
    execution_minutes: int = 60  # Total execution window in minutes
    
    # Market impact parameters
    impact_alpha: float = 0.1  # Linear impact coefficient
    impact_beta: float = 0.5  # Nonlinear impact exponent
    avg_daily_volume: int = 1000000  # Average daily volume
    
    # Risk parameters
    risk_aversion: float = 0.5  # Risk aversion parameter
    volatility: float = 0.02  # Volatility
    
    # Constraints
    max_participation_rate: float = 0.2  # Maximum 20% participation
    min_slice_size: int = 100  # Minimum slice size
    
    # Algorithm selection
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.VWAP


class TWAPOptimizer:
    """
    TWAP (Time-Weighted Average Price) Optimizer
    
    Executes orders evenly over time to minimize market impact.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
    
    def generate_schedule(self) -> pd.DataFrame:
        """
        Generate TWAP execution schedule
        
        Returns:
            DataFrame with execution schedule
        """
        n_intervals = self.config.execution_minutes
        quantity_per_interval = self.config.total_quantity / n_intervals
        
        schedule = []
        current_time = self.config.start_time or datetime.now()
        time_delta = timedelta(minutes=1)
        
        for i in range(n_intervals):
            schedule.append({
                "time": current_time + timedelta(minutes=i),
                "quantity": int(quantity_per_interval),
                "cumulative_quantity": int(quantity_per_interval * (i + 1))
            })
        
        return pd.DataFrame(schedule)
    
    def optimize_with_volume_profile(self, volume_profile: pd.Series) -> pd.DataFrame:
        """
        Optimize TWAP with volume profile
        
        Args:
            volume_profile: Volume profile (time -> volume)
            
        Returns:
            Optimized schedule
        """
        n_intervals = self.config.execution_minutes
        
        # Normalize volume profile
        total_volume = volume_profile.sum()
        volume_weights = volume_profile / total_volume
        
        # Allocate quantity based on volume
        schedule = []
        current_time = self.config.start_time or datetime.now()
        
        cumulative_qty = 0
        for i in range(n_intervals):
            if i < len(volume_weights):
                qty = int(self.config.total_quantity * volume_weights.iloc[i])
            else:
                qty = int(self.config.total_quantity / n_intervals)
            
            cumulative_qty += qty
            schedule.append({
                "time": current_time + timedelta(minutes=i),
                "quantity": qty,
                "cumulative_quantity": cumulative_qty
            })
        
        return pd.DataFrame(schedule)


class VAPOptimizer:
    """
    VWAP (Volume-Weighted Average Price) Optimizer
    
    Executes orders proportional to historical volume patterns.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
    
    def generate_schedule(self, volume_profile: pd.Series) -> pd.DataFrame:
        """
        Generate VWAP execution schedule
        
        Args:
            volume_profile: Historical volume profile (time -> volume)
            
        Returns:
            DataFrame with execution schedule
        """
        # Normalize volume profile
        total_volume = volume_profile.sum()
        volume_weights = volume_profile / total_volume
        
        # Generate schedule
        schedule = []
        current_time = self.config.start_time or datetime.now()
        
        cumulative_qty = 0
        for i, (time_idx, volume) in enumerate(volume_profile.items()):
            qty = int(self.config.total_quantity * volume_weights.iloc[i] if hasattr(volume_weights, 'iloc') else volume_weights[i])
            cumulative_qty += qty
            
            schedule.append({
                "time": current_time + timedelta(minutes=i),
                "quantity": qty,
                "volume": volume,
                "cumulative_quantity": cumulative_qty
            })
        
        return pd.DataFrame(schedule)
    
    def optimize_with_impact(self, volume_profile: pd.Series) -> pd.DataFrame:
        """
        Optimize VWAP considering market impact
        
        Args:
            volume_profile: Volume profile
            
        Returns:
            Optimized schedule
        """
        n_intervals = len(volume_profile)
        
        # Calculate optimal allocation considering impact
        # Minimize: impact_cost + risk_cost
        # Simplified: allocate inversely proportional to volume to reduce impact
        inv_volume = 1.0 / (volume_profile + 1e-8)
        inv_weights = inv_volume / inv_volume.sum()
        
        schedule = []
        current_time = self.config.start_time or datetime.now()
        
        cumulative_qty = 0
        for i, (time_idx, volume) in enumerate(volume_profile.items()):
            qty = int(self.config.total_quantity * inv_weights.iloc[i] if hasattr(inv_weights, 'iloc') else inv_weights[i])
            cumulative_qty += qty
            
            schedule.append({
                "time": current_time + timedelta(minutes=i),
                "quantity": qty,
                "volume": volume,
                "cumulative_quantity": cumulative_qty
            })
        
        return pd.DataFrame(schedule)


class ExecutionOptimizer:
    """
    Execution Optimizer
    
    Combines TWAP, VWAP, and other algorithms for optimal execution.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        
        self.twap_optimizer = TWAPOptimizer(config)
        self.vwap_optimizer = VAPOptimizer(config)
        
        # Performance tracking
        self.execution_history: List[Dict] = []
    
    def execute(self, volume_profile: Optional[pd.Series] = None) -> Dict:
        """
        Execute order using selected algorithm
        
        Args:
            volume_profile: Volume profile (required for VWAP)
            
        Returns:
            Execution results
        """
        if self.config.algorithm == ExecutionAlgorithm.TWAP:
            if volume_profile is not None:
                schedule = self.twap_optimizer.optimize_with_volume_profile(volume_profile)
            else:
                schedule = self.twap_optimizer.generate_schedule()
        elif self.config.algorithm == ExecutionAlgorithm.VWAP:
            if volume_profile is None:
                volume_profile = self._generate_default_volume_profile()
            schedule = self.vwap_optimizer.generate_schedule(volume_profile)
        else:
            # Default to TWAP
            schedule = self.twap_optimizer.generate_schedule()
        
        # Calculate expected costs
        impact_cost = self._calculate_impact_cost(schedule)
        risk_cost = self._calculate_risk_cost(schedule)
        
        result = {
            "algorithm": self.config.algorithm.value,
            "schedule": schedule,
            "total_quantity": schedule["quantity"].sum(),
            "impact_cost": impact_cost,
            "risk_cost": risk_cost,
            "total_cost": impact_cost + risk_cost
        }
        
        self.execution_history.append(result)
        
        return result
    
    def _generate_default_volume_profile(self) -> pd.Series:
        """Generate default volume profile (U-shaped)"""
        n_intervals = self.config.execution_minutes
        
        # U-shaped volume profile (more volume at start and end)
        times = np.arange(n_intervals)
        volume = 1.0 + 0.5 * np.sin(2 * np.pi * times / n_intervals)
        volume = volume / volume.sum()
        
        return pd.Series(volume)
    
    def _calculate_impact_cost(self, schedule: pd.DataFrame) -> float:
        """Calculate market impact cost"""
        total_cost = 0.0
        
        for _, row in schedule.iterrows():
            quantity = row["quantity"]
            
            # Market impact model
            participation_rate = quantity / self.config.avg_daily_volume
            impact = self.config.impact_alpha * (participation_rate ** self.config.impact_beta)
            
            total_cost += impact * quantity
        
        return total_cost
    
    def _calculate_risk_cost(self, schedule: pd.DataFrame) -> float:
        """Calculate risk cost (price uncertainty)"""
        # Simplified: risk proportional to square of execution time
        n_intervals = len(schedule)
        risk_cost = self.config.risk_aversion * self.config.volatility ** 2 * n_intervals
        
        return risk_cost
    
    def compare_algorithms(self, volume_profile: pd.Series) -> Dict:
        """
        Compare different execution algorithms
        
        Args:
            volume_profile: Volume profile
            
        Returns:
            Comparison results
        """
        results = {}
        
        # Test TWAP
        self.config.algorithm = ExecutionAlgorithm.TWAP
        twap_result = self.execute(volume_profile)
        results["TWAP"] = twap_result["total_cost"]
        
        # Test VWAP
        self.config.algorithm = ExecutionAlgorithm.VWAP
        vwap_result = self.execute(volume_profile)
        results["VWAP"] = vwap_result["total_cost"]
        
        # Find best
        best_algorithm = min(results, key=results.get)
        
        return {
            "comparison": results,
            "best_algorithm": best_algorithm,
            "best_cost": results[best_algorithm]
        }
    
    def get_execution_summary(self) -> Dict:
        """Get execution summary"""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        avg_cost = np.mean([e["total_cost"] for e in self.execution_history])
        
        return {
            "total_executions": total_executions,
            "average_cost": avg_cost,
            "algorithm_distribution": {
                alg: sum(1 for e in self.execution_history if e["algorithm"] == alg)
                for alg in ["twap", "vwap", "pov", "is"]
            }
        }


def simulate_volume_profile(n_minutes: int = 60) -> pd.Series:
    """Simulate intraday volume profile"""
    np.random.seed(42)
    
    # U-shaped pattern with noise
    times = np.arange(n_minutes)
    base_volume = 1.0 + 0.5 * np.sin(2 * np.pi * times / n_minutes)
    noise = np.random.randn(n_minutes) * 0.2
    volume = base_volume + noise
    volume = np.maximum(volume, 0.1)  # Ensure positive
    
    return pd.Series(volume)


if __name__ == "__main__":
    # Example usage
    config = ExecutionConfig(
        total_quantity=10000,
        side="buy",
        execution_minutes=60,
        algorithm=ExecutionAlgorithm.VWAP,
        avg_daily_volume=1000000
    )
    
    optimizer = ExecutionOptimizer(config)
    
    # Simulate volume profile
    print("Simulating volume profile...")
    volume_profile = simulate_volume_profile(60)
    
    # Execute with VWAP
    print("\nExecuting with VWAP...")
    result = optimizer.execute(volume_profile)
    
    print(f"\nExecution Results:")
    print(f"  Algorithm: {result['algorithm']}")
    print(f"  Total Quantity: {result['total_quantity']}")
    print(f"  Impact Cost: {result['impact_cost']:.4f}")
    print(f"  Risk Cost: {result['risk_cost']:.4f}")
    print(f"  Total Cost: {result['total_cost']:.4f}")
    
    print(f"\nExecution Schedule (first 10 intervals):")
    print(result['schedule'].head(10).to_string())
    
    # Compare algorithms
    print("\nComparing algorithms...")
    comparison = optimizer.compare_algorithms(volume_profile)
    
    print(f"\nAlgorithm Comparison:")
    for alg, cost in comparison["comparison"].items():
        print(f"  {alg}: {cost:.4f}")
    
    print(f"\nBest Algorithm: {comparison['best_algorithm']} (Cost: {comparison['best_cost']:.4f})")
    
    # Execution summary
    print("\nExecution Summary:")
    summary = optimizer.get_execution_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
