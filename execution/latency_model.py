"""
Latency Model
Models execution latency based on colocation and infrastructure.

Critical for realistic backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ColocationTier(Enum):
    """Colocation tiers"""
    NONE = "none"  # No colocation (100ms+)
    BASIC = "basic"  # Basic colocation (50ms)
    PREMIUM = "premium"  # Premium colocation (20ms)
    EXCHANGE = "exchange"  # Exchange colocation (10ms)
    FPGA = "fpga"  # FPGA/ASIC (sub-microsecond)


class ExecutionStage(Enum):
    """Execution pipeline stages"""
    SIGNAL_GENERATION = "signal_generation"
    ORDER_ROUTING = "order_routing"
    EXCHANGE_PROCESSING = "exchange_processing"
    FILL_CONFIRMATION = "fill_confirmation"
    TOTAL = "total"


@dataclass
class LatencyConfig:
    """Configuration for latency model"""
    colocation_tier: ColocationTier = ColocationTier.BASIC
    base_latency_ms: float = 50.0  # Base latency without colocation
    network_latency_ms: float = 10.0  # Network latency
    processing_latency_ms: float = 5.0  # Local processing
    exchange_latency_ms: float = 20.0  # Exchange processing
    jitter_ms: float = 2.0  # Latency jitter (random variation)
    
    # Stage-specific latencies
    signal_gen_ms: float = 5.0
    order_routing_ms: float = 10.0
    exchange_processing_ms: float = 20.0
    fill_confirmation_ms: float = 5.0


@dataclass
class LatencyMeasurement:
    """Latency measurement for a single execution"""
    timestamp: datetime
    stage: ExecutionStage
    latency_ms: float
    expected_latency_ms: float
    deviation_ms: float


class LatencyModel:
    """
    Latency Model
    
    Models execution latency based on colocation tier and infrastructure.
    Accounts for jitter and random variation.
    
    Latency ranges:
    - No colocation: 100-200ms
    - Basic colocation: 50-100ms
    - Premium colocation: 20-50ms
    - Exchange colocation: 10-20ms
    - FPGA: <1ms
    """
    
    def __init__(self, config: LatencyConfig):
        self.config = config
        self.measurements: List[LatencyMeasurement] = []
        
        # Colocation multipliers
        self.colocation_multipliers = {
            ColocationTier.NONE: 2.0,
            ColocationTier.BASIC: 1.0,
            ColocationTier.PREMIUM: 0.4,
            ColocationTier.EXCHANGE: 0.2,
            ColocationTier.FPGA: 0.02
        }
    
    def get_expected_latency(self, stage: ExecutionStage = ExecutionStage.TOTAL) -> float:
        """
        Get expected latency for a stage.
        
        Args:
            stage: Execution stage
        
        Returns:
            Expected latency in milliseconds
        """
        multiplier = self.colocation_multipliers[self.config.colocation_tier]
        
        if stage == ExecutionStage.SIGNAL_GENERATION:
            return self.config.signal_gen_ms * multiplier
        elif stage == ExecutionStage.ORDER_ROUTING:
            return self.config.order_routing_ms * multiplier
        elif stage == ExecutionStage.EXCHANGE_PROCESSING:
            return self.config.exchange_processing_ms * multiplier
        elif stage == ExecutionStage.FILL_CONFIRMATION:
            return self.config.fill_confirmation_ms * multiplier
        elif stage == ExecutionStage.TOTAL:
            return (self.config.signal_gen_ms + self.config.order_routing_ms +
                   self.config.exchange_processing_ms + self.config.fill_confirmation_ms) * multiplier
        else:
            return self.config.base_latency_ms * multiplier
    
    def simulate_latency(self, stage: ExecutionStage = ExecutionStage.TOTAL) -> float:
        """
        Simulate latency with jitter.
        
        Args:
            stage: Execution stage
        
        Returns:
            Simulated latency in milliseconds
        """
        expected = self.get_expected_latency(stage)
        
        # Add jitter (random variation)
        jitter = np.random.normal(0, self.config.jitter_ms)
        
        # Ensure non-negative
        simulated = max(0, expected + jitter)
        
        # Record measurement
        measurement = LatencyMeasurement(
            timestamp=datetime.now(),
            stage=stage,
            latency_ms=simulated,
            expected_latency_ms=expected,
            deviation_ms=simulated - expected
        )
        self.measurements.append(measurement)
        
        return simulated
    
    def get_latency_distribution(self, stage: ExecutionStage = ExecutionStage.TOTAL,
                               n_samples: int = 1000) -> Dict:
        """
        Get latency distribution by simulating multiple samples.
        
        Args:
            stage: Execution stage
            n_samples: Number of samples to simulate
        
        Returns:
            Dictionary with distribution statistics
        """
        samples = [self.simulate_latency(stage) for _ in range(n_samples)]
        
        return {
            "mean_ms": np.mean(samples),
            "std_ms": np.std(samples),
            "min_ms": np.min(samples),
            "max_ms": np.max(samples),
            "p50_ms": np.percentile(samples, 50),
            "p95_ms": np.percentile(samples, 95),
            "p99_ms": np.percentile(samples, 99)
        }
    
    def get_p99_latency(self, stage: ExecutionStage = ExecutionStage.TOTAL) -> float:
        """Get P99 latency (99th percentile)"""
        dist = self.get_latency_distribution(stage, n_samples=1000)
        return dist["p99_ms"]
    
    def update_colocation(self, new_tier: ColocationTier):
        """Update colocation tier"""
        self.config.colocation_tier = new_tier
    
    def get_average_latency(self, stage: ExecutionStage = ExecutionStage.TOTAL,
                          n_recent: int = 100) -> Optional[float]:
        """Get average recent latency"""
        recent = [m for m in self.measurements if m.stage == stage][-n_recent:]
        if not recent:
            return None
        return np.mean([m.latency_ms for m in recent])
    
    def get_latency_breakdown(self) -> Dict[str, float]:
        """Get latency breakdown by stage"""
        return {
            "signal_generation_ms": self.get_expected_latency(ExecutionStage.SIGNAL_GENERATION),
            "order_routing_ms": self.get_expected_latency(ExecutionStage.ORDER_ROUTING),
            "exchange_processing_ms": self.get_expected_latency(ExecutionStage.EXCHANGE_PROCESSING),
            "fill_confirmation_ms": self.get_expected_latency(ExecutionStage.FILL_CONFIRMATION),
            "total_ms": self.get_expected_latency(ExecutionStage.TOTAL)
        }
    
    def generate_report(self) -> str:
        """Generate latency report"""
        breakdown = self.get_latency_breakdown()
        dist = self.get_latency_distribution()
        
        report = f"""
Latency Model Report
{'=' * 50}
Colocation Tier: {self.config.colocation_tier.value}
Base Latency: {self.config.base_latency_ms} ms
Jitter: {self.config.jitter_ms} ms

Latency Breakdown:
{'-' * 50}
Signal Generation: {breakdown['signal_generation_ms']:.2f} ms
Order Routing: {breakdown['order_routing_ms']:.2f} ms
Exchange Processing: {breakdown['exchange_processing_ms']:.2f} ms
Fill Confirmation: {breakdown['fill_confirmation_ms']:.2f} ms
Total: {breakdown['total_ms']:.2f} ms

Latency Distribution (1000 samples):
{'-' * 50}
Mean: {dist['mean_ms']:.2f} ms
Std: {dist['std_ms']:.2f} ms
Min: {dist['min_ms']:.2f} ms
Max: {dist['max_ms']:.2f} ms
P50: {dist['p50_ms']:.2f} ms
P95: {dist['p95_ms']:.2f} ms
P99: {dist['p99_ms']:.2f} ms

Total Measurements: {len(self.measurements)}
"""
        
        return report


def estimate_adverse_selection_cost(latency_ms: float, volatility_annual: float,
                                    price: float = 100.0) -> float:
    """
    Estimate adverse selection cost due to latency.
    
    Args:
        latency_ms: Execution latency in milliseconds
        volatility_annual: Annualized volatility
        price: Current price
    
    Returns:
        Estimated adverse selection cost in bps
    """
    # Convert latency to fraction of trading day
    latency_fraction = latency_ms / (6.5 * 60 * 60 * 1000)  # 6.5 hour trading day
    
    # Volatility per unit time
    vol_per_fraction = volatility_annual * np.sqrt(latency_fraction / 252)
    
    # Adverse selection cost (simplified)
    cost_bps = vol_per_fraction * 10000 * 0.5  # Half the volatility move
    
    return cost_bps


if __name__ == "__main__":
    # Example usage
    config = LatencyConfig(colocation_tier=ColocationTier.BASIC)
    model = LatencyModel(config)
    
    print("Latency Model Report:")
    print(model.generate_report())
    
    # Test different colocation tiers
    print("\n=== Colocation Comparison ===")
    for tier in ColocationTier:
        model.update_colocation(tier)
        total_latency = model.get_expected_latency()
        p99_latency = model.get_p99_latency()
        print(f"{tier.value:15s}: {total_latency:.2f} ms (P99: {p99_latency:.2f} ms)")
    
    # Estimate adverse selection cost
    print("\n=== Adverse Selection Cost ===")
    for tier in ColocationTier:
        model.update_colocation(tier)
        latency = model.get_expected_latency()
        cost = estimate_adverse_selection_cost(latency, volatility_annual=0.20)
        print(f"{tier.value:15s}: {cost:.2f} bps adverse selection cost")
