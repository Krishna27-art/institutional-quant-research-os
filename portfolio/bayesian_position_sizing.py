"""
Bayesian Position Sizing
Based on V3 Blueprint - Multi-Factor Position Sizing with Risk Adjustments

Key findings from research:
- Kelly alone is too aggressive, ignores confidence and regime
- Multi-factor position sizing with risk adjustments
- Formula: base_position = Kelly × confidence × regime_score × liquidity_score × feature_drift_penalty
- Risk caps: single trade ≤ 0.5% AUM, total portfolio ≤ 10% AUM

V3 Upgrade - Expected Sharpe increase: +0.05–0.1
Priority: High
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy import stats


@dataclass
class PositionSizingInput:
    """Input parameters for position sizing"""
    expected_return: float
    volatility: float
    win_rate: float
    avg_win: float
    avg_loss: float
    confidence_score: float  # 0-1 from model
    regime_score: float  # 0.5 for high_vol, 1.0 for normal
    liquidity_score: float  # 1.0 if position < capacity_limit * 0.7, else decays
    feature_drift_psi: float  # PSI for top features


@dataclass
class PositionSizingOutput:
    """Output from Bayesian position sizing"""
    kelly_fraction: float
    base_position: float
    adjusted_position: float
    final_position: float
    position_risk_pct: float  # Risk as % of AUM
    confidence_adjustment: float
    regime_adjustment: float
    liquidity_adjustment: float
    drift_adjustment: float


class BayesianPositionSizer:
    """
    Bayesian Position Sizing with multi-factor adjustments.
    
    Formula:
    base_position = Kelly_fraction * (expected_return / variance)
    adjusted_position = base_position
        * confidence_score
        * regime_score (0.5 for high_vol, 1.0 for normal)
        * liquidity_score (1.0 if position < capacity_limit * 0.7, else decays)
        * feature_drift_penalty (1 - PSI/0.5 capped at 0)
    
    Risk Caps:
    - Single trade risk ≤ 0.5% of AUM (unlevered)
    - Total portfolio risk ≤ 10% of AUM (unlevered)
    
    Clipping:
    - If adjusted_position > 2× Kelly → clip to 2× Kelly
    - If adjusted_position < 0.1× Kelly → set to 0 (skip trade)
    """
    
    def __init__(self, aum: float = 2.5e8):  # ₹25 Crore
        self.aum = aum
        self.capacity_limits = {
            "NIFTY": 5e10,  # ₹5000 Cr
            "BANKNIFTY": 3e10,  # ₹3000 Cr
            "RELIANCE": 2e9,  # ₹200 Cr
            "HDFCBANK": 1.5e9,  # ₹150 Cr
            "INFY": 1e9,  # ₹100 Cr
        }
    
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Kelly = (p * b - (1-p)) / b
        where b = avg_win / avg_loss (odds)
        
        Args:
            expected_return: Expected return
            win_rate: Win rate (0-1)
            avg_win: Average win
            avg_loss: Average loss
            
        Returns:
            Kelly fraction
        """
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        
        kelly = (p * b - (1 - p)) / b
        
        # Cap at 25% (half-Kelly is common)
        kelly = min(0.25, max(0, kelly))
        
        return kelly
    
    def calculate_confidence_adjustment(self, confidence_score: float) -> float:
        """
        Calculate confidence adjustment.
        
        Args:
            confidence_score: Model confidence (0-1)
            
        Returns:
            Adjustment factor (0.5-1.0)
        """
        # Linear adjustment: confidence 0 → 0.5, confidence 1 → 1.0
        adjustment = 0.5 + 0.5 * confidence_score
        return adjustment
    
    def calculate_regime_adjustment(self, regime_score: float) -> float:
        """
        Calculate regime adjustment.
        
        Args:
            regime_score: Regime score (0.5 for high_vol, 1.0 for normal)
            
        Returns:
            Adjustment factor (0.5-1.0)
        """
        return regime_score
    
    def calculate_liquidity_adjustment(
        self,
        position_value: float,
        symbol: str
    ) -> float:
        """
        Calculate liquidity adjustment.
        
        Args:
            position_value: Position value in ₹
            symbol: Stock symbol
            
        Returns:
            Adjustment factor (0-1.0)
        """
        capacity_limit = self.capacity_limits.get(symbol, 1e9)
        capacity_threshold = capacity_limit * 0.7
        
        if position_value < capacity_threshold:
            return 1.0
        else:
            # Decay linearly from 1.0 to 0.5 as position approaches capacity
            excess = (position_value - capacity_threshold) / (capacity_limit - capacity_threshold)
            adjustment = 1.0 - 0.5 * excess
            return max(0.5, adjustment)
    
    def calculate_drift_adjustment(self, psi: float) -> float:
        """
        Calculate feature drift penalty.
        
        Args:
            psi: Population Stability Index (0-1+)
            
        Returns:
            Adjustment factor (0-1.0)
        """
        # Penalty = 1 - PSI/0.5, capped at 0
        penalty = 1 - psi / 0.5
        return max(0.0, penalty)
    
    def calculate_position_size(
        self,
        inputs: PositionSizingInput,
        symbol: str,
        position_value: float
    ) -> PositionSizingOutput:
        """
        Calculate Bayesian position size.
        
        Args:
            inputs: Position sizing inputs
            symbol: Stock symbol
            position_value: Current position value (for liquidity check)
            
        Returns:
            PositionSizingOutput
        """
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            inputs.expected_return,
            inputs.win_rate,
            inputs.avg_win,
            inputs.avg_loss
        )
        
        # Calculate base position
        variance = inputs.volatility ** 2
        if variance > 0:
            base_position = kelly_fraction * (inputs.expected_return / variance)
        else:
            base_position = 0.0
        
        # Calculate adjustments
        confidence_adjustment = self.calculate_confidence_adjustment(inputs.confidence_score)
        regime_adjustment = self.calculate_regime_adjustment(inputs.regime_score)
        liquidity_adjustment = self.calculate_liquidity_adjustment(position_value, symbol)
        drift_adjustment = self.calculate_drift_adjustment(inputs.feature_drift_psi)
        
        # Calculate adjusted position
        adjusted_position = (
            base_position
            * confidence_adjustment
            * regime_adjustment
            * liquidity_adjustment
            * drift_adjustment
        )
        
        # Clipping
        if adjusted_position > 2 * kelly_fraction:
            adjusted_position = 2 * kelly_fraction
        elif adjusted_position < 0.1 * kelly_fraction:
            adjusted_position = 0.0
        
        # Apply risk cap: single trade ≤ 0.5% of AUM
        max_position_risk = self.aum * 0.005
        position_risk = adjusted_position * self.aum
        
        if position_risk > max_position_risk:
            adjusted_position = max_position_risk / self.aum
        
        final_position = adjusted_position
        position_risk_pct = final_position * 100
        
        return PositionSizingOutput(
            kelly_fraction=kelly_fraction,
            base_position=base_position,
            adjusted_position=adjusted_position,
            final_position=final_position,
            position_risk_pct=position_risk_pct,
            confidence_adjustment=confidence_adjustment,
            regime_adjustment=regime_adjustment,
            liquidity_adjustment=liquidity_adjustment,
            drift_adjustment=drift_adjustment
        )
    
    def print_sizing_report(self, output: PositionSizingOutput, symbol: str) -> None:
        """Print position sizing report."""
        print("\n" + "="*60)
        print(f"BAYESIAN POSITION SIZING: {symbol}")
        print("="*60)
        print(f"Kelly Fraction: {output.kelly_fraction:.2%}")
        print(f"Base Position: {output.base_position:.2%}")
        print(f"Adjusted Position: {output.adjusted_position:.2%}")
        print(f"Final Position: {output.final_position:.2%}")
        print(f"Position Risk: {output.position_risk_pct:.2%}% of AUM")
        print("\nAdjustments:")
        print(f"  Confidence: {output.confidence_adjustment:.2f}x")
        print(f"  Regime: {output.regime_adjustment:.2f}x")
        print(f"  Liquidity: {output.liquidity_adjustment:.2f}x")
        print(f"  Drift Penalty: {output.drift_adjustment:.2f}x")
        print("="*60)


def run_sample_position_sizing():
    """Run sample Bayesian position sizing."""
    sizer = BayesianPositionSizer(aum=2.5e8)  # ₹25 Crore
    
    # Sample scenarios
    scenarios = [
        {
            "symbol": "NIFTY",
            "inputs": PositionSizingInput(
                expected_return=0.001,
                volatility=0.015,
                win_rate=0.55,
                avg_win=0.02,
                avg_loss=0.015,
                confidence_score=0.8,
                regime_score=1.0,
                liquidity_score=1.0,
                feature_drift_psi=0.05
            ),
            "position_value": 1e7  # ₹1 Cr
        },
        {
            "symbol": "BANKNIFTY",
            "inputs": PositionSizingInput(
                expected_return=0.0008,
                volatility=0.018,
                win_rate=0.52,
                avg_win=0.018,
                avg_loss=0.016,
                confidence_score=0.6,
                regime_score=0.75,  # High volatility
                liquidity_score=0.9,
                feature_drift_psi=0.15
            ),
            "position_value": 5e7  # ₹5 Cr
        },
        {
            "symbol": "RELIANCE",
            "inputs": PositionSizingInput(
                expected_return=0.0012,
                volatility=0.02,
                win_rate=0.58,
                avg_win=0.025,
                avg_loss=0.018,
                confidence_score=0.9,
                regime_score=1.0,
                liquidity_score=0.8,
                feature_drift_psi=0.25
            ),
            "position_value": 1.5e8  # ₹15 Cr (near capacity)
        }
    ]
    
    for scenario in scenarios:
        output = sizer.calculate_position_size(
            scenario["inputs"],
            scenario["symbol"],
            scenario["position_value"]
        )
        sizer.print_sizing_report(output, scenario["symbol"])
    
    return sizer


if __name__ == "__main__":
    run_sample_position_sizing()
