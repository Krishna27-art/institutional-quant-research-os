"""
Triple Barrier Labelling for ML Training

Based on machine-learning-for-trading (Stefan Jansen) Chapter 4.
This implements the correct way to label financial time series for ML models.

Key Concepts:
- Instead of hold-forever labels, use triple barrier method
- Upper barrier: take-profit target
- Lower barrier: stop-loss
- Time barrier: maximum holding period
- Label is whichever barrier is hit first

This prevents the common mistake of training models on hold-forever returns
which don't reflect actual trading behavior.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BarrierType(Enum):
    """Which barrier was hit"""
    UPPER = "upper"  # Take-profit hit
    LOWER = "lower"  # Stop-loss hit
    TIME = "time"    # Time barrier hit (expired)
    NONE = "none"    # No barrier hit yet


@dataclass
class TripleBarrierLabel:
    """Result of triple barrier labelling"""
    label: int  # 1 for upper (profit), -1 for lower (loss), 0 for time (neutral)
    barrier_type: BarrierType
    hit_time: Optional[pd.Timestamp] = None
    return_at_hit: Optional[float] = None
    holding_period: Optional[int] = None  # Number of bars held


class TripleBarrierLabeller:
    """
    Triple Barrier Labelling for financial time series.
    
    Based on De Prado's "Advances in Financial Machine Learning".
    This method accounts for the fact that in real trading, positions
    are closed when either:
    1. A profit target is hit (upper barrier)
    2. A stop-loss is hit (lower barrier)
    3. A maximum holding period is reached (time barrier)
    
    This is the correct way to generate labels for ML models that
    will be used for actual trading.
    """
    
    def __init__(
        self,
        upper_barrier_pct: float = 0.02,  # 2% take-profit
        lower_barrier_pct: float = 0.01,  # 1% stop-loss
        time_barrier_days: int = 5,  # 5-day max hold
        min_return: float = 0.0,  # Minimum return to consider as signal
        vol_adjusted: bool = True,  # Adjust barriers by volatility
        vol_window: int = 20  # Window for volatility calculation
    ):
        """
        Initialize triple barrier labeller.
        
        Args:
            upper_barrier_pct: Take-profit as percentage of entry price
            lower_barrier_pct: Stop-loss as percentage of entry price
            time_barrier_days: Maximum holding period in days
            min_return: Minimum return to label as signal (vs noise)
            vol_adjusted: If True, adjust barriers by realized volatility
            vol_window: Window for volatility calculation
        """
        self.upper_barrier_pct = upper_barrier_pct
        self.lower_barrier_pct = lower_barrier_pct
        self.time_barrier_days = time_barrier_days
        self.min_return = min_return
        self.vol_adjusted = vol_adjusted
        self.vol_window = vol_window
    
    def get_volatility(self, prices: pd.Series, idx: int) -> float:
        """
        Calculate realized volatility at a point in time.
        
        Args:
            prices: Price series
            idx: Index in the series
            
        Returns:
            Realized volatility (annualized)
        """
        if idx < self.vol_window:
            # Not enough data, use default
            return 0.15  # 15% annual vol
        
        window_prices = prices.iloc[idx - self.vol_window:idx]
        returns = np.log(window_prices / window_prices.shift(1)).dropna()
        
        if len(returns) == 0:
            return 0.15
        
        vol = returns.std() * np.sqrt(252)  # Annualized
        return max(vol, 0.05)  # Minimum 5% vol
    
    def get_barriers(
        self,
        entry_price: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate upper and lower barrier prices.
        
        Args:
            entry_price: Entry price
            volatility: Realized volatility
            
        Returns:
            Tuple of (upper_barrier, lower_barrier)
        """
        if self.vol_adjusted:
            # Adjust barriers by volatility (1-day vol)
            daily_vol = volatility / np.sqrt(252)
            upper = entry_price * (1 + self.upper_barrier_pct * daily_vol * 10)
            lower = entry_price * (1 - self.lower_barrier_pct * daily_vol * 10)
        else:
            upper = entry_price * (1 + self.upper_barrier_pct)
            lower = entry_price * (1 - self.lower_barrier_pct)
        
        return upper, lower
    
    def label_single(
        self,
        prices: pd.Series,
        entry_idx: int,
        entry_price: Optional[float] = None
    ) -> TripleBarrierLabel:
        """
        Apply triple barrier labelling to a single entry point.
        
        Args:
            prices: Price series
            entry_idx: Index of entry point
            entry_price: Entry price (if None, uses prices[entry_idx])
            
        Returns:
            TripleBarrierLabel with the result
        """
        if entry_price is None:
            entry_price = prices.iloc[entry_idx]
        
        # Get volatility at entry
        volatility = self.get_volatility(prices, entry_idx)
        
        # Calculate barriers
        upper_barrier, lower_barrier = self.get_barriers(entry_price, volatility)
        
        # Check future prices
        max_idx = min(entry_idx + self.time_barrier_days, len(prices) - 1)
        future_prices = prices.iloc[entry_idx + 1:max_idx + 1]
        
        if len(future_prices) == 0:
            # No future data available
            return TripleBarrierLabel(
                label=0,
                barrier_type=BarrierType.NONE,
                return_at_hit=0.0,
                holding_period=0
            )
        
        # Check which barrier is hit first
        for i, price in enumerate(future_prices):
            if price >= upper_barrier:
                # Upper barrier hit (profit)
                return_at_hit = (price - entry_price) / entry_price
                return TripleBarrierLabel(
                    label=1,
                    barrier_type=BarrierType.UPPER,
                    hit_time=future_prices.index[i],
                    return_at_hit=return_at_hit,
                    holding_period=i + 1
                )
            elif price <= lower_barrier:
                # Lower barrier hit (loss)
                return_at_hit = (price - entry_price) / entry_price
                return TripleBarrierLabel(
                    label=-1,
                    barrier_type=BarrierType.LOWER,
                    hit_time=future_prices.index[i],
                    return_at_hit=return_at_hit,
                    holding_period=i + 1
                )
        
        # Time barrier hit (expired)
        final_price = future_prices.iloc[-1]
        return_at_hit = (final_price - entry_price) / entry_price
        
        # Label based on whether return exceeds minimum threshold
        if return_at_hit >= self.min_return:
            label = 1
        elif return_at_hit <= -self.min_return:
            label = -1
        else:
            label = 0
        
        return TripleBarrierLabel(
            label=label,
            barrier_type=BarrierType.TIME,
            hit_time=future_prices.index[-1],
            return_at_hit=return_at_hit,
            holding_period=len(future_prices)
        )
    
    def label_series(
        self,
        prices: pd.Series,
        entry_points: Optional[pd.DatetimeIndex] = None,
        step: int = 1
    ) -> pd.DataFrame:
        """
        Apply triple barrier labelling to a price series.
        
        Args:
            prices: Price series
            entry_points: Specific entry points (if None, uses every step)
            step: Step size for entry points (if entry_points is None)
            
        Returns:
            DataFrame with labels and metadata
        """
        if entry_points is None:
            # Use regular sampling
            entry_indices = range(0, len(prices) - self.time_barrier_days, step)
        else:
            # Use specific entry points
            entry_indices = [prices.index.get_loc(dt) for dt in entry_points 
                           if dt in prices.index]
        
        results = []
        for idx in entry_indices:
            try:
                label = self.label_single(prices, idx)
                results.append({
                    'entry_time': prices.index[idx],
                    'entry_price': prices.iloc[idx],
                    'label': label.label,
                    'barrier_type': label.barrier_type.value,
                    'hit_time': label.hit_time,
                    'return_at_hit': label.return_at_hit,
                    'holding_period': label.holding_period
                })
            except Exception as e:
                logger.warning(f"Failed to label entry at index {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def get_label_distribution(self, labels: pd.Series) -> dict:
        """
        Get distribution of labels.
        
        Args:
            labels: Series of labels (-1, 0, 1)
            
        Returns:
            Dictionary with label counts and percentages
        """
        counts = labels.value_counts()
        total = len(labels)
        
        distribution = {
            'total': total,
            'profit': counts.get(1, 0),
            'loss': counts.get(-1, 0),
            'neutral': counts.get(0, 0),
            'profit_pct': counts.get(1, 0) / total if total > 0 else 0,
            'loss_pct': counts.get(-1, 0) / total if total > 0 else 0,
            'neutral_pct': counts.get(0, 0) / total if total > 0 else 0
        }
        
        return distribution


class MetaLabeller:
    """
    Meta-Labelling for signal quality prediction.
    
    Based on machine-learning-for-trading Chapter 4.
    This adds a second layer that predicts whether the first model's
    signal is correct, improving overall accuracy.
    
    The idea: instead of just predicting direction, predict
    the probability that a directional signal will be correct.
    """
    
    def __init__(
        self,
        primary_labels: pd.Series,
        returns: pd.Series,
        confidence_window: int = 20
    ):
        """
        Initialize meta-labeller.
        
        Args:
            primary_labels: Primary model labels (-1, 0, 1)
            returns: Actual returns
            confidence_window: Window for calculating confidence
        """
        self.primary_labels = primary_labels
        self.returns = returns
        self.confidence_window = confidence_window
    
    def get_meta_labels(self) -> pd.Series:
        """
        Generate meta-labels based on primary label accuracy.
        
        Returns:
            Series of meta-labels (1 if primary was correct, 0 otherwise)
        """
        meta_labels = []
        
        for i in range(len(self.primary_labels)):
            if i < self.confidence_window:
                # Not enough history
                meta_labels.append(0)
                continue
            
            # Check if primary label was correct
            primary_label = self.primary_labels.iloc[i]
            actual_return = self.returns.iloc[i]
            
            # Primary label is correct if direction matches
            if primary_label == 1 and actual_return > 0:
                meta_labels.append(1)
            elif primary_label == -1 and actual_return < 0:
                meta_labels.append(1)
            elif primary_label == 0:
                meta_labels.append(0.5)  # Neutral
            else:
                meta_labels.append(0)
        
        return pd.Series(meta_labels, index=self.primary_labels.index)
    
    def get_confidence_scores(self) -> pd.Series:
        """
        Calculate rolling confidence scores for primary labels.
        
        Returns:
            Series of confidence scores (0 to 1)
        """
        meta_labels = self.get_meta_labels()
        confidence = meta_labels.rolling(
            window=self.confidence_window,
            min_periods=1
        ).mean()
        
        return confidence


def get_triple_barrier_labeller(
    upper_barrier_pct: float = 0.02,
    lower_barrier_pct: float = 0.01,
    time_barrier_days: int = 5,
    vol_adjusted: bool = True
) -> TripleBarrierLabeller:
    """
    Factory function to get a triple barrier labeller with sensible defaults.
    
    Args:
        upper_barrier_pct: Take-profit percentage
        lower_barrier_pct: Stop-loss percentage
        time_barrier_days: Maximum holding period
        vol_adjusted: Whether to adjust barriers by volatility
        
    Returns:
        TripleBarrierLabeller instance
    """
    return TripleBarrierLabeller(
        upper_barrier_pct=upper_barrier_pct,
        lower_barrier_pct=lower_barrier_pct,
        time_barrier_days=time_barrier_days,
        vol_adjusted=vol_adjusted
    )


if __name__ == "__main__":
    # Test the triple barrier labeller
    print("Testing Triple Barrier Labeller...")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates
    )
    
    # Initialize labeller
    labeller = get_triple_barrier_labeller(
        upper_barrier_pct=0.02,
        lower_barrier_pct=0.01,
        time_barrier_days=5,
        vol_adjusted=True
    )
    
    # Label the series
    labels_df = labeller.label_series(prices, step=5)
    
    print(f"\nGenerated {len(labels_df)} labels")
    print(f"\nLabel distribution:")
    dist = labeller.get_label_distribution(labels_df['label'])
    for key, value in dist.items():
        if 'pct' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nFirst 5 labels:")
    print(labels_df.head())
