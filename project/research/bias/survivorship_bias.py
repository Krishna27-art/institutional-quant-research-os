"""
Survivorship Bias Adjustment

This module implements survivorship bias adjustment to correct for
the overestimation of returns that occurs when only including currently
listed stocks in historical analysis.

Key Features:
- Delisted stock identification
- Historical universe reconstruction
- Return adjustment for delisted stocks
- Bias quantification
- Adjusted performance metrics

Based on Audit Report Priority 1: Research Quality
Research Papers: Brown et al (1992), Shumway & Warther (1999)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DelistingEvent:
    """Delisting event for a stock."""
    symbol: str
    delisting_date: datetime
    delisting_reason: str
    last_price: float
    delisting_return: float
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BiasAdjustment:
    """Survivorship bias adjustment result."""
    original_return: float
    adjusted_return: float
    bias_amount: float
    bias_percentage: float
    delisted_stocks_count: int
    surviving_stocks_count: int
    adjustment_method: str


class SurvivorshipBiasAdjuster:
    """
    Survivorship bias adjuster.
    
    This class identifies and adjusts for survivorship bias in
    historical returns data.
    """
    
    def __init__(self):
        """Initialize survivorship bias adjuster."""
        self.delisting_history: Dict[str, List[DelistingEvent]] = {}
        self.historical_universe: Dict[datetime, List[str]] = {}
        
        logger.info("SurvivorshipBiasAdjuster initialized")
    
    def add_delisting_event(
        self,
        symbol: str,
        delisting_date: datetime,
        delisting_reason: str,
        last_price: float,
        delisting_return: Optional[float] = None
    ) -> None:
        """
        Add a delisting event.
        
        Args:
            symbol: Stock symbol
            delisting_date: Date of delisting
            delisting_reason: Reason for delisting
            last_price: Last trading price
            delisting_return: Return at delisting (default -100%)
        """
        if delisting_return is None:
            delisting_return = -1.0  # Assume total loss
        
        event = DelistingEvent(
            symbol=symbol,
            delisting_date=delisting_date,
            delisting_reason=delisting_reason,
            last_price=last_price,
            delisting_return=delisting_return
        )
        
        if symbol not in self.delisting_history:
            self.delisting_history[symbol] = []
        
        self.delisting_history[symbol].append(event)
        
        logger.info(f"Added delisting event for {symbol} on {delisting_date}")
    
    def reconstruct_historical_universe(
        self,
        current_universe: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[datetime, List[str]]:
        """
        Reconstruct historical universe including delisted stocks.
        
        Args:
            current_universe: Currently listed stocks
            start_date: Start date for reconstruction
            end_date: End date for reconstruction
            
        Returns:
            Dictionary mapping dates to stock lists
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='1M')
        universe_over_time = {}
        
        for date in dates:
            # Start with current universe
            universe = set(current_universe)
            
            # Add stocks that were listed on this date
            for symbol, events in self.delisting_history.items():
                for event in events:
                    # Stock was listed before delisting date
                    if date < event.delisting_date:
                        universe.add(symbol)
            
            universe_over_time[date] = list(universe)
        
        self.historical_universe = universe_over_time
        
        logger.info(f"Reconstructed historical universe from {start_date} to {end_date}")
        return universe_over_time
    
    def calculate_adjusted_returns(
        self,
        returns_data: pd.DataFrame,
        method: str = "include_delisted"
    ) -> BiasAdjustment:
        """
        Calculate survivorship bias-adjusted returns.
        
        Args:
            returns_data: DataFrame with returns data
            method: Adjustment method ('include_delisted', 'weight_adjustment')
            
        Returns:
            BiasAdjustment with adjustment details
        """
        if method == "include_delisted":
            return self._adjust_by_including_delisted(returns_data)
        elif method == "weight_adjustment":
            return self._adjust_by_weighting(returns_data)
        else:
            logger.warning(f"Unknown method: {method}, using include_delisted")
            return self._adjust_by_including_delisted(returns_data)
    
    def _adjust_by_including_delisted(
        self,
        returns_data: pd.DataFrame
    ) -> BiasAdjustment:
        """
        Adjust returns by including delisted stocks.
        
        Simulates returns if delisted stocks were included in the universe.
        """
        original_returns = returns_data.mean(axis=1)
        
        # Create adjusted returns including delisted stocks
        adjusted_returns = []
        
        for date in returns_data.index:
            # Get surviving stocks for this date
            surviving_stocks = returns_data.columns[returns_data.loc[date].notna()]
            
            # Get delisted stocks that should be included
            delisted_returns = []
            for symbol, events in self.delisting_history.items():
                for event in events:
                    # If this date is after delisting, include delisting return
                    if date >= event.delisting_date:
                        delisted_returns.append(event.delisting_return)
            
            # Combine surviving and delisted returns
            if len(surviving_stocks) > 0:
                surviving_return = returns_data.loc[date, surviving_stocks].mean()
            else:
                surviving_return = 0.0
            
            if delisted_returns:
                delisted_return = np.mean(delisted_returns)
            else:
                delisted_return = 0.0
            
            # Weighted average
            total_stocks = len(surviving_stocks) + len(delisted_returns)
            if total_stocks > 0:
                adjusted_return = (surviving_return * len(surviving_stocks) + 
                                 delisted_return * len(delisted_returns)) / total_stocks
            else:
                adjusted_return = 0.0
            
            adjusted_returns.append(adjusted_return)
        
        adjusted_returns = pd.Series(adjusted_returns, index=returns_data.index)
        
        # Calculate bias
        original_cumulative = (1 + original_returns).prod() - 1
        adjusted_cumulative = (1 + adjusted_returns).prod() - 1
        bias_amount = original_cumulative - adjusted_cumulative
        bias_percentage = (bias_amount / abs(adjusted_cumulative)) * 100 if adjusted_cumulative != 0 else 0
        
        return BiasAdjustment(
            original_return=original_cumulative,
            adjusted_return=adjusted_cumulative,
            bias_amount=bias_amount,
            bias_percentage=bias_percentage,
            delisted_stocks_count=len(self.delisting_history),
            surviving_stocks_count=len(returns_data.columns),
            adjustment_method="include_delisted"
        )
    
    def _adjust_by_weighting(
        self,
        returns_data: pd.DataFrame
    ) -> BiasAdjustment:
        """
        Adjust returns by weighting for survivorship bias.
        
        Applies a statistical adjustment factor based on historical delisting rates.
        """
        original_returns = returns_data.mean(axis=1)
        
        # Calculate historical delisting rate
        total_periods = len(returns_data)
        delisting_rate = len(self.delisting_history) / (len(returns_data.columns) + len(self.delisting_history))
        
        # Apply adjustment factor (simplified)
        # Higher delisting rate = larger downward adjustment
        adjustment_factor = 1.0 - (delisting_rate * 0.5)  # Conservative adjustment
        
        adjusted_returns = original_returns * adjustment_factor
        
        # Calculate bias
        original_cumulative = (1 + original_returns).prod() - 1
        adjusted_cumulative = (1 + adjusted_returns).prod() - 1
        bias_amount = original_cumulative - adjusted_cumulative
        bias_percentage = (bias_amount / abs(adjusted_cumulative)) * 100 if adjusted_cumulative != 0 else 0
        
        return BiasAdjustment(
            original_return=original_cumulative,
            adjusted_return=adjusted_cumulative,
            bias_amount=bias_amount,
            bias_percentage=bias_percentage,
            delisted_stocks_count=len(self.delisting_history),
            surviving_stocks_count=len(returns_data.columns),
            adjustment_method="weight_adjustment"
        )
    
    def quantify_bias(
        self,
        returns_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Quantify survivorship bias using multiple methods.
        
        Args:
            returns_data: DataFrame with returns data
            
        Returns:
            Dictionary of bias metrics
        """
        metrics = {}
        
        # Method 1: Include delisted
        adjustment1 = self._adjust_by_including_delisted(returns_data)
        metrics['bias_include_delisted'] = adjustment1.bias_amount
        metrics['bias_pct_include_delisted'] = adjustment1.bias_percentage
        
        # Method 2: Weight adjustment
        adjustment2 = self._adjust_by_weighting(returns_data)
        metrics['bias_weight_adjustment'] = adjustment2.bias_amount
        metrics['bias_pct_weight_adjustment'] = adjustment2.bias_percentage
        
        # Average bias
        metrics['average_bias'] = (adjustment1.bias_amount + adjustment2.bias_amount) / 2
        metrics['average_bias_pct'] = (adjustment1.bias_percentage + adjustment2.bias_percentage) / 2
        
        return metrics
    
    def get_delisting_statistics(self) -> Dict:
        """Get statistics about delisting events."""
        if not self.delisting_history:
            return {}
        
        total_delistings = sum(len(events) for events in self.delisting_history.values())
        
        reasons = {}
        for events in self.delisting_history.values():
            for event in events:
                reason = event.delisting_reason
                reasons[reason] = reasons.get(reason, 0) + 1
        
        avg_delisting_return = np.mean([
            event.delisting_return for events in self.delisting_history.values()
            for event in events
        ])
        
        return {
            'total_delistings': total_delistings,
            'unique_symbols': len(self.delisting_history),
            'delisting_reasons': reasons,
            'average_delisting_return': avg_delisting_return
        }
    
    def print_bias_report(self, returns_data: pd.DataFrame) -> None:
        """Print survivorship bias report."""
        print("\n" + "="*60)
        print("SURVIVORSHIP BIAS REPORT")
        print("="*60)
        
        # Delisting statistics
        stats = self.get_delisting_statistics()
        print(f"\nDelisting Statistics:")
        print(f"  Total Delistings: {stats.get('total_delistings', 0)}")
        print(f"  Unique Symbols: {stats.get('unique_symbols', 0)}")
        print(f"  Average Delisting Return: {stats.get('average_delisting_return', 0):.2%}")
        
        if stats.get('delisting_reasons'):
            print(f"\n  Delisting Reasons:")
            for reason, count in stats['delisting_reasons'].items():
                print(f"    {reason}: {count}")
        
        # Bias quantification
        bias_metrics = self.quantify_bias(returns_data)
        print(f"\nBias Quantification:")
        print(f"  Bias (Include Delisted): {bias_metrics['bias_include_delisted']:.4f} "
              f"({bias_metrics['bias_pct_include_delisted']:.2f}%)")
        print(f"  Bias (Weight Adjustment): {bias_metrics['bias_weight_adjustment']:.4f} "
              f"({bias_metrics['bias_pct_weight_adjustment']:.2f}%)")
        print(f"  Average Bias: {bias_metrics['average_bias']:.4f} "
              f"({bias_metrics['average_bias_pct']:.2f}%)")
        
        print("\n" + "="*60)


# Singleton instance
_survivorship_bias_adjuster = None

def get_survivorship_bias_adjuster() -> SurvivorshipBiasAdjuster:
    """Get the singleton survivorship bias adjuster instance."""
    global _survivorship_bias_adjuster
    if _survivorship_bias_adjuster is None:
        _survivorship_bias_adjuster = SurvivorshipBiasAdjuster()
    return _survivorship_bias_adjuster


if __name__ == "__main__":
    # Test survivorship bias adjuster
    print("Testing Survivorship Bias Adjuster...")
    
    adjuster = SurvivorshipBiasAdjuster()
    
    # Add some delisting events
    adjuster.add_delisting_event(
        "DELISTED1",
        datetime(2023, 6, 1),
        "bankruptcy",
        100.0,
        -0.8
    )
    
    adjuster.add_delisting_event(
        "DELISTED2",
        datetime(2023, 9, 1),
        "merger",
        150.0,
        0.3
    )
    
    # Create sample returns data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1M')
    np.random.seed(42)
    
    returns_data = pd.DataFrame(
        np.random.normal(0.01, 0.05, (12, 10)),
        index=dates,
        columns=[f'STOCK{i}' for i in range(10)]
    )
    
    # Print bias report
    adjuster.print_bias_report(returns_data)
