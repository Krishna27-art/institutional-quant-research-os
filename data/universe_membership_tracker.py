"""
Universe Membership Tracker for Survivorship-Free Backtesting

This module tracks point-in-time universe membership to eliminate survivorship bias
in backtesting. It ensures that backtests only use stocks that were actually in the
universe at each point in time.

Key Features:
- NIFTY 50 constituent history tracking
- Point-in-time universe queries
- Survivorship bias detection
- Universe reconstruction from historical data
- Integration with point-in-time reconstruction

Based on V4 Blueprint - Institutional Architecture
Priority: High (Phase 0.1)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniverseType(Enum):
    """Types of universes."""
    NIFTY_50 = "nifty_50"
    NIFTY_100 = "nifty_100"
    NIFTY_200 = "nifty_200"
    BANKNIFTY = "banknifty"
    CUSTOM = "custom"


@dataclass
class UniverseMembership:
    """Universe membership for a symbol."""
    symbol: str
    universe_type: UniverseType
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_reason: str = ""
    exit_reason: str = ""
    
    def is_active(self, date: datetime) -> bool:
        """Check if symbol is active in universe on given date."""
        if self.entry_date > date:
            return False
        if self.exit_date is not None and self.exit_date < date:
            return False
        return True


class UniverseMembershipTracker:
    """
    Tracker for point-in-time universe membership.
    
    This class maintains the history of which stocks were in which universes
    at each point in time, enabling survivorship-free backtesting.
    """
    
    def __init__(self):
        self.universe_history: Dict[UniverseType, List[UniverseMembership]] = {}
        self.reconstitution_dates: Dict[UniverseType, List[datetime]] = {}
        
        # Initialize universe histories
        for universe_type in UniverseType:
            self.universe_history[universe_type] = []
            self.reconstitution_dates[universe_type] = []
        
        logger.info("UniverseMembershipTracker initialized")
    
    def add_membership(
        self,
        symbol: str,
        universe_type: UniverseType,
        entry_date: datetime,
        exit_date: Optional[datetime] = None,
        entry_reason: str = "",
        exit_reason: str = ""
    ) -> None:
        """
        Add universe membership record.
        
        Args:
            symbol: Stock symbol
            universe_type: Universe type
            entry_date: Date when symbol entered universe
            exit_date: Date when symbol exited universe (optional)
            entry_reason: Reason for entry
            exit_reason: Reason for exit
        """
        membership = UniverseMembership(
            symbol=symbol,
            universe_type=universe_type,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_reason=entry_reason,
            exit_reason=exit_reason
        )
        
        self.universe_history[universe_type].append(membership)
        
        logger.info(f"Added membership: {symbol} in {universe_type.value} from {entry_date} to {exit_date}")
    
    def get_universe_at_date(
        self,
        universe_type: UniverseType,
        date: datetime
    ) -> Set[str]:
        """
        Get all symbols in universe at a specific date.
        
        Args:
            universe_type: Universe type
            date: Query date
            
        Returns:
            Set of symbols in universe at date
        """
        active_symbols = set()
        
        for membership in self.universe_history[universe_type]:
            if membership.is_active(date):
                active_symbols.add(membership.symbol)
        
        return active_symbols
    
    def get_universe_history(
        self,
        universe_type: UniverseType,
        start_date: datetime,
        end_date: datetime,
        frequency: str = 'M'
    ) -> Dict[datetime, Set[str]]:
        """
        Get universe membership history over a date range.
        
        Args:
            universe_type: Universe type
            start_date: Start date
            end_date: End date
            frequency: Frequency ('D' daily, 'W' weekly, 'M' monthly)
            
        Returns:
            Dict mapping dates to sets of symbols
        """
        history = {}
        dates = pd.date_range(start_date, end_date, freq=frequency)
        
        for date in dates:
            history[date] = self.get_universe_at_date(universe_type, date)
        
        return history
    
    def detect_survivorship_bias(
        self,
        backtest_symbols: Set[str],
        universe_type: UniverseType,
        backtest_start: datetime,
        backtest_end: datetime
    ) -> Dict[str, any]:
        """
        Detect survivorship bias in backtest.
        
        Args:
            backtest_symbols: Symbols used in backtest
            universe_type: Universe type
            backtest_start: Backtest start date
            backtest_end: Backtest end date
            
        Returns:
            Dict with bias detection results
        """
        results = {
            'has_survivorship_bias': False,
            'symbols_not_in_universe': [],
            'symbols_with_future_membership': [],
            'bias_score': 0.0
        }
        
        # Get universe at backtest start
        universe_at_start = self.get_universe_at_date(universe_type, backtest_start)
        
        # Check for symbols not in universe at start
        for symbol in backtest_symbols:
            if symbol not in universe_at_start:
                results['symbols_not_in_universe'].append(symbol)
                results['has_survivorship_bias'] = True
        
        # Check for symbols that entered universe during backtest
        universe_at_end = self.get_universe_at_date(universe_type, backtest_end)
        for symbol in backtest_symbols:
            if symbol in universe_at_end and symbol not in universe_at_start:
                results['symbols_with_future_membership'].append(symbol)
                results['has_survivorship_bias'] = True
        
        # Calculate bias score
        if len(backtest_symbols) > 0:
            bias_count = len(results['symbols_not_in_universe']) + len(results['symbols_with_future_membership'])
            results['bias_score'] = bias_count / len(backtest_symbols)
        
        return results
    
    def load_nifty_50_history(self, data_path: str = None) -> None:
        """
        Load NIFTY 50 constituent history from data.
        
        Args:
            data_path: Path to constituent history data (CSV/Excel)
        """
        # Sample NIFTY 50 reconstitution dates (actual data should come from NSE)
        # This is a placeholder - in production, load from official NSE data
        
        sample_reconstitutions = [
            # (date, added_symbols, removed_symbols)
            (datetime(2020, 1, 1), ['TATAMOTORS', 'BAJFINANCE'], ['M&M', 'ZEEL']),
            (datetime(2021, 1, 1), ['TATASTEEL', 'HINDALCO'], ['BHARTIARTL', 'WIPRO']),
            (datetime(2022, 1, 1), ['ADANIENTERPRISE', 'SBILIFE'], ['TATAMOTORS', 'BAJFINANCE']),
            (datetime(2023, 1, 1), ['LICI', 'JSWSTEEL'], ['ADANIENTERPRISE', 'SBILIFE']),
            (datetime(2024, 1, 1), ['JIOFINANCIAL', 'TRENT'], ['LICI', 'JSWSTEEL']),
        ]
        
        # Build membership history from reconstitutions
        current_universe = self._get_initial_nifty_50()
        
        for date, added, removed in sample_reconstitutions:
            self.reconstitution_dates[UniverseType.NIFTY_50].append(date)
            
            # Add new symbols
            for symbol in added:
                self.add_membership(
                    symbol=symbol,
                    universe_type=UniverseType.NIFTY_50,
                    entry_date=date,
                    entry_reason="reconstitution"
                )
            
            # Remove old symbols
            for symbol in removed:
                # Find existing membership and set exit date
                for membership in self.universe_history[UniverseType.NIFTY_50]:
                    if membership.symbol == symbol and membership.exit_date is None:
                        membership.exit_date = date
                        membership.exit_reason = "reconstitution"
                        break
        
        logger.info(f"Loaded NIFTY 50 history with {len(self.reconstitution_dates[UniverseType.NIFTY_50])} reconstitutions")
    
    def _get_initial_nifty_50(self) -> Set[str]:
        """Get initial NIFTY 50 constituents (placeholder)."""
        # Sample initial NIFTY 50 stocks
        return {
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK',
            'LICI', 'LT', 'AXISBANK', 'BAJFINANCE', 'HCLTECH',
            'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'BAJAJFINSV',
            'DMART', 'WIPRO', 'ULTRACEMCO', 'NTPC', 'POWERGRID',
            'TATAMOTORS', 'TATASTEEL', 'HINDALCO', 'TATACONSUM', 'ONGC',
            'COALINDIA', 'IOC', 'BPCL', 'GAIL', 'M&M',
            'ZEEL', 'JSWSTEEL', 'GRASIM', 'ADANIENTERPRISE', 'SBILIFE',
            'NESTLEIND', 'BRITANNIA', 'DIVISLAB', 'DRREDDY', 'CIPLA',
            'SUNPHARMA', 'AUROPHARMA', 'TRENT', 'JIOFINANCIAL'
        }
    
    def export_universe_history(
        self,
        universe_type: UniverseType,
        output_path: str
    ) -> None:
        """
        Export universe history to CSV.
        
        Args:
            universe_type: Universe type
            output_path: Output file path
        """
        data = []
        
        for membership in self.universe_history[universe_type]:
            data.append({
                'symbol': membership.symbol,
                'universe_type': membership.universe_type.value,
                'entry_date': membership.entry_date,
                'exit_date': membership.exit_date,
                'entry_reason': membership.entry_reason,
                'exit_reason': membership.exit_reason
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported universe history to {output_path}")
    
    def import_universe_history(
        self,
        input_path: str,
        universe_type: UniverseType
    ) -> None:
        """
        Import universe history from CSV.
        
        Args:
            input_path: Input file path
            universe_type: Universe type
        """
        df = pd.read_csv(input_path)
        
        for _, row in df.iterrows():
            self.add_membership(
                symbol=row['symbol'],
                universe_type=universe_type,
                entry_date=pd.to_datetime(row['entry_date']),
                exit_date=pd.to_datetime(row['exit_date']) if pd.notna(row['exit_date']) else None,
                entry_reason=row.get('entry_reason', ''),
                exit_reason=row.get('exit_reason', '')
            )
        
        logger.info(f"Imported universe history from {input_path}")
    
    def print_universe_report(
        self,
        universe_type: UniverseType,
        date: datetime
    ) -> None:
        """Print universe report for a specific date."""
        print("\n" + "="*60)
        print(f"UNIVERSE REPORT: {universe_type.value.upper()} at {date.date()}")
        print("="*60)
        
        universe = self.get_universe_at_date(universe_type, date)
        
        print(f"\nTotal symbols: {len(universe)}")
        print(f"\nSymbols:")
        for symbol in sorted(universe):
            print(f"  {symbol}")
        
        print("\n" + "="*60)


def sample_universe_tracking():
    """Demonstrate universe membership tracking."""
    print("=== Universe Membership Tracker Demo ===\n")
    
    tracker = UniverseMembershipTracker()
    
    # Load sample NIFTY 50 history
    print("Loading NIFTY 50 history...")
    tracker.load_nifty_50_history()
    
    # Get universe at different dates
    dates = [
        datetime(2020, 6, 1),
        datetime(2021, 6, 1),
        datetime(2022, 6, 1),
        datetime(2023, 6, 1),
        datetime(2024, 6, 1)
    ]
    
    for date in dates:
        universe = tracker.get_universe_at_date(UniverseType.NIFTY_50, date)
        print(f"\nNIFTY 50 at {date.date()}: {len(universe)} symbols")
    
    # Detect survivorship bias
    print("\n\nDetecting survivorship bias...")
    backtest_symbols = {'TATAMOTORS', 'TCS', 'RELIANCE', 'JIOFINANCIAL'}  # Mix of old and new
    bias_results = tracker.detect_survivorship_bias(
        backtest_symbols=backtest_symbols,
        universe_type=UniverseType.NIFTY_50,
        backtest_start=datetime(2019, 1, 1),
        backtest_end=datetime(2024, 12, 31)
    )
    
    print(f"Has survivorship bias: {bias_results['has_survivorship_bias']}")
    print(f"Bias score: {bias_results['bias_score']:.2%}")
    print(f"Symbols not in universe at start: {bias_results['symbols_not_in_universe']}")
    print(f"Symbols with future membership: {bias_results['symbols_with_future_membership']}")
    
    # Print universe report
    tracker.print_universe_report(UniverseType.NIFTY_50, datetime(2022, 6, 1))
    
    print("\n=== Universe Membership Tracker Demo Complete ===")
    print("Key capabilities:")
    print("- Point-in-time universe queries")
    print("- Survivorship bias detection")
    print("- Universe history tracking")
    print("- Reconstitution date tracking")


if __name__ == "__main__":
    sample_universe_tracking()
