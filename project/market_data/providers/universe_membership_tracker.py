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
        Load NIFTY 50 constituent history from a CSV file.

        Expected CSV columns: symbol, entry_date, exit_date (optional),
        entry_reason (optional), exit_reason (optional)

        Data source:
          NSE website → Market Data → Indices → NIFTY 50 →
          "Historical Constituent Changes"
          URL: https://www.nseindia.com/market-data/indices-production-statistics-indices-listing

        Parameters
        ----------
        data_path : str, optional
            Path to CSV file. Defaults to
            'data/raw/nifty50_constituents_history.csv'
        """
        if data_path is None:
            data_path = 'data/raw/nifty50_constituents_history.csv'

        import os
        if not os.path.exists(data_path):
            logger.warning(
                "NIFTY 50 constituent history file not found at '%s'. "
                "SURVIVORSHIP BIAS WILL AFFECT ALL BACKTESTS. "
                "Download the official NSE constituent history CSV from: "
                "https://www.nseindia.com/market-data/indices-production-statistics-indices-listing "
                "and save it to '%s'. "
                "A template with the required column format has been created at "
                "'data/raw/nifty50_constituents_history_template.csv'.",
                data_path, data_path
            )
            self._create_template_csv(
                'data/raw/nifty50_constituents_history_template.csv'
            )
            return

        import pandas as pd
        try:
            df = pd.read_csv(data_path)
            required = {'symbol', 'entry_date'}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(
                    f"NIFTY 50 history CSV missing required columns: {missing}"
                )

            df['entry_date'] = pd.to_datetime(df['entry_date'])
            if 'exit_date' in df.columns:
                df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
            else:
                df['exit_date'] = pd.NaT

            entry_reason_col = df.get('entry_reason', pd.Series([''] * len(df)))
            exit_reason_col  = df.get('exit_reason',  pd.Series([''] * len(df)))

            for _, row in df.iterrows():
                exit_date = None
                if 'exit_date' in df.columns and pd.notna(row['exit_date']):
                    exit_date = row['exit_date'].to_pydatetime()
                self.add_membership(
                    symbol=str(row['symbol']).upper(),
                    universe_type=UniverseType.NIFTY_50,
                    entry_date=row['entry_date'].to_pydatetime(),
                    exit_date=exit_date,
                    entry_reason=str(row.get('entry_reason', '')),
                    exit_reason=str(row.get('exit_reason', '')),
                )

            logger.info(
                "Loaded NIFTY 50 history: %d membership records from '%s'",
                len(df), data_path
            )
        except Exception as exc:
            logger.error(
                "Failed to load NIFTY 50 history from '%s': %s", data_path, exc
            )
            raise

    def _create_template_csv(self, path: str) -> None:
        """Create a template CSV showing the expected format."""
        import os
        import csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [
            ['symbol', 'entry_date', 'exit_date', 'entry_reason', 'exit_reason'],
            ['RELIANCE', '2000-01-01', '', 'initial_constituent', ''],
            ['TCS', '2004-08-25', '', 'ipo_addition', ''],
            ['ZEEL', '2000-01-01', '2020-06-26', 'initial_constituent', 'reconstitution'],
            ['SHREECEM', '2012-03-30', '2021-09-24', 'reconstitution', 'reconstitution'],
            ['JIOFINANCIAL', '2023-10-13', '', 'demerger_addition', ''],
        ]
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
        logger.info("Created NIFTY 50 history template at '%s'", path)
    
    def _get_initial_nifty_50(self) -> Set[str]:
        """Get initial NIFTY 50 constituents (placeholder — replaced by CSV loader)."""
        return set()

    # ------------------------------------------------------------------
    # Survivorship bias helpers
    # ------------------------------------------------------------------

    def get_delisted_symbols(
        self,
        start_date: datetime,
        end_date: datetime,
        universe_type: UniverseType,
    ) -> Set[str]:
        """
        Return symbols that were in the universe at start_date but exited
        (were delisted/removed) before end_date.

        These are the "survivorship-biased" stocks: if you only look at
        today's universe, you miss every stock that failed between
        start_date and end_date, inflating historical returns by 4–8%/year.

        Parameters
        ----------
        start_date : datetime
        end_date : datetime
        universe_type : UniverseType

        Returns
        -------
        Set[str] of symbols that were active at start but exited before end.
        """
        active_at_start = self.get_universe_at_date(universe_type, start_date)
        active_at_end   = self.get_universe_at_date(universe_type, end_date)
        # Symbols present at start but gone by end = delisted/removed
        return active_at_start - active_at_end

    def get_full_historical_universe(
        self,
        start_date: datetime,
        end_date: datetime,
        universe_type: UniverseType,
    ) -> Set[str]:
        """
        Return ALL symbols that were EVER active in the universe between
        start_date and end_date (inclusive).

        = currently_active_at_end UNION delisted_in_period

        This is the correct backtest universe: it includes every stock that
        could have been traded during the period, preventing survivorship bias.
        """
        active_at_end  = self.get_universe_at_date(universe_type, end_date)
        delisted       = self.get_delisted_symbols(start_date, end_date, universe_type)
        return active_at_end | delisted

    def validate_backtest_universe(
        self,
        symbols: Set[str],
        start_date: datetime,
        universe_type: UniverseType,
    ) -> Dict:
        """
        Check whether any symbol in `symbols` was NOT in the universe at
        start_date (i.e., it only became available later, or never existed).

        Returns
        -------
        dict:
            survivorship_bias_detected : bool
            offending_symbols : List[str]
            bias_pct : float  (fraction of symbols that are offenders)
        """
        universe_at_start = self.get_universe_at_date(universe_type, start_date)
        upper_universe    = {s.upper() for s in universe_at_start}
        offenders         = [
            s for s in symbols if s.upper() not in upper_universe
        ]
        return {
            'survivorship_bias_detected': len(offenders) > 0,
            'offending_symbols': sorted(offenders),
            'bias_pct': len(offenders) / max(len(symbols), 1),
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
