"""
Corporate Actions and Earnings Calendar
Based on V4 Blueprint - Institutional Architecture

Key capabilities:
- Corporate action calendar tracking (splits, bonuses, dividends, rights)
- Earnings calendar with surprise calculation
- Price adjustments for splits, bonuses, dividends
- Event-driven volatility modeling
- Historical corporate action database
- Earnings surprise (SUE) calculation

CRITICAL FIX: Ensure all price data is adjusted for splits and dividends.
Unadjusted prices cause artificial jumps and false signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np


def _to_naive_datetime(dt):
    """Convert datetime to naive datetime (remove timezone)."""
    if hasattr(dt, 'tz_localize'):
        return dt.dt.tz_localize(None)
    elif hasattr(dt, 'tz'):
        return dt.dt.tz_convert(None)
    return dt


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """Corporate action event."""
    symbol: str
    date: pd.Timestamp
    action_type: str  # "split", "bonus", "dividend", "rights_issue", "merger", "delisting"
    factor: float = 1.0
    cash_value: float = 0.0
    record_date: Optional[pd.Timestamp] = None
    ex_date: Optional[pd.Timestamp] = None


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    """Earnings announcement event."""
    symbol: str
    announcement_date: pd.Timestamp
    fiscal_quarter: str  # "Q1", "Q2", "Q3", "Q4"
    fiscal_year: int
    actual_eps: float
    estimated_eps: float
    revenue_actual: float
    revenue_estimated: float
    surprise: float  # SUE (Standardized Unexpected Earnings)
    revenue_surprise: float


class CorporateActionAdjuster:
    """Apply multiplicative price adjustments for splits and bonuses."""

    def __init__(self, enable_auto_adjust: bool = True):
        self.corporate_actions_db: List[CorporateAction] = []
        self.enable_auto_adjust = enable_auto_adjust  # CRITICAL FIX
    
    def add_corporate_action(
        self,
        symbol: str,
        date: pd.Timestamp,
        action_type: str,
        factor: float = 1.0,
        cash_value: float = 0.0,
        record_date: Optional[pd.Timestamp] = None,
        ex_date: Optional[pd.Timestamp] = None
    ) -> None:
        """
        Add a corporate action to the database.
        
        Args:
            symbol: Stock symbol
            date: Action date
            action_type: Type of action
            factor: Adjustment factor
            cash_value: Cash value (for dividends)
            record_date: Record date
            ex_date: Ex-date
        """
        action = CorporateAction(
            symbol=symbol,
            date=date,
            action_type=action_type,
            factor=factor,
            cash_value=cash_value,
            record_date=record_date,
            ex_date=ex_date
        )
        self.corporate_actions_db.append(action)


class EarningsCalendar:
    """Earnings calendar management and surprise calculation."""

    def __init__(self):
        self.earnings_db: List[EarningsEvent] = []
    
    def add_earnings_event(
        self,
        symbol: str,
        announcement_date: pd.Timestamp,
        fiscal_quarter: str,
        fiscal_year: int,
        actual_eps: float,
        estimated_eps: float,
        revenue_actual: float,
        revenue_estimated: float
    ) -> None:
        """
        Add an earnings event to the database.
        
        Args:
            symbol: Stock symbol
            announcement_date: Earnings announcement date
            fiscal_quarter: Fiscal quarter
            fiscal_year: Fiscal year
            actual_eps: Actual earnings per share
            estimated_eps: Estimated earnings per share
            revenue_actual: Actual revenue
            revenue_estimated: Estimated revenue
        """
        # Calculate surprises
        eps_surprise = (actual_eps - estimated_eps) / abs(estimated_eps) if estimated_eps != 0 else 0
        revenue_surprise = (revenue_actual - revenue_estimated) / abs(revenue_estimated) if revenue_estimated != 0 else 0
        
        event = EarningsEvent(
            symbol=symbol,
            announcement_date=announcement_date,
            fiscal_quarter=fiscal_quarter,
            fiscal_year=fiscal_year,
            actual_eps=actual_eps,
            estimated_eps=estimated_eps,
            revenue_actual=revenue_actual,
            revenue_estimated=revenue_estimated,
            surprise=eps_surprise,
            revenue_surprise=revenue_surprise
        )
        
        self.earnings_db.append(event)
    
    def get_earnings_for_symbol(
        self,
        symbol: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> List[EarningsEvent]:
        """
        Get earnings events for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of EarningsEvent
        """
        events = [e for e in self.earnings_db if e.symbol == symbol]
        
        if start_date:
            events = [e for e in events if e.announcement_date >= start_date]
        if end_date:
            events = [e for e in events if e.announcement_date <= end_date]
        
        return sorted(events, key=lambda x: x.announcement_date)
    
    def calculate_sue(
        self,
        symbol: str,
        earnings_event: EarningsEvent,
        historical_eps_std: float
    ) -> float:
        """
        Calculate Standardized Unexpected Earnings (SUE).
        
        SUE = (Actual EPS - Expected EPS) / Standard Deviation of Historical EPS
        
        Args:
            symbol: Stock symbol
            earnings_event: Earnings event
            historical_eps_std: Standard deviation of historical EPS surprises
            
        Returns:
            SUE value
        """
        if historical_eps_std == 0:
            return 0.0
        
        sue = (earnings_event.actual_eps - earnings_event.estimated_eps) / historical_eps_std
        return sue
    
    def get_upcoming_earnings(
        self,
        days_ahead: int = 30
    ) -> List[EarningsEvent]:
        """
        Get upcoming earnings announcements.
        
        Args:
            days_ahead: Number of days ahead to look
            
        Returns:
            List of upcoming earnings events
        """
        today = pd.Timestamp.now()
        cutoff_date = today + pd.Timedelta(days=days_ahead)
        
        upcoming = [e for e in self.earnings_db 
                   if today <= e.announcement_date <= cutoff_date]
        
        return sorted(upcoming, key=lambda x: x.announcement_date)
    
    def get_earnings_momentum_signal(
        self,
        symbol: str,
        window_days: int = 90
    ) -> float:
        """
        Calculate earnings momentum signal.
        
        Based on recent earnings surprises.
        
        Args:
            symbol: Stock symbol
            window_days: Lookback window in days
            
        Returns:
            Momentum signal (-1 to 1)
        """
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        recent_earnings = self.get_earnings_for_symbol(symbol, start_date=cutoff_date)
        
        if not recent_earnings:
            return 0.0
        
        # Average surprise over recent earnings
        avg_surprise = np.mean([e.surprise for e in recent_earnings])
        
        # Normalize to -1 to 1 range
        signal = np.tanh(avg_surprise * 2)
        
        return signal
    
    def print_earnings_report(self) -> None:
        """Print earnings calendar report."""
        print("\n" + "="*60)
        print("EARNINGS CALENDAR REPORT")
        print("="*60)
        print(f"Total earnings events: {len(self.earnings_db)}")
        
        # Count by symbol
        symbols = {}
        for event in self.earnings_db:
            symbols[event.symbol] = symbols.get(event.symbol, 0) + 1
        
        print("\nTop 10 Symbols by Earnings Count:")
        for symbol, count in sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {symbol}: {count}")
        
        # Surprise statistics
        surprises = [e.surprise for e in self.earnings_db]
        if surprises:
            print("\nEarnings Surprise Statistics:")
            print(f"  Mean: {np.mean(surprises):.4f}")
            print(f"  Std: {np.std(surprises):.4f}")
            print(f"  Min: {np.min(surprises):.4f}")
            print(f"  Max: {np.max(surprises):.4f}")
        
        # Upcoming earnings
        upcoming = self.get_upcoming_earnings(days_ahead=30)
        print(f"\nUpcoming Earnings (next 30 days): {len(upcoming)}")
        for event in upcoming[:5]:
            print(f"  {event.symbol}: {event.announcement_date.date()} ({event.fiscal_quarter} FY{event.fiscal_year})")
        
        print("="*60)


class IntradayCorporateActionAdjuster:
    """
    Apply intraday price adjustments for splits, bonuses, and dividends.
    
    This class handles corporate action adjustments at the intraday (tick/minute) level,
    ensuring that adjustments are applied at the exact time of the corporate action.
    """
    
    def __init__(self):
        self.corporate_actions_db: List[CorporateAction] = []
        self.adjustment_cache: Dict[str, Dict[pd.Timestamp, float]] = {}
    
    def add_corporate_action(
        self,
        symbol: str,
        date: pd.Timestamp,
        action_type: str,
        factor: float = 1.0,
        cash_value: float = 0.0,
        record_date: Optional[pd.Timestamp] = None,
        ex_date: Optional[pd.Timestamp] = None,
        action_time: Optional[pd.Timestamp] = None
    ) -> None:
        """
        Add a corporate action to the database with intraday time.
        
        Args:
            symbol: Stock symbol
            date: Action date
            action_type: Type of action
            factor: Adjustment factor
            cash_value: Cash value (for dividends)
            record_date: Record date
            ex_date: Ex-date
            action_time: Exact time of action (for intraday adjustments)
        """
        action = CorporateAction(
            symbol=symbol,
            date=date,
            action_type=action_type,
            factor=factor,
            cash_value=cash_value,
            record_date=record_date,
            ex_date=ex_date
        )
        self.corporate_actions_db.append(action)
        
        # Cache adjustment factor for quick lookup
        if symbol not in self.adjustment_cache:
            self.adjustment_cache[symbol] = {}
        
        # Use ex_date if available, otherwise use action_date
        effective_date = ex_date if ex_date else date
        if action_time:
            effective_datetime = pd.Timestamp.combine(effective_date.date(), action_time.time())
        else:
            effective_datetime = effective_date
        
        if action_type.lower() in {"split", "bonus", "reverse_split"}:
            self.adjustment_cache[symbol][effective_datetime] = factor
        elif action_type.lower() == "dividend":
            self.adjustment_cache[symbol][effective_datetime] = cash_value
    
    def adjust_intraday(
        self,
        frame: pd.DataFrame,
        symbol: str,
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Apply intraday price adjustments for a symbol.
        
        Args:
            frame: Intraday price data DataFrame with timestamps
            symbol: Stock symbol
            timestamp_col: Name of timestamp column
            
        Returns:
            Adjusted DataFrame
        """
        df = frame.copy()
        
        if symbol not in self.adjustment_cache:
            return df
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Get actions for this symbol
        actions = self.adjustment_cache[symbol]
        
        if not actions:
            return df
        
        # Apply adjustments in chronological order
        for action_datetime, adjustment_value in sorted(actions.items()):
            # Determine adjustment type based on value
            if adjustment_value > 1.0 or adjustment_value < 1.0:
                # Factor adjustment (split/bonus)
                mask = df[timestamp_col] < action_datetime
                self._apply_factor_intraday(df, mask, adjustment_value)
            else:
                # Cash adjustment (dividend)
                mask = df[timestamp_col] < action_datetime
                self._apply_dividend_intraday(df, mask, adjustment_value)
        
        return df
    
    def _apply_factor_intraday(self, df: pd.DataFrame, mask: pd.Series, factor: float) -> None:
        """Apply multiplicative factor to intraday prices and volume."""
        price_cols = ["open", "high", "low", "close", "bid", "ask", "mid"]
        for col in price_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] / factor
        if "volume" in df.columns:
            df.loc[mask, "volume"] = df.loc[mask, "volume"] * factor
        if "bid_volume" in df.columns:
            df.loc[mask, "bid_volume"] = df.loc[mask, "bid_volume"] * factor
        if "ask_volume" in df.columns:
            df.loc[mask, "ask_volume"] = df.loc[mask, "ask_volume"] * factor
    
    def _apply_dividend_intraday(self, df: pd.DataFrame, mask: pd.Series, dividend: float) -> None:
        """Apply dividend adjustment to intraday prices."""
        price_cols = ["open", "high", "low", "close", "bid", "ask", "mid"]
        for col in price_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] - dividend
    
    def get_adjustment_factor(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> float:
        """
        Get the cumulative adjustment factor for a symbol at a timestamp.
        
        Args:
            symbol: Stock symbol
            timestamp: Query timestamp
            
        Returns:
            Cumulative adjustment factor (1.0 if no adjustments)
        """
        if symbol not in self.adjustment_cache:
            return 1.0
        
        cumulative_factor = 1.0
        
        for action_datetime, adjustment_value in self.adjustment_cache[symbol].items():
            if action_datetime <= timestamp:
                if adjustment_value > 1.0 or adjustment_value < 1.0:
                    cumulative_factor *= adjustment_value
        
        return cumulative_factor
    
    def adjust_tick(
        self,
        tick: Dict[str, any],
        symbol: str
    ) -> Dict[str, any]:
        """
        Adjust a single tick for corporate actions.
        
        Args:
            tick: Tick data dictionary
            symbol: Stock symbol
            
        Returns:
            Adjusted tick
        """
        adjusted_tick = tick.copy()
        
        if symbol not in self.adjustment_cache:
            return adjusted_tick
        
        timestamp = pd.Timestamp(tick.get('timestamp', tick.get('datetime')))
        cumulative_factor = self.get_adjustment_factor(symbol, timestamp)
        
        # Apply cumulative factor to prices
        price_fields = ['price', 'bid', 'ask', 'mid']
        for field in price_fields:
            if field in adjusted_tick and adjusted_tick[field] is not None:
                adjusted_tick[field] = adjusted_tick[field] / cumulative_factor
        
        # Apply inverse factor to volume
        volume_fields = ['volume', 'bid_volume', 'ask_volume']
        for field in volume_fields:
            if field in adjusted_tick and adjusted_tick[field] is not None:
                adjusted_tick[field] = adjusted_tick[field] * cumulative_factor
        
        return adjusted_tick


class CorporateActionAdjuster:
    """Apply multiplicative price adjustments for splits and bonuses."""

    def __init__(self):
        self.corporate_actions_db: List[CorporateAction] = []
    
    def add_corporate_action(
        self,
        symbol: str,
        date: pd.Timestamp,
        action_type: str,
        factor: float = 1.0,
        cash_value: float = 0.0,
        record_date: Optional[pd.Timestamp] = None,
        ex_date: Optional[pd.Timestamp] = None
    ) -> None:
        """
        Add a corporate action to the database.
        
        Args:
            symbol: Stock symbol
            date: Action date
            action_type: Type of action
            factor: Adjustment factor
            cash_value: Cash value (for dividends)
            record_date: Record date
            ex_date: Ex-date
        """
        action = CorporateAction(
            symbol=symbol,
            date=date,
            action_type=action_type,
            factor=factor,
            cash_value=cash_value,
            record_date=record_date,
            ex_date=ex_date
        )
        self.corporate_actions_db.append(action)
    
    def get_actions_for_symbol(
        self,
        symbol: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> List[CorporateAction]:
        """
        Get corporate actions for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            List of CorporateAction
        """
        actions = [a for a in self.corporate_actions_db if a.symbol == symbol]
        
        if start_date:
            actions = [a for a in actions if a.date >= start_date]
        if end_date:
            actions = [a for a in actions if a.date <= end_date]
        
        return sorted(actions, key=lambda x: x.date)
    
    def adjust(self, frame: pd.DataFrame, actions: Iterable[CorporateAction]) -> pd.DataFrame:
        """
        Apply multiplicative price adjustments for splits and bonuses.
        
        CRITICAL FIX: Ensure all price data is adjusted for splits and dividends.
        Unadjusted prices cause artificial jumps and false signals.
        
        Args:
            frame: Price data DataFrame
            actions: Corporate actions to apply
            
        Returns:
            Adjusted DataFrame
        """
        if not self.enable_auto_adjust:
            return frame
            
        df = frame.copy()
        df["date"] = _to_naive_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for action in sorted(actions, key=lambda item: item.date):
            if action.factor <= 0:
                raise ValueError("corporate action factor must be positive")
            action_date = pd.Timestamp(action.date)
            if getattr(action_date, "tzinfo", None) is not None:
                action_date = action_date.tz_convert(None)
            mask = df["date"] < action_date
            if action.action_type.lower() in {"split", "bonus", "reverse_split"}:
                self._apply_factor(df, mask, action.factor)
            elif action.action_type.lower() == "dividend":
                self._apply_dividend(df, mask, action.cash_value)
        return df
    
    def _apply_factor(self, df: pd.DataFrame, mask: pd.Series, factor: float) -> None:
        """Apply multiplicative factor to prices and volume."""
        price_cols = ["open", "high", "low", "close", "adjusted_close"]
        for col in price_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] / factor
        if "volume" in df.columns:
            df.loc[mask, "volume"] = df.loc[mask, "volume"] * factor
    
    def _apply_dividend(self, df: pd.DataFrame, mask: pd.Series, dividend: float) -> None:
        """Apply dividend adjustment to prices."""
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col] - dividend
    
    def estimate_event_volatility(
        self,
        symbol: str,
        action_type: str,
        days_before: int = 5,
        days_after: int = 5
    ) -> Dict[str, float]:
        """
        Estimate volatility around corporate action events.
        
        Args:
            symbol: Stock symbol
            action_type: Type of action
            days_before: Days before event
            days_after: Days after event
            
        Returns:
            Volatility statistics
        """
        actions = [a for a in self.corporate_actions_db 
                  if a.symbol == symbol and a.action_type == action_type]
        
        if not actions:
            return {}
        
        # Placeholder volatility estimation
        # In production, use actual price data
        vol_before = np.random.uniform(0.02, 0.05)
        vol_after = np.random.uniform(0.03, 0.08)
        vol_spike = vol_after / vol_before
        
        return {
            "volatility_before": vol_before,
            "volatility_after": vol_after,
            "volatility_spike": vol_spike,
            "num_events": len(actions)
        }
    
    def print_calendar_report(self) -> None:
        """Print corporate actions calendar report."""
        print("\n" + "="*60)
        print("CORPORATE ACTIONS CALENDAR REPORT")
        print("="*60)
        print(f"Total corporate actions: {len(self.corporate_actions_db)}")
        
        # Count by action type
        action_types = {}
        for action in self.corporate_actions_db:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
        
        print("\nAction Types:")
        for action_type, count in sorted(action_types.items()):
            print(f"  {action_type}: {count}")
        
        # Count by symbol
        symbols = {}
        for action in self.corporate_actions_db:
            symbols[action.symbol] = symbols.get(action.symbol, 0) + 1
        
        print("\nTop 10 Symbols by Action Count:")
        for symbol, count in sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {symbol}: {count}")
        
        print("="*60)
