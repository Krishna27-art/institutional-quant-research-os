"""
Corporate Actions Database
Handles splits, dividends, bonuses, mergers, delistings, and IPOs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of corporate actions."""
    SPLIT = "split"
    DIVIDEND = "dividend"
    BONUS = "bonus"
    MERGER = "merger"
    DELISTING = "delisting"
    IPO = "ipo"
    RIGHTS_ISSUE = "rights_issue"


@dataclass
class CorporateAction:
    """Represents a corporate action event."""
    symbol: str
    action_type: ActionType
    effective_date: date
    ratio: float = 1.0  # For splits/bonus
    amount: float = 0.0  # For dividends
    new_symbol: Optional[str] = None  # For mergers
    record_date: Optional[date] = None
    announcement_date: Optional[date] = None
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'action_type': self.action_type.value,
            'effective_date': self.effective_date,
            'ratio': self.ratio,
            'amount': self.amount,
            'new_symbol': self.new_symbol,
            'record_date': self.record_date,
            'announcement_date': self.announcement_date
        }


class CorporateActionsDatabase:
    """
    Database for corporate actions with point-in-time reconstruction.
    
    Algorithms:
    1. Action Storage: Store all corporate actions with metadata
    2. Price Adjustment: Adjust historical prices for splits/bonus
    3. Dividend Adjustment: Adjust for dividend payments
    4. Universe Reconstruction: Build tradable universe at each point in time
    5. Survivorship Correction: Include delisted stocks in historical backtests
    """
    
    def __init__(self):
        self.actions: Dict[str, List[CorporateAction]] = {}
        self.ipo_dates: Dict[str, date] = {}
        self.delist_dates: Dict[str, date] = {}
        self.mergers: Dict[str, Tuple[str, date]] = {}  # old_symbol -> (new_symbol, date)
        
    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action to the database."""
        symbol = action.symbol
        
        if symbol not in self.actions:
            self.actions[symbol] = []
        
        self.actions[symbol].append(action)
        
        # Update special databases
        if action.action_type == ActionType.IPO:
            self.ipo_dates[symbol] = action.effective_date
        elif action.action_type == ActionType.DELISTING:
            self.delist_dates[symbol] = action.effective_date
        elif action.action_type == ActionType.MERGER and action.new_symbol:
            self.mergers[symbol] = (action.new_symbol, action.effective_date)
        
        logger.info(f"Added corporate action: {symbol} {action.action_type.value} on {action.effective_date}")
    
    def get_actions(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[CorporateAction]:
        """Get corporate actions for a symbol within date range."""
        if symbol not in self.actions:
            return []
        
        actions = self.actions[symbol]
        
        if start_date:
            actions = [a for a in actions if a.effective_date >= start_date]
        if end_date:
            actions = [a for a in actions if a.effective_date <= end_date]
        
        return sorted(actions, key=lambda a: a.effective_date)
    
    def adjust_prices_for_split(
        self,
        data: pd.DataFrame,
        symbol: str,
        split_ratio: float,
        split_date: date
    ) -> pd.DataFrame:
        """
        Adjust historical prices for stock split.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol being adjusted
            split_ratio: Split ratio (e.g., 2.0 for 2-for-1 split)
            split_date: Effective date of split
            
        Returns:
            Adjusted DataFrame
        """
        if data.empty:
            return data
        
        # Adjust prices before split date
        mask = pd.to_datetime(data.index).date < split_date
        data.loc[mask, ['open', 'high', 'low', 'close']] /= split_ratio
        data.loc[mask, 'volume'] *= split_ratio
        
        logger.info(f"Adjusted {symbol} prices for {split_ratio}:1 split on {split_date}")
        return data
    
    def adjust_prices_for_bonus(
        self,
        data: pd.DataFrame,
        symbol: str,
        bonus_ratio: float,
        bonus_date: date
    ) -> pd.DataFrame:
        """
        Adjust historical prices for bonus issue.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol being adjusted
            bonus_ratio: Bonus ratio (e.g., 1.0 for 1:1 bonus)
            bonus_date: Ex-date of bonus
            
        Returns:
            Adjusted DataFrame
        """
        if data.empty:
            return data
        
        # Adjust prices before bonus date
        adjustment_factor = 1 + bonus_ratio
        mask = pd.to_datetime(data.index).date < bonus_date
        data.loc[mask, ['open', 'high', 'low', 'close']] /= adjustment_factor
        data.loc[mask, 'volume'] *= adjustment_factor
        
        logger.info(f"Adjusted {symbol} prices for {bonus_ratio}:1 bonus on {bonus_date}")
        return data
    
    def adjust_prices_for_dividend(
        self,
        data: pd.DataFrame,
        symbol: str,
        dividend_amount: float,
        ex_date: date
    ) -> pd.DataFrame:
        """
        Adjust prices for dividend payment.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol being adjusted
            dividend_amount: Dividend per share
            ex_date: Ex-dividend date
            
        Returns:
            Adjusted DataFrame
        """
        if data.empty:
            return data
        
        # Adjust prices on and after ex-date
        mask = pd.to_datetime(data.index).date >= ex_date
        data.loc[mask, ['open', 'high', 'low', 'close']] -= dividend_amount
        
        logger.info(f"Adjusted {symbol} prices for ₹{dividend_amount} dividend on {ex_date}")
        return data
    
    def apply_all_actions(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply all corporate actions for a symbol to price data."""
        if symbol not in self.actions:
            return data
        
        actions = sorted(self.actions[symbol], key=lambda a: a.effective_date)
        
        for action in actions:
            if action.action_type == ActionType.SPLIT:
                data = self.adjust_prices_for_split(
                    data, symbol, action.ratio, action.effective_date
                )
            elif action.action_type == ActionType.BONUS:
                data = self.adjust_prices_for_bonus(
                    data, symbol, action.ratio, action.effective_date
                )
            elif action.action_type == ActionType.DIVIDEND:
                data = self.adjust_prices_for_dividend(
                    data, symbol, action.amount, action.effective_date
                )
        
        return data
    
    def was_symbol_tradable(self, symbol: str, check_date: date) -> bool:
        """
        Check if a symbol was tradable on a given date.
        
        Considers IPO dates and delisting dates.
        """
        # Check IPO
        if symbol in self.ipo_dates:
            if check_date < self.ipo_dates[symbol]:
                return False  # Not yet listed
        
        # Check delisting
        if symbol in self.delist_dates:
            if check_date >= self.delist_dates[symbol]:
                return False  # Already delisted
        
        # Check merger
        if symbol in self.mergers:
            merger_date = self.mergers[symbol][1]
            if check_date >= merger_date:
                return False  # Merged away
        
        return True
    
    def get_universe_at_date(self, all_symbols: List[str], check_date: date) -> List[str]:
        """Get list of tradable symbols at a specific date."""
        tradable = []
        for symbol in all_symbols:
            if self.was_symbol_tradable(symbol, check_date):
                tradable.append(symbol)
        return tradable
    
    def get_survivorship_corrected_universe(
        self,
        all_symbols: List[str],
        start_date: date,
        end_date: date
    ) -> Dict[date, List[str]]:
        """
        Build survivorship-corrected universe over date range.
        
        Includes delisted stocks that were tradable at each point in time.
        """
        date_range = pd.date_range(start_date, end_date, freq='D')
        universe = {}
        
        for dt in date_range:
            check_date = dt.date()
            universe[check_date] = self.get_universe_at_date(all_symbols, check_date)
        
        return universe
    
    def load_from_csv(self, csv_path: str) -> None:
        """Load corporate actions from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            
            for _, row in df.iterrows():
                action = CorporateAction(
                    symbol=row['symbol'],
                    action_type=ActionType(row['action_type']),
                    effective_date=pd.to_datetime(row['effective_date']).date(),
                    ratio=row.get('ratio', 1.0),
                    amount=row.get('amount', 0.0),
                    new_symbol=row.get('new_symbol'),
                    record_date=pd.to_datetime(row['record_date']).date() if pd.notna(row.get('record_date')) else None,
                    announcement_date=pd.to_datetime(row['announcement_date']).date() if pd.notna(row.get('announcement_date')) else None
                )
                self.add_action(action)
            
            logger.info(f"Loaded {len(df)} corporate actions from {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to load corporate actions from CSV: {e}")
    
    def save_to_csv(self, csv_path: str) -> None:
        """Save corporate actions to CSV file."""
        rows = []
        
        for symbol, actions in self.actions.items():
            for action in actions:
                rows.append(action.to_dict())
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(rows)} corporate actions to {csv_path}")


# Example usage and testing
if __name__ == "__main__":
    # Test Corporate Actions Database
    print("Testing Corporate Actions Database...")
    
    db = CorporateActionsDatabase()
    
    # Add some sample actions
    from datetime import date
    
    # RELIANCE 1:1 bonus
    bonus_action = CorporateAction(
        symbol="RELIANCE",
        action_type=ActionType.BONUS,
        effective_date=date(2017, 9, 7),
        ratio=1.0,
        announcement_date=date(2017, 8, 10)
    )
    db.add_action(bonus_action)
    
    # HDFCBANK split
    split_action = CorporateAction(
        symbol="HDFCBANK",
        action_type=ActionType.SPLIT,
        effective_date=date(2019, 9, 19),
        ratio=5.0,  # 5-for-1 split
        announcement_date=date(2019, 7, 23)
    )
    db.add_action(split_action)
    
    # Sample IPO
    ipo_action = CorporateAction(
        symbol="NEWSTOCK",
        action_type=ActionType.IPO,
        effective_date=date(2020, 1, 15),
        announcement_date=date(2019, 12, 20)
    )
    db.add_action(ipo_action)
    
    # Test price adjustment
    dates = pd.date_range("2019-08-01", "2019-10-31", freq="D")
    prices = np.random.normal(1000, 50, len(dates)).cumsum()
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    print(f"\nOriginal data shape: {data.shape}")
    print(f"Original close price (first): {data['close'].iloc[0]:.2f}")
    
    # Apply split adjustment
    adjusted = db.adjust_prices_for_split(data, "HDFCBANK", 5.0, date(2019, 9, 19))
    
    print(f"Adjusted close price (first): {adjusted['close'].iloc[0]:.2f}")
    print(f"Expected: ~{1000 / 5:.2f}")
    
    # Test tradability check
    print(f"\nNEWSTOCK tradable on 2019-12-01: {db.was_symbol_tradable('NEWSTOCK', date(2019, 12, 1))}")
    print(f"NEWSTOCK tradable on 2020-02-01: {db.was_symbol_tradable('NEWSTOCK', date(2020, 2, 1))}")
    
    print("\nCorporate Actions Database test completed.")
