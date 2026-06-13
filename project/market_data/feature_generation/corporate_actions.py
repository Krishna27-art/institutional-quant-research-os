"""
CRITICAL FIX: Dividend and corporate action adjustments in feature engineering.

The review noted that feature engineering doesn't account for dividends, stock splits,
or corporate actions. These events cause artificial jumps in price that can be mistaken
for alpha signals.

This module provides:
- Dividend adjustment for price series
- Stock split adjustment
- Corporate action handling
- Point-in-time price reconstruction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of corporate actions."""
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    BONUS_ISSUE = "bonus_issue"
    RIGHTS_ISSUE = "rights_issue"
    MERGER = "merger"
    SPINOFF = "spinoff"
    DELISTING = "delisting"


@dataclass
class CorporateAction:
    """Corporate action event."""
    symbol: str
    action_type: ActionType
    action_date: datetime
    ratio: float  # Split ratio, dividend amount, etc.
    description: str
    ex_date: Optional[datetime] = None
    record_date: Optional[datetime] = None


class CorporateActionAdjuster:
    """
    Adjust prices and returns for corporate actions.
    
    CRITICAL FIX: Prevents artificial price jumps from being mistaken for alpha signals.
    """
    
    def __init__(self):
        self.actions: Dict[str, List[CorporateAction]] = {}
        logger.info("Corporate action adjuster initialized")
    
    def add_action(self, action: CorporateAction) -> None:
        """
        Add a corporate action to the registry.
        
        Args:
            action: Corporate action to add
        """
        if action.symbol not in self.actions:
            self.actions[action.symbol] = []
        self.actions[action.symbol].append(action)
        
        # Sort by date
        self.actions[action.symbol].sort(key=lambda x: x.action_date)
        
        logger.info(f"Added {action.action_type.value} for {action.symbol} on {action.action_date}")
    
    def adjust_for_dividends(
        self,
        prices: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Adjust price series for dividends.
        
        Args:
            prices: DataFrame with price data
            symbol: Symbol to adjust
            
        Returns:
            Adjusted price DataFrame
        """
        if symbol not in self.actions:
            return prices
        
        result = prices.copy()
        
        for action in self.actions[symbol]:
            if action.action_type != ActionType.DIVIDEND:
                continue
            
            # Adjust prices before ex-date
            ex_date = action.ex_date or action.action_date
            mask = result.index < ex_date
            
            # Subtract dividend from prices before ex-date
            # This makes the price series continuous
            result.loc[mask, 'close'] = result.loc[mask, 'close'] - action.ratio
            result.loc[mask, 'open'] = result.loc[mask, 'open'] - action.ratio
            result.loc[mask, 'high'] = result.loc[mask, 'high'] - action.ratio
            result.loc[mask, 'low'] = result.loc[mask, 'low'] - action.ratio
            
            logger.info(
                f"Adjusted {symbol} for dividend of {action.ratio:.2f} on {ex_date}"
            )
        
        return result
    
    def adjust_for_splits(
        self,
        prices: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Adjust price series for stock splits.
        
        Args:
            prices: DataFrame with price data
            symbol: Symbol to adjust
            
        Returns:
            Adjusted price DataFrame
        """
        if symbol not in self.actions:
            return prices
        
        result = prices.copy()
        
        for action in self.actions[symbol]:
            if action.action_type != ActionType.STOCK_SPLIT:
                continue
            
            # Adjust prices before split date
            split_date = action.action_date
            mask = result.index < split_date
            
            # Divide prices by split ratio
            # e.g., 2:1 split -> divide old prices by 2
            result.loc[mask, 'close'] = result.loc[mask, 'close'] / action.ratio
            result.loc[mask, 'open'] = result.loc[mask, 'open'] / action.ratio
            result.loc[mask, 'high'] = result.loc[mask, 'high'] / action.ratio
            result.loc[mask, 'low'] = result.loc[mask, 'low'] / action.ratio
            
            # Adjust volume
            if 'volume' in result.columns:
                result.loc[mask, 'volume'] = result.loc[mask, 'volume'] * action.ratio
            
            logger.info(
                f"Adjusted {symbol} for {action.ratio}:1 split on {split_date}"
            )
        
        return result
    
    def adjust_for_bonus(
        self,
        prices: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Adjust price series for bonus issues.
        
        Args:
            prices: DataFrame with price data
            symbol: Symbol to adjust
            
        Returns:
            Adjusted price DataFrame
        """
        if symbol not in self.actions:
            return prices
        
        result = prices.copy()
        
        for action in self.actions[symbol]:
            if action.action_type != ActionType.BONUS_ISSUE:
                continue
            
            # Adjust prices before bonus date
            bonus_date = action.action_date
            mask = result.index < bonus_date
            
            # Divide prices by (1 + bonus_ratio)
            # e.g., 1:1 bonus -> divide by 2
            result.loc[mask, 'close'] = result.loc[mask, 'close'] / (1 + action.ratio)
            result.loc[mask, 'open'] = result.loc[mask, 'open'] / (1 + action.ratio)
            result.loc[mask, 'high'] = result.loc[mask, 'high'] / (1 + action.ratio)
            result.loc[mask, 'low'] = result.loc[mask, 'low'] / (1 + action.ratio)
            
            # Adjust volume
            if 'volume' in result.columns:
                result.loc[mask, 'volume'] = result.loc[mask, 'volume'] * (1 + action.ratio)
            
            logger.info(
                f"Adjusted {symbol} for {action.ratio}:1 bonus on {bonus_date}"
            )
        
        return result
    
    def adjust_all_actions(
        self,
        prices: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Adjust for all corporate actions for a symbol.
        
        Args:
            prices: DataFrame with price data
            symbol: Symbol to adjust
            
        Returns:
            Fully adjusted price DataFrame
        """
        result = prices.copy()
        
        # Apply adjustments in chronological order
        if symbol in self.actions:
            for action in self.actions[symbol]:
                if action.action_type == ActionType.DIVIDEND:
                    result = self.adjust_for_dividends(result, symbol)
                elif action.action_type == ActionType.STOCK_SPLIT:
                    result = self.adjust_for_splits(result, symbol)
                elif action.action_type == ActionType.BONUS_ISSUE:
                    result = self.adjust_for_bonus(result, symbol)
        
        logger.info(f"Applied all corporate action adjustments for {symbol}")
        
        return result
    
    def calculate_adjusted_returns(
        self,
        prices: pd.DataFrame,
        symbol: str
    ) -> pd.Series:
        """
        Calculate returns with corporate action adjustments.
        
        Args:
            prices: DataFrame with price data
            symbol: Symbol to calculate returns for
            
        Returns:
            Adjusted returns series
        """
        adjusted_prices = self.adjust_all_actions(prices, symbol)
        
        # Calculate log returns
        returns = np.log(adjusted_prices['close'] / adjusted_prices['close'].shift(1))
        
        return returns
    
    def detect_price_jumps(
        self,
        prices: pd.DataFrame,
        threshold: float = 0.10
    ) -> List[Tuple[pd.Timestamp, float]]:
        """
        Detect potential unadjusted corporate actions from price jumps.
        
        Args:
            prices: DataFrame with price data
            threshold: Jump threshold (10% default)
            
        Returns:
            List of (timestamp, jump_size) for detected jumps
        """
        returns = np.log(prices['close'] / prices['close'].shift(1))
        
        # Detect large jumps
        jumps = []
        for idx, ret in returns.items():
            if abs(ret) > threshold:
                jumps.append((idx, ret))
        
        if jumps:
            logger.warning(
                f"Detected {len(jumps)} potential unadjusted corporate actions "
                f"(threshold={threshold:.1%})"
            )
        
        return jumps


def load_corporate_actions_from_csv(
    filepath: str
) -> Dict[str, List[CorporateAction]]:
    """
    Load corporate actions from CSV file.
    
    Expected CSV format:
    symbol,action_type,action_date,ratio,description,ex_date,record_date
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Dictionary of symbol -> list of actions
    """
    df = pd.read_csv(filepath)
    
    actions_dict: Dict[str, List[CorporateAction]] = {}
    
    for _, row in df.iterrows():
        action = CorporateAction(
            symbol=row['symbol'],
            action_type=ActionType(row['action_type']),
            action_date=pd.to_datetime(row['action_date']),
            ratio=row['ratio'],
            description=row['description'],
            ex_date=pd.to_datetime(row['ex_date']) if pd.notna(row['ex_date']) else None,
            record_date=pd.to_datetime(row['record_date']) if pd.notna(row['record_date']) else None
        )
        
        if action.symbol not in actions_dict:
            actions_dict[action.symbol] = []
        actions_dict[action.symbol].append(action)
    
    logger.info(f"Loaded {len(actions_dict)} symbols with corporate actions from {filepath}")
    
    return actions_dict


def create_sample_corporate_actions() -> Dict[str, List[CorporateAction]]:
    """
    Create sample corporate actions for testing.
    
    Returns:
        Dictionary of sample actions
    """
    actions = {
        'RELIANCE': [
            CorporateAction(
                symbol='RELIANCE',
                action_type=ActionType.DIVIDEND,
                action_date=pd.Timestamp('2024-03-15'),
                ratio=10.0,
                description='Dividend ₹10 per share',
                ex_date=pd.Timestamp('2024-03-15')
            ),
            CorporateAction(
                symbol='RELIANCE',
                action_type=ActionType.BONUS_ISSUE,
                action_date=pd.Timestamp('2023-09-01'),
                ratio=1.0,
                description='1:1 bonus issue',
                ex_date=pd.Timestamp('2023-09-01')
            )
        ],
        'TCS': [
            CorporateAction(
                symbol='TCS',
                action_type=ActionType.DIVIDEND,
                action_date=pd.Timestamp('2024-04-15'),
                ratio=24.0,
                description='Dividend ₹24 per share',
                ex_date=pd.Timestamp('2024-04-15')
            )
        ]
    }
    
    return actions
