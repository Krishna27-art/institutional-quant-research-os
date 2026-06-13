"""
Agency Theory Monitor - Level 1 Foundation

This module provides monitoring of corporate events based on agency theory (Jensen & Meckling 1976):
- Share repurchase announcements
- Insider trading activity
- M&A announcements
- CEO changes
- Corporate governance events

Based on Audit Report Priority 1: Economics & Market Microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of corporate events."""
    BUYBACK = "buyback"
    INSIDER_BUY = "insider_buy"
    INSIDER_SELL = "insider_sell"
    MA_ANNOUNCEMENT = "ma_announcement"
    CEO_CHANGE = "ceo_change"
    SECONDARY_OFFERING = "secondary_offering"
    DIVIDEND_CHANGE = "dividend_change"
    EARNINGS_SURPRISE = "earnings_surprise"


@dataclass
class CorporateEvent:
    """Corporate event data."""
    symbol: str
    event_type: EventType
    announcement_date: datetime
    details: Dict[str, Union[str, float, int]]
    
    def __post_init__(self):
        """Validate event data."""
        if not isinstance(self.symbol, str) or len(self.symbol) == 0:
            raise ValueError("Symbol must be a non-empty string")
        if not isinstance(self.event_type, EventType):
            raise ValueError("Event type must be EventType enum")


class AgencyTheoryMonitor:
    """
    Agency theory event monitor.
    
    This class monitors corporate events that create agency-related alpha
    based on Jensen & Meckling (1976) theory that managers may not act
    in shareholder interest.
    """
    
    def __init__(self):
        """Initialize agency theory monitor."""
        self.events: List[CorporateEvent] = []
        self.event_cache: Dict[str, List[CorporateEvent]] = {}
    
    def corporate_actions_calendar(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[CorporateEvent]:
        """
        Get corporate actions calendar for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for calendar
            end_date: End date for calendar
            
        Returns:
            List of corporate events in date range
        """
        # In production, this would query a database or API
        # For now, return cached events
        if symbol in self.event_cache:
            return [
                event for event in self.event_cache[symbol]
                if start_date <= event.announcement_date <= end_date
            ]
        
        return []
    
    def buyback_detector(
        self,
        announcements: List[Dict[str, Union[str, float, int]]]
    ) -> List[CorporateEvent]:
        """
        Detect share repurchase announcements.
        
        Args:
            announcements: List of buyback announcements
            
        Returns:
            List of buyback events
        """
        events = []
        
        for announcement in announcements:
            symbol = announcement.get('symbol')
            date_str = announcement.get('date')
            amount = announcement.get('amount', 0)
            pct_of_shares = announcement.get('pct_of_shares', 0)
            
            if symbol and date_str:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    event = CorporateEvent(
                        symbol=symbol,
                        event_type=EventType.BUYBACK,
                        announcement_date=date,
                        details={
                            'amount': amount,
                            'pct_of_shares': pct_of_shares,
                            'announcement_date': date_str,
                        }
                    )
                    events.append(event)
                    
                    # Add to cache
                    if symbol not in self.event_cache:
                        self.event_cache[symbol] = []
                    self.event_cache[symbol].append(event)
                    
                except ValueError:
                    logger.warning(f"Invalid date format for buyback: {date_str}")
        
        return events
    
    def insider_trading_monitor(
        self,
        insider_data: List[Dict[str, Union[str, float, int]]]
    ) -> List[CorporateEvent]:
        """
        Monitor insider trading activity.
        
        Args:
            insider_data: List of insider trading records
            
        Returns:
            List of insider trading events
        """
        events = []
        
        for record in insider_data:
            symbol = record.get('symbol')
            date_str = record.get('date')
            transaction_type = record.get('transaction_type', '').lower()
            shares = record.get('shares', 0)
            price = record.get('price', 0)
            
            if symbol and date_str and transaction_type:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if 'buy' in transaction_type:
                        event_type = EventType.INSIDER_BUY
                    elif 'sell' in transaction_type:
                        event_type = EventType.INSIDER_SELL
                    else:
                        continue
                    
                    event = CorporateEvent(
                        symbol=symbol,
                        event_type=event_type,
                        announcement_date=date,
                        details={
                            'shares': shares,
                            'price': price,
                            'transaction_type': transaction_type,
                            'date': date_str,
                        }
                    )
                    events.append(event)
                    
                    # Add to cache
                    if symbol not in self.event_cache:
                        self.event_cache[symbol] = []
                    self.event_cache[symbol].append(event)
                    
                except ValueError:
                    logger.warning(f"Invalid date format for insider trade: {date_str}")
        
        return events
    
    def ma_detector(
        self,
        news: List[Dict[str, str]],
        filings: List[Dict[str, str]]
    ) -> List[CorporateEvent]:
        """
        Detect M&A announcements from news and filings.
        
        Args:
            news: List of news articles
            filings: List of SEC filings
            
        Returns:
            List of M&A events
        """
        events = []
        
        # Check news for M&A keywords
        ma_keywords = ['acquisition', 'merger', 'takeover', 'buyout', 'acquire']
        
        for article in news:
            symbol = article.get('symbol')
            date_str = article.get('date')
            headline = article.get('headline', '').lower()
            
            if symbol and date_str and any(kw in headline for kw in ma_keywords):
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    event = CorporateEvent(
                        symbol=symbol,
                        event_type=EventType.MA_ANNOUNCEMENT,
                        announcement_date=date,
                        details={
                            'headline': article.get('headline'),
                            'date': date_str,
                        }
                    )
                    events.append(event)
                    
                    # Add to cache
                    if symbol not in self.event_cache:
                        self.event_cache[symbol] = []
                    self.event_cache[symbol].append(event)
                    
                except ValueError:
                    logger.warning(f"Invalid date format for M&A news: {date_str}")
        
        # Check filings for M&A
        for filing in filings:
            symbol = filing.get('symbol')
            date_str = filing.get('date')
            filing_type = filing.get('type', '').lower()
            
            if symbol and date_str and 's-4' in filing_type or 'definitive agreement' in filing.get('description', '').lower():
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    event = CorporateEvent(
                        symbol=symbol,
                        event_type=EventType.MA_ANNOUNCEMENT,
                        announcement_date=date,
                        details={
                            'filing_type': filing.get('type'),
                            'description': filing.get('description'),
                            'date': date_str,
                        }
                    )
                    events.append(event)
                    
                    # Add to cache
                    if symbol not in self.event_cache:
                        self.event_cache[symbol] = []
                    self.event_cache[symbol].append(event)
                    
                except ValueError:
                    logger.warning(f"Invalid date format for M&A filing: {date_str}")
        
        return events
    
    def ceo_change_detector(
        self,
        news: List[Dict[str, str]]
    ) -> List[CorporateEvent]:
        """
        Detect CEO changes from news.
        
        Args:
            news: List of news articles
            
        Returns:
            List of CEO change events
        """
        events = []
        
        ceo_keywords = ['ceo', 'chief executive officer', 'chief executive', 'resign', 'appoint', 'replace']
        
        for article in news:
            symbol = article.get('symbol')
            date_str = article.get('date')
            headline = article.get('headline', '').lower()
            
            if symbol and date_str and any(kw in headline for kw in ceo_keywords):
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    event = CorporateEvent(
                        symbol=symbol,
                        event_type=EventType.CEO_CHANGE,
                        announcement_date=date,
                        details={
                            'headline': article.get('headline'),
                            'date': date_str,
                        }
                    )
                    events.append(event)
                    
                    # Add to cache
                    if symbol not in self.event_cache:
                        self.event_cache[symbol] = []
                    self.event_cache[symbol].append(event)
                    
                except ValueError:
                    logger.warning(f"Invalid date format for CEO change: {date_str}")
        
        return events
    
    def calculate_agency_alpha(
        self,
        symbol: str,
        events: List[CorporateEvent],
        lookback_days: int = 30
    ) -> float:
        """
        Calculate agency-based alpha for a stock.
        
        Based on Jensen & Meckling (1976), certain events indicate
        management alignment or misalignment with shareholders.
        
        Args:
            symbol: Stock symbol
            events: List of corporate events
            lookback_days: Lookback period in days
            
        Returns:
            Expected return from agency events
        """
        if not events:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_events = [e for e in events if e.announcement_date >= cutoff_date]
        
        alpha = 0.0
        
        for event in recent_events:
            if event.event_type == EventType.BUYBACK:
                # Buybacks are positive - management returning capital
                pct_shares = event.details.get('pct_of_shares', 0)
                alpha += 0.02 * pct_shares  # 2% per 1% of shares repurchased
            
            elif event.event_type == EventType.INSIDER_BUY:
                # Insider buying is positive signal
                shares = event.details.get('shares', 0)
                alpha += 0.01  # 1% expected return from insider buying
            
            elif event.event_type == EventType.INSIDER_SELL:
                # Insider selling is negative signal
                shares = event.details.get('shares', 0)
                alpha -= 0.015  # -1.5% expected return from insider selling
            
            elif event.event_type == EventType.MA_ANNOUNCEMENT:
                # M&A can be positive or negative depending on context
                alpha += 0.02  # Simplified: assume positive
            
            elif event.event_type == EventType.CEO_CHANGE:
                # CEO change is uncertain - small positive or negative
                alpha += 0.005  # 0.5% expected return (uncertain)
            
            elif event.event_type == EventType.SECONDARY_OFFERING:
                # Secondary offering is negative - dilution
                alpha -= 0.015  # -1.5% expected return
        
        # Cap alpha at reasonable limits
        alpha = max(min(alpha, 0.10), -0.10)
        
        return alpha
    
    def event_driven_signals(
        self,
        events: List[CorporateEvent],
        min_confidence: float = 0.5
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Generate event-driven trading signals.
        
        Args:
            events: List of corporate events
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary of signals by symbol
        """
        signals = {}
        
        # Group events by symbol
        events_by_symbol = {}
        for event in events:
            if event.symbol not in events_by_symbol:
                events_by_symbol[event.symbol] = []
            events_by_symbol[event.symbol].append(event)
        
        # Generate signals for each symbol
        for symbol, symbol_events in events_by_symbol.items():
            alpha = self.calculate_agency_alpha(symbol, symbol_events)
            
            if abs(alpha) > 0.01:  # Only signal if alpha > 1%
                confidence = min(abs(alpha) / 0.05, 1.0)  # Scale alpha to confidence
                
                if confidence >= min_confidence:
                    signals[symbol] = {
                        'signal': 'LONG' if alpha > 0 else 'SHORT',
                        'expected_return': alpha,
                        'confidence': confidence,
                        'event_count': len(symbol_events),
                        'latest_event_date': max(e.announcement_date for e in symbol_events).isoformat(),
                    }
        
        return signals
    
    def get_recent_events(
        self,
        symbol: str,
        days: int = 30
    ) -> List[CorporateEvent]:
        """
        Get recent events for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            List of recent events
        """
        if symbol not in self.event_cache:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = [
            event for event in self.event_cache[symbol]
            if event.announcement_date >= cutoff_date
        ]
        
        return sorted(recent_events, key=lambda e: e.announcement_date, reverse=True)
    
    def add_event(
        self,
        event: Optional[CorporateEvent] = None,
        symbol: Optional[str] = None,
        event_type: Optional[EventType] = None,
        date: Optional[datetime] = None,
        details: Optional[Dict] = None
    ) -> None:
        """
        Add a corporate event to the monitor.
        
        Args:
            event: Corporate event to add
            symbol: Stock symbol (alternative)
            event_type: EventType (alternative)
            date: Event date (alternative)
            details: Event details (alternative)
        """
        if event is None:
            if symbol is None or event_type is None or date is None:
                raise ValueError("Must provide either event object or symbol, event_type, and date")
            event = CorporateEvent(
                symbol=symbol,
                event_type=event_type,
                announcement_date=date,
                details=details or {}
            )
            
        self.events.append(event)
        
        if event.symbol not in self.event_cache:
            self.event_cache[event.symbol] = []
        self.event_cache[event.symbol].append(event)
    
    def clear_cache(self) -> None:
        """Clear the event cache."""
        self.events = []
        self.event_cache = {}
