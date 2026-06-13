"""
Corporate Event Database
Based on the critique: Build data moat with corporate event data

Critical for institutional edge:
- Earnings announcements
- Dividend declarations
- Stock splits/bonus issues
- M&A announcements
- Board meetings
- Corporate actions
- Regulatory filings
- Management guidance

Data Sources:
- Company disclosures (NSE/BSE)
- Regulatory filings (SEBI)
- Earnings transcripts
- News aggregators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class EventType(Enum):
    """Types of corporate events."""
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    STOCK_SPLIT = "stock_split"
    BONUS_ISSUE = "bonus_issue"
    RIGHTS_ISSUE = "rights_issue"
    MERGER_ACQUISITION = "merger_acquisition"
    BOARD_MEETING = "board_meeting"
    AGM = "annual_general_meeting"
    BUYBACK = "buyback"
    DELISTING = "delisting"
    SPINOFF = "spinoff"
    REGULATORY = "regulatory"
    GUIDANCE = "guidance"


class EventImpact(Enum):
    """Expected impact of event."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class CorporateEvent:
    """Corporate event data."""
    event_id: str
    symbol: str
    event_type: EventType
    announcement_date: datetime
    event_date: datetime
    impact: EventImpact
    description: str
    details: Dict
    confidence: float  # 0 to 1
    
    # Post-event metrics (filled after event)
    actual_price_change: Optional[float] = None
    actual_volume_change: Optional[float] = None
    actual_volatility_change: Optional[float] = None


@dataclass
class EarningsData:
    """Earnings data."""
    event_id: str
    symbol: str
    announcement_date: datetime
    quarter: str
    fiscal_year: int
    reported_eps: float
    expected_eps: float
    revenue: float
    expected_revenue: float
    guidance: str
    transcript_url: Optional[str] = None
    
    @property
    def eps_surprise(self) -> float:
        """EPS surprise as percentage."""
        return (self.reported_eps - self.expected_eps) / self.expected_eps if self.expected_eps > 0 else 0
    
    @property
    def revenue_surprise(self) -> float:
        """Revenue surprise as percentage."""
        return (self.revenue - self.expected_revenue) / self.expected_revenue if self.expected_revenue > 0 else 0


class CorporateEventDatabase:
    """
    Database for corporate events.
    
    Features:
    - Event tracking and storage
    - Event impact prediction
    - Historical event analysis
    - Earnings surprise calculation
    - Event-based signal generation
    - Pre/post event analysis
    """
    
    def __init__(self):
        self.events: Dict[str, List[CorporateEvent]] = {}  # symbol -> events
        self.earnings_data: Dict[str, List[EarningsData]] = {}  # symbol -> earnings
        self.event_signals: List[Dict] = []
        
        # Configuration
        self.lookback_days = 90  # For event history
        self.impact_threshold = 0.02  # 2% price change threshold
    
    def add_event(self, event: CorporateEvent) -> None:
        """Add corporate event."""
        symbol = event.symbol
        
        if symbol not in self.events:
            self.events[symbol] = []
        
        self.events[symbol].append(event)
    
    def add_earnings(self, earnings: EarningsData) -> None:
        """Add earnings data."""
        symbol = earnings.symbol
        
        if symbol not in self.earnings_data:
            self.earnings_data[symbol] = []
        
        self.earnings_data[symbol].append(earnings)
    
    def get_upcoming_events(
        self,
        symbol: str,
        days_ahead: int = 30
    ) -> List[CorporateEvent]:
        """
        Get upcoming events for a symbol.
        
        Args:
            symbol: Trading symbol
            days_ahead: Number of days ahead to look
            
        Returns:
            List of upcoming events
        """
        if symbol not in self.events:
            return []
        
        now = datetime.now()
        end_date = now + timedelta(days=days_ahead)
        
        return [
            event for event in self.events[symbol]
            if now < event.event_date <= end_date
        ]
    
    def get_past_events(
        self,
        symbol: str,
        days_back: int = 90
    ) -> List[CorporateEvent]:
        """
        Get past events for a symbol.
        
        Args:
            symbol: Trading symbol
            days_back: Number of days back to look
            
        Returns:
            List of past events
        """
        if symbol not in self.events:
            return []
        
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        
        return [
            event for event in self.events[symbol]
            if start_date <= event.event_date <= now
        ]
    
    def analyze_event_impact(
        self,
        symbol: str,
        event_type: EventType,
        days_before: int = 5,
        days_after: int = 5
    ) -> Dict[str, float]:
        """
        Analyze historical impact of event type.
        
        Args:
            symbol: Trading symbol
            event_type: Type of event to analyze
            days_before: Days before event to measure
            days_after: Days after event to measure
            
        Returns:
            Dictionary with impact metrics
        """
        if symbol not in self.events:
            return {}
        
        events = [
            event for event in self.events[symbol]
            if event.event_type == event_type
            and event.actual_price_change is not None
        ]
        
        if not events:
            return {}
        
        price_changes = [event.actual_price_change for event in events]
        
        return {
            'avg_price_change': np.mean(price_changes),
            'median_price_change': np.median(price_changes),
            'positive_events': len([c for c in price_changes if c > 0]),
            'negative_events': len([c for c in price_changes if c < 0]),
            'win_rate': len([c for c in price_changes if c > 0]) / len(price_changes),
            'num_events': len(events)
        }
    
    def generate_earnings_signal(
        self,
        symbol: str,
        earnings: EarningsData
    ) -> Dict:
        """
        Generate signal from earnings data.
        
        Args:
            symbol: Trading symbol
            earnings: Earnings data
            
        Returns:
            Dictionary with signal
        """
        # Calculate surprises
        eps_surprise = earnings.eps_surprise
        revenue_surprise = earnings.revenue_surprise
        
        # Determine direction
        if eps_surprise > 0.05 and revenue_surprise > 0.05:
            direction = "bullish"
            strength = min((eps_surprise + revenue_surprise) / 0.2, 1.0)
            reasoning = f"Strong beat: EPS surprise {eps_surprise:.1%}, Revenue surprise {revenue_surprise:.1%}"
        elif eps_surprise < -0.05 or revenue_surprise < -0.05:
            direction = "bearish"
            strength = min(abs(eps_surprise + revenue_surprise) / 0.2, 1.0)
            reasoning = f"Miss: EPS surprise {eps_surprise:.1%}, Revenue surprise {revenue_surprise:.1%}"
        else:
            direction = "neutral"
            strength = 0.0
            reasoning = f"In-line results: EPS surprise {eps_surprise:.1%}, Revenue surprise {revenue_surprise:.1%}"
        
        # Check historical impact
        historical_impact = self.analyze_event_impact(symbol, EventType.EARNINGS)
        win_rate = historical_impact.get('win_rate', 0.5)
        
        signal = {
            'symbol': symbol,
            'event_type': 'earnings',
            'direction': direction,
            'strength': strength,
            'confidence': win_rate,
            'reasoning': reasoning,
            'eps_surprise': eps_surprise,
            'revenue_surprise': revenue_surprise,
            'historical_win_rate': win_rate
        }
        
        self.event_signals.append(signal)
        return signal
    
    def get_event_calendar(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get event calendar for date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with events
        """
        data = []
        
        for symbol, events in self.events.items():
            for event in events:
                if start_date <= event.event_date <= end_date:
                    data.append({
                        'date': event.event_date,
                        'symbol': symbol,
                        'event_type': event.event_type.value,
                        'impact': event.impact.value,
                        'description': event.description
                    })
        
        return pd.DataFrame(data)
    
    def get_earnings_calendar(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get earnings calendar for date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with earnings
        """
        data = []
        
        for symbol, earnings_list in self.earnings_data.items():
            for earnings in earnings_list:
                if start_date <= earnings.announcement_date <= end_date:
                    data.append({
                        'date': earnings.announcement_date,
                        'symbol': symbol,
                        'quarter': earnings.quarter,
                        'fiscal_year': earnings.fiscal_year,
                        'expected_eps': earnings.expected_eps,
                        'guidance': earnings.guidance
                    })
        
        return pd.DataFrame(data)
    
    def update_event_outcome(
        self,
        event_id: str,
        price_change: float,
        volume_change: float,
        volatility_change: float
    ) -> bool:
        """
        Update event with actual outcome.
        
        Args:
            event_id: Event ID
            price_change: Actual price change
            volume_change: Actual volume change
            volatility_change: Actual volatility change
            
        Returns:
            True if updated successfully
        """
        for symbol, events in self.events.items():
            for event in events:
                if event.event_id == event_id:
                    event.actual_price_change = price_change
                    event.actual_volume_change = volume_change
                    event.actual_volatility_change = volatility_change
                    return True
        return False
    
    def get_event_summary(self) -> pd.DataFrame:
        """Get summary of recent event signals."""
        data = []
        
        for signal in self.event_signals[-20:]:  # Last 20 signals
            data.append({
                'symbol': signal['symbol'],
                'event_type': signal['event_type'],
                'direction': signal['direction'],
                'strength': signal['strength'],
                'confidence': signal['confidence'],
                'reasoning': signal['reasoning']
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the Corporate Event Database
    print("Testing Corporate Event Database...")
    
    db = CorporateEventDatabase()
    
    # Generate sample corporate events
    print("\nGenerating sample corporate events...")
    base_date = datetime.now()
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    
    for i in range(30):
        for symbol in symbols:
            event = CorporateEvent(
                event_id=f"{symbol}_event_{i}",
                symbol=symbol,
                event_type=EventType.EARNINGS if i % 3 == 0 else EventType.DIVIDEND if i % 3 == 1 else EventType.BOARD_MEETING,
                announcement_date=base_date - timedelta(days=i),
                event_date=base_date - timedelta(days=i) + timedelta(days=1),
                impact=EventImpact.POSITIVE if i % 2 == 0 else EventImpact.NEGATIVE,
                description=f"Sample event for {symbol}",
                details={'value': np.random.uniform(100, 1000)},
                confidence=np.random.uniform(0.7, 0.9)
            )
            db.add_event(event)
    
    print(f"Added events for {len(db.events)} symbols")
    
    # Generate sample earnings data
    print("\nGenerating sample earnings data...")
    for i in range(10):
        for symbol in symbols:
            expected_eps = np.random.uniform(10, 20)
            expected_revenue = np.random.uniform(10000, 20000)
            
            earnings = EarningsData(
                event_id=f"{symbol}_earnings_{i}",
                symbol=symbol,
                announcement_date=base_date - timedelta(days=i * 30),
                quarter=f"Q{(i % 4) + 1}",
                fiscal_year=2024,
                reported_eps=expected_eps * np.random.uniform(0.9, 1.1),
                expected_eps=expected_eps,
                revenue=expected_revenue * np.random.uniform(0.9, 1.1),
                expected_revenue=expected_revenue,
                guidance="positive" if i % 2 == 0 else "neutral"
            )
            db.add_earnings(earnings)
    
    print(f"Added earnings data for {len(db.earnings_data)} symbols")
    
    # Get upcoming events
    print("\nGetting upcoming events for RELIANCE...")
    upcoming = db.get_upcoming_events('RELIANCE', days_ahead=30)
    print(f"Upcoming events: {len(upcoming)}")
    for event in upcoming[:5]:
        print(f"  {event.event_type.value} on {event.event_date.date()}")
    
    # Get past events
    print("\nGetting past events for RELIANCE...")
    past = db.get_past_events('RELIANCE', days_back=90)
    print(f"Past events: {len(past)}")
    
    # Analyze event impact
    print("\nAnalyzing earnings impact for RELIANCE...")
    impact = db.analyze_event_impact('RELIANCE', EventType.EARNINGS)
    for key, value in impact.items():
        print(f"  {key}: {value}")
    
    # Generate earnings signal
    print("\nGenerating earnings signal for RELIANCE...")
    latest_earnings = db.earnings_data.get('RELIANCE', [])[0] if db.earnings_data.get('RELIANCE') else None
    if latest_earnings:
        signal = db.generate_earnings_signal('RELIANCE', latest_earnings)
        for key, value in signal.items():
            print(f"  {key}: {value}")
    
    # Get event calendar
    print("\nGetting event calendar...")
    start = datetime.now() - timedelta(days=30)
    end = datetime.now() + timedelta(days=30)
    calendar = db.get_event_calendar(start, end)
    print(f"Calendar events: {len(calendar)}")
    print(calendar.head())
    
    # Get earnings calendar
    print("\nGetting earnings calendar...")
    earnings_calendar = db.get_earnings_calendar(start, end)
    print(f"Earnings events: {len(earnings_calendar)}")
    print(earnings_calendar.head())
    
    # Get event summary
    print("\nEvent signal summary:")
    summary = db.get_event_summary()
    print(summary.to_string(index=False))
