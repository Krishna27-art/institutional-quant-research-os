"""
Market Event Store.
Moves the system from CSV/Parquet OHLCV to an event-centric paradigm.
All data is reconstructed from atomic events: Trades, Quotes (Order Book), Corporate Actions.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Iterator
from datetime import datetime
import pandas as pd

class EventType(Enum):
    TRADE = "trade"
    QUOTE_UPDATE = "quote_update"
    AUCTION = "auction"
    CORPORATE_ACTION = "corporate_action"

@dataclass
class MarketEvent:
    timestamp: datetime
    symbol: str
    event_type: EventType
    
@dataclass
class TradeEvent(MarketEvent):
    price: float
    quantity: int
    is_buyer_maker: bool

@dataclass
class QuoteEvent(MarketEvent):
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    book_level: int = 1

class EventStore:
    """Central repository for all market events. Immutable and append-only."""
    def __init__(self):
        # In a real institutional system, this wraps a highly optimized time-series DB
        # like kdb+/q, ClickHouse, or Arctic.
        self._events: List[MarketEvent] = []
        
    def append(self, event: MarketEvent) -> None:
        self._events.append(event)
        
    def stream_events(
        self, 
        start: datetime, 
        end: datetime, 
        symbols: Optional[List[str]] = None
    ) -> Iterator[MarketEvent]:
        """
        Yields events sequentially to allow perfect tick-by-tick state reconstruction.
        """
        # Sort is simulated here; real systems index by time and symbol.
        sorted_events = sorted(self._events, key=lambda e: e.timestamp)
        for event in sorted_events:
            if start <= event.timestamp <= end:
                if symbols is None or event.symbol in symbols:
                    yield event
                    
    def load_from_parquet_ticks(self, file_path: str) -> None:
        """Utility to ingest raw tick data into the Event Store."""
        # Institutional integration point for historical ticks.
        pass
