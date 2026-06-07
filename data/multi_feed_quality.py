"""
Multi-Feed Data Quality Engine (Jane Street style)
Three independent feeds: primary (NSE WebSocket), secondary (Bloomberg API),
tertiary (Yahoo Finance fallback). Each tick gets a confidence score.

This implements production-grade data quality validation with:
- Latency-based feed selection
- Cross-feed validation (3-sigma outlier detection)
- Automatic fallback mechanisms
- Real-time inconsistency alerts
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Protocol
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeedQuote:
    """Quote from a single feed with metadata"""
    feed_name: str
    price: float
    timestamp: datetime
    latency_ms: float
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class ValidatedQuote:
    """Validated quote with confidence score"""
    price: float
    confidence: float
    source_feed: str
    validation_method: str
    all_quotes: List[FeedQuote]
    timestamp: datetime


class PriceFeed(Protocol):
    """Minimal interface required by MultiFeedDataQuality."""

    last_update: datetime
    last_latency: float

    def get_price(self, symbol: str) -> float:
        ...


class NSEWebSocket:
    """Mock NSE WebSocket feed"""
    def __init__(self):
        self.last_latency = 50.0  # ms
        self.last_update = datetime.now()
        
    def get_price(self, symbol: str) -> float:
        # Mock implementation - in production, connect to actual NSE WebSocket
        self.last_latency = np.random.uniform(20, 80)
        self.last_update = datetime.now()
        return np.random.uniform(100, 2000)
    


class BloombergAPI:
    """Mock Bloomberg API feed"""
    def __init__(self):
        self.last_latency = 100.0  # ms
        self.last_update = datetime.now()
        
    def get_price(self, symbol: str) -> float:
        # Mock implementation - in production, connect to Bloomberg API
        self.last_latency = np.random.uniform(80, 150)
        self.last_update = datetime.now()
        return np.random.uniform(100, 2000)
    


class YahooFinance:
    """Mock Yahoo Finance fallback feed"""
    def __init__(self):
        self.last_latency = 500.0  # ms
        self.last_update = datetime.now()
        
    def get_price(self, symbol: str) -> float:
        # Mock implementation - in production, use yfinance
        self.last_latency = np.random.uniform(400, 600)
        self.last_update = datetime.now()
        return np.random.uniform(100, 2000)
    


class MultiFeedDataQuality:
    """
    Three independent feeds: primary (NSE WebSocket), secondary (Bloomberg API),
    tertiary (Yahoo Finance fallback). Each tick gets a confidence score.
    
    Jane Street-style data quality engine with:
    - Latency-based feed selection
    - Cross-feed validation (3-sigma outlier detection)
    - Automatic fallback mechanisms
    - Real-time inconsistency alerts
    """
    
    def __init__(self, max_latency_window: int = 1000, feeds: Optional[Dict[str, PriceFeed]] = None):
        self.feeds = feeds or {
            'nse': NSEWebSocket(),
            'bloomberg': BloombergAPI(),
            'yahoo': YahooFinance()
        }
        self.latency_window = deque(maxlen=max_latency_window)
        self.activity_scores = {name: deque(maxlen=100) for name in self.feeds}
        self.alert_history = deque(maxlen=1000)
        
        # Feed priority (lower = higher priority)
        self.feed_priority = {name: idx + 1 for idx, name in enumerate(self.feeds)}
        self.last_valid: Dict[str, ValidatedQuote] = {}
        
        # Validation thresholds
        self.outlier_threshold_sigma = 3.0
        self.max_latency_ms = 1000.0
        self.stale_threshold_seconds = 60.0
        
    def _activity_score(self, feed_name: str) -> float:
        """Calculate activity score based on recent updates"""
        activity = self.activity_scores[feed_name]
        if not activity:
            return 0.0
        # Exponential decay of recent activity
        now = datetime.now()
        score = 0.0
        for timestamp in activity:
            age = (now - timestamp).total_seconds()
            score += np.exp(-age / 10.0)  # 10-second half-life
        return score
    
    def _update_activity(self, feed_name: str):
        """Update activity score for a feed"""
        self.activity_scores[feed_name].append(datetime.now())
    
    def alert(self, alert_type: str, symbol: str, details: dict):
        """Log an alert for data quality issues"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'symbol': symbol,
            'details': details
        }
        self.alert_history.append(alert)
        logger.warning(f"Data Quality Alert: {alert_type} for {symbol} - {details}")
    
    def get_best_price(self, symbol: str) -> ValidatedQuote:
        """
        Get best validated price from all feeds.
        
        Algorithm:
        1. Fetch quotes from all available feeds
        2. Filter by latency and staleness
        3. Sort by (latency, activity_score, priority)
        4. Validate best quote against other feeds (3-sigma check)
        5. Return validated quote with confidence score
        """
        quotes = []
        now = datetime.now()
        
        # Fetch from all feeds
        for name, feed in self.feeds.items():
            try:
                price = feed.get_price(symbol)
                latency = feed.last_latency
                
                # Check staleness
                age = (now - feed.last_update).total_seconds()
                if age > self.stale_threshold_seconds:
                    logger.warning(f"Feed {name} stale for {symbol}: {age:.1f}s old")
                    continue
                
                # Check latency
                if latency > self.max_latency_ms:
                    logger.warning(f"Feed {name} high latency for {symbol}: {latency:.1f}ms")
                    continue
                
                quote = FeedQuote(
                    feed_name=name,
                    price=price,
                    timestamp=now,
                    latency_ms=latency
                )
                quotes.append(quote)
                self._update_activity(name)
                
            except Exception as e:
                logger.error(f"Failed to get price from {name} for {symbol}: {e}")
                continue
        
        if not quotes:
            raise ValueError(f"No valid quotes available for {symbol}")
        
        # Sort by (latency, -activity_score, priority)
        quotes.sort(key=lambda q: (
            q.latency_ms,
            -self._activity_score(q.feed_name),
            self.feed_priority[q.feed_name]
        ))
        
        best = quotes[0]
        prices = np.asarray([q.price for q in quotes], dtype=float)
        
        # Cross-feed validation
        validation_method = "single_feed"
        confidence = 0.5
        
        if len(prices) > 1:
            # Median absolute deviation is robust when one feed is broken.
            median_price = float(np.median(prices))
            mad = float(np.median(np.abs(prices - median_price)))
            robust_sigma = 1.4826 * mad
            std_price = float(np.std(prices))
            scale = robust_sigma if robust_sigma > 1e-12 else max(abs(median_price) * 0.001, 1e-8)

            if abs(best.price - median_price) > self.outlier_threshold_sigma * scale:
                # Outlier detected - fall back to median
                self.alert("data_inconsistency", symbol, {
                    'best_feed': best.feed_name,
                    'best_price': best.price,
                    'median_price': median_price,
                    'std_price': std_price,
                    'mad_price': mad,
                    'deviation_sigma': abs(best.price - median_price) / scale
                })
                
                # Use median price
                best_price = median_price
                validation_method = "median_fallback"
                confidence = 0.7  # Lower confidence due to inconsistency
            else:
                # Consistent across feeds
                best_price = best.price
                validation_method = "cross_feed_validated"
                confidence = 0.95  # High confidence
        else:
            best_price = best.price
            validation_method = "single_feed"
            confidence = 0.8  # Medium confidence (only one feed available)
        
        # Track latency
        self.latency_window.append(best.latency_ms)
        
        validated = ValidatedQuote(
            price=best_price,
            confidence=confidence,
            source_feed=best.feed_name,
            validation_method=validation_method,
            all_quotes=quotes,
            timestamp=now
        )
        self.last_valid[symbol] = validated
        return validated

    def get_price(self, symbol: str, max_stale_seconds: float = 300.0) -> float:
        """Return scalar best price, falling back to recent last-valid quote."""
        try:
            return self.get_best_price(symbol).price
        except ValueError:
            cached = self.last_valid.get(symbol)
            if cached is None:
                raise
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age <= max_stale_seconds:
                self.alert("stale_last_valid_fallback", symbol, {"age_seconds": age, "price": cached.price})
                return cached.price
            raise
    
    def get_feed_health(self) -> Dict[str, dict]:
        """Get health status of all feeds"""
        health = {}
        now = datetime.now()
        
        for name, feed in self.feeds.items():
            age = (now - feed.last_update).total_seconds()
            activity = self._activity_score(name)
            
            health[name] = {
                'last_update': feed.last_update,
                'age_seconds': age,
                'last_latency_ms': feed.last_latency,
                'activity_score': activity,
                'is_stale': age > self.stale_threshold_seconds,
                'is_slow': feed.last_latency > self.max_latency_ms,
                'priority': self.feed_priority[name]
            }
        
        return health
    
    def get_average_latency(self) -> float:
        """Get average latency across all feeds"""
        if not self.latency_window:
            return 0.0
        return np.mean(self.latency_window)


# Singleton instance
_multi_feed_instance = None

def get_multi_feed_quality() -> MultiFeedDataQuality:
    """Get the singleton multi-feed data quality instance"""
    global _multi_feed_instance
    if _multi_feed_instance is None:
        _multi_feed_instance = MultiFeedDataQuality()
    return _multi_feed_instance
