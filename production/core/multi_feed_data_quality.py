"""
Multi-Feed Data Quality Engine (Jane Street Style)

Three independent feeds: primary (NSE WebSocket), secondary (Bloomberg API),
tertiary (Yahoo Finance fallback). Each tick gets a confidence score.

Time complexity: O(F) per symbol, F = number of feeds (3-5)
Space: O(1) per symbol (caches last valid)

Based on institutional blueprint specification
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import pytz
from collections import deque
import requests
import json

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


class FeedStatus(Enum):
    """Feed health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class FeedQuote:
    """Quote from a single feed"""
    feed_name: str
    price: float
    timestamp: datetime
    latency_ms: float
    volume: Optional[float] = None
    is_valid: bool = True


@dataclass
class QualityScore:
    """Quality score for a feed"""
    feed_name: str
    score: float
    latency_score: float
    consistency_score: float
    activity_score: float


class BaseFeed:
    """Base class for data feeds"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_update_times: Dict[str, datetime] = {}
        self.latency_window: deque = deque(maxlen=100)
        self.status = FeedStatus.HEALTHY
        
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price for symbol - to be implemented by subclasses"""
        raise NotImplementedError
        
    def last_latency(self) -> float:
        """Get average latency in milliseconds"""
        if not self.latency_window:
            return 1000.0  # Default high latency
        return np.mean(self.latency_window)
    
    def last_update_time(self, symbol: str) -> Optional[datetime]:
        """Get last update time for symbol"""
        return self.last_update_times.get(symbol)
    
    def update_latency(self, latency_ms: float):
        """Update latency window"""
        self.latency_window.append(latency_ms)
        
    def mark_update(self, symbol: str, timestamp: datetime):
        """Mark symbol as updated"""
        self.last_update_times[symbol] = timestamp


class NSEWebSocketFeed(BaseFeed):
    """NSE WebSocket feed (primary)"""
    
    def __init__(self):
        super().__init__("nse_websocket")
        # In production, this would connect to actual NSE WebSocket
        self.connected = False
        
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get price from NSE WebSocket.
        In production, this would read from live WebSocket connection.
        For now, simulate with API call.
        """
        try:
            # Simulate WebSocket latency
            start = datetime.now()
            
            # Use NSE API as proxy
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('priceInfo', {}).get('lastPrice')
                
                if price:
                    latency = (datetime.now() - start).total_seconds() * 1000
                    self.update_latency(latency)
                    self.mark_update(symbol, datetime.now(IST))
                    self.status = FeedStatus.HEALTHY
                    return price
                    
        except Exception as e:
            logger.warning(f"NSE WebSocket feed error for {symbol}: {e}")
            self.status = FeedStatus.DEGRADED
            
        return None


class BloombergAPIFeed(BaseFeed):
    """Bloomberg API feed (secondary)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("bloomberg_api")
        self.api_key = api_key
        
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get price from Bloomberg API.
        In production, this would use actual Bloomberg API.
        For now, use Yahoo Finance as proxy.
        """
        try:
            start = datetime.now()
            
            # Use Yahoo Finance as proxy for Bloomberg
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                price = data['chart']['result'][0]['meta']['regularMarketPrice']
                
                if price:
                    latency = (datetime.now() - start).total_seconds() * 1000
                    self.update_latency(latency)
                    self.mark_update(symbol, datetime.now(IST))
                    self.status = FeedStatus.HEALTHY
                    return price
                    
        except Exception as e:
            logger.warning(f"Bloomberg API feed error for {symbol}: {e}")
            self.status = FeedStatus.DEGRADED
            
        return None


class YahooFinanceFeed(BaseFeed):
    """Yahoo Finance feed (tertiary/fallback)"""
    
    def __init__(self):
        super().__init__("yahoo_finance")
        
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price from Yahoo Finance"""
        try:
            start = datetime.now()
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = data['chart']['result'][0]['meta']['regularMarketPrice']
                
                if price:
                    latency = (datetime.now() - start).total_seconds() * 1000
                    self.update_latency(latency)
                    self.mark_update(symbol, datetime.now(IST))
                    self.status = FeedStatus.HEALTHY
                    return price
                    
        except Exception as e:
            logger.warning(f"Yahoo Finance feed error for {symbol}: {e}")
            self.status = FeedStatus.DEGRADED
            
        return None


class AlphaVantageFeed(BaseFeed):
    """Alpha Vantage feed (additional fallback)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("alpha_vantage")
        self.api_key = api_key or "demo"  # Use demo key if none provided
        
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price from Alpha Vantage"""
        try:
            start = datetime.now()
            
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}.NS&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('Global Quote', {}).get('05. price', 0))
                
                if price > 0:
                    latency = (datetime.now() - start).total_seconds() * 1000
                    self.update_latency(latency)
                    self.mark_update(symbol, datetime.now(IST))
                    self.status = FeedStatus.HEALTHY
                    return price
                    
        except Exception as e:
            logger.warning(f"Alpha Vantage feed error for {symbol}: {e}")
            self.status = FeedStatus.DEGRADED
            
        return None


class MultiFeedDataQuality:
    """
    Multi-Feed Data Quality Engine (Jane Street Style)
    
    Three independent feeds: primary (NSE WebSocket), secondary (Bloomberg API),
    tertiary (Yahoo Finance fallback). Each tick gets a confidence score.
    
    Time complexity: O(F) per symbol, F = number of feeds (3-5)
    Space: O(1) per symbol (caches last valid)
    """
    
    def __init__(self, bloomberg_api_key: Optional[str] = None, alpha_vantage_key: Optional[str] = None):
        """
        Initialize multi-feed data quality engine.
        
        Args:
            bloomberg_api_key: Optional Bloomberg API key
            alpha_vantage_key: Optional Alpha Vantage API key
        """
        self.feeds: Dict[str, BaseFeed] = {
            'nse_websocket': NSEWebSocketFeed(),
            'bloomberg_api': BloombergAPIFeed(bloomberg_api_key),
            'yahoo_finance': YahooFinanceFeed(),
            'alpha_vantage': AlphaVantageFeed(alpha_vantage_key)
        }
        
        # Cache last valid prices
        self.last_valid: Dict[str, Dict] = {}
        
        # Track feed health
        self.feed_health: Dict[str, FeedStatus] = {
            name: FeedStatus.HEALTHY for name in self.feeds.keys()
        }
        
        # Latency tracking
        self.latency_window: deque = deque(maxlen=1000)
        
        # Alert thresholds
        self.staleness_threshold = 300  # 5 minutes
        self.outlier_threshold_sigma = 3.0  # 3-sigma for outlier detection
        
    def get_best_price(self, symbol: str, timestamp: Optional[datetime] = None) -> Tuple[float, str, float]:
        """
        Get best price from all feeds with confidence scoring.
        
        Args:
            symbol: Stock/index symbol
            timestamp: Current timestamp (uses now if None)
            
        Returns:
            Tuple of (price, feed_used, confidence_score)
        """
        if timestamp is None:
            timestamp = datetime.now(IST)
        
        # Collect quotes from all feeds
        quotes: List[FeedQuote] = []
        for name, feed in self.feeds.items():
            try:
                price = feed.get_price(symbol)
                if price and price > 0:
                    latency = feed.last_latency()
                    quotes.append(FeedQuote(
                        feed_name=name,
                        price=price,
                        timestamp=timestamp,
                        latency_ms=latency,
                        is_valid=True
                    ))
            except Exception as e:
                logger.warning(f"Feed {name} failed for {symbol}: {e}")
                continue
        
        if len(quotes) == 0:
            # Fallback to last valid with staleness warning
            if symbol in self.last_valid:
                last = self.last_valid[symbol]
                age = (timestamp - last['timestamp']).total_seconds()
                if age < self.staleness_threshold:
                    self.alert("DATA_STALE", symbol, age)
                    return last['price'], last['feed'], 0.3  # Low confidence
            raise NoDataError(f"No feed data for {symbol}")
        
        # Compute median and MAD for outlier detection
        prices = [q.price for q in quotes]
        median = np.median(prices)
        mad = np.median(np.abs(prices - median))
        
        # Score each feed: lower latency + consistency + recent activity
        scored_quotes: List[Tuple[FeedQuote, QualityScore]] = []
        for quote in quotes:
            score = self._compute_feed_score(quote, median, mad, quotes)
            scored_quotes.append((quote, score))
        
        # Choose best (highest score)
        best_quote, best_score = max(scored_quotes, key=lambda x: x[1].score)
        
        # Validate against other feeds (no 3-sigma deviations)
        if len(prices) > 1 and abs(best_quote.price - median) > self.outlier_threshold_sigma * mad:
            self.alert("DATA_INCONSISTENCY", symbol, best_quote.price, prices)
            # Fall back to median
            best_price = median
            best_feed = "median_fallback"
            confidence = 0.5
        else:
            best_price = best_quote.price
            best_feed = best_quote.feed_name
            confidence = best_score.score
        
        # Cache last valid
        self.last_valid[symbol] = {
            'price': best_price,
            'timestamp': timestamp,
            'feed': best_feed
        }
        
        return best_price, best_feed, confidence
    
    def _compute_feed_score(
        self, 
        quote: FeedQuote, 
        median: float, 
        mad: float,
        all_quotes: List[FeedQuote]
    ) -> QualityScore:
        """
        Compute quality score for a feed.
        
        Scoring components:
        - Latency score: lower is better (inverse)
        - Consistency score: within 3-sigma of median
        - Activity score: recent updates (within 10 seconds)
        
        Final score: 0.5 * latency + 0.3 * consistency + 0.2 * activity
        """
        # Latency score (inverse, normalized)
        latency_score = 1.0 / (quote.latency_ms / 1000.0 + 1.0)
        latency_score = np.clip(latency_score, 0, 1)
        
        # Consistency score
        if mad > 0:
            z_score = abs(quote.price - median) / mad
            consistency_score = 1.0 if z_score < self.outlier_threshold_sigma else 0.0
        else:
            consistency_score = 1.0
        
        # Activity score (recent updates)
        feed = self.feeds[quote.feed_name]
        last_update = feed.last_update_time(quote.feed_name)
        if last_update:
            age = (datetime.now(IST) - last_update).total_seconds()
            activity_score = max(0, 1 - age / 10.0)  # Decay over 10 seconds
        else:
            activity_score = 0.0
        
        # Final weighted score
        final_score = 0.5 * latency_score + 0.3 * consistency_score + 0.2 * activity_score
        
        return QualityScore(
            feed_name=quote.feed_name,
            score=final_score,
            latency_score=latency_score,
            consistency_score=consistency_score,
            activity_score=activity_score
        )
    
    def alert(self, alert_type: str, symbol: str, *args):
        """Send alert for data quality issue"""
        if alert_type == "DATA_STALE":
            age = args[0]
            logger.warning(f"DATA_STALE: {symbol} is {age:.0f}s old")
        elif alert_type == "DATA_INCONSISTENCY":
            price, prices = args
            logger.warning(f"DATA_INCONSISTENCY: {symbol} price {price} deviates from median {np.median(prices)}")
        
        # In production, send to monitoring system (PagerDuty, etc.)
    
    def get_feed_health(self) -> Dict[str, FeedStatus]:
        """Get health status of all feeds"""
        return self.feed_health.copy()
    
    def get_quality_summary(self) -> Dict:
        """Get summary of data quality metrics"""
        return {
            'total_feeds': len(self.feeds),
            'healthy_feeds': sum(1 for s in self.feed_health.values() if s == FeedStatus.HEALTHY),
            'degraded_feeds': sum(1 for s in self.feed_health.values() if s == FeedStatus.DEGRADED),
            'cached_symbols': len(self.last_valid),
            'avg_latency_ms': np.mean([f.last_latency() for f in self.feeds.values()]) if self.feeds else 0
        }


class NoDataError(Exception):
    """Exception raised when no data is available from any feed"""
    pass


# Singleton instance
_multi_feed_engine = None

def get_multi_feed_engine(bloomberg_api_key: Optional[str] = None, alpha_vantage_key: Optional[str] = None) -> MultiFeedDataQuality:
    """Get the singleton multi-feed data quality engine instance"""
    global _multi_feed_engine
    if _multi_feed_engine is None:
        _multi_feed_engine = MultiFeedDataQuality(bloomberg_api_key, alpha_vantage_key)
    return _multi_feed_engine


if __name__ == "__main__":
    # Test the multi-feed data quality engine
    print("Testing Multi-Feed Data Quality Engine...")
    
    engine = MultiFeedDataQuality()
    
    # Test getting price for a symbol
    try:
        price, feed, confidence = engine.get_best_price("RELIANCE")
        print(f"Price: {price}, Feed: {feed}, Confidence: {confidence:.3f}")
    except NoDataError as e:
        print(f"Error: {e}")
    
    # Print feed health
    print("\nFeed Health:")
    for name, status in engine.get_feed_health().items():
        print(f"  {name}: {status.value}")
    
    # Print quality summary
    print("\nQuality Summary:")
    summary = engine.get_quality_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
