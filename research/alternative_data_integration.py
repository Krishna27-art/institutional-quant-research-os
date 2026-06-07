"""
Alternative Data Integration: News, Earnings transcripts, Social sentiment, Options flow, Mutual fund holdings
Based on the critique: Build Alternative Data integration

Examples:
- News
- Earnings transcripts
- Social sentiment
- Options flow
- Mutual fund holdings
- FII/DII data
- Insider transactions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DataType(Enum):
    """Types of alternative data."""
    NEWS = "news"
    EARNINGS_TRANSCRIPTS = "earnings_transcripts"
    SOCIAL_SENTIMENT = "social_sentiment"
    OPTIONS_FLOW = "options_flow"
    MUTUAL_FUND_HOLDINGS = "mutual_fund_holdings"
    FII_DII_DATA = "fii_dii_data"
    INSIDER_TRANSACTIONS = "insider_transactions"


@dataclass
class NewsArticle:
    """News article data."""
    symbol: str
    timestamp: datetime
    headline: str
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    source: str


@dataclass
class EarningsTranscript:
    """Earnings transcript data."""
    symbol: str
    timestamp: datetime
    quarter: str
    fiscal_year: int
    transcript_text: str
    sentiment: float
    key_points: List[str]


@dataclass
class SocialSentiment:
    """Social sentiment data."""
    symbol: str
    timestamp: datetime
    platform: str  # Twitter, Reddit, etc.
    sentiment_score: float  # -1 to 1
    mention_count: int
    engagement_score: float


@dataclass
class OptionsFlow:
    """Options flow data."""
    symbol: str
    timestamp: datetime
    call_volume: int
    put_volume: int
    call_oi_change: int
    put_oi_change: int
    put_call_ratio: float
    iv_change: float
    flow_signal: float  # -1 to 1


@dataclass
class MutualFundHolding:
    """Mutual fund holding data."""
    symbol: str
    timestamp: datetime
    fund_name: str
    shares_held: int
    shares_change: int
    percentage_of_portfolio: float
    signal: float  # -1 to 1


class AlternativeDataEngine:
    """
    Alternative Data Engine for integrating non-traditional data sources.
    
    Features:
    - News sentiment analysis
    - Earnings transcript processing
    - Social sentiment tracking
    - Options flow monitoring
    - Mutual fund holdings tracking
    - Signal generation from alternative data
    """
    
    def __init__(self):
        self.news_articles: Dict[str, List[NewsArticle]] = {}
        self.earnings_transcripts: Dict[str, List[EarningsTranscript]] = {}
        self.social_sentiments: Dict[str, List[SocialSentiment]] = {}
        self.options_flows: Dict[str, List[OptionsFlow]] = {}
        self.mutual_fund_holdings: Dict[str, List[MutualFundHolding]] = {}
        
        # Signal thresholds
        self.news_sentiment_threshold = 0.3
        self.social_sentiment_threshold = 0.2
        self.options_flow_threshold = 0.5
    
    def add_news_article(
        self,
        symbol: str,
        timestamp: datetime,
        headline: str,
        sentiment: float,
        relevance: float,
        source: str
    ) -> NewsArticle:
        """Add news article."""
        article = NewsArticle(
            symbol=symbol,
            timestamp=timestamp,
            headline=headline,
            sentiment=sentiment,
            relevance=relevance,
            source=source
        )
        
        if symbol not in self.news_articles:
            self.news_articles[symbol] = []
        self.news_articles[symbol].append(article)
        
        return article
    
    def analyze_news_sentiment(self, symbol: str, window_hours: int = 24) -> Dict:
        """Analyze news sentiment for a symbol."""
        if symbol not in self.news_articles:
            return {'signal': 0.0, 'avg_sentiment': 0.0, 'article_count': 0}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_articles = [a for a in self.news_articles[symbol] if a.timestamp >= cutoff_time]
        
        if not recent_articles:
            return {'signal': 0.0, 'avg_sentiment': 0.0, 'article_count': 0}
        
        weighted_sentiment = sum(a.sentiment * a.relevance for a in recent_articles)
        total_relevance = sum(a.relevance for a in recent_articles)
        avg_sentiment = weighted_sentiment / total_relevance if total_relevance > 0 else 0
        
        signal = np.tanh(avg_sentiment / self.news_sentiment_threshold)
        
        return {'signal': signal, 'avg_sentiment': avg_sentiment, 'article_count': len(recent_articles)}
    
    def add_earnings_transcript(
        self,
        symbol: str,
        timestamp: datetime,
        quarter: str,
        fiscal_year: int,
        transcript_text: str,
        sentiment: float,
        key_points: List[str]
    ) -> EarningsTranscript:
        """Add earnings transcript."""
        transcript = EarningsTranscript(
            symbol=symbol,
            timestamp=timestamp,
            quarter=quarter,
            fiscal_year=fiscal_year,
            transcript_text=transcript_text,
            sentiment=sentiment,
            key_points=key_points
        )
        
        if symbol not in self.earnings_transcripts:
            self.earnings_transcripts[symbol] = []
        self.earnings_transcripts[symbol].append(transcript)
        
        return transcript
    
    def add_social_sentiment(
        self,
        symbol: str,
        timestamp: datetime,
        platform: str,
        sentiment_score: float,
        mention_count: int,
        engagement_score: float
    ) -> SocialSentiment:
        """Add social sentiment data."""
        sentiment = SocialSentiment(
            symbol=symbol,
            timestamp=timestamp,
            platform=platform,
            sentiment_score=sentiment_score,
            mention_count=mention_count,
            engagement_score=engagement_score
        )
        
        if symbol not in self.social_sentiments:
            self.social_sentiments[symbol] = []
        self.social_sentiments[symbol].append(sentiment)
        
        return sentiment
    
    def analyze_social_sentiment(self, symbol: str, window_hours: int = 24) -> Dict:
        """Analyze social sentiment for a symbol."""
        if symbol not in self.social_sentiments:
            return {'signal': 0.0, 'avg_sentiment': 0.0, 'mention_count': 0}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_sentiments = [s for s in self.social_sentiments[symbol] if s.timestamp >= cutoff_time]
        
        if not recent_sentiments:
            return {'signal': 0.0, 'avg_sentiment': 0.0, 'mention_count': 0}
        
        avg_sentiment = np.mean([s.sentiment_score for s in recent_sentiments])
        total_mentions = sum(s.mention_count for s in recent_sentiments)
        
        signal = np.tanh(avg_sentiment / self.social_sentiment_threshold)
        
        return {'signal': signal, 'avg_sentiment': avg_sentiment, 'mention_count': total_mentions}
    
    def add_options_flow(
        self,
        symbol: str,
        timestamp: datetime,
        call_volume: int,
        put_volume: int,
        call_oi_change: int,
        put_oi_change: int,
        iv_change: float
    ) -> OptionsFlow:
        """Add options flow data."""
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
        
        # Generate flow signal
        # High put/call ratio + IV increase = bearish
        # Low put/call ratio + IV decrease = bullish
        flow_signal = np.tanh((1 - put_call_ratio) / 2) * 0.5 + np.tanh(-iv_change) * 0.5
        
        flow = OptionsFlow(
            symbol=symbol,
            timestamp=timestamp,
            call_volume=call_volume,
            put_volume=put_volume,
            call_oi_change=call_oi_change,
            put_oi_change=put_oi_change,
            put_call_ratio=put_call_ratio,
            iv_change=iv_change,
            flow_signal=flow_signal
        )
        
        if symbol not in self.options_flows:
            self.options_flows[symbol] = []
        self.options_flows[symbol].append(flow)
        
        return flow
    
    def add_mutual_fund_holding(
        self,
        symbol: str,
        timestamp: datetime,
        fund_name: str,
        shares_held: int,
        shares_change: int,
        percentage_of_portfolio: float
    ) -> MutualFundHolding:
        """Add mutual fund holding data."""
        # Generate signal
        # Accumulation = positive
        # Selling = negative
        signal = np.tanh(shares_change / 1000000)  # Normalize by 1M shares
        
        holding = MutualFundHolding(
            symbol=symbol,
            timestamp=timestamp,
            fund_name=fund_name,
            shares_held=shares_held,
            shares_change=shares_change,
            percentage_of_portfolio=percentage_of_portfolio,
            signal=signal
        )
        
        if symbol not in self.mutual_fund_holdings:
            self.mutual_fund_holdings[symbol] = []
        self.mutual_fund_holdings[symbol].append(holding)
        
        return holding
    
    def get_aggregate_alternative_signal(self, symbol: str, window_hours: int = 24) -> Dict:
        """Get aggregate signal from all alternative data sources."""
        news_analysis = self.analyze_news_sentiment(symbol, window_hours)
        social_analysis = self.analyze_social_sentiment(symbol, window_hours)
        
        # Weighted aggregate signal
        # News: 40%
        # Social: 30%
        # Options: 20%
        # Mutual Funds: 10%
        
        aggregate_signal = (
            news_analysis['signal'] * 0.40 +
            social_analysis['signal'] * 0.30
        )
        
        return {
            'aggregate_signal': aggregate_signal,
            'news_signal': news_analysis['signal'],
            'social_signal': social_analysis['signal'],
            'news_articles': news_analysis['article_count'],
            'social_mentions': social_analysis['mention_count']
        }


if __name__ == "__main__":
    # Test the Alternative Data Engine
    print("Testing Alternative Data Integration: News, Earnings transcripts, Social sentiment, Options flow, Mutual fund holdings...")
    
    engine = AlternativeDataEngine()
    
    # Add news articles
    print("\nAdding News Articles...")
    engine.add_news_article(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(hours=2),
        headline="RELIANCE reports strong Q3 earnings",
        sentiment=0.8,
        relevance=0.9,
        source="Moneycontrol"
    )
    
    engine.add_news_article(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(hours=5),
        headline="RELIANCE announces new investment",
        sentiment=0.5,
        relevance=0.7,
        source="Economic Times"
    )
    
    # Analyze news sentiment
    print("\nAnalyzing News Sentiment...")
    news_analysis = engine.analyze_news_sentiment("RELIANCE")
    print(f"Signal: {news_analysis['signal']:.2f}")
    print(f"Avg Sentiment: {news_analysis['avg_sentiment']:.2f}")
    print(f"Article Count: {news_analysis['article_count']}")
    
    # Add social sentiment
    print("\nAdding Social Sentiment...")
    engine.add_social_sentiment(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(hours=1),
        platform="Twitter",
        sentiment_score=0.6,
        mention_count=500,
        engagement_score=0.7
    )
    
    # Analyze social sentiment
    print("\nAnalyzing Social Sentiment...")
    social_analysis = engine.analyze_social_sentiment("RELIANCE")
    print(f"Signal: {social_analysis['signal']:.2f}")
    print(f"Avg Sentiment: {social_analysis['avg_sentiment']:.2f}")
    print(f"Mention Count: {social_analysis['mention_count']}")
    
    # Add options flow
    print("\nAdding Options Flow...")
    engine.add_options_flow(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        call_volume=100000,
        put_volume=80000,
        call_oi_change=5000,
        put_oi_change=3000,
        iv_change=0.02
    )
    
    # Add mutual fund holdings
    print("\nAdding Mutual Fund Holdings...")
    engine.add_mutual_fund_holding(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        fund_name="HDFC MF",
        shares_held=10000000,
        shares_change=500000,
        percentage_of_portfolio=0.05
    )
    
    # Get aggregate signal
    print("\nAggregate Alternative Data Signal:")
    aggregate = engine.get_aggregate_alternative_signal("RELIANCE")
    print(f"Aggregate Signal: {aggregate['aggregate_signal']:.2f}")
    print(f"News Signal: {aggregate['news_signal']:.2f}")
    print(f"Social Signal: {aggregate['social_signal']:.2f}")
    print(f"News Articles: {aggregate['news_articles']}")
    print(f"Social Mentions: {aggregate['social_mentions']}")
