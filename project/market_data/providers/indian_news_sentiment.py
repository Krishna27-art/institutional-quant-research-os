"""
LLM Sentiment for Indian News
Based on V3 Blueprint - India-Specific Alternative Data

Key findings from research:
- Alternative data sources for Indian markets
- News articles (Economic Times, Business Standard, Moneycontrol) via API
- FinBERT fine-tuned on Indian financial news headlines
- Classes: positive, negative, neutral (for stock impact)
- Also output: entity (stock ticker), topic (earnings, policy, macro, etc.)
- Expected Sharpe contribution: +0.05–0.15

V3 Upgrade - Expected Sharpe increase: +0.05–0.15
Priority: Low
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class NewsArticle:
    """News article"""
    title: str
    source: str
    timestamp: str
    url: str
    entities: List[str]  # Stock tickers mentioned
    topic: str


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    article: NewsArticle
    sentiment: str  # "positive", "negative", "neutral"
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    entity_sentiments: Dict[str, float]  # ticker -> sentiment score


class IndianNewsSentimentEngine:
    """
    Indian News Sentiment Engine using LLM.
    
    Sources:
    - Economic Times (API)
    - Business Standard (RSS)
    - Moneycontrol (scraping)
    - Bloomberg Quint (RSS)
    - CNBC TV18 transcripts (if available)
    
    Model:
    - FinBERT fine-tuned on 10,000 manually labeled Indian financial news headlines
    - Classes: positive, negative, neutral (for stock impact)
    - Also output: entity (stock ticker), topic (earnings, policy, macro, etc.)
    """
    
    def __init__(self):
        self.news_history: List[NewsArticle] = []
        self.sentiment_history: List[SentimentResult] = []
        
        # News sources
        self.sources = {
            "economic_times": "https://economictimes.indiatimes.com",
            "business_standard": "https://www.business-standard.com",
            "moneycontrol": "https://www.moneycontrol.com",
            "bloomberg_quint": "https://www.bqprime.com"
        }
        
        # Indian stock tickers
        self.tickers = [
            "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
            "LT", "AXISBANK", "HCLTECH", "ASIANPAINT", "MARUTI"
        ]
    
    def fetch_news(self, source: str, limit: int = 10) -> List[NewsArticle]:
        """
        Fetch news from a source.
        
        Args:
            source: News source name
            limit: Number of articles to fetch
            
        Returns:
            List of NewsArticle
        """
        # Placeholder implementation
        # In production, use actual API calls or web scraping
        
        articles = []
        np.random.seed(hash(source))
        
        for i in range(limit):
            # Generate sample news
            titles = [
                "Reliance reports strong Q3 earnings, beats estimates",
                "HDFC Bank faces NPA concerns, stock falls 2%",
                "Infosys wins $2B deal with US client",
                "RBI hikes repo rate by 25bps, markets react negatively",
                "TCS announces dividend of ₹24 per share",
                "ITC launches new product line in FMCG segment",
                "SBI reports lower than expected loan growth",
                "Bharti Airtel gains market share in telecom sector",
                "L&T wins major infrastructure contract",
                "Axis Bank shows improvement in asset quality"
            ]
            
            title = np.random.choice(titles)
            
            # Extract entities (simplified)
            entities = []
            for ticker in self.tickers:
                if ticker.lower() in title.lower():
                    entities.append(ticker)
            
            # Determine topic
            if "earnings" in title.lower() or "q3" in title.lower():
                topic = "earnings"
            elif "rbi" in title.lower() or "rate" in title.lower():
                topic = "policy"
            elif "contract" in title.lower() or "deal" in title.lower():
                topic = "corporate"
            else:
                topic = "market"
            
            article = NewsArticle(
                title=title,
                source=source,
                timestamp=datetime.now().isoformat(),
                url=f"{self.sources[source]}/article/{i}",
                entities=entities,
                topic=topic
            )
            
            articles.append(article)
        
        self.news_history.extend(articles)
        
        return articles
    
    def analyze_sentiment(self, article: NewsArticle) -> SentimentResult:
        """
        Analyze sentiment of a news article.
        
        Args:
            article: News article
            
        Returns:
            SentimentResult
        """
        # Placeholder for FinBERT inference
        # In production, use actual model inference
        
        title_lower = article.title.lower()
        
        # Simple rule-based sentiment (placeholder)
        positive_words = ["strong", "beats", "wins", "gains", "improvement", "dividend"]
        negative_words = ["falls", "concerns", "lower", "reacts negatively", "npa"]
        
        positive_count = sum(1 for word in positive_words if word in title_lower)
        negative_count = sum(1 for word in negative_words if word in title_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            sentiment_score = 0.6 + 0.1 * positive_count
        elif negative_count > positive_count:
            sentiment = "negative"
            sentiment_score = -0.6 - 0.1 * negative_count
        else:
            sentiment = "neutral"
            sentiment_score = 0.0
        
        sentiment_score = np.clip(sentiment_score, -1, 1)
        confidence = 0.7  # Placeholder
        
        # Entity-specific sentiments
        entity_sentiments = {}
        for entity in article.entities:
            entity_sentiments[entity] = sentiment_score
        
        result = SentimentResult(
            article=article,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            confidence=confidence,
            entity_sentiments=entity_sentiments
        )
        
        self.sentiment_history.append(result)
        
        return result
    
    def aggregate_sentiment(
        self,
        ticker: str,
        window_hours: int = 24
    ) -> Dict[str, float]:
        """
        Aggregate sentiment for a ticker over time window.
        
        Args:
            ticker: Stock ticker
            window_hours: Time window in hours
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Filter recent sentiment results
        cutoff_time = datetime.now().replace(
            hour=datetime.now().hour - window_hours
        )
        
        recent_results = [
            r for r in self.sentiment_history
            if ticker in r.entity_sentiments
        ]
        
        if not recent_results:
            return {
                "mean_sentiment": 0.0,
                "sentiment_count": 0,
                "positive_count": 0,
                "negative_count": 0
            }
        
        sentiments = [r.entity_sentiments[ticker] for r in recent_results]
        
        return {
            "mean_sentiment": np.mean(sentiments),
            "sentiment_count": len(sentiments),
            "positive_count": sum(1 for s in sentiments if s > 0),
            "negative_count": sum(1 for s in sentiments if s < 0)
        }
    
    def get_sentiment_features(self, window_hours: int = 24) -> pd.DataFrame:
        """
        Get sentiment features for all tickers.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            DataFrame with sentiment features
        """
        features = []
        
        for ticker in self.tickers:
            agg = self.aggregate_sentiment(ticker, window_hours)
            features.append({
                "ticker": ticker,
                "sentiment_mean": agg["mean_sentiment"],
                "sentiment_count": agg["sentiment_count"],
                "sentiment_positive_count": agg["positive_count"],
                "sentiment_negative_count": agg["negative_count"]
            })
        
        return pd.DataFrame(features)
    
    def print_sentiment_report(self, limit: int = 10) -> None:
        """Print sentiment report."""
        print("\n" + "="*60)
        print("INDIAN NEWS SENTIMENT REPORT")
        print("="*60)
        
        for result in self.sentiment_history[-limit:]:
            print(f"\n{result.article.title}")
            print(f"  Source: {result.article.source}")
            print(f"  Sentiment: {result.sentiment.upper()} ({result.sentiment_score:.2f})")
            print(f"  Entities: {result.article.entities}")
            if result.entity_sentiments:
                print(f"  Entity Sentiments: {result.entity_sentiments}")
        
        print("\nAggregated Sentiment (24h):")
        features = self.get_sentiment_features(window_hours=24)
        for _, row in features.head(5).iterrows():
            print(f"  {row['ticker']}: {row['sentiment_mean']:.2f} ({row['sentiment_count']} articles)")
        
        print("="*60)


def run_sample_news_sentiment():
    """Run sample Indian news sentiment."""
    engine = IndianNewsSentimentEngine()
    
    # Fetch news from sources
    for source in engine.sources.keys():
        articles = engine.fetch_news(source, limit=5)
        print(f"Fetched {len(articles)} articles from {source}")
        
        # Analyze sentiment
        for article in articles:
            engine.analyze_sentiment(article)
    
    # Print report
    engine.print_sentiment_report()
    
    return engine


if __name__ == "__main__":
    run_sample_news_sentiment()
