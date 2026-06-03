"""
LLM-Extracted Sentiment from Earnings Calls

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#16)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- FinBERT fine-tuned on financial news
- Earnings call tone analysis (CEO vs. analyst Q&A)
- Sentiment momentum, reversal, surprise factors
- Alternative data alpha source
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")


@dataclass
class SentimentConfig:
    """Configuration for LLM Sentiment Analysis"""
    # Model
    model_name: str = "ProsusAI/finbert"  # FinBERT model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Sentiment thresholds
    positive_threshold: float = 0.6
    negative_threshold: float = 0.4
    
    # Feature engineering
    sentiment_window: int = 5  # Days for sentiment momentum
    surprise_window: int = 20  # Days for sentiment surprise
    
    # Earnings call specific
    analyze_ceo_tone: bool = True
    analyze_analyst_tone: bool = True
    compare_ceo_analyst: bool = True


class FinBERTSentiment:
    """FinBERT-based sentiment analyzer"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load FinBERT model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()
            print(f"FinBERT loaded on {self.config.device}")
        except Exception as e:
            print(f"Failed to load FinBERT: {e}")
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_score, sentiment_label)
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            # Fallback: simple keyword-based sentiment
            return self._fallback_sentiment(text)
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiment_score = predictions[0][2].item()  # Positive class
            
            # Determine label
            if sentiment_score > self.config.positive_threshold:
                label = "positive"
            elif sentiment_score < self.config.negative_threshold:
                label = "negative"
            else:
                label = "neutral"
            
            return sentiment_score, label
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Tuple[float, str]:
        """Fallback keyword-based sentiment"""
        positive_words = ["growth", "profit", "increase", "strong", "excellent", "beat"]
        negative_words = ["decline", "loss", "decrease", "weak", "concern", "miss"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5, "neutral"
        
        sentiment_score = positive_count / total
        if sentiment_score > 0.6:
            label = "positive"
        elif sentiment_score < 0.4:
            label = "negative"
        else:
            label = "neutral"
        
        return sentiment_score, label


class EarningsCallAnalyzer:
    """Earnings call sentiment analyzer"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.sentiment_analyzer = FinBERTSentiment(config)
        
        # Sentiment history
        self.sentiment_history: Dict[str, List[Tuple[datetime, float]]] = {}
    
    def analyze_earnings_call(self, 
                             ceo_speech: str,
                             analyst_questions: List[str],
                             company: str,
                             date: datetime) -> Dict:
        """
        Analyze earnings call sentiment
        
        Args:
            ceo_speech: CEO's prepared remarks
            analyst_questions: List of analyst questions
            company: Company name
            date: Earnings call date
            
        Returns:
            Dictionary with sentiment analysis results
        """
        results = {
            "company": company,
            "date": date,
            "ceo_sentiment": 0.0,
            "ceo_label": "",
            "analyst_sentiment": 0.0,
            "analyst_label": "",
            "sentiment_gap": 0.0,
            "overall_sentiment": 0.0
        }
        
        # Analyze CEO tone
        if self.config.analyze_ceo_tone:
            ceo_score, ceo_label = self.sentiment_analyzer.analyze_sentiment(ceo_speech)
            results["ceo_sentiment"] = ceo_score
            results["ceo_label"] = ceo_label
        
        # Analyze analyst tone
        if self.config.analyze_analyst_tone and analyst_questions:
            analyst_scores = []
            for question in analyst_questions:
                score, _ = self.sentiment_analyzer.analyze_sentiment(question)
                analyst_scores.append(score)
            
            if analyst_scores:
                results["analyst_sentiment"] = np.mean(analyst_scores)
                results["analyst_label"] = "positive" if np.mean(analyst_scores) > 0.5 else "negative"
        
        # Calculate sentiment gap
        if self.config.compare_ceo_analyst:
            results["sentiment_gap"] = results["ceo_sentiment"] - results["analyst_sentiment"]
        
        # Overall sentiment
        results["overall_sentiment"] = (results["ceo_sentiment"] + results["analyst_sentiment"]) / 2
        
        # Store in history
        if company not in self.sentiment_history:
            self.sentiment_history[company] = []
        self.sentiment_history[company].append((date, results["overall_sentiment"]))
        
        return results
    
    def compute_sentiment_features(self, company: str) -> Dict[str, float]:
        """
        Compute sentiment-based features
        
        Args:
            company: Company name
            
        Returns:
            Dictionary of sentiment features
        """
        if company not in self.sentiment_history or len(self.sentiment_history[company]) < 2:
            return {}
        
        history = sorted(self.sentiment_history[company], key=lambda x: x[0])
        sentiments = [s for _, s in history]
        dates = [d for d, _ in history]
        
        features = {}
        
        # Current sentiment
        features["sentiment_current"] = sentiments[-1]
        
        # Sentiment momentum (change over window)
        if len(sentiments) >= self.config.sentiment_window:
            momentum = sentiments[-1] - sentiments[-self.config.sentiment_window]
            features["sentiment_momentum"] = momentum
        
        # Sentiment surprise (deviation from average)
        if len(sentiments) >= self.config.surprise_window:
            avg_sentiment = np.mean(sentiments[-self.config.surprise_window:])
            surprise = sentiments[-1] - avg_sentiment
            features["sentiment_surprise"] = surprise
        
        # Sentiment trend
        if len(sentiments) >= 3:
            recent_trend = np.polyfit(range(len(sentiments[-5:])), sentiments[-5:], 1)[0]
            features["sentiment_trend"] = recent_trend
        
        # Sentiment volatility
        if len(sentiments) >= 5:
            features["sentiment_volatility"] = np.std(sentiments[-10:])
        
        return features


class LLMSentimentPipeline:
    """
    LLM Sentiment Pipeline for Alternative Data
    
    Extracts sentiment from earnings calls and news using FinBERT.
    Generates alpha signals from sentiment features.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.earnings_analyzer = EarningsCallAnalyzer(config)
        
        # Company sentiment database
        self.company_sentiments: Dict[str, List[Dict]] = {}
    
    def process_earnings_call(self, 
                             ceo_speech: str,
                             analyst_questions: List[str],
                             company: str,
                             date: datetime) -> Dict:
        """
        Process earnings call and store results
        
        Args:
            ceo_speech: CEO's prepared remarks
            analyst_questions: List of analyst questions
            company: Company name
            date: Earnings call date
            
        Returns:
            Analysis results
        """
        results = self.earnings_analyzer.analyze_earnings_call(
            ceo_speech, analyst_questions, company, date
        )
        
        # Store in database
        if company not in self.company_sentiments:
            self.company_sentiments[company] = []
        self.company_sentiments[company].append(results)
        
        return results
    
    def generate_alpha_signal(self, company: str) -> Dict[str, float]:
        """
        Generate alpha signal from sentiment features
        
        Args:
            company: Company name
            
        Returns:
            Alpha signal with confidence
        """
        features = self.earnings_analyzer.compute_sentiment_features(company)
        
        if not features:
            return {"signal": 0.0, "confidence": 0.0}
        
        # Simple signal: positive sentiment momentum -> buy
        signal = 0.0
        confidence = 0.0
        
        if "sentiment_momentum" in features:
            signal += features["sentiment_momentum"] * 2.0
            confidence += 0.3
        
        if "sentiment_surprise" in features:
            signal += features["sentiment_surprise"] * 1.5
            confidence += 0.3
        
        if "sentiment_trend" in features:
            signal += features["sentiment_trend"] * 10.0
            confidence += 0.2
        
        if "sentiment_current" in features:
            signal += (features["sentiment_current"] - 0.5) * 0.5
            confidence += 0.2
        
        # Clip signal
        signal = np.clip(signal, -1.0, 1.0)
        confidence = min(confidence, 1.0)
        
        return {"signal": signal, "confidence": confidence, "features": features}
    
    def get_company_sentiment_history(self, company: str) -> List[Dict]:
        """Get sentiment history for a company"""
        return self.company_sentiments.get(company, [])


def simulate_earnings_call() -> Tuple[str, List[str]]:
    """Simulate earnings call data for testing"""
    ceo_speech = """
    We are pleased to report strong quarterly results. Revenue increased by 15% year-over-year,
    driven by robust growth in our core markets. Our margins expanded significantly due to
    operational efficiency improvements. We remain confident in our long-term strategy and
    expect continued growth in the coming quarters. Our balance sheet remains strong with
    healthy cash flows.
    """
    
    analyst_questions = [
        "Can you provide more details on the margin expansion?",
        "What are your expectations for the next quarter?",
        "How are you managing the competitive landscape?",
        "What about the impact of rising costs?"
    ]
    
    return ceo_speech, analyst_questions


if __name__ == "__main__":
    # Example usage
    config = SentimentConfig(
        model_name="ProsusAI/finbert",
        analyze_ceo_tone=True,
        analyze_analyst_tone=True
    )
    
    pipeline = LLMSentimentPipeline(config)
    
    # Simulate earnings call
    print("Simulating earnings call...")
    ceo_speech, analyst_questions = simulate_earnings_call()
    
    # Process earnings call
    print("\nProcessing earnings call...")
    results = pipeline.process_earnings_call(
        ceo_speech, analyst_questions, "RELIANCE", datetime.now()
    )
    
    print(f"\nEarnings Call Analysis:")
    print(f"  CEO Sentiment: {results['ceo_sentiment']:.4f} ({results['ceo_label']})")
    print(f"  Analyst Sentiment: {results['analyst_sentiment']:.4f} ({results['analyst_label']})")
    print(f"  Sentiment Gap: {results['sentiment_gap']:.4f}")
    print(f"  Overall Sentiment: {results['overall_sentiment']:.4f}")
    
    # Generate alpha signal
    print("\nGenerating alpha signal...")
    signal = pipeline.generate_alpha_signal("RELIANCE")
    
    print(f"\nAlpha Signal:")
    print(f"  Signal: {signal['signal']:.4f}")
    print(f"  Confidence: {signal['confidence']:.4f}")
    if "features" in signal:
        print(f"  Features:")
        for key, value in signal["features"].items():
            print(f"    {key}: {value:.4f}")
