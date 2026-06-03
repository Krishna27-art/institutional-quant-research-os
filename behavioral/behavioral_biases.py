"""
Behavioral Finance Biases

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#29)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Loss aversion
- Overconfidence bias
- Herding behavior
- Disposition effect
- Used to model and exploit market inefficiencies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class BiasType(Enum):
    LOSS_AVERSION = "loss_aversion"
    OVERCONFIDENCE = "overconfidence"
    HERDING = "herding"
    DISPOSITION = "disposition"
    ANCHORING = "anchoring"
    CONFIRMATION = "confirmation"


@dataclass
class BiasConfig:
    """Configuration for Behavioral Bias Analysis"""
    # Loss aversion parameters
    loss_aversion_coefficient: float = 2.25  # Kahneman-Tversky value
    reference_point: float = 0.0  # Reference point for gains/losses
    
    # Overconfidence parameters
    confidence_interval_width: float = 0.8  # Width of confidence interval
    overconfidence_factor: float = 1.2  # Overconfidence multiplier
    
    # Herding parameters
    herding_threshold: float = 0.7  # Correlation threshold for herding
    window_size: int = 20  # Window for herding detection
    
    # Disposition effect parameters
    holding_period_days: int = 30  # Lookback for disposition effect
    gain_threshold: float = 0.1  # 10% gain threshold
    loss_threshold: float = -0.1  # -10% loss threshold


class BehavioralBiasAnalyzer:
    """
    Behavioral Bias Analyzer
    
    Detects and quantifies behavioral biases in market data.
    Used to identify exploitable inefficiencies.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: BiasConfig):
        self.config = config
        
        # Bias scores
        self.bias_scores: Dict[str, float] = {}
        
        # Bias history
        self.bias_history: Dict[str, List[float]] = {}
    
    def detect_loss_aversion(self, returns: pd.Series) -> float:
        """
        Detect loss aversion bias
        
        Loss aversion: investors feel losses more than gains
        Measured by asymmetry in reaction to gains vs losses
        
        Args:
            returns: Asset returns
            
        Returns:
            Loss aversion score (0-1)
        """
        # Separate gains and losses
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(gains) == 0 or len(losses) == 0:
            return 0.0
        
        # Calculate average reaction
        avg_gain_reaction = gains.mean()
        avg_loss_reaction = abs(losses.mean())
        
        # Loss aversion ratio
        loss_aversion = avg_loss_reaction / (avg_gain_reaction + 1e-8)
        
        # Normalize to 0-1
        score = min(loss_aversion / self.config.loss_aversion_coefficient, 1.0)
        
        self.bias_scores["loss_aversion"] = score
        return score
    
    def detect_overconfidence(self, predictions: pd.Series, actuals: pd.Series) -> float:
        """
        Detect overconfidence bias
        
        Overconfidence: predictions are too precise
        Measured by calibration of prediction intervals
        
        Args:
            predictions: Predicted returns
            actuals: Actual returns
            
        Returns:
            Overconfidence score (0-1)
        """
        # Calculate prediction errors
        errors = predictions - actuals
        
        # Check if errors are larger than expected
        expected_error = self.config.confidence_interval_width * predictions.std()
        actual_error = errors.std()
        
        # Overconfidence ratio
        overconfidence = actual_error / (expected_error + 1e-8)
        
        # Normalize to 0-1
        score = min(overconfidence / self.config.overconfidence_factor, 1.0)
        
        self.bias_scores["overconfidence"] = score
        return score
    
    def detect_herding(self, returns: pd.DataFrame) -> float:
        """
        Detect herding behavior
        
        Herding: stocks move together more than fundamentals suggest
        Measured by correlation clustering
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Herding score (0-1)
        """
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Average correlation
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Herding score
        score = min(avg_corr / self.config.herding_threshold, 1.0)
        
        self.bias_scores["herding"] = score
        return score
    
    def detect_disposition_effect(self, positions: pd.DataFrame, returns: pd.Series) -> float:
        """
        Detect disposition effect
        
        Disposition effect: investors hold losers too long, sell winners too early
        Measured by holding period vs return relationship
        
        Args:
            positions: Position changes
            returns: Asset returns
            
        Returns:
            Disposition effect score (0-1)
        """
        # Calculate holding periods
        # Simplified: check if negative correlation between return and position change
        position_changes = positions.diff()
        
        # Calculate correlation
        correlation = position_changes.corr(returns)
        
        # Disposition effect: negative correlation (sell winners, hold losers)
        score = max(-correlation, 0) if correlation < 0 else 0
        
        self.bias_scores["disposition"] = score
        return score
    
    def detect_anchoring(self, prices: pd.Series) -> float:
        """
        Detect anchoring bias
        
        Anchoring: prices anchored to recent levels
        Measured by mean reversion tendency
        
        Args:
            prices: Price series
            
        Returns:
            Anchoring score (0-1)
        """
        # Calculate mean reversion
        mean_price = prices.rolling(self.config.window_size).mean()
        deviation = (prices - mean_price) / mean_price
        
        # Anchoring: prices tend to revert to mean
        # Measure by autocorrelation of deviations
        autocorr = deviation.autocorr(lag=1)
        
        # Negative autocorrelation indicates mean reversion (anchoring)
        score = max(-autocorr, 0)
        
        self.bias_scores["anchoring"] = score
        return score
    
    def detect_confirmation_bias(self, news_sentiment: pd.Series, price_changes: pd.Series) -> float:
        """
        Detect confirmation bias
        
        Confirmation bias: investors seek information that confirms beliefs
        Measured by correlation between sentiment and price changes
        
        Args:
            news_sentiment: News sentiment scores
            price_changes: Price changes
            
        Returns:
            Confirmation bias score (0-1)
        """
        # Align data
        common_index = news_sentiment.index.intersection(price_changes.index)
        sentiment = news_sentiment.loc[common_index]
        prices = price_changes.loc[common_index]
        
        # Calculate correlation
        correlation = sentiment.corr(prices)
        
        # High positive correlation indicates confirmation bias
        score = min(abs(correlation), 1.0)
        
        self.bias_scores["confirmation"] = score
        return score
    
    def get_bias_summary(self) -> Dict:
        """Get summary of all biases"""
        return {
            "loss_aversion": self.bias_scores.get("loss_aversion", 0.0),
            "overconfidence": self.bias_scores.get("overconfidence", 0.0),
            "herding": self.bias_scores.get("herding", 0.0),
            "disposition": self.bias_scores.get("disposition", 0.0),
            "anchoring": self.bias_scores.get("anchoring", 0.0),
            "confirmation": self.bias_scores.get("confirmation", 0.0)
        }
    
    def generate_bias_signals(self) -> Dict[str, float]:
        """
        Generate trading signals based on biases
        
        Returns:
            Dictionary of bias -> signal
        """
        signals = {}
        
        # Loss aversion: fade moves after losses
        if self.bias_scores.get("loss_aversion", 0) > 0.5:
            signals["loss_aversion_fade"] = 1.0
        
        # Overconfidence: fade consensus
        if self.bias_scores.get("overconfidence", 0) > 0.5:
            signals["overconfidence_fade"] = 1.0
        
        # Herding: contrarian to herd
        if self.bias_scores.get("herding", 0) > 0.5:
            signals["herding_contrarian"] = 1.0
        
        # Disposition: hold winners, sell losers (contrarian)
        if self.bias_scores.get("disposition", 0) > 0.5:
            signals["disposition_contrarian"] = 1.0
        
        return signals


def simulate_market_data(n_assets: int = 50, n_days: int = 252) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Simulate market data for testing"""
    np.random.seed(42)
    
    # Generate correlated returns with some behavioral bias
    base_returns = np.random.randn(n_days)
    
    # Add herding (common factor)
    herding_factor = 0.3
    returns = np.zeros((n_days, n_assets))
    
    for i in range(n_assets):
        # Individual component
        idiosyncratic = np.random.randn(n_days) * 0.7
        # Herding component
        herding = base_returns * herding_factor
        returns[:, i] = herding + idiosyncratic
    
    # Add loss aversion (asymmetric reaction)
    returns[returns < 0] *= 1.5
    
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    # Simulate prices
    prices = 100 * np.cumprod(1 + returns_df / 100, axis=0)
    
    # Simulate news sentiment
    news_sentiment = pd.Series(np.random.randn(n_days), index=dates)
    
    return returns_df, news_sentiment, prices


if __name__ == "__main__":
    # Example usage
    config = BiasConfig(
        loss_aversion_coefficient=2.25,
        herding_threshold=0.7
    )
    
    analyzer = BehavioralBiasAnalyzer(config)
    
    # Simulate data
    print("Simulating market data...")
    returns, news_sentiment, prices = simulate_market_data(50, 252)
    
    # Detect biases
    print("\nDetecting behavioral biases...")
    
    loss_aversion = analyzer.detect_loss_aversion(returns.iloc[:, 0])
    print(f"  Loss Aversion: {loss_aversion:.4f}")
    
    herding = analyzer.detect_herding(returns)
    print(f"  Herding: {herding:.4f}")
    
    anchoring = analyzer.detect_anchoring(prices.iloc[:, 0])
    print(f"  Anchoring: {anchoring:.4f}")
    
    confirmation = analyzer.detect_confirmation_bias(news_sentiment, returns.iloc[:, 0])
    print(f"  Confirmation Bias: {confirmation:.4f}")
    
    # Bias summary
    print("\nBias Summary:")
    summary = analyzer.get_bias_summary()
    for bias, score in summary.items():
        print(f"  {bias}: {score:.4f}")
    
    # Generate signals
    print("\nGenerating bias-based signals...")
    signals = analyzer.generate_bias_signals()
    for signal, value in signals.items():
        print(f"  {signal}: {value:.2f}")
