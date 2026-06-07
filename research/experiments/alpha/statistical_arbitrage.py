"""
Statistical Arbitrage - Eigen-Portfolio Strategy

Based on Profit-Centric Audit - High ROI Addition (#2)
Expected ΔSharpe: +0.25
Capacity: 10x
Difficulty: Medium

Methodology:
- Use PCA on stock returns to identify eigen-portfolios
- Trade the first eigen-portfolio (market factor) and residual components
- Mean reversion on residuals
- High capacity, uncorrelated with existing alphas
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class StatArbConfig:
    """Configuration for Statistical Arbitrage"""
    lookback_days: int = 252  # 1 year for PCA estimation
    n_components: int = 10  # Number of PCA components
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5  # Z-score threshold for exit
    position_size_pct: float = 0.05  # 5% of AUM per position
    max_positions: int = 20  # Max number of concurrent positions
    stop_loss_pct: float = 0.03  # 3% stop loss
    target_pct: float = 0.02  # 2% target


@dataclass
class StatArbSignal:
    """Signal from statistical arbitrage"""
    symbol: str
    direction: str  # "long" or "short"
    z_score: float
    eigen_component: int
    confidence: float
    entry_price: float
    stop_loss: float
    target: float


@dataclass
class StatArbPosition:
    """Position from statistical arbitrage"""
    symbol: str
    direction: str
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    target: float
    eigen_component: int


class StatisticalArbEngine:
    """
    Statistical Arbitrage Engine using PCA Eigen-Portfolios
    
    Methodology:
    1. Compute returns for all stocks in universe
    2. Apply PCA to identify eigen-portfolios
    3. Residualize returns against first eigen-portfolio (market factor)
    4. Trade residuals when they deviate significantly from mean
    5. Mean reversion on residuals provides alpha
    """
    
    def __init__(self, config: StatArbConfig):
        self.config = config
        
        # PCA model
        self.pca = PCA(n_components=config.n_components)
        self.scaler = StandardScaler()
        
        # Eigen-portfolio weights
        self.eigen_weights: Optional[np.ndarray] = None
        self.eigen_means: Optional[np.ndarray] = None
        
        # Active positions
        self.positions: Dict[str, StatArbPosition] = {}
        
        # Training data
        self.returns_history: pd.DataFrame = None
        
        # Last training date
        self.last_train_date: Optional[datetime] = None
    
    def train(self, returns_data: pd.DataFrame) -> None:
        """
        Train PCA model on historical returns
        
        Args:
            returns_data: DataFrame with symbols as columns, dates as index
        """
        if len(returns_data) < self.config.lookback_days:
            print(f"Not enough data: {len(returns_data)} < {self.config.lookback_days}")
            return
        
        # Use lookback window
        recent_returns = returns_data.tail(self.config.lookback_days)
        
        # Standardize returns
        scaled_returns = self.scaler.fit_transform(recent_returns)
        
        # Fit PCA
        self.pca.fit(scaled_returns)
        
        # Store eigen-portfolio weights
        self.eigen_weights = self.pca.components_
        
        # Store mean returns for each stock
        self.eigen_means = recent_returns.mean()
        
        # Store training data
        self.returns_history = returns_data
        self.last_train_date = datetime.now()
        
        print(f"PCA trained on {len(recent_returns)} days, {self.config.n_components} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
    
    def compute_residuals(self, current_returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute residuals after removing first eigen-portfolio (market factor)
        
        Args:
            current_returns: Series of current returns for all symbols
            
        Returns:
            residuals: Residual returns after removing market factor
            z_scores: Z-scores of residuals
        """
        if self.eigen_weights is None:
            return np.zeros(len(current_returns)), np.zeros(len(current_returns))
        
        # Standardize current returns
        scaled_returns = self.scaler.transform(current_returns.values.reshape(1, -1))
        
        # Project onto first eigen-portfolio (market factor)
        market_factor = np.dot(scaled_returns, self.eigen_weights[0])
        
        # Residualize: remove market factor
        residuals = scaled_returns - market_factor * self.eigen_weights[0]
        
        # Compute z-scores of residuals
        z_scores = (residuals - self.eigen_means.values) / self.eigen_means.values.std()
        
        return residuals.flatten(), z_scores.flatten()
    
    def generate_signals(self, current_returns: pd.Series) -> List[StatArbSignal]:
        """
        Generate trading signals based on residual z-scores
        
        Args:
            current_returns: Series of current returns for all symbols
            
        Returns:
            List of trading signals
        """
        if self.eigen_weights is None:
            return []
        
        residuals, z_scores = self.compute_residuals(current_returns)
        
        signals = []
        
        for i, (symbol, z_score) in enumerate(zip(current_returns.index, z_scores)):
            # Skip if already in position
            if symbol in self.positions:
                continue
            
            # Long signal: residual too negative (undervalued)
            if z_score < -self.config.entry_threshold:
                signal = StatArbSignal(
                    symbol=symbol,
                    direction="long",
                    z_score=z_score,
                    eigen_component=0,
                    confidence=min(abs(z_score) / self.config.entry_threshold, 1.0),
                    entry_price=current_returns[symbol],
                    stop_loss=current_returns[symbol] * (1 - self.config.stop_loss_pct),
                    target=current_returns[symbol] * (1 + self.config.target_pct)
                )
                signals.append(signal)
            
            # Short signal: residual too positive (overvalued)
            elif z_score > self.config.entry_threshold:
                signal = StatArbSignal(
                    symbol=symbol,
                    direction="short",
                    z_score=z_score,
                    eigen_component=0,
                    confidence=min(abs(z_score) / self.config.entry_threshold, 1.0),
                    entry_price=current_returns[symbol],
                    stop_loss=current_returns[symbol] * (1 + self.config.stop_loss_pct),
                    target=current_returns[symbol] * (1 - self.config.target_pct)
                )
                signals.append(signal)
        
        # Sort by confidence and limit to max positions
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:self.config.max_positions]
    
    def should_retrain(self, current_date: datetime) -> bool:
        """Check if PCA should be retrained (weekly)"""
        if self.last_train_date is None:
            return True
        
        days_since_train = (current_date - self.last_train_date).days
        return days_since_train >= 7  # Weekly retraining
    
    def update_positions(self, current_prices: pd.Series) -> List[str]:
        """
        Update positions based on current prices
        
        Args:
            current_prices: Series of current prices for all symbols
            
        Returns:
            List of symbols to close
        """
        to_close = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                to_close.append(symbol)
                continue
            
            current_price = current_prices[symbol]
            
            # Check stop loss
            if position.direction == "long":
                if current_price < position.stop_loss:
                    to_close.append(symbol)
                elif current_price > position.target:
                    to_close.append(symbol)
            else:  # short
                if current_price > position.stop_loss:
                    to_close.append(symbol)
                elif current_price < position.target:
                    to_close.append(symbol)
        
        # Close positions
        for symbol in to_close:
            del self.positions[symbol]
        
        return to_close
    
    def get_portfolio_exposure(self) -> Dict[str, float]:
        """Get current portfolio exposure by symbol"""
        exposure = {}
        for symbol, position in self.positions.items():
            exposure[symbol] = position.quantity * position.entry_price
        return exposure


class PairsTradingEngine:
    """
    Pairs Trading Engine using Cointegration
    
    Methodology:
    1. Identify cointegrated pairs of stocks
    2. Trade the spread when it deviates from mean
    3. Mean reversion on spread provides alpha
    """
    
    def __init__(self, config: StatArbConfig):
        self.config = config
        
        # Cointegrated pairs
        self.cointegrated_pairs: List[Tuple[str, str]] = []
        
        # Spread statistics
        self.spread_means: Dict[Tuple[str, str], float] = {}
        self.spread_stds: Dict[Tuple[str, str], float] = {}
        
        # Active positions
        self.positions: Dict[Tuple[str, str], Dict] = {}
    
    def find_cointegrated_pairs(self, returns_data: pd.DataFrame) -> None:
        """
        Find cointegrated pairs using Engle-Granger test
        
        Args:
            returns_data: DataFrame with symbols as columns, dates as index
        """
        from statsmodels.tsa.stattools import coint
        
        symbols = returns_data.columns
        n_symbols = len(symbols)
        
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                # Skip if insufficient data
                if returns_data[symbol1].isna().any() or returns_data[symbol2].isna().any():
                    continue
                
                # Engle-Granger test
                try:
                    score, pvalue, _ = coint(returns_data[symbol1], returns_data[symbol2])
                    
                    # If p-value < 0.05, pairs are cointegrated
                    if pvalue < 0.05:
                        self.cointegrated_pairs.append((symbol1, symbol2))
                        
                        # Compute spread statistics
                        spread = returns_data[symbol1] - returns_data[symbol2]
                        self.spread_means[(symbol1, symbol2)] = spread.mean()
                        self.spread_stds[(symbol1, symbol2)] = spread.std()
                except:
                    continue
        
        print(f"Found {len(self.cointegrated_pairs)} cointegrated pairs")
    
    def generate_signals(self, current_returns: pd.Series) -> List[Dict]:
        """
        Generate trading signals based on spread deviations
        
        Args:
            current_returns: Series of current returns for all symbols
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for symbol1, symbol2 in self.cointegrated_pairs:
            if symbol1 not in current_returns or symbol2 not in current_returns:
                continue
            
            # Skip if already in position
            if (symbol1, symbol2) in self.positions:
                continue
            
            # Compute current spread
            spread = current_returns[symbol1] - current_returns[symbol2]
            spread_mean = self.spread_means[(symbol1, symbol2)]
            spread_std = self.spread_stds[(symbol1, symbol2)]
            
            # Compute z-score
            z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Generate signal if deviation is significant
            if abs(z_score) > self.config.entry_threshold:
                direction = "long_spread" if z_score < 0 else "short_spread"
                
                signal = {
                    "pair": (symbol1, symbol2),
                    "direction": direction,
                    "z_score": z_score,
                    "confidence": min(abs(z_score) / self.config.entry_threshold, 1.0)
                }
                signals.append(signal)
        
        return signals


def backtest_stat_arb(
    returns_data: pd.DataFrame,
    config: StatArbConfig
) -> Dict:
    """
    Simple backtest for statistical arbitrage
    
    Args:
        returns_data: DataFrame with symbols as columns, dates as index
        config: Configuration for statistical arbitrage
        
    Returns:
        Dictionary with backtest results
    """
    engine = StatisticalArbEngine(config)
    
    # Train on first year
    train_data = returns_data.iloc[:config.lookback_days]
    engine.train(train_data)
    
    # Test on remaining data
    test_data = returns_data.iloc[config.lookback_days:]
    
    # Simulate trading
    returns = []
    
    for date, row in test_data.iterrows():
        # Generate signals
        signals = engine.generate_signals(row)
        
        # Simple simulation: assume we take all signals
        if signals:
            # Compute average return of signals
            signal_returns = [row[s.symbol] for s in signals if s.symbol in row.index]
            if signal_returns:
                returns.append(np.mean(signal_returns))
        else:
            returns.append(0.0)
    
    # Compute metrics
    returns_array = np.array(returns)
    
    sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
    
    return {
        "total_return": np.sum(returns_array),
        "sharpe_ratio": sharpe,
        "num_trades": len([r for r in returns if r != 0])
    }


if __name__ == "__main__":
    # Example usage
    config = StatArbConfig()
    
    # Generate synthetic returns data for testing
    np.random.seed(42)
    n_symbols = 50
    n_days = 500
    
    synthetic_returns = pd.DataFrame(
        np.random.randn(n_days, n_symbols) * 0.01,
        index=pd.date_range(start="2020-01-01", periods=n_days),
        columns=[f"STOCK_{i}" for i in range(n_symbols)]
    )
    
    # Add some correlation structure
    synthetic_returns.iloc[:, :10] += synthetic_returns.iloc[:, 0].values.reshape(-1, 1) * 0.5
    
    print("Running Statistical Arbitrage Backtest...")
    results = backtest_stat_arb(synthetic_returns, config)
    print(f"Results: {results}")
