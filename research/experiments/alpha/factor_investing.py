"""
Factor Investing (Fama-French + Custom)

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#27)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Fama-French 5-factor model (Market, Size, Value, Profitability, Investment)
- Custom factors for Indian market
- Factor construction and portfolio formation
- Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class FactorConfig:
    """Configuration for Factor Investing"""
    # Factor selection
    use_market_factor: bool = True
    use_size_factor: bool = True
    use_value_factor: bool = True
    use_momentum_factor: bool = True
    use_profitability_factor: bool = True
    use_investment_factor: bool = True
    
    # Custom Indian market factors
    use_liquidity_factor: bool = True
    use_volatility_factor: bool = True
    use_sector_factor: bool = True
    
    # Portfolio construction
    n_quantiles: int = 5  # Number of quantiles for portfolio formation
    rebalance_frequency: str = "monthly"  # "monthly", "quarterly"
    
    # Risk model
    factor_model_type: str = "fama_french"  # "fama_french", "barra", "custom"
    
    # Constraints
    max_factor_exposure: float = 2.0  # Maximum factor exposure
    min_stocks: int = 20  # Minimum stocks per portfolio


class FactorCalculator:
    """Calculate factor exposures for stocks"""
    
    def __init__(self, config: FactorConfig):
        self.config = config
    
    def calculate_size_factor(self, market_cap: pd.Series) -> pd.Series:
        """
        Calculate size factor (log market cap)
        
        Args:
            market_cap: Market capitalization
            
        Returns:
            Size factor
        """
        return np.log(market_cap)
    
    def calculate_value_factor(self, book_to_market: pd.Series) -> pd.Series:
        """
        Calculate value factor (book-to-market ratio)
        
        Args:
            book_to_market: Book-to-market ratio
            
        Returns:
            Value factor
        """
        return book_to_market
    
    def calculate_momentum_factor(self, returns: pd.DataFrame, window: int = 12) -> pd.Series:
        """
        Calculate momentum factor (12-month return)
        
        Args:
            returns: Historical returns
            window: Lookback window in months
            
        Returns:
            Momentum factor
        """
        return returns.tail(window * 20).sum()  # Approximate monthly to daily
    
    def calculate_profitability_factor(self, roe: pd.Series) -> pd.Series:
        """
        Calculate profitability factor (ROE)
        
        Args:
            roe: Return on equity
            
        Returns:
            Profitability factor
        """
        return roe
    
    def calculate_investment_factor(self, asset_growth: pd.Series) -> pd.Series:
        """
        Calculate investment factor (asset growth)
        
        Args:
            asset_growth: Asset growth rate
            
        Returns:
            Investment factor (inverted)
        """
        return -asset_growth  # High investment -> low returns
    
    def calculate_liquidity_factor(self, volume: pd.Series, market_cap: pd.Series) -> pd.Series:
        """
        Calculate liquidity factor (turnover)
        
        Args:
            volume: Trading volume
            market_cap: Market capitalization
            
        Returns:
            Liquidity factor
        """
        return volume / market_cap
    
    def calculate_volatility_factor(self, returns: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate volatility factor
        
        Args:
            returns: Historical returns
            window: Lookback window
            
        Returns:
            Volatility factor (inverted)
        """
        vol = returns.tail(window).std()
        return -vol  # High volatility -> low returns


class FactorModel:
    """
    Factor Model for Portfolio Construction
    
    Implements Fama-French 5-factor model with custom factors.
    """
    
    def __init__(self, config: FactorConfig):
        self.config = config
        self.factor_calculator = FactorCalculator(config)
        
        # Factor exposures
        self.factor_exposures: pd.DataFrame = None
        
        # Factor returns
        self.factor_returns: pd.Series = None
        
        # Model
        self.model = None
    
    def construct_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct factor exposures for all stocks
        
        Args:
            data: DataFrame with stock characteristics
            
        Returns:
            DataFrame with factor exposures
        """
        factors = pd.DataFrame(index=data.index)
        
        # Fama-French factors
        if self.config.use_market_factor:
            factors["market"] = 1.0  # Market factor is constant for all stocks
        
        if self.config.use_size_factor and "market_cap" in data.columns:
            factors["size"] = self.factor_calculator.calculate_size_factor(data["market_cap"])
        
        if self.config.use_value_factor and "book_to_market" in data.columns:
            factors["value"] = self.factor_calculator.calculate_value_factor(data["book_to_market"])
        
        if self.config.use_momentum_factor and "returns" in data.columns:
            factors["momentum"] = self.factor_calculator.calculate_momentum_factor(data["returns"])
        
        if self.config.use_profitability_factor and "roe" in data.columns:
            factors["profitability"] = self.factor_calculator.calculate_profitability_factor(data["roe"])
        
        if self.config.use_investment_factor and "asset_growth" in data.columns:
            factors["investment"] = self.factor_calculator.calculate_investment_factor(data["asset_growth"])
        
        # Custom Indian market factors
        if self.config.use_liquidity_factor and "volume" in data.columns and "market_cap" in data.columns:
            factors["liquidity"] = self.factor_calculator.calculate_liquidity_factor(data["volume"], data["market_cap"])
        
        if self.config.use_volatility_factor and "returns" in data.columns:
            factors["volatility"] = self.factor_calculator.calculate_volatility_factor(data["returns"])
        
        self.factor_exposures = factors
        return factors
    
    def estimate_factor_returns(self, stock_returns: pd.DataFrame, factor_exposures: pd.DataFrame) -> pd.Series:
        """
        Estimate factor returns using regression
        
        Args:
            stock_returns: Stock returns
            factor_exposures: Factor exposures
            
        Returns:
            Factor returns
        """
        if not SKLEARN_AVAILABLE:
            return pd.Series(0, index=factor_exposures.columns)
        
        # Align data
        common_index = stock_returns.index.intersection(factor_exposures.index)
        returns = stock_returns.loc[common_index]
        exposures = factor_exposures.loc[common_index]
        
        # Run regression for each stock
        factor_returns_list = []
        
        for stock in returns.columns:
            y = returns[stock].dropna()
            X = exposures.loc[y.index].dropna()
            
            if len(y) < 10 or len(X) < 10:
                continue
            
            # Align X and y
            common_idx = y.index.intersection(X.index)
            y = y.loc[common_idx]
            X = X.loc[common_idx]
            
            model = LinearRegression()
            model.fit(X, y)
            
            factor_returns_list.append(model.coef_)
        
        if factor_returns_list:
            factor_returns_array = np.array(factor_returns_list)
            avg_factor_returns = factor_returns_array.mean(axis=0)
            return pd.Series(avg_factor_returns, index=exposures.columns)
        
        return pd.Series(0, index=exposures.columns)
    
    def form_factor_portfolios(self, factor_exposures: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Form factor-based portfolios (long-short)
        
        Args:
            factor_exposures: Factor exposures
            returns: Stock returns
            
        Returns:
            Dictionary of factor -> portfolio returns
        """
        portfolio_returns = {}
        
        for factor in factor_exposures.columns:
            # Sort stocks by factor exposure
            factor_values = factor_exposures[factor].dropna()
            
            if len(factor_values) < self.config.n_quantiles * self.config.min_stocks:
                continue
            
            # Form quantiles
            quantiles = pd.qcut(factor_values, self.config.n_quantiles, labels=False, duplicates='drop')
            
            # Long top quantile, short bottom quantile
            top_stocks = quantiles[quantiles == quantiles.max()].index
            bottom_stocks = quantiles[quantiles == quantiles.min()].index
            
            # Calculate portfolio returns
            top_returns = returns[top_stocks].mean(axis=1)
            bottom_returns = returns[bottom_stocks].mean(axis=1)
            
            portfolio_returns[factor] = top_returns - bottom_returns
        
        return portfolio_returns
    
    def calculate_factor_correlations(self, portfolio_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calculate correlations between factor portfolios
        
        Args:
            portfolio_returns: Dictionary of factor portfolio returns
            
        Returns:
            Correlation matrix
        """
        returns_df = pd.DataFrame(portfolio_returns)
        return returns_df.corr()


class FactorInvestor:
    """
    Factor Investor
    
    Implements factor-based investing strategy using
    Fama-French and custom factors.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: FactorConfig):
        self.config = config
        self.factor_model = FactorModel(config)
        
        # Portfolio weights
        self.portfolio_weights: pd.Series = None
        
        # Performance tracking
        self.portfolio_returns: pd.Series = None
    
    def construct_portfolio(self, data: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """
        Construct factor-based portfolio
        
        Args:
            data: Stock characteristics
            returns: Stock returns
            
        Returns:
            Portfolio weights
        """
        # Construct factors
        factor_exposures = self.factor_model.construct_factors(data)
        
        # Estimate factor returns
        factor_returns = self.factor_model.estimate_factor_returns(returns, factor_exposures)
        
        # Calculate factor scores for each stock
        factor_scores = factor_exposures * factor_returns
        total_score = factor_scores.sum(axis=1)
        
        # Normalize weights
        weights = total_score / total_score.abs().sum()
        
        self.portfolio_weights = weights
        return weights
    
    def backtest(self, returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """
        Backtest factor portfolio
        
        Args:
            returns: Stock returns
            weights: Portfolio weights
            
        Returns:
            Portfolio returns
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        self.portfolio_returns = portfolio_returns
        return portfolio_returns
    
    def get_performance_metrics(self) -> Dict:
        """Get portfolio performance metrics"""
        if self.portfolio_returns is None:
            return {}
        
        returns = self.portfolio_returns
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        # Drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }


def simulate_stock_data(n_stocks: int = 100, n_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate stock data for testing"""
    np.random.seed(42)
    
    # Generate stock characteristics
    market_cap = np.random.lognormal(20, 1, n_stocks)
    book_to_market = np.random.uniform(0.5, 2.0, n_stocks)
    roe = np.random.uniform(0.05, 0.25, n_stocks)
    asset_growth = np.random.uniform(-0.1, 0.2, n_stocks)
    volume = np.random.exponential(1000000, n_stocks)
    
    characteristics = pd.DataFrame({
        "market_cap": market_cap,
        "book_to_market": book_to_market,
        "roe": roe,
        "asset_growth": asset_growth,
        "volume": volume
    })
    
    # Generate returns with factor structure
    factor_loadings = np.random.randn(n_stocks, 5)  # 5 factors
    factor_returns = np.random.randn(n_days, 5) * 0.01
    idiosyncratic = np.random.randn(n_days, n_stocks) * 0.02
    
    returns = factor_returns @ factor_loadings.T + idiosyncratic
    
    stock_names = [f"STOCK_{i}" for i in range(n_stocks)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    returns_df = pd.DataFrame(returns, index=dates, columns=stock_names)
    
    return characteristics, returns_df


if __name__ == "__main__":
    # Example usage
    config = FactorConfig(
        use_market_factor=True,
        use_size_factor=True,
        use_value_factor=True,
        use_momentum_factor=True,
        n_quantiles=5
    )
    
    investor = FactorInvestor(config)
    
    # Simulate data
    print("Simulating stock data...")
    characteristics, returns = simulate_stock_data(100, 252)
    
    # Construct portfolio
    print("\nConstructing factor portfolio...")
    weights = investor.construct_portfolio(characteristics, returns)
    
    print(f"\nPortfolio Weights (top 10):")
    print(weights.nlargest(10))
    
    # Backtest
    print("\nBacktesting portfolio...")
    portfolio_returns = investor.backtest(returns, weights)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = investor.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Factor correlations
    print("\nFactor Portfolio Correlations:")
    factor_exposures = investor.factor_model.construct_factors(characteristics)
    portfolio_returns_dict = investor.factor_model.form_factor_portfolios(factor_exposures, returns)
    correlations = investor.factor_model.calculate_factor_correlations(portfolio_returns_dict)
    print(correlations.to_string())
