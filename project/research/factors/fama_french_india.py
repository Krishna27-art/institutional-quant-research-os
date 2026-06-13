"""
Fama-French 5-Factor Model for India

Implements the Fama-French 5-factor model adapted for the Indian market.
The model includes:
1. Market factor (Rm - Rf)
2. Size factor (SMB - Small Minus Big)
3. Value factor (HML - High Minus Low)
4. Profitability factor (RMW - Robust Minus Weak)
5. Investment factor (CMA - Conservative Minus Aggressive)

Key Features:
- Factor construction for Indian market
- Size and value sorting
- Profitability and investment metrics
- Factor return calculation
- Regression analysis for asset pricing
- Sector-specific adjustments for India

Based on Blueprint Week 5-6: Alpha Models (Classical)
References:
- Fama & French (2015) - A Five-Factor Asset Pricing Model
- Adapted for Indian market (NSE indices)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FamaFrenchIndia:
    """
    Fama-French 5-Factor Model for Indian Market.
    
    This class implements the Fama-French 5-factor model adapted for
    the Indian market using NSE indices and stock characteristics.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.06,  # 6% annual risk-free rate for India
        rebalance_frequency: str = 'monthly'
    ):
        """
        Initialize Fama-French India model.
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 10-year government bond)
            rebalance_frequency: Rebalancing frequency ('monthly', 'quarterly')
        """
        self.risk_free_rate = risk_free_rate
        self.rebalance_frequency = rebalance_frequency
        
        # Factor returns
        self.market_factor = None
        self.size_factor = None
        self.value_factor = None
        self.profitability_factor = None
        self.investment_factor = None
    
    def calculate_market_factor(
        self,
        market_returns: pd.Series,
        risk_free_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate market factor (Rm - Rf).
        
        Args:
            market_returns: Market index returns (e.g., NIFTY 50)
            risk_free_returns: Risk-free rate returns
            
        Returns:
            Market factor series
        """
        market_factor = market_returns - risk_free_returns
        self.market_factor = market_factor
        return market_factor
    
    def calculate_size_factor(
        self,
        small_cap_returns: pd.Series,
        large_cap_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate size factor (SMB - Small Minus Big).
        
        For India, we use NIFTY Smallcap 100 and NIFTY 50.
        
        Args:
            small_cap_returns: Small-cap index returns
            large_cap_returns: Large-cap index returns
            
        Returns:
            Size factor series
        """
        size_factor = small_cap_returns - large_cap_returns
        self.size_factor = size_factor
        return size_factor
    
    def calculate_value_factor(
        self,
        value_returns: pd.Series,
        growth_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate value factor (HML - High Minus Low).
        
        For India, we construct value and growth portfolios based on P/B ratios.
        
        Args:
            value_returns: Value portfolio returns
            growth_returns: Growth portfolio returns
            
        Returns:
            Value factor series
        """
        value_factor = value_returns - growth_returns
        self.value_factor = value_factor
        return value_factor
    
    def calculate_profitability_factor(
        self,
        robust_returns: pd.Series,
        weak_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate profitability factor (RMW - Robust Minus Weak).
        
        RMW is based on operating profitability (OP/BE).
        
        Args:
            robust_returns: High profitability portfolio returns
            weak_returns: Low profitability portfolio returns
            
        Returns:
            Profitability factor series
        """
        profitability_factor = robust_returns - weak_returns
        self.profitability_factor = profitability_factor
        return profitability_factor
    
    def calculate_investment_factor(
        self,
        conservative_returns: pd.Series,
        aggressive_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate investment factor (CMA - Conservative Minus Aggressive).
        
        CMA is based on investment (change in total assets / total assets).
        
        Args:
            conservative_returns: Low investment portfolio returns
            aggressive_returns: High investment portfolio returns
            
        Returns:
            Investment factor series
        """
        investment_factor = conservative_returns - aggressive_returns
        self.investment_factor = investment_factor
        return investment_factor
    
    def construct_factors(
        self,
        market_returns: pd.Series,
        small_cap_returns: pd.Series,
        large_cap_returns: pd.Series,
        value_returns: pd.Series,
        growth_returns: pd.Series,
        robust_returns: pd.Series,
        weak_returns: pd.Series,
        conservative_returns: pd.Series,
        aggressive_returns: pd.Series,
        risk_free_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Construct all 5 factors.
        
        Args:
            market_returns: Market index returns
            small_cap_returns: Small-cap returns
            large_cap_returns: Large-cap returns
            value_returns: Value portfolio returns
            growth_returns: Growth portfolio returns
            robust_returns: High profitability returns
            weak_returns: Low profitability returns
            conservative_returns: Low investment returns
            aggressive_returns: High investment returns
            risk_free_returns: Risk-free rate returns
            
        Returns:
            DataFrame with all 5 factors
        """
        factors = pd.DataFrame(index=market_returns.index)
        
        factors['MKT'] = self.calculate_market_factor(market_returns, risk_free_returns)
        factors['SMB'] = self.calculate_size_factor(small_cap_returns, large_cap_returns)
        factors['HML'] = self.calculate_value_factor(value_returns, growth_returns)
        factors['RMW'] = self.calculate_profitability_factor(robust_returns, weak_returns)
        factors['CMA'] = self.calculate_investment_factor(conservative_returns, aggressive_returns)
        
        return factors
    
    def run_regression(
        self,
        asset_returns: pd.Series,
        factors: pd.DataFrame
    ) -> Dict:
        """
        Run Fama-French 5-factor regression for an asset.
        
        R_i - R_f = α + β_MKT * MKT + β_SMB * SMB + β_HML * HML + β_RMW * RMW + β_CMA * CMA + ε
        
        Args:
            asset_returns: Asset excess returns
            factors: Factor returns DataFrame
            
        Returns:
            Dictionary with regression results
        """
        # Align data
        aligned_data = pd.concat([asset_returns, factors], axis=1).dropna()
        
        if len(aligned_data) < 30:
            logger.warning("Insufficient data for regression")
            return {}
        
        y = aligned_data.iloc[:, 0].values
        X = aligned_data.iloc[:, 1:].values
        X = np.column_stack([np.ones(len(X)), X])  # Add constant
        
        # OLS regression
        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            residuals = y - X @ beta
            r_squared = 1 - (residuals @ residuals) / (y.T @ y)
            
            # Standard errors
            n = len(y)
            k = X.shape[1]
            sigma2 = (residuals @ residuals) / (n - k)
            cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # t-statistics
            t_stats = beta / std_errors
            
            # Factor names
            factor_names = ['Alpha', 'MKT', 'SMB', 'HML', 'RMW', 'CMA']
            
            results = {}
            for i, name in enumerate(factor_names):
                results[name] = {
                    'coefficient': beta[i],
                    'std_error': std_errors[i],
                    't_stat': t_stats[i],
                    'p_value': 2 * (1 - stats.t.cdf(abs(t_stats[i]), n - k))
                }
            
            results['R_squared'] = r_squared
            results['n_observations'] = n
            
            return results
            
        except np.linalg.LinAlgError:
            logger.error("Matrix inversion failed in regression")
            return {}
    
    def calculate_expected_return(
        self,
        factor_betas: Dict,
        factor_premiums: Dict,
        risk_free_rate: float
    ) -> float:
        """
        Calculate expected return using factor model.
        
        E[R_i] = R_f + β_MKT * E[MKT] + β_SMB * E[SMB] + β_HML * E[HML] + β_RMW * E[RMW] + β_CMA * E[CMA]
        
        Args:
            factor_betas: Factor betas from regression
            factor_premiums: Expected factor premiums
            risk_free_rate: Risk-free rate
            
        Returns:
            Expected annual return
        """
        expected_return = risk_free_rate
        
        for factor in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
            beta = factor_betas.get(factor, {}).get('coefficient', 0)
            premium = factor_premiums.get(factor, 0)
            expected_return += beta * premium
        
        return expected_return


class IndiaFactorConstruction:
    """
    Helper class for constructing Indian market factors.
    
    Provides methods to sort stocks into portfolios based on
    characteristics for the Indian market.
    """
    
    @staticmethod
    def sort_by_market_cap(
        prices: pd.DataFrame,
        market_caps: pd.Series,
        n_portfolios: int = 6
    ) -> Dict[str, List[str]]:
        """
        Sort stocks by market cap into portfolios.
        
        Args:
            prices: Price DataFrame
            market_caps: Market cap series
            n_portfolios: Number of portfolios
            
        Returns:
            Dictionary mapping portfolio names to stock lists
        """
        # Sort by market cap
        sorted_stocks = market_caps.sort_values()
        
        # Divide into portfolios
        portfolio_size = len(sorted_stocks) // n_portfolios
        portfolios = {}
        
        for i in range(n_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < n_portfolios - 1 else len(sorted_stocks)
            stocks = sorted_stocks.iloc[start_idx:end_idx].index.tolist()
            portfolios[f'Size_{i+1}'] = stocks
        
        return portfolios
    
    @staticmethod
    def sort_by_book_to_market(
        prices: pd.DataFrame,
        book_values: pd.Series,
        n_portfolios: int = 3
    ) -> Dict[str, List[str]]:
        """
        Sort stocks by book-to-market ratio.
        
        Args:
            prices: Price DataFrame
            book_values: Book value per share
            n_portfolios: Number of portfolios
            
        Returns:
            Dictionary mapping portfolio names to stock lists
        """
        # Calculate B/M ratio
        latest_prices = prices.iloc[-1]
        bm_ratios = book_values / latest_prices
        
        # Sort by B/M
        sorted_stocks = bm_ratios.sort_values()
        
        # Divide into portfolios
        portfolio_size = len(sorted_stocks) // n_portfolios
        portfolios = {}
        
        for i in range(n_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < n_portfolios - 1 else len(sorted_stocks)
            stocks = sorted_stocks.iloc[start_idx:end_idx].index.tolist()
            portfolios[f'BM_{i+1}'] = stocks
        
        return portfolios
    
    @staticmethod
    def sort_by_profitability(
        operating_profit: pd.Series,
        book_equity: pd.Series,
        n_portfolios: int = 3
    ) -> Dict[str, List[str]]:
        """
        Sort stocks by operating profitability (OP/BE).
        
        Args:
            operating_profit: Operating profit
            book_equity: Book equity
            n_portfolios: Number of portfolios
            
        Returns:
            Dictionary mapping portfolio names to stock lists
        """
        # Calculate OP/BE
        profitability = operating_profit / book_equity
        
        # Sort by profitability
        sorted_stocks = profitability.sort_values()
        
        # Divide into portfolios
        portfolio_size = len(sorted_stocks) // n_portfolios
        portfolios = {}
        
        for i in range(n_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < n_portfolios - 1 else len(sorted_stocks)
            stocks = sorted_stocks.iloc[start_idx:end_idx].index.tolist()
            portfolios[f'OP_{i+1}'] = stocks
        
        return portfolios
    
    @staticmethod
    def sort_by_investment(
        total_assets: pd.DataFrame,
        n_portfolios: int = 3
    ) -> Dict[str, List[str]]:
        """
        Sort stocks by investment (change in assets).
        
        Args:
            total_assets: Total assets over time
            n_portfolios: Number of portfolios
            
        Returns:
            Dictionary mapping portfolio names to stock lists
        """
        # Calculate investment (change in assets / lagged assets)
        asset_growth = total_assets.pct_change(periods=12, axis=0).iloc[-1]
        
        # Sort by investment
        sorted_stocks = asset_growth.sort_values()
        
        # Divide into portfolios
        portfolio_size = len(sorted_stocks) // n_portfolios
        portfolios = {}
        
        for i in range(n_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < n_portfolios - 1 else len(sorted_stocks)
            stocks = sorted_stocks.iloc[start_idx:end_idx].index.tolist()
            portfolios[f'Inv_{i+1}'] = stocks
        
        return portfolios
    
    @staticmethod
    def calculate_portfolio_returns(
        prices: pd.DataFrame,
        portfolio_stocks: Dict[str, List[str]]
    ) -> Dict[str, pd.Series]:
        """
        Calculate equal-weighted portfolio returns.
        
        Args:
            prices: Price DataFrame
            portfolio_stocks: Dictionary of portfolio stock lists
            
        Returns:
            Dictionary of portfolio return series
        """
        portfolio_returns = {}
        
        for portfolio_name, stocks in portfolio_stocks.items():
            # Filter available stocks
            available_stocks = [s for s in stocks if s in prices.columns]
            
            if len(available_stocks) == 0:
                continue
            
            # Calculate returns
            portfolio_prices = prices[available_stocks]
            portfolio_returns_series = portfolio_prices.pct_change(axis=0).mean(axis=1)
            
            portfolio_returns[portfolio_name] = portfolio_returns_series
        
        return portfolio_returns


if __name__ == "__main__":
    # Test Fama-French India
    print("Testing Fama-French 5-Factor Model for India...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 252  # 1 year of daily data
    
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate factor returns
    market_returns = pd.Series(np.random.normal(0.0005, 0.015, n_samples), index=dates)
    small_cap_returns = pd.Series(np.random.normal(0.0006, 0.02, n_samples), index=dates)
    large_cap_returns = pd.Series(np.random.normal(0.0004, 0.012, n_samples), index=dates)
    value_returns = pd.Series(np.random.normal(0.0005, 0.014, n_samples), index=dates)
    growth_returns = pd.Series(np.random.normal(0.0003, 0.016, n_samples), index=dates)
    robust_returns = pd.Series(np.random.normal(0.0006, 0.013, n_samples), index=dates)
    weak_returns = pd.Series(np.random.normal(0.0002, 0.017, n_samples), index=dates)
    conservative_returns = pd.Series(np.random.normal(0.0004, 0.014, n_samples), index=dates)
    aggressive_returns = pd.Series(np.random.normal(0.0003, 0.018, n_samples), index=dates)
    risk_free_returns = pd.Series(np.full(n_samples, 0.06/252), index=dates)
    
    # Create model
    ff_india = FamaFrenchIndia(risk_free_rate=0.06)
    
    # Construct factors
    factors = ff_india.construct_factors(
        market_returns, small_cap_returns, large_cap_returns,
        value_returns, growth_returns, robust_returns, weak_returns,
        conservative_returns, aggressive_returns, risk_free_returns
    )
    
    print(f"\nFactor Statistics:")
    print(factors.describe())
    
    # Test regression on a sample asset
    asset_returns = pd.Series(np.random.normal(0.0005, 0.02, n_samples), index=dates)
    results = ff_india.run_regression(asset_returns, factors)
    
    print(f"\nRegression Results:")
    for factor, result in results.items():
        if isinstance(result, dict):
            print(f"{factor}: {result['coefficient']:.4f} (t={result['t_stat']:.2f}, p={result['p_value']:.4f})")
        else:
            print(f"{factor}: {result}")
    
    # Test expected return calculation
    factor_betas = {k: v for k, v in results.items() if isinstance(v, dict)}
    factor_premiums = {
        'MKT': 0.08,
        'SMB': 0.03,
        'HML': 0.04,
        'RMW': 0.03,
        'CMA': 0.02
    }
    
    expected_return = ff_india.calculate_expected_return(
        factor_betas, factor_premiums, 0.06
    )
    print(f"\nExpected Annual Return: {expected_return:.2%}")
    
    # Test factor construction
    print("\nTesting India Factor Construction...")
    
    n_stocks = 50
    stock_names = [f'STOCK_{i}' for i in range(n_stocks)]
    
    prices = pd.DataFrame(
        np.random.uniform(100, 1000, (n_samples, n_stocks)),
        index=dates,
        columns=stock_names
    )
    
    market_caps = pd.Series(np.random.uniform(1000, 100000, n_stocks), index=stock_names)
    
    constructor = IndiaFactorConstruction()
    size_portfolios = constructor.sort_by_market_cap(prices, market_caps, n_portfolios=6)
    
    print(f"Size Portfolios:")
    for name, stocks in size_portfolios.items():
        print(f"{name}: {len(stocks)} stocks")
    
    portfolio_returns = constructor.calculate_portfolio_returns(prices, size_portfolios)
    print(f"\nPortfolio Returns Calculated: {len(portfolio_returns)} portfolios")
    
    print("\nFama-French 5-Factor Model for India test completed.")
