"""
Top 50 Features Catalog

This module implements a comprehensive catalog of the top 50 features
required for quantitative trading models and alpha generation.

Based on the Quant Research Intelligence System document.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Feature category types."""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VALUE = "value"
    QUALITY = "quality"
    MICROSTRUCTURE = "microstructure"
    OPTIONS = "options"
    ALTERNATIVE = "alternative"


class FeatureComplexity(Enum):
    """Feature complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class Feature:
    """Feature definition."""
    id: str
    name: str
    category: FeatureCategory
    complexity: FeatureComplexity
    description: str
    calculation_method: str
    data_requirements: List[str]
    update_frequency: str


class FeaturesCatalog:
    """
    Catalog of top 50 features.
    
    This class provides a comprehensive catalog of features
    with their characteristics and calculation requirements.
    """
    
    def __init__(self):
        """Initialize features catalog."""
        self.features: Dict[str, Feature] = {}
        self._initialize_catalog()
        
        logger.info(f"FeaturesCatalog initialized with {len(self.features)} features")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with top 50 features."""
        
        # Price Features
        self.features['returns_1d'] = Feature(
            id='returns_1d',
            name='1-Day Returns',
            category=FeatureCategory.PRICE,
            complexity=FeatureComplexity.SIMPLE,
            description='Daily percentage returns',
            calculation_method='(P_t - P_{t-1}) / P_{t-1}',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['returns_5d'] = Feature(
            id='returns_5d',
            name='5-Day Returns',
            category=FeatureCategory.PRICE,
            complexity=FeatureComplexity.SIMPLE,
            description='5-day percentage returns',
            calculation_method='(P_t - P_{t-5}) / P_{t-5}',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['log_returns'] = Feature(
            id='log_returns',
            name='Log Returns',
            category=FeatureCategory.PRICE,
            complexity=FeatureComplexity.SIMPLE,
            description='Logarithmic returns',
            calculation_method='ln(P_t / P_{t-1})',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['price_momentum_12m'] = Feature(
            id='price_momentum_12m',
            name='12-Month Price Momentum',
            category=FeatureCategory.MOMENTUM,
            complexity=FeatureComplexity.SIMPLE,
            description='12-month price momentum (excluding last month)',
            calculation_method='(P_t - P_{t-12}) / P_{t-12}',
            data_requirements=['Monthly prices'],
            update_frequency='Monthly'
        )
        
        self.features['price_momentum_6m'] = Feature(
            id='price_momentum_6m',
            name='6-Month Price Momentum',
            category=FeatureCategory.MOMENTUM,
            complexity=FeatureComplexity.SIMPLE,
            description='6-month price momentum',
            calculation_method='(P_t - P_{t-6}) / P_{t-6}',
            data_requirements=['Monthly prices'],
            update_frequency='Monthly'
        )
        
        # Volume Features
        self.features['volume_20d_avg'] = Feature(
            id='volume_20d_avg',
            name='20-Day Average Volume',
            category=FeatureCategory.VOLUME,
            complexity=FeatureComplexity.SIMPLE,
            description='20-day moving average of volume',
            calculation_method='MA(Volume, 20)',
            data_requirements=['Daily volume'],
            update_frequency='Daily'
        )
        
        self.features['volume_ratio'] = Feature(
            id='volume_ratio',
            name='Volume Ratio',
            category=FeatureCategory.VOLUME,
            complexity=FeatureComplexity.SIMPLE,
            description='Current volume / average volume',
            calculation_method='Volume_t / MA(Volume, 20)',
            data_requirements=['Daily volume'],
            update_frequency='Daily'
        )
        
        self.features['relative_volume'] = Feature(
            id='relative_volume',
            name='Relative Volume',
            category=FeatureCategory.VOLUME,
            complexity=FeatureComplexity.SIMPLE,
            description='Current volume / 60-day average volume',
            calculation_method='Volume_t / MA(Volume, 60)',
            data_requirements=['Daily volume'],
            update_frequency='Daily'
        )
        
        self.features['volume_trend'] = Feature(
            id='volume_trend',
            name='Volume Trend',
            category=FeatureCategory.VOLUME,
            complexity=FeatureComplexity.MEDIUM,
            description='Linear regression slope of volume over 20 days',
            calculation_method='Slope(Volume, 20)',
            data_requirements=['Daily volume'],
            update_frequency='Daily'
        )
        
        # Volatility Features
        self.features['volatility_20d'] = Feature(
            id='volatility_20d',
            name='20-Day Volatility',
            category=FeatureCategory.VOLATILITY,
            complexity=FeatureComplexity.SIMPLE,
            description='20-day realized volatility (annualized)',
            calculation_method='Std(Returns, 20) * sqrt(252)',
            data_requirements=['Daily returns'],
            update_frequency='Daily'
        )
        
        self.features['volatility_60d'] = Feature(
            id='volatility_60d',
            name='60-Day Volatility',
            category=FeatureCategory.VOLATILITY,
            complexity=FeatureComplexity.SIMPLE,
            description='60-day realized volatility (annualized)',
            calculation_method='Std(Returns, 60) * sqrt(252)',
            data_requirements=['Daily returns'],
            update_frequency='Daily'
        )
        
        self.features['volatility_ratio'] = Feature(
            id='volatility_ratio',
            name='Volatility Ratio',
            category=FeatureCategory.VOLATILITY,
            complexity=FeatureComplexity.SIMPLE,
            description='Short-term / long-term volatility',
            calculation_method='Vol_20d / Vol_60d',
            data_requirements=['Daily returns'],
            update_frequency='Daily'
        )
        
        self.features['garch_volatility'] = Feature(
            id='garch_volatility',
            name='GARCH Volatility',
            category=FeatureCategory.VOLATILITY,
            complexity=FeatureComplexity.COMPLEX,
            description='GARCH(1,1) conditional volatility',
            calculation_method='GARCH(1,1) model',
            data_requirements=['Daily returns'],
            update_frequency='Daily'
        )
        
        # Momentum Features
        self.features['rsi_14'] = Feature(
            id='rsi_14',
            name='RSI (14)',
            category=FeatureCategory.MOMENTUM,
            complexity=FeatureComplexity.MEDIUM,
            description='14-period Relative Strength Index',
            calculation_method='RSI formula',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['macd'] = Feature(
            id='macd',
            name='MACD',
            category=FeatureCategory.MOMENTUM,
            complexity=FeatureComplexity.MEDIUM,
            description='Moving Average Convergence Divergence',
            calculation_method='EMA(12) - EMA(26)',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['adx'] = Feature(
            id='adx',
            name='ADX',
            category=FeatureCategory.MOMENTUM,
            complexity=FeatureComplexity.MEDIUM,
            description='Average Directional Index',
            calculation_method='ADX formula',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        # Reversal Features
        self.features['bollinger_band_position'] = Feature(
            id='bollinger_band_position',
            name='Bollinger Band Position',
            category=FeatureCategory.REVERSAL,
            complexity=FeatureComplexity.MEDIUM,
            description='Price position relative to Bollinger Bands',
            calculation_method='(P - BB_lower) / (BB_upper - BB_lower)',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['stochastic'] = Feature(
            id='stochastic',
            name='Stochastic Oscillator',
            category=FeatureCategory.REVERSAL,
            complexity=FeatureComplexity.MEDIUM,
            description='Stochastic oscillator %K and %D',
            calculation_method='Stochastic formula',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        self.features['williams_r'] = Feature(
            id='williams_r',
            name="Williams %R",
            category=FeatureCategory.REVERSAL,
            complexity=FeatureComplexity.MEDIUM,
            description="Williams %R indicator",
            calculation_method='Williams %R formula',
            data_requirements=['Daily prices'],
            update_frequency='Daily'
        )
        
        # Value Features
        self.features['pe_ratio'] = Feature(
            id='pe_ratio',
            name='P/E Ratio',
            category=FeatureCategory.VALUE,
            complexity=FeatureComplexity.SIMPLE,
            description='Price-to-earnings ratio',
            calculation_method='Price / EPS',
            data_requirements=['Price', 'EPS'],
            update_frequency='Quarterly'
        )
        
        self.features['pb_ratio'] = Feature(
            id='pb_ratio',
            name='P/B Ratio',
            category=FeatureCategory.VALUE,
            complexity=FeatureComplexity.SIMPLE,
            description='Price-to-book ratio',
            calculation_method='Price / Book Value per Share',
            data_requirements=['Price', 'Book Value'],
            update_frequency='Quarterly'
        )
        
        self.features['pcf_ratio'] = Feature(
            id='pcf_ratio',
            name='P/CF Ratio',
            category=FeatureCategory.VALUE,
            complexity=FeatureComplexity.SIMPLE,
            description='Price-to-cash flow ratio',
            calculation_method='Price / Cash Flow per Share',
            data_requirements=['Price', 'Cash Flow'],
            update_frequency='Quarterly'
        )
        
        self.features['ev_ebitda'] = Feature(
            id='ev_ebitda',
            name='EV/EBITDA',
            category=FeatureCategory.VALUE,
            complexity=FeatureComplexity.MEDIUM,
            description='Enterprise Value to EBITDA',
            calculation_method='EV / EBITDA',
            data_requirements=['Market Cap', 'Debt', 'Cash', 'EBITDA'],
            update_frequency='Quarterly'
        )
        
        # Quality Features
        self.features['roe'] = Feature(
            id='roe',
            name='Return on Equity (ROE)',
            category=FeatureCategory.QUALITY,
            complexity=FeatureComplexity.SIMPLE,
            description='Return on equity',
            calculation_method='Net Income / Shareholder Equity',
            data_requirements=['Financial Statements'],
            update_frequency='Quarterly'
        )
        
        self.features['roa'] = Feature(
            id='roa',
            name='Return on Assets (ROA)',
            category=FeatureCategory.QUALITY,
            complexity=FeatureComplexity.SIMPLE,
            description='Return on assets',
            calculation_method='Net Income / Total Assets',
            data_requirements=['Financial Statements'],
            update_frequency='Quarterly'
        )
        
        self.features['profit_margin'] = Feature(
            id='profit_margin',
            name='Profit Margin',
            category=FeatureCategory.QUALITY,
            complexity=FeatureComplexity.SIMPLE,
            description='Net profit margin',
            calculation_method='Net Income / Revenue',
            data_requirements=['Financial Statements'],
            update_frequency='Quarterly'
        )
        
        self.features['debt_equity'] = Feature(
            id='debt_equity',
            name='Debt-to-Equity Ratio',
            category=FeatureCategory.QUALITY,
            complexity=FeatureComplexity.SIMPLE,
            description='Debt-to-equity ratio',
            calculation_method='Total Debt / Shareholder Equity',
            data_requirements=['Financial Statements'],
            update_frequency='Quarterly'
        )
        
        self.features['accruals'] = Feature(
            id='accruals',
            name='Accruals',
            category=FeatureCategory.QUALITY,
            complexity=FeatureComplexity.MEDIUM,
            description='Accruals component of earnings',
            calculation_method='Net Income - Cash Flow from Operations',
            data_requirements=['Financial Statements'],
            update_frequency='Quarterly'
        )
        
        # Microstructure Features
        self.features['order_flow_imbalance'] = Feature(
            id='order_flow_imbalance',
            name='Order Flow Imbalance',
            category=FeatureCategory.MICROSTRUCTURE,
            complexity=FeatureComplexity.MEDIUM,
            description='Order book imbalance',
            calculation_method='(Bid Volume - Ask Volume) / Total Volume',
            data_requirements=['Order book data'],
            update_frequency='Real-time'
        )
        
        self.features['bid_ask_spread'] = Feature(
            id='bid_ask_spread',
            name='Bid-Ask Spread',
            category=FeatureCategory.MICROSTRUCTURE,
            complexity=FeatureComplexity.SIMPLE,
            description='Bid-ask spread',
            calculation_method='(Ask - Bid) / Mid',
            data_requirements=['Quote data'],
            update_frequency='Real-time'
        )
        
        self.features['depth_imbalance'] = Feature(
            id='depth_imbalance',
            name='Depth Imbalance',
            category=FeatureCategory.MICROSTRUCTURE,
            complexity=FeatureComplexity.MEDIUM,
            description='Order book depth imbalance',
            calculation_method='Depth imbalance calculation',
            data_requirements=['Order book data'],
            update_frequency='Real-time'
        )
        
        self.features['vpin'] = Feature(
            id='vpin',
            name='VPIN (Volume-Synchronized PIN)',
            category=FeatureCategory.MICROSTRUCTURE,
            complexity=FeatureComplexity.COMPLEX,
            description='Volume-synchronized probability of informed trading',
            calculation_method='VPIN algorithm',
            data_requirements=['Trade data', 'volume buckets'],
            update_frequency='Intraday'
        )
        
        # Options Features
        self.features['implied_volatility'] = Feature(
            id='implied_volatility',
            name='Implied Volatility',
            category=FeatureCategory.OPTIONS,
            complexity=FeatureComplexity.MEDIUM,
            description='Option implied volatility',
            calculation_method='Black-Scholes inversion',
            data_requirements=['Option prices', 'underlying price'],
            update_frequency='Real-time'
        )
        
        self.features['skew'] = Feature(
            id='skew',
            name='Volatility Skew',
            category=FeatureCategory.OPTIONS,
            complexity=FeatureComplexity.MEDIUM,
            description='Volatility skew (25-delta)',
            calculation_method='IV(25d Put) - IV(25d Call)',
            data_requirements=['Option chain'],
            update_frequency='Real-time'
        )
        
        self.features['term_structure'] = Feature(
            id='term_structure',
            name='Volatility Term Structure',
            category=FeatureCategory.OPTIONS,
            complexity=FeatureComplexity.MEDIUM,
            description='Volatility term structure slope',
            calculation_method='IV(3m) - IV(1m)',
            data_requirements=['Option chain'],
            update_frequency='Daily'
        )
        
        # Alternative Features
        self.features['news_sentiment'] = Feature(
            id='news_sentiment',
            name='News Sentiment',
            category=FeatureCategory.ALTERNATIVE,
            complexity=FeatureComplexity.VERY_COMPLEX,
            description='News article sentiment score',
            calculation_method='NLP sentiment analysis',
            data_requirements=['News data', 'NLP model'],
            update_frequency='Real-time'
        )
        
        self.features['social_sentiment'] = Feature(
            id='social_sentiment',
            name='Social Media Sentiment',
            category=FeatureCategory.ALTERNATIVE,
            complexity=FeatureComplexity.VERY_COMPLEX,
            description='Social media sentiment score',
            calculation_method='NLP sentiment analysis',
            data_requirements=['Social media data', 'NLP model'],
            update_frequency='Real-time'
        )
        
        # Add more features to reach 50 (abbreviated for brevity)
        additional_features = [
            ('earnings_surprise', 'Earnings Surprise', FeatureCategory.ALTERNATIVE, FeatureComplexity.SIMPLE),
            ('analyst_revision', 'Analyst Revision', FeatureCategory.ALTERNATIVE, FeatureComplexity.SIMPLE),
            ('insider_trading', 'Insider Trading', FeatureCategory.ALTERNATIVE, FeatureComplexity.SIMPLE),
            ('short_interest', 'Short Interest', FeatureCategory.ALTERNATIVE, FeatureComplexity.SIMPLE),
            ('institutional_ownership', 'Institutional Ownership', FeatureCategory.ALTERNATIVE, FeatureComplexity.SIMPLE),
            ('beta', 'Beta', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('alpha', 'Alpha', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('tracking_error', 'Tracking Error', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('information_ratio', 'Information Ratio', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('sharpe_ratio', 'Sharpe Ratio', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('sortino_ratio', 'Sortino Ratio', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('max_drawdown', 'Maximum Drawdown', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('calmar_ratio', 'Calmar Ratio', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('up_capture', 'Up Capture Ratio', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('down_capture', 'Down Capture Ratio', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('var_95', 'VaR 95%', FeatureCategory.VOLATILITY, FeatureComplexity.MEDIUM),
            ('cvar_95', 'CVaR 95%', FeatureCategory.VOLATILITY, FeatureComplexity.MEDIUM),
            ('expected_shortfall', 'Expected Shortfall', FeatureCategory.VOLATILITY, FeatureComplexity.MEDIUM),
            ('skewness', 'Return Skewness', FeatureCategory.VOLATILITY, FeatureComplexity.SIMPLE),
            ('kurtosis', 'Return Kurtosis', FeatureCategory.VOLATILITY, FeatureComplexity.SIMPLE),
            ('autocorrelation', 'Return Autocorrelation', FeatureCategory.VOLATILITY, FeatureComplexity.SIMPLE),
            ('hurst_exponent', 'Hurst Exponent', FeatureCategory.VOLATILITY, FeatureComplexity.COMPLEX),
            ('fractal_dimension', 'Fractal Dimension', FeatureCategory.VOLATILITY, FeatureComplexity.COMPLEX),
            ('entropy', 'Entropy', FeatureCategory.VOLATILITY, FeatureComplexity.COMPLEX),
            ('liquidity_ratio', 'Liquidity Ratio', FeatureCategory.MICROSTRUCTURE, FeatureComplexity.MEDIUM),
            ('turnover', 'Turnover', FeatureCategory.VOLUME, FeatureComplexity.SIMPLE),
            ('amihud_illiquidity', 'Amihud Illiquidity', FeatureCategory.MICROSTRUCTURE, FeatureComplexity.SIMPLE),
            ('market_impact', 'Market Impact', FeatureCategory.MICROSTRUCTURE, FeatureComplexity.COMPLEX),
            ('price_impact', 'Price Impact', FeatureCategory.MICROSTRUCTURE, FeatureComplexity.COMPLEX),
            ('volume_weighted_price', 'Volume-Weighted Price', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('time_weighted_price', 'Time-Weighted Price', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('vwap', 'VWAP', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('twap', 'TWAP', FeatureCategory.PRICE, FeatureComplexity.MEDIUM),
            ('high_low', 'High-Low Range', FeatureCategory.PRICE, FeatureComplexity.SIMPLE),
            ('close_open', 'Close-Open', FeatureCategory.PRICE, FeatureComplexity.SIMPLE),
            ('gap', 'Gap', FeatureCategory.PRICE, FeatureComplexity.SIMPLE),
        ]
        
        for i, (feat_id, name, category, complexity) in enumerate(additional_features, start=30):
            self.features[feat_id] = Feature(
                id=feat_id,
                name=name,
                category=category,
                complexity=complexity,
                description=f'Feature for {name}',
                calculation_method='Calculation method TBD',
                data_requirements=['Data requirements TBD'],
                update_frequency='Variable'
            )
    
    def get_feature(self, feature_id: str) -> Optional[Feature]:
        """Get a feature by ID."""
        return self.features.get(feature_id)
    
    def get_features_by_category(self, category: FeatureCategory) -> List[Feature]:
        """Get features by category."""
        return [f for f in self.features.values() if f.category == category]
    
    def get_features_by_complexity(self, complexity: FeatureComplexity) -> List[Feature]:
        """Get features by complexity."""
        return [f for f in self.features.values() if f.complexity == complexity]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("TOP 50 FEATURES CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Features: {len(self.features)}")
        
        print(f"\nBy Category:")
        for category in FeatureCategory:
            count = len(self.get_features_by_category(category))
            if count > 0:
                print(f"  {category.value}: {count}")
        
        print(f"\nBy Complexity:")
        for complexity in FeatureComplexity:
            count = len(self.get_features_by_complexity(complexity))
            if count > 0:
                print(f"  {complexity.value}: {count}")
        
        print(f"\nSample Features by Category:")
        print(f"{'ID':<25} {'Name':<40} {'Category':<15} {'Complexity':<15}")
        print("-" * 100)
        for feature in list(self.features.values())[:15]:
            print(f"{feature.id:<25} {feature.name:<40} {feature.category.value:<15} {feature.complexity.value:<15}")
        
        print("\n" + "="*80)


def sample_features_catalog():
    """Demonstrate features catalog."""
    print("=== Top 50 Features Catalog Demo ===\n")
    
    catalog = FeaturesCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Top 50 Features Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of top 50 features")
    print("- Classification by category (price, volume, volatility, etc.)")
    print("- Classification by complexity (simple, medium, complex, very complex)")
    print("- Calculation methods and data requirements")
    print("- Update frequency for each feature")


if __name__ == "__main__":
    sample_features_catalog()
