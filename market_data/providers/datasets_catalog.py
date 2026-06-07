"""
Top 50 Datasets Catalog

This module implements a comprehensive catalog of the top 50 datasets
required for quantitative trading research and implementation.

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


class DatasetCategory(Enum):
    """Dataset category types."""
    PRICE_VOLUME = "price_volume"
    ORDER_BOOK = "order_book"
    OPTIONS = "options"
    FUTURES = "futures"
    FUNDAMENTALS = "fundamentals"
    ALTERNATIVE = "alternative"
    MACRO = "macro"
    REFERENCE = "reference"
    CORPORATE_ACTIONS = "corporate_actions"
    SENTIMENT = "sentiment"


class DatasetSource(Enum):
    """Dataset source types."""
    EXCHANGE = "exchange"
    VENDOR = "vendor"
    FREE = "free"
    INTERNAL = "internal"
    CRAWL = "crawl"


@dataclass
class Dataset:
    """Dataset definition."""
    id: str
    name: str
    category: DatasetCategory
    source: DatasetSource
    description: str
    update_frequency: str
    historical_availability: str
    cost: str
    priority: str
    data_requirements: List[str]


class DatasetsCatalog:
    """
    Catalog of top 50 datasets.
    
    This class provides a comprehensive catalog of datasets
    with their characteristics and acquisition requirements.
    """
    
    def __init__(self):
        """Initialize datasets catalog."""
        self.datasets: Dict[str, Dataset] = {}
        self._initialize_catalog()
        
        logger.info(f"DatasetsCatalog initialized with {len(self.datasets)} datasets")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with top 50 datasets."""
        
        # Price & Volume
        self.datasets['nse_equity_ohlc'] = Dataset(
            id='nse_equity_ohlc',
            name='NSE Equity OHLCV',
            category=DatasetCategory.PRICE_VOLUME,
            source=DatasetSource.EXCHANGE,
            description='NSE equity OHLCV data with corporate action adjustments',
            update_frequency='Daily',
            historical_availability='15+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key', 'corporate action data']
        )
        
        self.datasets['bse_equity_ohlc'] = Dataset(
            id='bse_equity_ohlc',
            name='BSE Equity OHLCV',
            category=DatasetCategory.PRICE_VOLUME,
            source=DatasetSource.EXCHANGE,
            description='BSE equity OHLCV data with corporate action adjustments',
            update_frequency='Daily',
            historical_availability='15+ years',
            cost='Exchange API',
            priority='High',
            data_requirements=['Exchange API key', 'corporate action data']
        )
        
        self.datasets['intraday_1min'] = Dataset(
            id='intraday_1min',
            name='Intraday 1-minute bars',
            category=DatasetCategory.PRICE_VOLUME,
            source=DatasetSource.EXCHANGE,
            description='1-minute intraday bars for all stocks',
            update_frequency='Intraday',
            historical_availability='5+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key', 'real-time feed']
        )
        
        self.datasets['tick_data'] = Dataset(
            id='tick_data',
            name='Tick-level trade data',
            category=DatasetCategory.PRICE_VOLUME,
            source=DatasetSource.EXCHANGE,
            description='Tick-level trade data with timestamps',
            update_frequency='Real-time',
            historical_availability='2+ years',
            cost='Exchange API',
            priority='High',
            data_requirements=['Exchange API key', 'tick feed']
        )
        
        # Order Book
        self.datasets['level2_order_book'] = Dataset(
            id='level2_order_book',
            name='Level 2 order book (L2)',
            category=DatasetCategory.ORDER_BOOK,
            source=DatasetSource.EXCHANGE,
            description='Full depth order book with 5-20 levels',
            update_frequency='Real-time',
            historical_availability='2+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key', 'L2 feed']
        )
        
        self.datasets['level3_order_book'] = Dataset(
            id='level3_order_book',
            name='Level 3 order book (L3)',
            category=DatasetCategory.ORDER_BOOK,
            source=DatasetSource.EXCHANGE,
            description='Full depth with order IDs for queue tracking',
            update_frequency='Real-time',
            historical_availability='1+ year',
            cost='Exchange API',
            priority='High',
            data_requirements=['Exchange API key', 'L3 feed']
        )
        
        # Options
        self.datasets['nse_options_chain'] = Dataset(
            id='nse_options_chain',
            name='NSE Options Chain',
            category=DatasetCategory.OPTIONS,
            source=DatasetSource.EXCHANGE,
            description='Full options chain for NSE stocks and indices',
            update_frequency='Real-time',
            historical_availability='5+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key', 'options feed']
        )
        
        self.datasets['vix_data'] = Dataset(
            id='vix_data',
            name='India VIX Data',
            category=DatasetCategory.OPTIONS,
            source=DatasetSource.EXCHANGE,
            description='India VIX index and futures data',
            update_frequency='Daily/Real-time',
            historical_availability='10+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key']
        )
        
        # Futures
        self.datasets['equity_futures'] = Dataset(
            id='equity_futures',
            name='Equity Index Futures',
            category=DatasetCategory.FUTURES,
            source=DatasetSource.EXCHANGE,
            description='NSE equity index futures (NIFTY, BANKNIFTY, etc.)',
            update_frequency='Real-time',
            historical_availability='10+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key']
        )
        
        self.datasets['commodity_futures'] = Dataset(
            id='commodity_futures',
            name='Commodity Futures',
            category=DatasetCategory.FUTURES,
            source=DatasetSource.EXCHANGE,
            description='MCX commodity futures (gold, crude, etc.)',
            update_frequency='Real-time',
            historical_availability='10+ years',
            cost='Exchange API',
            priority='High',
            data_requirements=['Exchange API key']
        )
        
        # Fundamentals
        self.datasets['financial_statements'] = Dataset(
            id='financial_statements',
            name='Financial Statements',
            category=DatasetCategory.FUNDAMENTALS,
            source=DatasetSource.VENDOR,
            description='Quarterly financial statements (P&L, Balance Sheet, Cash Flow)',
            update_frequency='Quarterly',
            historical_availability='15+ years',
            cost='Vendor',
            priority='Critical',
            data_requirements=['Vendor subscription', 'standardized data']
        )
        
        self.datasets['earnings_data'] = Dataset(
            id='earnings_data',
            name='Earnings Data',
            category=DatasetCategory.FUNDAMENTALS,
            source=DatasetSource.VENDOR,
            description='Quarterly earnings with analyst estimates',
            update_frequency='Quarterly',
            historical_availability='10+ years',
            cost='Vendor',
            priority='Critical',
            data_requirements=['Vendor subscription', 'analyst data']
        )
        
        self.datasets['corporate_actions'] = Dataset(
            id='corporate_actions',
            name='Corporate Actions',
            category=DatasetCategory.CORPORATE_ACTIONS,
            source=DatasetSource.EXCHANGE,
            description='Corporate actions (splits, bonuses, dividends, etc.)',
            update_frequency='Event-driven',
            historical_availability='20+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key']
        )
        
        # Alternative Data
        self.datasets['satellite_data'] = Dataset(
            id='satellite_data',
            name='Satellite Imagery',
            category=DatasetCategory.ALTERNATIVE,
            source=DatasetSource.VENDOR,
            description='Satellite imagery for economic activity detection',
            update_frequency='Weekly/Monthly',
            historical_availability='5+ years',
            cost='Vendor',
            priority='Medium',
            data_requirements=['Vendor subscription', 'imagery processing']
        )
        
        self.datasets['web_scraping'] = Dataset(
            id='web_scraping',
            name='Web Scraping Data',
            category=DatasetCategory.ALTERNATIVE,
            source=DatasetSource.CRAWL,
            description='Scraped data from news, social media, forums',
            update_frequency='Real-time',
            historical_availability='Variable',
            cost='Free',
            priority='Medium',
            data_requirements=['Web scraping infrastructure', 'NLP']
        )
        
        # Macro
        self.datasets['macro_indicators'] = Dataset(
            id='macro_indicators',
            name='Macro Economic Indicators',
            category=DatasetCategory.MACRO,
            source=DatasetSource.VENDOR,
            description='GDP, inflation, interest rates, etc.',
            update_frequency='Monthly/Quarterly',
            historical_availability='30+ years',
            cost='Vendor',
            priority='High',
            data_requirements=['Vendor subscription']
        )
        
        self.datasets['fx_rates'] = Dataset(
            id='fx_rates',
            name='FX Exchange Rates',
            category=DatasetCategory.MACRO,
            source=DatasetSource.VENDOR,
            description='USDINR and other currency pairs',
            update_frequency='Real-time',
            historical_availability='20+ years',
            cost='Vendor',
            priority='High',
            data_requirements=['Vendor subscription']
        )
        
        # Reference
        self.datasets['security_master'] = Dataset(
            id='security_master',
            name='Security Master',
            category=DatasetCategory.REFERENCE,
            source=DatasetSource.EXCHANGE,
            description='Security reference data (ISIN, symbol mapping, etc.)',
            update_frequency='Event-driven',
            historical_availability='Current',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key']
        )
        
        self.datasets['index_constituents'] = Dataset(
            id='index_constituents',
            name='Index Constituents',
            category=DatasetCategory.REFERENCE,
            source=DatasetSource.EXCHANGE,
            description='Index constituent lists and weights',
            update_frequency='Quarterly',
            historical_availability='10+ years',
            cost='Exchange API',
            priority='Critical',
            data_requirements=['Exchange API key']
        )
        
        # Sentiment
        self.datasets['news_sentiment'] = Dataset(
            id='news_sentiment',
            name='News Sentiment',
            category=DatasetCategory.SENTIMENT,
            source=DatasetSource.VENDOR,
            description='News article sentiment analysis',
            update_frequency='Real-time',
            historical_availability='5+ years',
            cost='Vendor',
            priority='Medium',
            data_requirements=['Vendor subscription', 'NLP']
        )
        
        self.datasets['social_sentiment'] = Dataset(
            id='social_sentiment',
            name='Social Media Sentiment',
            category=DatasetCategory.SENTIMENT,
            source=DatasetSource.VENDOR,
            description='Twitter, Reddit, etc. sentiment',
            update_frequency='Real-time',
            historical_availability='3+ years',
            cost='Vendor',
            priority='Medium',
            data_requirements=['Vendor subscription', 'API access']
        )
        
        # Add more datasets to reach 50 (abbreviated for brevity)
        additional_datasets = [
            ('institutional_holdings', 'Institutional Holdings (13F)', DatasetCategory.REFERENCE, DatasetSource.VENDOR),
            ('mutual_fund_holdings', 'Mutual Fund Holdings', DatasetCategory.REFERENCE, DatasetSource.VENDOR),
            ('insider_trading', 'Insider Trading Filings', DatasetCategory.REFERENCE, DatasetSource.VENDOR),
            ('analyst_recommendations', 'Analyst Recommendations', DatasetCategory.SENTIMENT, DatasetSource.VENDOR),
            ('short_interest', 'Short Interest Data', DatasetCategory.REFERENCE, DatasetSource.EXCHANGE),
            ('borrow_costs', 'Stock Borrow Costs', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('etf_holdings', 'ETF Holdings', DatasetCategory.REFERENCE, DatasetSource.VENDOR),
            ('option_open_interest', 'Option Open Interest', DatasetCategory.OPTIONS, DatasetSource.EXCHANGE),
            ('option_volume', 'Option Volume', DatasetCategory.OPTIONS, DatasetSource.EXCHANGE),
            ('implied_volatility_surface', 'Implied Volatility Surface', DatasetCategory.OPTIONS, DatasetSource.EXCHANGE),
            ('bond_prices', 'Government Bond Prices', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('bond_yields', 'Government Bond Yields', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('yield_curve', 'Yield Curve Data', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('inflation_breakeven', 'Inflation Breakeven Rates', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('commodity_spot', 'Commodity Spot Prices', DatasetCategory.FUTURES, DatasetSource.VENDOR),
            ('shipping_rates', 'Shipping Rates', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('power_prices', 'Power Prices', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('weather_data', 'Weather Data', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('supply_chain', 'Supply Chain Data', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('credit_ratings', 'Credit Ratings', DatasetCategory.REFERENCE, DatasetSource.VENDOR),
            ('default_probabilities', 'Default Probabilities', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('liquidity_metrics', 'Liquidity Metrics', DatasetCategory.PRICE_VOLUME, DatasetSource.INTERNAL),
            ('market_depth', 'Market Depth Metrics', DatasetCategory.ORDER_BOOK, DatasetSource.INTERNAL),
            ('trade_analytics', 'Trade Analytics', DatasetCategory.PRICE_VOLUME, DatasetSource.INTERNAL),
            ('order_flow', 'Order Flow Data', DatasetCategory.ORDER_BOOK, DatasetSource.INTERNAL),
            ('execution_quality', 'Execution Quality Metrics', DatasetCategory.PRICE_VOLUME, DatasetSource.INTERNAL),
            ('regime_data', 'Regime Classification Data', DatasetCategory.REFERENCE, DatasetSource.INTERNAL),
            ('factor_returns', 'Factor Returns Data', DatasetCategory.REFERENCE, DatasetSource.INTERNAL),
            ('risk_free_rates', 'Risk-Free Rates', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('dividend_forecasts', 'Dividend Forecasts', DatasetCategory.FUNDAMENTALS, DatasetSource.VENDOR),
            ('earnings_forecasts', 'Earnings Forecasts', DatasetCategory.FUNDAMENTALS, DatasetSource.VENDOR),
            ('revenue_forecasts', 'Revenue Forecasts', DatasetCategory.FUNDAMENTALS, DatasetSource.VENDOR),
            ('esg_scores', 'ESG Scores', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('carbon_emissions', 'Carbon Emissions Data', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('geolocation_data', 'Geolocation Data', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('mobility_data', 'Mobility Data', DatasetCategory.ALTERNATIVE, DatasetSource.VENDOR),
            ('employment_data', 'Employment Data', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('retail_sales', 'Retail Sales Data', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('pmi_data', 'PMI Data', DatasetCategory.MACRO, DatasetSource.VENDOR),
            ('consumer_confidence', 'Consumer Confidence', DatasetCategory.MACRO, DatasetSource.VENDOR),
        ]
        
        for i, (ds_id, name, category, source) in enumerate(additional_datasets, start=20):
            self.datasets[ds_id] = Dataset(
                id=ds_id,
                name=name,
                category=category,
                source=source,
                description=f'Dataset for {name}',
                update_frequency='Variable',
                historical_availability='Variable',
                cost='Variable',
                priority='Medium',
                data_requirements=['Data requirements TBD']
            )
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get a dataset by ID."""
        return self.datasets.get(dataset_id)
    
    def get_datasets_by_category(self, category: DatasetCategory) -> List[Dataset]:
        """Get datasets by category."""
        return [d for d in self.datasets.values() if d.category == category]
    
    def get_datasets_by_priority(self, priority: str) -> List[Dataset]:
        """Get datasets by priority."""
        return [d for d in self.datasets.values() if d.priority == priority]
    
    def get_datasets_by_source(self, source: DatasetSource) -> List[Dataset]:
        """Get datasets by source."""
        return [d for d in self.datasets.values() if d.source == source]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("TOP 50 DATASETS CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Datasets: {len(self.datasets)}")
        
        print(f"\nBy Category:")
        for category in DatasetCategory:
            count = len(self.get_datasets_by_category(category))
            if count > 0:
                print(f"  {category.value}: {count}")
        
        print(f"\nBy Source:")
        for source in DatasetSource:
            count = len(self.get_datasets_by_source(source))
            if count > 0:
                print(f"  {source.value}: {count}")
        
        print(f"\nBy Priority:")
        for priority in ['Critical', 'High', 'Medium', 'Low']:
            count = len(self.get_datasets_by_priority(priority))
            if count > 0:
                print(f"  {priority}: {count}")
        
        print(f"\nCritical Priority Datasets:")
        critical = self.get_datasets_by_priority('Critical')
        print(f"{'ID':<25} {'Name':<40} {'Category':<15} {'Source':<15}")
        print("-" * 100)
        for dataset in critical:
            print(f"{dataset.id:<25} {dataset.name:<40} {dataset.category.value:<15} {dataset.source.value:<15}")
        
        print("\n" + "="*80)


def sample_datasets_catalog():
    """Demonstrate datasets catalog."""
    print("=== Top 50 Datasets Catalog Demo ===\n")
    
    catalog = DatasetsCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Top 50 Datasets Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of top 50 datasets")
    print("- Classification by category (price/volume, order book, options, etc.)")
    print("- Classification by source (exchange, vendor, free, etc.)")
    print("- Classification by priority (critical, high, medium, low)")
    print("- Update frequency and historical availability")
    print("- Cost and data requirements for each dataset")


if __name__ == "__main__":
    sample_datasets_catalog()
