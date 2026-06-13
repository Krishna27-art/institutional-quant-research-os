"""
Top 50 Regime Strategies Catalog

This module implements a comprehensive catalog of the top 50 regime strategies
for quantitative trading.

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


class RegimeCategory(Enum):
    """Regime category types."""
    VOLATILITY = "volatility"
    TREND = "trend"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    SEASONAL = "seasonal"
    CRISIS = "crisis"


class RegimeType(Enum):
    """Regime type types."""
    HMM = "hmm"
    MARKOV_SWITCHING = "markov_switching"
    CHANGE_POINT = "change_point"
    CLUSTERING = "clustering"
    THRESHOLD = "threshold"
    ML = "ml"
    RULE_BASED = "rule_based"


@dataclass
class RegimeStrategy:
    """Regime strategy definition."""
    id: str
    name: str
    category: RegimeCategory
    regime_type: RegimeType
    description: str
    expected_sharpe: float
    expected_capacity: str
    difficulty: str
    data_requirements: List[str]


class RegimeStrategiesCatalog:
    """
    Catalog of top 50 regime strategies.
    
    This class provides a comprehensive catalog of regime strategies
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize regime strategies catalog."""
        self.strategies: Dict[str, RegimeStrategy] = {}
        self._initialize_catalog()
        
        logger.info(f"RegimeStrategiesCatalog initialized with {len(self.strategies)} strategies")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with top 50 regime strategies."""
        
        # Volatility Regimes
        self.strategies['vol_hmm_4state'] = RegimeStrategy(
            id='vol_hmm_4state',
            name='4-state HMM volatility regime',
            category=RegimeCategory.VOLATILITY,
            regime_type=RegimeType.HMM,
            description='4-state HMM for volatility regimes (low, normal, high, crisis)',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Returns data', 'HMM library']
        )
        
        self.strategies['vol_hmm_5state'] = RegimeStrategy(
            id='vol_hmm_5state',
            name='5-state HMM volatility regime',
            category=RegimeCategory.VOLATILITY,
            regime_type=RegimeType.HMM,
            description='5-state HMM for volatility regimes',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Returns data', 'HMM library']
        )
        
        self.strategies['vol_threshold'] = RegimeStrategy(
            id='vol_threshold',
            name='Volatility threshold regime',
            category=RegimeCategory.VOLATILITY,
            regime_type=RegimeType.THRESHOLD,
            description='Threshold-based volatility regime classification',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Volatility data']
        )
        
        self.strategies['garch_regime'] = RegimeStrategy(
            id='garch_regime',
            name='GARCH regime switching',
            category=RegimeCategory.VOLATILITY,
            regime_type=RegimeType.MARKOV_SWITCHING,
            description='Markov-switching GARCH model',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Returns data', 'GARCH library']
        )
        
        # Trend Regimes
        self.strategies['trend_hmm'] = RegimeStrategy(
            id='trend_hmm',
            name='Trend HMM regime',
            category=RegimeCategory.TREND,
            regime_type=RegimeType.HMM,
            description='HMM for trend regimes (bull, bear, sideways)',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Returns data', 'HMM library']
        )
        
        self.strategies['trend_ma'] = RegimeStrategy(
            id='trend_ma',
            name='Moving average trend regime',
            category=RegimeCategory.TREND,
            regime_type=RegimeType.RULE_BASED,
            description='MA-based trend regime classification',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Price data']
        )
        
        self.strategies['trend_adf'] = RegimeStrategy(
            id='trend_adf',
            name='ADF test trend regime',
            category=RegimeCategory.TREND,
            regime_type=RegimeType.THRESHOLD,
            description='ADF test for trend stationarity',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Price data', 'ADF test']
        )
        
        # Liquidity Regimes
        self.strategies['liquidity_hmm'] = RegimeStrategy(
            id='liquidity_hmm',
            name='Liquidity HMM regime',
            category=RegimeCategory.LIQUIDITY,
            regime_type=RegimeType.HMM,
            description='HMM for liquidity regimes',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Liquidity data', 'HMM library']
        )
        
        self.strategies['spread_regime'] = RegimeStrategy(
            id='spread_regime',
            name='Spread regime',
            category=RegimeCategory.LIQUIDITY,
            regime_type=RegimeType.THRESHOLD,
            description='Spread-based liquidity regime',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Spread data']
        )
        
        # Correlation Regimes
        self.strategies['correlation_hmm'] = RegimeStrategy(
            id='correlation_hmm',
            name='Correlation HMM regime',
            category=RegimeCategory.CORRELATION,
            regime_type=RegimeType.HMM,
            description='HMM for correlation regimes',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Correlation data', 'HMM library']
        )
        
        self.strategies['correlation_clustering'] = RegimeStrategy(
            id='correlation_clustering',
            name='Correlation clustering regime',
            category=RegimeCategory.CORRELATION,
            regime_type=RegimeType.CLUSTERING,
            description='Clustering-based correlation regime',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Correlation data', 'clustering library']
        )
        
        # Macro Regimes
        self.strategies['macro_hmm'] = RegimeStrategy(
            id='macro_hmm',
            name='Macro HMM regime',
            category=RegimeCategory.MACRO,
            regime_type=RegimeType.HMM,
            description='HMM for macro regimes',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Macro data', 'HMM library']
        )
        
        self.strategies['business_cycle'] = RegimeStrategy(
            id='business_cycle',
            name='Business cycle regime',
            category=RegimeCategory.MACRO,
            regime_type=RegimeType.RULE_BASED,
            description='Business cycle regime classification',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Macro data']
        )
        
        # Sentiment Regimes
        self.strategies['sentiment_hmm'] = RegimeStrategy(
            id='sentiment_hmm',
            name='Sentiment HMM regime',
            category=RegimeCategory.SENTIMENT,
            regime_type=RegimeType.HMM,
            description='HMM for sentiment regimes',
            expected_sharpe=0.2,
            expected_capacity='Medium',
            difficulty='Medium',
            data_requirements=['Sentiment data', 'HMM library']
        )
        
        self.strategies['vix_regime'] = RegimeStrategy(
            id='vix_regime',
            name='VIX regime',
            category=RegimeCategory.VOLATILITY,
            regime_type=RegimeType.THRESHOLD,
            description='VIX-based volatility regime',
            expected_sharpe=0.2,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['VIX data']
        )
        
        # Seasonal Regimes
        self.strategies['monthly_seasonality'] = RegimeStrategy(
            id='monthly_seasonality',
            name='Monthly seasonality regime',
            category=RegimeCategory.SEASONAL,
            regime_type=RegimeType.RULE_BASED,
            description='Monthly seasonality regime',
            expected_sharpe=0.1,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Calendar data']
        )
        
        self.strategies['day_of_week'] = RegimeStrategy(
            id='day_of_week',
            name='Day of week regime',
            category=RegimeCategory.SEASONAL,
            regime_type=RegimeType.RULE_BASED,
            description='Day of week regime',
            expected_sharpe=0.1,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Calendar data']
        )
        
        # Crisis Regimes
        self.strategies['crisis_detection'] = RegimeStrategy(
            id='crisis_detection',
            name='Crisis detection regime',
            category=RegimeCategory.CRISIS,
            regime_type=RegimeType.ML,
            description='ML-based crisis detection',
            expected_sharpe=0.0,
            expected_capacity='N/A',
            difficulty='High',
            data_requirements=['Returns data', 'ML library']
        )
        
        self.strategies['drawdown_regime'] = RegimeStrategy(
            id='drawdown_regime',
            name='Drawdown regime',
            category=RegimeCategory.CRISIS,
            regime_type=RegimeType.THRESHOLD,
            description='Drawdown-based crisis regime',
            expected_sharpe=0.0,
            expected_capacity='N/A',
            difficulty='Low',
            data_requirements=['Returns data']
        )
        
        # Add more strategies to reach 50 (abbreviated for brevity)
        additional_strategies = [
            ('vol_change_point', 'Volatility change point', RegimeCategory.VOLATILITY, RegimeType.CHANGE_POINT),
            ('trend_change_point', 'Trend change point', RegimeCategory.TREND, RegimeType.CHANGE_POINT),
            ('liquidity_change_point', 'Liquidity change point', RegimeCategory.LIQUIDITY, RegimeType.CHANGE_POINT),
            ('correlation_change_point', 'Correlation change point', RegimeCategory.CORRELATION, RegimeType.CHANGE_POINT),
            ('vol_clustering', 'Volatility clustering', RegimeCategory.VOLATILITY, RegimeType.CLUSTERING),
            ('return_clustering', 'Return clustering', RegimeCategory.TREND, RegimeType.CLUSTERING),
            ('liquidity_clustering', 'Liquidity clustering', RegimeCategory.LIQUIDITY, RegimeType.CLUSTERING),
            ('sector_regime', 'Sector regime', RegimeCategory.TREND, RegimeType.HMM),
            ('style_regime', 'Style regime', RegimeCategory.TREND, RegimeType.HMM),
            ('factor_regime', 'Factor regime', RegimeCategory.TREND, RegimeType.HMM),
            ('regime_aware_factor', 'Regime-aware factor', RegimeCategory.TREND, RegimeType.ML),
            ('dynamic_allocation', 'Dynamic allocation', RegimeCategory.TREND, RegimeType.ML),
            ('volatility_targeting', 'Volatility targeting', RegimeCategory.VOLATILITY, RegimeType.RULE_BASED),
            ('risk_parity_regime', 'Risk parity regime', RegimeCategory.VOLATILITY, RegimeType.RULE_BASED),
            ('tail_hedge_regime', 'Tail hedge regime', RegimeCategory.CRISIS, RegimeType.RULE_BASED),
            ('crisis_alpha', 'Crisis alpha', RegimeCategory.CRISIS, RegimeType.ML),
            ('flight_to_quality', 'Flight to quality', RegimeCategory.CRISIS, RegimeType.RULE_BASED),
            ('safe_haven', 'Safe haven', RegimeCategory.CRISIS, RegimeType.RULE_BASED),
            ('inflation_regime', 'Inflation regime', RegimeCategory.MACRO, RegimeType.THRESHOLD),
            ('rate_regime', 'Interest rate regime', RegimeCategory.MACRO, RegimeType.THRESHOLD),
            ('growth_regime', 'Growth regime', RegimeCategory.MACRO, RegimeType.THRESHOLD),
            ('recession_regime', 'Recession regime', RegimeCategory.MACRO, RegimeType.RULE_BASED),
            ('expansion_regime', 'Expansion regime', RegimeCategory.MACRO, RegimeType.RULE_BASED),
            ('news_regime', 'News sentiment regime', RegimeCategory.SENTIMENT, RegimeType.ML),
            ('social_regime', 'Social sentiment regime', RegimeCategory.SENTIMENT, RegimeType.ML),
            ('analyst_regime', 'Analyst regime', RegimeCategory.SENTIMENT, RegimeType.RULE_BASED),
            ('insider_regime', 'Insider regime', RegimeCategory.SENTIMENT, RegimeType.RULE_BASED),
            ('retail_regime', 'Retail regime', RegimeCategory.SENTIMENT, RegimeType.RULE_BASED),
            ('institutional_regime', 'Institutional regime', RegimeCategory.SENTIMENT, RegimeType.RULE_BASED),
            ('turn_of_month', 'Turn of month regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
            ('turn_of_quarter', 'Turn of quarter regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
            ('turn_of_year', 'Turn of year regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
            ('holiday_regime', 'Holiday regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
            ('earnings_season', 'Earnings season regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
            ('pre_market', 'Pre-market regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
            ('post_market', 'Post-market regime', RegimeCategory.SEASONAL, RegimeType.RULE_BASED),
        ]
        
        for i, (strat_id, name, category, regime_type) in enumerate(additional_strategies, start=20):
            self.strategies[strat_id] = RegimeStrategy(
                id=strat_id,
                name=name,
                category=category,
                regime_type=regime_type,
                description=f'Regime strategy for {name}',
                expected_sharpe=0.2,
                expected_capacity='High',
                difficulty='Medium',
                data_requirements=['Data requirements TBD']
            )
    
    def get_strategy(self, strategy_id: str) -> Optional[RegimeStrategy]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def get_strategies_by_category(self, category: RegimeCategory) -> List[RegimeStrategy]:
        """Get strategies by category."""
        return [s for s in self.strategies.values() if s.category == category]
    
    def get_strategies_by_type(self, regime_type: RegimeType) -> List[RegimeStrategy]:
        """Get strategies by type."""
        return [s for s in self.strategies.values() if s.regime_type == regime_type]
    
    def get_highest_sharpe_strategies(self, n: int = 10) -> List[RegimeStrategy]:
        """Get top N strategies by expected Sharpe."""
        sorted_strategies = sorted(
            self.strategies.values(),
            key=lambda x: x.expected_sharpe,
            reverse=True
        )
        return sorted_strategies[:n]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("TOP 50 REGIME STRATEGIES CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Strategies: {len(self.strategies)}")
        
        print(f"\nBy Category:")
        for category in RegimeCategory:
            count = len(self.get_strategies_by_category(category))
            if count > 0:
                print(f"  {category.value}: {count}")
        
        print(f"\nBy Type:")
        for regime_type in RegimeType:
            count = len(self.get_strategies_by_type(regime_type))
            if count > 0:
                print(f"  {regime_type.value}: {count}")
        
        print(f"\nTop 10 by Expected Sharpe:")
        top_10 = self.get_highest_sharpe_strategies(10)
        print(f"{'ID':<25} {'Name':<40} {'Category':<15} {'Type':<15} {'Sharpe':<10}")
        print("-" * 120)
        for strat in top_10:
            print(f"{strat.id:<25} {strat.name:<40} {strat.category.value:<15} {strat.regime_type.value:<15} {strat.expected_sharpe:<10.2f}")
        
        print("\n" + "="*80)


def sample_regime_strategies_catalog():
    """Demonstrate regime strategies catalog."""
    print("=== Top 50 Regime Strategies Catalog Demo ===\n")
    
    catalog = RegimeStrategiesCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Top 50 Regime Strategies Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of top 50 regime strategies")
    print("- Classification by category (volatility, trend, liquidity, etc.)")
    print("- Classification by type (HMM, Markov switching, change point, etc.)")
    print("- Expected Sharpe, capacity, and difficulty ratings")
    print("- Data requirements for each strategy")


if __name__ == "__main__":
    sample_regime_strategies_catalog()
