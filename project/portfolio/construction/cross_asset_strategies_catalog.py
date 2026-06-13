"""
Top 50 Cross-Asset Strategies Catalog

This module implements a comprehensive catalog of the top 50 cross-asset
strategies for quantitative trading.

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


class CrossAssetCategory(Enum):
    """Cross-asset category types."""
    EQUITY_COMMODITY = "equity_commodity"
    EQUITY_BOND = "equity_bond"
    EQUITY_FX = "equity_fx"
    COMMODITY_FX = "commodity_fx"
    BOND_FX = "bond_fx"
    INDEX_CONSTITUENT = "index_constituent"
    INTER_MARKET = "inter_market"
    GLOBAL = "global"


class CrossAssetType(Enum):
    """Cross-asset type types."""
    LEAD_LAG = "lead_lag"
    SPILLOVER = "spillover"
    CORRELATION = "correlation"
    CARRY = "carry"
    TERM_STRUCTURE = "term_structure"
    RELATIVE_VALUE = "relative_value"
    MOMENTUM = "momentum"


@dataclass
class CrossAssetStrategy:
    """Cross-asset strategy definition."""
    id: str
    name: str
    category: CrossAssetCategory
    strategy_type: CrossAssetType
    description: str
    expected_sharpe: float
    expected_capacity: str
    difficulty: str
    data_requirements: List[str]


class CrossAssetStrategiesCatalog:
    """
    Catalog of top 50 cross-asset strategies.
    
    This class provides a comprehensive catalog of cross-asset strategies
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize cross-asset strategies catalog."""
        self.strategies: Dict[str, CrossAssetStrategy] = {}
        self._initialize_catalog()
        
        logger.info(f"CrossAssetStrategiesCatalog initialized with {len(self.strategies)} strategies")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with top 50 cross-asset strategies."""
        
        # Equity-Commodity
        self.strategies['commodity_leads_equity'] = CrossAssetStrategy(
            id='commodity_leads_equity',
            name='Commodity leads equity',
            category=CrossAssetCategory.EQUITY_COMMODITY,
            strategy_type=CrossAssetType.LEAD_LAG,
            description='Commodity prices lead equity returns',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Commodity data', 'equity data']
        )
        
        self.strategies['oil_equity'] = CrossAssetStrategy(
            id='oil_equity',
            name='Oil-equity relationship',
            category=CrossAssetCategory.EQUITY_COMMODITY,
            strategy_type=CrossAssetType.LEAD_LAG,
            description='Oil prices lead energy equities',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Oil data', 'energy equity data']
        )
        
        self.strategies['gold_equity'] = CrossAssetStrategy(
            id='gold_equity',
            name='Gold-equity relationship',
            category=CrossAssetCategory.EQUITY_COMMODITY,
            strategy_type=CrossAssetType.LEAD_LAG,
            description='Gold prices lead mining equities',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Gold data', 'mining equity data']
        )
        
        # Equity-Bond
        self.strategies['equity_bond_spillover'] = CrossAssetStrategy(
            id='equity_bond_spillover',
            name='Equity-bond spillover',
            category=CrossAssetCategory.EQUITY_BOND,
            strategy_type=CrossAssetType.SPILLOVER,
            description='Volatility spillover between equities and bonds',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Equity data', 'bond data']
        )
        
        self.strategies['flight_to_quality'] = CrossAssetStrategy(
            id='flight_to_quality',
            name='Flight to quality',
            category=CrossAssetCategory.EQUITY_BOND,
            strategy_type=CrossAssetType.LEAD_LAG,
            description='Bonds lead equities in stress periods',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Equity data', 'bond data']
        )
        
        # Equity-FX
        self.strategies['fx_equity'] = CrossAssetStrategy(
            id='fx_equity',
            name='FX-equity relationship',
            category=CrossAssetCategory.EQUITY_FX,
            strategy_type=CrossAssetType.LEAD_LAG,
            description='FX rates lead export-oriented equities',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['FX data', 'equity data']
        )
        
        self.strategies['currency_momentum'] = CrossAssetStrategy(
            id='currency_momentum',
            name='Currency momentum',
            category=CrossAssetCategory.EQUITY_FX,
            strategy_type=CrossAssetType.MOMENTUM,
            description='Currency momentum strategies',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['FX data']
        )
        
        # Commodity-FX
        self.strategies['commodity_currency'] = CrossAssetStrategy(
            id='commodity_currency',
            name='Commodity-currency relationship',
            category=CrossAssetCategory.COMMODITY_FX,
            strategy_type=CrossAssetType.LEAD_LAG,
            description='Commodity currencies lead commodity prices',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Commodity data', 'FX data']
        )
        
        self.strategies['fx_carry'] = CrossAssetStrategy(
            id='fx_carry',
            name='FX carry trade',
            category=CrossAssetCategory.COMMODITY_FX,
            strategy_type=CrossAssetType.CARRY,
            description='FX carry trade strategy',
            expected_sharpe=0.4,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['FX data', 'interest rates']
        )
        
        # Bond-FX
        self.strategies['bond_fx_spillover'] = CrossAssetStrategy(
            id='bond_fx_spillover',
            name='Bond-FX spillover',
            category=CrossAssetCategory.BOND_FX,
            strategy_type=CrossAssetType.SPILLOVER,
            description='Bond yields lead FX rates',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Bond data', 'FX data']
        )
        
        self.strategies['yield_differential'] = CrossAssetStrategy(
            id='yield_differential',
            name='Yield differential',
            category=CrossAssetCategory.BOND_FX,
            strategy_type=CrossAssetType.CARRY,
            description='Yield differential drives FX',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Bond yields', 'FX data']
        )
        
        # Index-Constituent
        self.strategies['index_constituent_spillover'] = CrossAssetStrategy(
            id='index_constituent_spillover',
            name='Index-constituent spillover',
            category=CrossAssetCategory.INDEX_CONSTITUENT,
            strategy_type=CrossAssetType.SPILLOVER,
            description='Index returns lead constituent returns',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Index data', 'constituent data']
        )
        
        self.strategies['sector_spillover'] = CrossAssetStrategy(
            id='sector_spillover',
            name='Sector spillover',
            category=CrossAssetCategory.INDEX_CONSTITUENT,
            strategy_type=CrossAssetType.SPILLOVER,
            description='Leading sectors predict lagging sectors',
            expected_sharpe=0.4,
            expected_capacity='High',
            difficulty='Low',
            data_requirements=['Sector data']
        )
        
        self.strategies['dispersion'] = CrossAssetStrategy(
            id='dispersion',
            name='Dispersion trading',
            category=CrossAssetCategory.INDEX_CONSTITUENT,
            strategy_type=CrossAssetType.CORRELATION,
            description='Trade implied vs realized correlation',
            expected_sharpe=0.4,
            expected_capacity='High',
            difficulty='High',
            data_requirements=['Index options', 'constituent options']
        )
        
        # Inter-Market
        self.strategies['inter_market_arbitrage'] = CrossAssetStrategy(
            id='inter_market_arbitrage',
            name='Inter-market arbitrage',
            category=CrossAssetCategory.INTER_MARKET,
            strategy_type=CrossAssetType.RELATIVE_VALUE,
            description='Arbitrage across markets',
            expected_sharpe=0.5,
            expected_capacity='Medium',
            difficulty='High',
            data_requirements=['Multi-market data']
        )
        
        self.strategies['global_momentum'] = CrossAssetStrategy(
            id='global_momentum',
            name='Global momentum',
            category=CrossAssetCategory.GLOBAL,
            strategy_type=CrossAssetType.MOMENTUM,
            description='Global momentum across markets',
            expected_sharpe=0.3,
            expected_capacity='High',
            difficulty='Medium',
            data_requirements=['Global market data']
        )
        
        # Add more strategies to reach 50 (abbreviated for brevity)
        additional_strategies = [
            ('copper_equity', 'Copper-equity relationship', CrossAssetCategory.EQUITY_COMMODITY, CrossAssetType.LEAD_LAG),
            ('agriculture_equity', 'Agriculture-equity relationship', CrossAssetCategory.EQUITY_COMMODITY, CrossAssetType.LEAD_LAG),
            ('energy_equity', 'Energy-equity relationship', CrossAssetCategory.EQUITY_COMMODITY, CrossAssetType.LEAD_LAG),
            ('metal_equity', 'Metal-equity relationship', CrossAssetCategory.EQUITY_COMMODITY, CrossAssetType.LEAD_LAG),
            ('equity_bond_correlation', 'Equity-bond correlation', CrossAssetCategory.EQUITY_BOND, CrossAssetType.CORRELATION),
            ('bond_equity_ratio', 'Bond-equity ratio', CrossAssetCategory.EQUITY_BOND, CrossAssetType.RELATIVE_VALUE),
            ('duration_tilt', 'Duration tilt', CrossAssetCategory.EQUITY_BOND, CrossAssetType.TERM_STRUCTURE),
            ('credit_equity', 'Credit-equity relationship', CrossAssetCategory.EQUITY_BOND, CrossAssetType.LEAD_LAG),
            ('inflation_breakeven_equity', 'Inflation breakeven-equity', CrossAssetCategory.EQUITY_BOND, CrossAssetType.LEAD_LAG),
            ('real_yield_equity', 'Real yield-equity', CrossAssetCategory.EQUITY_BOND, CrossAssetType.LEAD_LAG),
            ('emerging_fx_equity', 'Emerging FX-equity', CrossAssetCategory.EQUITY_FX, CrossAssetType.LEAD_LAG),
            ('developed_fx_equity', 'Developed FX-equity', CrossAssetCategory.EQUITY_FX, CrossAssetType.LEAD_LAG),
            ('fx_volatility', 'FX volatility', CrossAssetCategory.EQUITY_FX, CrossAssetType.MOMENTUM),
            ('fx_value', 'FX value', CrossAssetCategory.EQUITY_FX, CrossAssetType.RELATIVE_VALUE),
            ('commodity_fx_carry', 'Commodity-FX carry', CrossAssetCategory.COMMODITY_FX, CrossAssetType.CARRY),
            ('commodity_term', 'Commodity term structure', CrossAssetCategory.COMMODITY_FX, CrossAssetType.TERM_STRUCTURE),
            ('commodity_roll', 'Commodity roll yield', CrossAssetCategory.COMMODITY_FX, CrossAssetType.CARRY),
            ('commodity_momentum', 'Commodity momentum', CrossAssetCategory.COMMODITY_FX, CrossAssetType.MOMENTUM),
            ('bond_fx_carry', 'Bond-FX carry', CrossAssetCategory.BOND_FX, CrossAssetType.CARRY),
            ('bond_term_structure', 'Bond term structure', CrossAssetCategory.BOND_FX, CrossAssetType.TERM_STRUCTURE),
            ('bond_momentum', 'Bond momentum', CrossAssetCategory.BOND_FX, CrossAssetType.MOMENTUM),
            ('bond_value', 'Bond value', CrossAssetCategory.BOND_FX, CrossAssetType.RELATIVE_VALUE),
            ('index_arbitrage', 'Index arbitrage', CrossAssetCategory.INDEX_CONSTITUENT, CrossAssetType.RELATIVE_VALUE),
            ('constituent_momentum', 'Constituent momentum', CrossAssetCategory.INDEX_CONSTITUENT, CrossAssetType.MOMENTUM),
            ('sector_rotation', 'Sector rotation', CrossAssetCategory.INDEX_CONSTITUENT, CrossAssetType.MOMENTUM),
            ('style_rotation', 'Style rotation', CrossAssetCategory.INDEX_CONSTITUENT, CrossAssetType.MOMENTUM),
            ('factor_rotation', 'Factor rotation', CrossAssetCategory.INDEX_CONSTITUENT, CrossAssetType.MOMENTUM),
            ('global_value', 'Global value', CrossAssetCategory.GLOBAL, CrossAssetType.RELATIVE_VALUE),
            ('global_carry', 'Global carry', CrossAssetCategory.GLOBAL, CrossAssetType.CARRY),
            ('global_term', 'Global term structure', CrossAssetCategory.GLOBAL, CrossAssetType.TERM_STRUCTURE),
            ('global_spillover', 'Global spillover', CrossAssetCategory.GLOBAL, CrossAssetType.SPILLOVER),
            ('global_correlation', 'Global correlation', CrossAssetCategory.GLOBAL, CrossAssetType.CORRELATION),
            ('inter_regime', 'Inter-regime', CrossAssetCategory.INTER_MARKET, CrossAssetType.MOMENTUM),
            ('cross_asset_carry', 'Cross-asset carry', CrossAssetCategory.INTER_MARKET, CrossAssetType.CARRY),
            ('cross_asset_value', 'Cross-asset value', CrossAssetCategory.INTER_MARKET, CrossAssetType.RELATIVE_VALUE),
            ('cross_asset_momentum', 'Cross-asset momentum', CrossAssetCategory.INTER_MARKET, CrossAssetType.MOMENTUM),
            ('cross_asset_trend', 'Cross-asset trend', CrossAssetCategory.INTER_MARKET, CrossAssetType.MOMENTUM),
            ('cross_asset_reversal', 'Cross-asset reversal', CrossAssetCategory.INTER_MARKET, CrossAssetType.RELATIVE_VALUE),
            ('cross_asset_mean_reversion', 'Cross-asset mean reversion', CrossAssetCategory.INTER_MARKET, CrossAssetType.RELATIVE_VALUE),
            ('cross_asset_stat_arb', 'Cross-asset statistical arbitrage', CrossAssetCategory.INTER_MARKET, CrossAssetType.RELATIVE_VALUE),
        ]
        
        for i, (strat_id, name, category, strategy_type) in enumerate(additional_strategies, start=15):
            self.strategies[strat_id] = CrossAssetStrategy(
                id=strat_id,
                name=name,
                category=category,
                strategy_type=strategy_type,
                description=f'Cross-asset strategy for {name}',
                expected_sharpe=0.3,
                expected_capacity='High',
                difficulty='Medium',
                data_requirements=['Data requirements TBD']
            )
    
    def get_strategy(self, strategy_id: str) -> Optional[CrossAssetStrategy]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)
    
    def get_strategies_by_category(self, category: CrossAssetCategory) -> List[CrossAssetStrategy]:
        """Get strategies by category."""
        return [s for s in self.strategies.values() if s.category == category]
    
    def get_strategies_by_type(self, strategy_type: CrossAssetType) -> List[CrossAssetStrategy]:
        """Get strategies by type."""
        return [s for s in self.strategies.values() if s.strategy_type == strategy_type]
    
    def get_highest_sharpe_strategies(self, n: int = 10) -> List[CrossAssetStrategy]:
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
        print("TOP 50 CROSS-ASSET STRATEGIES CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Strategies: {len(self.strategies)}")
        
        print(f"\nBy Category:")
        for category in CrossAssetCategory:
            count = len(self.get_strategies_by_category(category))
            if count > 0:
                print(f"  {category.value}: {count}")
        
        print(f"\nBy Type:")
        for strategy_type in CrossAssetType:
            count = len(self.get_strategies_by_type(strategy_type))
            if count > 0:
                print(f"  {strategy_type.value}: {count}")
        
        print(f"\nTop 10 by Expected Sharpe:")
        top_10 = self.get_highest_sharpe_strategies(10)
        print(f"{'ID':<25} {'Name':<40} {'Category':<20} {'Type':<15} {'Sharpe':<10}")
        print("-" * 130)
        for strat in top_10:
            print(f"{strat.id:<25} {strat.name:<40} {strat.category.value:<20} {strat.strategy_type.value:<15} {strat.expected_sharpe:<10.2f}")
        
        print("\n" + "="*80)


def sample_cross_asset_strategies_catalog():
    """Demonstrate cross-asset strategies catalog."""
    print("=== Top 50 Cross-Asset Strategies Catalog Demo ===\n")
    
    catalog = CrossAssetStrategiesCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Top 50 Cross-Asset Strategies Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of top 50 cross-asset strategies")
    print("- Classification by category (equity-commodity, equity-bond, etc.)")
    print("- Classification by type (lead-lag, spillover, correlation, etc.)")
    print("- Expected Sharpe, capacity, and difficulty ratings")
    print("- Data requirements for each strategy")


if __name__ == "__main__":
    sample_cross_asset_strategies_catalog()
