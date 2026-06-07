"""
Liquidity Effects Catalog - 5 Liquidity Effect Models

This module implements 5 liquidity effect models that capture
the impact of liquidity on asset prices and returns.

Based on liquidity literature and empirical studies.
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


class LiquidityEffectType(Enum):
    """Types of liquidity effects."""
    ILLIQUIDITY_PREMIUM = "illiquidity_premium"
    AMIHUD_ILLIQUIDITY = "amihud_illiquidity"
    BID_ASK_SPREAD = "bid_ask_spread"
    TURNOVER = "turnover"
    MARKET_DEPTH = "market_depth"


@dataclass
class LiquidityEffect:
    """Liquidity effect definition."""
    id: str
    name: str
    effect_type: LiquidityEffectType
    description: str
    source: str
    expected_sharpe: float
    expected_capacity: str
    decay: str
    difficulty: str
    data_requirements: List[str]


class LiquidityEffectsCatalog:
    """
    Catalog of 5 liquidity effect models.
    
    This class provides a comprehensive catalog of liquidity effects
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize liquidity effects catalog."""
        self.effects: Dict[str, LiquidityEffect] = {}
        self._initialize_catalog()
        
        logger.info(f"LiquidityEffectsCatalog initialized with {len(self.effects)} effects")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with 5 liquidity effects."""
        
        self.effects['illiquidity_premium'] = LiquidityEffect(
            id='illiquidity_premium',
            name='Illiquidity premium',
            effect_type=LiquidityEffectType.ILLIQUIDITY_PREMIUM,
            description='Illiquid stocks earn higher expected returns to compensate for liquidity risk',
            source='Amihud & Mendelson 1986; Acharya & Pedersen 2005',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Trading volume', 'price data', 'illiquidity measures']
        )
        
        self.effects['amihud_illiquidity'] = LiquidityEffect(
            id='amihud_illiquidity',
            name='Amihud illiquidity ratio',
            effect_type=LiquidityEffectType.AMIHUD_ILLIQUIDITY,
            description='Daily price impact per dollar of trading volume',
            source='Amihud 2002',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Daily returns', 'dollar volume']
        )
        
        self.effects['bid_ask_spread'] = LiquidityEffect(
            id='bid_ask_spread',
            name='Bid-ask spread premium',
            effect_type=LiquidityEffectType.BID_ASK_SPREAD,
            description='Wide spreads predict higher future returns',
            source='Amihud & Mendelson 1986',
            expected_sharpe=0.2,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Bid-ask spreads', 'quote data']
        )
        
        self.effects['turnover'] = LiquidityEffect(
            id='turnover',
            name='Turnover effect',
            effect_type=LiquidityEffectType.TURNOVER,
            description='High turnover predicts lower future returns (overtrading)',
            source='Datar et al. 1998',
            expected_sharpe=0.2,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Trading volume', 'market cap']
        )
        
        self.effects['market_depth'] = LiquidityEffect(
            id='market_depth',
            name='Market depth effect',
            effect_type=LiquidityEffectType.MARKET_DEPTH,
            description='Shallow order books predict higher future returns',
            source='Market microstructure literature',
            expected_sharpe=0.3,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Order book depth', 'L2 data']
        )
    
    def get_effect(self, effect_id: str) -> Optional[LiquidityEffect]:
        """Get an effect by ID."""
        return self.effects.get(effect_id)
    
    def get_effects_by_type(self, effect_type: LiquidityEffectType) -> List[LiquidityEffect]:
        """Get effects by type."""
        return [e for e in self.effects.values() if e.effect_type == effect_type]
    
    def get_highest_sharpe_effects(self, n: int = 5) -> List[LiquidityEffect]:
        """Get top N effects by expected Sharpe."""
        sorted_effects = sorted(
            self.effects.values(),
            key=lambda x: x.expected_sharpe,
            reverse=True
        )
        return sorted_effects[:n]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("LIQUIDITY EFFECTS CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Effects: {len(self.effects)}")
        
        print(f"\nBy Type:")
        for etype in LiquidityEffectType:
            count = len(self.get_effects_by_type(etype))
            if count > 0:
                print(f"  {etype.value}: {count}")
        
        print(f"\nTop 5 by Expected Sharpe:")
        top_5 = self.get_highest_sharpe_effects(5)
        print(f"{'ID':<25} {'Name':<40} {'Sharpe':<10} {'Capacity':<20}")
        print("-" * 100)
        for effect in top_5:
            print(f"{effect.id:<25} {effect.name:<40} {effect.expected_sharpe:<10.2f} {effect.expected_capacity:<20}")
        
        print("\n" + "="*80)


def sample_liquidity_effects_catalog():
    """Demonstrate liquidity effects catalog."""
    print("=== Liquidity Effects Catalog Demo ===\n")
    
    catalog = LiquidityEffectsCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Liquidity Effects Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of 5 liquidity effect models")
    print("- Classification by type (illiquidity premium, Amihud, spread, etc.)")
    print("- Expected Sharpe, capacity, decay, and difficulty ratings")
    print("- Data requirements for each effect")
    print("- Source attribution for each effect")


if __name__ == "__main__":
    sample_liquidity_effects_catalog()
