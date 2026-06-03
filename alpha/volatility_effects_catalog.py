"""
Volatility Effects Catalog - 7 Volatility Effect Models

This module implements 7 volatility effect models that capture
the impact of volatility on asset prices and returns.

Based on volatility literature and empirical studies.
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


class VolatilityEffectType(Enum):
    """Types of volatility effects."""
    VOLATILITY_RISK_PREMIUM = "volatility_risk_premium"
    VOLATILITY_MEAN_REVERSION = "volatility_mean_reversion"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    VOLATILITY_SPILLOVER = "volatility_spillover"
    VOLATILITY_TERM_STRUCTURE = "volatility_term_structure"
    VOLATILITY_SKEW = "volatility_skew"
    VOLATILITY_OF_VOLATILITY = "volatility_of_volatility"


@dataclass
class VolatilityEffect:
    """Volatility effect definition."""
    id: str
    name: str
    effect_type: VolatilityEffectType
    description: str
    source: str
    expected_sharpe: float
    expected_capacity: str
    decay: str
    difficulty: str
    data_requirements: List[str]


class VolatilityEffectsCatalog:
    """
    Catalog of 7 volatility effect models.
    
    This class provides a comprehensive catalog of volatility effects
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize volatility effects catalog."""
        self.effects: Dict[str, VolatilityEffect] = {}
        self._initialize_catalog()
        
        logger.info(f"VolatilityEffectsCatalog initialized with {len(self.effects)} effects")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with 7 volatility effects."""
        
        self.effects['volatility_risk_premium'] = VolatilityEffect(
            id='volatility_risk_premium',
            name='Volatility risk premium (VRP)',
            effect_type=VolatilityEffectType.VOLATILITY_RISK_PREMIUM,
            description='Implied volatility exceeds realized volatility on average; selling volatility earns premium',
            source='Carr & Wu 2009; thesis 2010-2022',
            expected_sharpe=0.8,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Implied volatility', 'realized volatility', 'VIX futures']
        )
        
        self.effects['volatility_mean_reversion'] = VolatilityEffect(
            id='volatility_mean_reversion',
            name='Volatility mean reversion',
            effect_type=VolatilityEffectType.VOLATILITY_MEAN_REVERSION,
            description='Volatility tends to revert to long-term mean; high vol predicts low vol',
            source='GARCH literature',
            expected_sharpe=0.4,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Historical volatility', 'GARCH modeling']
        )
        
        self.effects['volatility_clustering'] = VolatilityEffect(
            id='volatility_clustering',
            name='Volatility clustering',
            effect_type=VolatilityEffectType.VOLATILITY_CLUSTERING,
            description='High volatility periods cluster together; persistence in vol',
            source='Engle 1982 (ARCH)',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Intraday volatility', 'clustering detection']
        )
        
        self.effects['volatility_spillover'] = VolatilityEffect(
            id='volatility_spillover',
            name='Volatility spillover',
            effect_type=VolatilityEffectType.VOLATILITY_SPILLOVER,
            description='Volatility spills over across markets and assets',
            source='2026 network study',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Cross-asset volatility', 'correlation data']
        )
        
        self.effects['volatility_term_structure'] = VolatilityEffect(
            id='volatility_term_structure',
            name='Volatility term structure',
            effect_type=VolatilityEffectType.VOLATILITY_TERM_STRUCTURE,
            description='Different maturities exhibit different premia; term structure slope predicts vol curve moves',
            source='VIX futures basis literature',
            expected_sharpe=0.6,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Full option surface', 'term structure data']
        )
        
        self.effects['volatility_skew'] = VolatilityEffect(
            id='volatility_skew',
            name='Volatility skew',
            effect_type=VolatilityEffectType.VOLATILITY_SKEW,
            description='OTM puts trade at higher vol than OTM calls; skew predicts returns',
            source='Kozhan, Neuberger, Schneider 2013',
            expected_sharpe=0.5,
            expected_capacity='High',
            decay='Persistent',
            difficulty='High',
            data_requirements=['Option surface', 'skew calculation']
        )
        
        self.effects['volatility_of_volatility'] = VolatilityEffect(
            id='volatility_of_volatility',
            name='Volatility of volatility (VoV)',
            effect_type=VolatilityEffectType.VOLATILITY_OF_VOLATILITY,
            description='Volatility itself is volatile; VoV premium can be harvested',
            source='Recent thesis 2025',
            expected_sharpe=0.5,
            expected_capacity='High',
            decay='Persistent',
            difficulty='High',
            data_requirements=['Realized vol of vol', 'options on volatility']
        )
    
    def get_effect(self, effect_id: str) -> Optional[VolatilityEffect]:
        """Get an effect by ID."""
        return self.effects.get(effect_id)
    
    def get_effects_by_type(self, effect_type: VolatilityEffectType) -> List[VolatilityEffect]:
        """Get effects by type."""
        return [e for e in self.effects.values() if e.effect_type == effect_type]
    
    def get_highest_sharpe_effects(self, n: int = 5) -> List[VolatilityEffect]:
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
        print("VOLATILITY EFFECTS CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Effects: {len(self.effects)}")
        
        print(f"\nBy Type:")
        for etype in VolatilityEffectType:
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


def sample_volatility_effects_catalog():
    """Demonstrate volatility effects catalog."""
    print("=== Volatility Effects Catalog Demo ===\n")
    
    catalog = VolatilityEffectsCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Volatility Effects Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of 7 volatility effect models")
    print("- Classification by type (VRP, mean reversion, clustering, etc.)")
    print("- Expected Sharpe, capacity, decay, and difficulty ratings")
    print("- Data requirements for each effect")
    print("- Source attribution for each effect")


if __name__ == "__main__":
    sample_volatility_effects_catalog()
