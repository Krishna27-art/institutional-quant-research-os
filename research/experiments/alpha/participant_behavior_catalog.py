"""
Participant Behavior Catalog - 6 Participant Behavior Models

This module implements 6 participant behavior models that capture
the impact of different market participant behaviors on prices.

Based on market participant literature and empirical studies.
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


class ParticipantType(Enum):
    """Types of market participants."""
    RETAIL = "retail"
    INSTITUTIONAL = "institutional"
    HFT = "hft"
    MARKET_MAKER = "market_maker"
    ARBITRAGEUR = "arbitrageur"
    CORPORATE = "corporate"


@dataclass
class ParticipantBehavior:
    """Participant behavior definition."""
    id: str
    name: str
    participant_type: ParticipantType
    description: str
    source: str
    expected_sharpe: float
    expected_capacity: str
    decay: str
    difficulty: str
    data_requirements: List[str]


class ParticipantBehaviorCatalog:
    """
    Catalog of 6 participant behavior models.
    
    This class provides a comprehensive catalog of participant behaviors
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize participant behavior catalog."""
        self.behaviors: Dict[str, ParticipantBehavior] = {}
        self._initialize_catalog()
        
        logger.info(f"ParticipantBehaviorCatalog initialized with {len(self.behaviors)} behaviors")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with 6 participant behaviors."""
        
        self.behaviors['retail_sentiment'] = ParticipantBehavior(
            id='retail_sentiment',
            name='Retail sentiment',
            participant_type=ParticipantType.RETAIL,
            description='Retail investors exhibit predictable sentiment patterns; contrarian to retail sentiment generates alpha',
            source='Barber & Odean 2008; retail flow studies',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Retail trading data', 'sentiment indicators']
        )
        
        self.behaviors['institutional_herding'] = ParticipantBehavior(
            id='institutional_herding',
            name='Institutional herding',
            participant_type=ParticipantType.INSTITUTIONAL,
            description='Institutional investors herd on popular stocks; herding predicts reversals',
            source='Lakonishok et al. 1992; institutional flow studies',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Institutional holdings', '13F filings']
        )
        
        self.behaviors['hft_impact'] = ParticipantBehavior(
            id='hft_impact',
            name='HFT impact',
            participant_type=ParticipantType.HFT,
            description='HFT activity creates predictable short-term price patterns',
            source='HFT literature',
            expected_sharpe=0.4,
            expected_capacity='Low',
            decay='Live (months)',
            difficulty='Very High',
            data_requirements=['High-frequency data', 'order book data']
        )
        
        self.behaviors['market_maker_inventory'] = ParticipantBehavior(
            id='market_maker_inventory',
            name='Market maker inventory',
            participant_type=ParticipantType.MARKET_MAKER,
            description='Market makers adjust quotes based on inventory; inventory predicts price moves',
            source='Madhavan & Seasholes 2007',
            expected_sharpe=0.3,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Order book data', 'inventory estimation']
        )
        
        self.behaviors['arbitrageur_activity'] = ParticipantBehavior(
            id='arbitrageur_activity',
            name='Arbitrageur activity',
            participant_type=ParticipantType.ARBITRAGEUR,
            description='Arbitrageur activity reveals mispricing; arbitrageur flows predict corrections',
            source='Arbitrage literature',
            expected_sharpe=0.4,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Arbitrage data', 'mispricing indicators']
        )
        
        self.behaviors['corporate_buybacks'] = ParticipantBehavior(
            id='corporate_buybacks',
            name='Corporate buybacks',
            participant_type=ParticipantType.CORPORATE,
            description='Corporate buybacks create predictable price support',
            source='Corporate finance literature',
            expected_sharpe=0.2,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Buyback announcements', 'execution data']
        )
    
    def get_behavior(self, behavior_id: str) -> Optional[ParticipantBehavior]:
        """Get a behavior by ID."""
        return self.behaviors.get(behavior_id)
    
    def get_behaviors_by_type(self, participant_type: ParticipantType) -> List[ParticipantBehavior]:
        """Get behaviors by participant type."""
        return [b for b in self.behaviors.values() if b.participant_type == participant_type]
    
    def get_highest_sharpe_behaviors(self, n: int = 5) -> List[ParticipantBehavior]:
        """Get top N behaviors by expected Sharpe."""
        sorted_behaviors = sorted(
            self.behaviors.values(),
            key=lambda x: x.expected_sharpe,
            reverse=True
        )
        return sorted_behaviors[:n]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("PARTICIPANT BEHAVIOR CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Behaviors: {len(self.behaviors)}")
        
        print(f"\nBy Participant Type:")
        for ptype in ParticipantType:
            count = len(self.get_behaviors_by_type(ptype))
            if count > 0:
                print(f"  {ptype.value}: {count}")
        
        print(f"\nTop 5 by Expected Sharpe:")
        top_5 = self.get_highest_sharpe_behaviors(5)
        print(f"{'ID':<25} {'Name':<40} {'Sharpe':<10} {'Capacity':<20}")
        print("-" * 100)
        for behavior in top_5:
            print(f"{behavior.id:<25} {behavior.name:<40} {behavior.expected_sharpe:<10.2f} {behavior.expected_capacity:<20}")
        
        print("\n" + "="*80)


def sample_participant_behavior_catalog():
    """Demonstrate participant behavior catalog."""
    print("=== Participant Behavior Catalog Demo ===\n")
    
    catalog = ParticipantBehaviorCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Participant Behavior Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of 6 participant behavior models")
    print("- Classification by participant type (retail, institutional, HFT, etc.)")
    print("- Expected Sharpe, capacity, decay, and difficulty ratings")
    print("- Data requirements for each behavior")
    print("- Source attribution for each behavior")


if __name__ == "__main__":
    sample_participant_behavior_catalog()
