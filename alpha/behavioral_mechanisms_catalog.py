"""
Behavioral Mechanisms Catalog - 10 Behavioral Mechanisms

This module implements 10 behavioral mechanism models that capture
psychological biases and patterns in market participant behavior.

Based on behavioral finance literature and empirical studies.
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


class BehavioralBias(Enum):
    """Types of behavioral biases."""
    HERDING = "herding"
    DISPOSITION_EFFECT = "disposition_effect"
    OVERCONFIDENCE = "overconfidence"
    ANCHORING = "anchoring"
    LOSS_AVERSION = "loss_aversion"
    RECENCY_BIAS = "recency_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    AVAILABILITY_BIAS = "availability_bias"
    MENTAL_ACCOUNTING = "mental_accounting"
    STATUS_QUO_BIAS = "status_quo_bias"


@dataclass
class BehavioralMechanism:
    """Behavioral mechanism definition."""
    id: str
    name: str
    bias: BehavioralBias
    description: str
    source: str
    expected_sharpe: float
    expected_capacity: str
    decay: str
    difficulty: str
    data_requirements: List[str]


class BehavioralMechanismsCatalog:
    """
    Catalog of 10 behavioral mechanisms.
    
    This class provides a comprehensive catalog of behavioral mechanisms
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize behavioral mechanisms catalog."""
        self.mechanisms: Dict[str, BehavioralMechanism] = {}
        self._initialize_catalog()
        
        logger.info(f"BehavioralMechanismsCatalog initialized with {len(self.mechanisms)} mechanisms")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with 10 behavioral mechanisms."""
        
        self.mechanisms['herding'] = BehavioralMechanism(
            id='herding',
            name='Herding behavior',
            bias=BehavioralBias.HERDING,
            description='Investors follow the crowd, leading to momentum and eventual reversals',
            source='Bikhchandani et al. 1992; Lakonishok et al. 1994',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Cross-sectional returns', 'correlation measures']
        )
        
        self.mechanisms['disposition_effect'] = BehavioralMechanism(
            id='disposition_effect',
            name='Disposition effect',
            bias=BehavioralBias.DISPOSITION_EFFECT,
            description='Investors hold losers too long and sell winners too soon',
            source='Shefrin & Statman 1985; Odean 1998',
            expected_sharpe=0.2,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Individual trading data', 'holding periods']
        )
        
        self.mechanisms['overconfidence'] = BehavioralMechanism(
            id='overconfidence',
            name='Overconfidence bias',
            bias=BehavioralBias.OVERCONFIDENCE,
            description='Investors overestimate their ability, leading to excessive trading',
            source='Barber & Odean 2001',
            expected_sharpe=0.2,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Trading volume', 'portfolio turnover']
        )
        
        self.mechanisms['anchoring'] = BehavioralMechanism(
            id='anchoring',
            name='Anchoring bias',
            bias=BehavioralBias.ANCHORING,
            description='Investors anchor to reference prices (52-week high, purchase price)',
            source='Tversky & Kahneman 1974; empirical studies',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['52-week high/low', 'purchase price data']
        )
        
        self.mechanisms['loss_aversion'] = BehavioralMechanism(
            id='loss_aversion',
            name='Loss aversion',
            bias=BehavioralBias.LOSS_AVERSION,
            description='Investors feel losses more intensely than gains, leading to risk aversion',
            source='Kahneman & Tversky 1979; prospect theory',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Return distributions', 'risk-taking behavior']
        )
        
        self.mechanisms['recency_bias'] = BehavioralMechanism(
            id='recency_bias',
            name='Recency bias',
            bias=BehavioralBias.RECENCY_BIAS,
            description='Investors overweight recent information, leading to momentum',
            source='Behavioral finance literature',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Moderate',
            difficulty='Low',
            data_requirements=['Recent returns', 'long-term returns']
        )
        
        self.mechanisms['confirmation_bias'] = BehavioralMechanism(
            id='confirmation_bias',
            name='Confirmation bias',
            bias=BehavioralBias.CONFIRMATION_BIAS,
            description='Investors seek information that confirms existing beliefs',
            source='Behavioral finance literature',
            expected_sharpe=0.2,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='High',
            data_requirements=['News sentiment', 'social media data']
        )
        
        self.mechanisms['availability_bias'] = BehavioralMechanism(
            id='availability_bias',
            name='Availability bias',
            bias=BehavioralBias.AVAILABILITY_BIAS,
            description='Investors judge probability by ease of recall',
            source='Tversky & Kahneman 1973',
            expected_sharpe=0.2,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='High',
            data_requirements=['News frequency', 'media coverage']
        )
        
        self.mechanisms['mental_accounting'] = BehavioralMechanism(
            id='mental_accounting',
            name='Mental accounting',
            bias=BehavioralBias.MENTAL_ACCOUNTING,
            description='Investors treat money differently based on source or use',
            source='Thaler 1985',
            expected_sharpe=0.2,
            expected_capacity='Medium',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Portfolio composition', 'trading patterns']
        )
        
        self.mechanisms['status_quo_bias'] = BehavioralMechanism(
            id='status_quo_bias',
            name='Status quo bias',
            bias=BehavioralBias.STATUS_QUO_BIAS,
            description='Investors prefer current holdings, leading to under-trading',
            source='Samuelson & Zeckhauser 1988',
            expected_sharpe=0.2,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Portfolio turnover', 'holding periods']
        )
    
    def get_mechanism(self, mechanism_id: str) -> Optional[BehavioralMechanism]:
        """Get a mechanism by ID."""
        return self.mechanisms.get(mechanism_id)
    
    def get_mechanisms_by_bias(self, bias: BehavioralBias) -> List[BehavioralMechanism]:
        """Get mechanisms by bias type."""
        return [m for m in self.mechanisms.values() if m.bias == bias]
    
    def get_highest_sharpe_mechanisms(self, n: int = 5) -> List[BehavioralMechanism]:
        """Get top N mechanisms by expected Sharpe."""
        sorted_mechanisms = sorted(
            self.mechanisms.values(),
            key=lambda x: x.expected_sharpe,
            reverse=True
        )
        return sorted_mechanisms[:n]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("BEHAVIORAL MECHANISMS CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Mechanisms: {len(self.mechanisms)}")
        
        print(f"\nBy Bias Type:")
        for bias in BehavioralBias:
            count = len(self.get_mechanisms_by_bias(bias))
            if count > 0:
                print(f"  {bias.value}: {count}")
        
        print(f"\nTop 5 by Expected Sharpe:")
        top_5 = self.get_highest_sharpe_mechanisms(5)
        print(f"{'ID':<25} {'Name':<40} {'Sharpe':<10} {'Capacity':<20}")
        print("-" * 100)
        for mech in top_5:
            print(f"{mech.id:<25} {mech.name:<40} {mech.expected_sharpe:<10.2f} {mech.expected_capacity:<20}")
        
        print("\n" + "="*80)


def sample_behavioral_mechanisms_catalog():
    """Demonstrate behavioral mechanisms catalog."""
    print("=== Behavioral Mechanisms Catalog Demo ===\n")
    
    catalog = BehavioralMechanismsCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Behavioral Mechanisms Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of 10 behavioral mechanisms")
    print("- Classification by bias type (herding, disposition, overconfidence, etc.)")
    print("- Expected Sharpe, capacity, decay, and difficulty ratings")
    print("- Data requirements for each mechanism")
    print("- Source attribution for each mechanism")


if __name__ == "__main__":
    sample_behavioral_mechanisms_catalog()
