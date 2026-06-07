"""
Market Inefficiencies Catalog - 20 Specific Inefficiencies Across Time Scales

This module implements 20 specific market inefficiencies across different
time scales as described in the research intelligence system.

Based on comprehensive research literature and empirical studies.
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


class TimeScale(Enum):
    """Time scale categories."""
    SUB_SECOND = "sub_second"
    ULTRA_SHORT = "ultra_short"
    SHORT = "short"
    INTRADAY = "intraday"
    DAILY_WEEKLY = "daily_weekly"
    MONTHLY_QUARTERLY = "monthly_quarterly"
    ANNUAL_PLUS = "annual_plus"


class InefficiencyType(Enum):
    """Types of market inefficiencies."""
    QUEUE_PRIORITY = "queue_priority"
    ORDER_FLOW_TOXICITY = "order_flow_toxicity"
    MICRO_ALPHA_QUEUE = "micro_alpha_queue"
    CANCEL_RATIO = "cancel_ratio"
    HIDDEN_ORDER = "hidden_order"
    VWAP_REVERSION = "vwap_reversion"
    ORB = "orb"
    ORDER_FLOW_AGGREGATED = "order_flow_aggregated"
    INFORMATION_LEAKAGE = "information_leakage"
    INTRADAY_MOMENTUM = "intraday_momentum"
    FIRST_HOUR_PREDICTS = "first_hour_predicts"
    VWAP_TREND = "vwap_trend"
    SECTOR_MOMENTUM = "sector_momentum"
    PEAD = "pead"
    SHORT_TERM_REVERSAL = "short_term_reversal"
    CALENDAR_EFFECTS = "calendar_effects"
    ANALYST_RECOMMENDATION = "analyst_recommendation"
    INSIDER_TRADING = "insider_trading"
    MUTUAL_FUND_FLOW = "mutual_fund_flow"
    MOMENTUM = "momentum"
    LOW_VOLATILITY = "low_volatility"
    VALUE = "value"
    QUALITY = "quality"
    SEASONALITY = "seasonality"
    OPTION_TERM_STRUCTURE = "option_term_structure"
    CROSS_ASSET_MOMENTUM = "cross_asset_momentum"


@dataclass
class Inefficiency:
    """Market inefficiency definition."""
    id: str
    name: str
    time_scale: TimeScale
    type: InefficiencyType
    description: str
    source: str
    expected_sharpe: float
    expected_capacity: str
    decay: str
    difficulty: str
    data_requirements: List[str]


class MarketInefficienciesCatalog:
    """
    Catalog of 20 market inefficiencies across time scales.
    
    This class provides a comprehensive catalog of market inefficiencies
    with their characteristics and implementation requirements.
    """
    
    def __init__(self):
        """Initialize market inefficiencies catalog."""
        self.inefficiencies: Dict[str, Inefficiency] = {}
        self._initialize_catalog()
        
        logger.info(f"MarketInefficienciesCatalog initialized with {len(self.inefficiencies)} inefficiencies")
    
    def _initialize_catalog(self) -> None:
        """Initialize the catalog with 20 inefficiencies."""
        
        # Sub-Second (0-1 sec)
        self.inefficiencies['queue_priority'] = Inefficiency(
            id='queue_priority',
            name='Queue position priority edge',
            time_scale=TimeScale.SUB_SECOND,
            type=InefficiencyType.QUEUE_PRIORITY,
            description='Orders at same price executed by queue order; earlier cancellation detection reveals future flow',
            source='HRT, Jane Street (public talks)',
            expected_sharpe=0.8,
            expected_capacity='Low (prop only)',
            decay='Live (months)',
            difficulty='Very High',
            data_requirements=['Full order book (L2/L3)', 'order IDs']
        )
        
        self.inefficiencies['order_flow_toxicity'] = Inefficiency(
            id='order_flow_toxicity',
            name='Order flow toxicity (VPIN)',
            time_scale=TimeScale.ULTRA_SHORT,
            type=InefficiencyType.ORDER_FLOW_TOXICITY,
            description='Toxic flow predicts near-term volatility and adverse selection risk',
            source='Easley et al. 2012, recent crypto validation 2025',
            expected_sharpe=0.3,
            expected_capacity='Medium-High (index futures)',
            decay='Moderate (years)',
            difficulty='Medium',
            data_requirements=['Volume-synchronized buckets', 'aggressive trade classification']
        )
        
        self.inefficiencies['micro_alpha_queue'] = Inefficiency(
            id='micro_alpha_queue',
            name='Micro-α from order book queue imbalance',
            time_scale=TimeScale.ULTRA_SHORT,
            type=InefficiencyType.MICRO_ALPHA_QUEUE,
            description='Conditioned probability shift in next price direction from queue dynamics (100ms–10s)',
            source='HTX Research 2025 (published), Chinese HFT literature',
            expected_sharpe=1.0,
            expected_capacity='Low (can be scaled with FPGA)',
            decay='Live (months)',
            difficulty='Very High',
            data_requirements=['10-level LOB updates at 100ms resolution']
        )
        
        self.inefficiencies['cancel_ratio'] = Inefficiency(
            id='cancel_ratio',
            name='Cancel ratio (CR) as spoofing signal',
            time_scale=TimeScale.ULTRA_SHORT,
            type=InefficiencyType.CANCEL_RATIO,
            description='Fake large orders placed and quickly cancelled manipulate perception; CR deviation predicts price reversal',
            source='HTX Research 2025',
            expected_sharpe=0.5,
            expected_capacity='Low (prop)',
            decay='Moderate (years)',
            difficulty='High',
            data_requirements=['Full order lifecycle (place, modify, cancel)']
        )
        
        self.inefficiencies['hidden_order'] = Inefficiency(
            id='hidden_order',
            name='Hidden/iceberg order detection',
            time_scale=TimeScale.ULTRA_SHORT,
            type=InefficiencyType.HIDDEN_ORDER,
            description='Large hidden orders reveal institutional demand; liquidity discovery as a signal',
            source='HRT, Citadel Securities micro-structure',
            expected_sharpe=0.6,
            expected_capacity='Low–Medium',
            decay='Live (months)',
            difficulty='Very High',
            data_requirements=['LOB depth with hidden order flags or inference from recurrence']
        )
        
        # Ultra-Short (1-60 sec)
        self.inefficiencies['vwap_reversion'] = Inefficiency(
            id='vwap_reversion',
            name='VWAP reversion',
            time_scale=TimeScale.SHORT,
            type=InefficiencyType.VWAP_REVERSION,
            description='Price tends to revert to volume-weighted average after deviations',
            source='Zarattini & Aziz 2023 (SSRN), empirical trading blogs',
            expected_sharpe=0.4,
            expected_capacity='Very High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['VWAP calculation on 1-min bars', 'real-time volume']
        )
        
        self.inefficiencies['orb'] = Inefficiency(
            id='orb',
            name='Opening range breakout (ORB)',
            time_scale=TimeScale.SHORT,
            type=InefficiencyType.ORB,
            description='Opening range breakout followed when relative volume > 100%, filtered to top 20 stocks',
            source='Zarattini et al. 2024 (in-depth 7000-stock study)',
            expected_sharpe=0.6,
            expected_capacity='Medium',
            decay='Years',
            difficulty='Medium',
            data_requirements=['First-5-min volume vs 14-day avg', 'opening range high/low']
        )
        
        self.inefficiencies['order_flow_aggregated'] = Inefficiency(
            id='order_flow_aggregated',
            name='Order flow imbalance aggregated by volume buckets',
            time_scale=TimeScale.SHORT,
            type=InefficiencyType.ORDER_FLOW_AGGREGATED,
            description='Order flow imbalance aggregated by volume buckets (not time)',
            source='Microstructure literature',
            expected_sharpe=0.5,
            expected_capacity='High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['Order book data', 'volume bucketing']
        )
        
        self.inefficiencies['information_leakage'] = Inefficiency(
            id='information_leakage',
            name='Information leakage prior to scheduled announcements',
            time_scale=TimeScale.SHORT,
            type=InefficiencyType.INFORMATION_LEAKAGE,
            description='Information leakage prior to scheduled announcements',
            source='Empirical',
            expected_sharpe=0.4,
            expected_capacity='High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['Earnings calendar', 'pre-announcement price action']
        )
        
        # Intraday (1 hour - close)
        self.inefficiencies['intraday_momentum'] = Inefficiency(
            id='intraday_momentum',
            name='Intraday momentum',
            time_scale=TimeScale.INTRADAY,
            type=InefficiencyType.INTRADAY_MOMENTUM,
            description='Morning momentum persists',
            source='Empirical',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['Intraday price data', 'time-of-day analysis']
        )
        
        self.inefficiencies['first_hour_predicts'] = Inefficiency(
            id='first_hour_predicts',
            name='First-hour return predicts day trend',
            time_scale=TimeScale.INTRADAY,
            type=InefficiencyType.FIRST_HOUR_PREDICTS,
            description='First-hour return predicts day trend (Heston & Korajczyk, 2021)',
            source='Heston & Korajczyk 2021',
            expected_sharpe=0.4,
            expected_capacity='High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['First-hour returns', 'day returns']
        )
        
        self.inefficiencies['vwap_trend'] = Inefficiency(
            id='vwap_trend',
            name='VWAP trend following',
            time_scale=TimeScale.INTRADAY,
            type=InefficiencyType.VWAP_TREND,
            description='Price above VWAP signals continuation',
            source='Empirical',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Years',
            difficulty='Low',
            data_requirements=['Real-time VWAP', 'price data']
        )
        
        self.inefficiencies['sector_momentum'] = Inefficiency(
            id='sector_momentum',
            name='Sector momentum spillover',
            time_scale=TimeScale.INTRADAY,
            type=InefficiencyType.SECTOR_MOMENTUM,
            description='Leading sectors predict lagging; works best when volatility is low (VIX < 15)',
            source='Moskowitz & Grinblatt 1999; 2025 study confirms',
            expected_sharpe=0.4,
            expected_capacity='High',
            decay='Moderate',
            difficulty='Low',
            data_requirements=['Sector ETF returns', 'VIX']
        )
        
        # Daily - Weekly
        self.inefficiencies['pead'] = Inefficiency(
            id='pead',
            name='Post-earnings announcement drift (PEAD)',
            time_scale=TimeScale.DAILY_WEEKLY,
            type=InefficiencyType.PEAD,
            description='Market under-reacts to earnings surprises, especially when considering historical streaks',
            source='Bernard & Thomas 1990; revived via ML (2025)',
            expected_sharpe=0.3,
            expected_capacity='Very High',
            decay='Decades',
            difficulty='Medium',
            data_requirements=['Earnings data', 'analyst expectations', 'earnings call transcripts']
        )
        
        self.inefficiencies['short_term_reversal'] = Inefficiency(
            id='short_term_reversal',
            name='Short-term reversal (1-day / 1-week)',
            time_scale=TimeScale.DAILY_WEEKLY,
            type=InefficiencyType.SHORT_TERM_REVERSAL,
            description='Short-term reversal (1-day / 1-week)',
            source='Standard academic anomalies',
            expected_sharpe=0.3,
            expected_capacity='Medium',
            decay='Moderate',
            difficulty='Low',
            data_requirements=['Daily returns', 'reversal calculation']
        )
        
        self.inefficiencies['calendar_effects'] = Inefficiency(
            id='calendar_effects',
            name='Calendar effects (Turn-of-month, January, holiday drift)',
            time_scale=TimeScale.DAILY_WEEKLY,
            type=InefficiencyType.CALENDAR_EFFECTS,
            description='Persistent seasonal patterns not fully arbitraged',
            source='Standard academic anomalies',
            expected_sharpe=0.2,
            expected_capacity='Very High',
            decay='Decades',
            difficulty='Very Low',
            data_requirements=['Calendar data']
        )
        
        self.inefficiencies['analyst_recommendation'] = Inefficiency(
            id='analyst_recommendation',
            name='Analyst recommendation changes',
            time_scale=TimeScale.DAILY_WEEKLY,
            type=InefficiencyType.ANALYST_RECOMMENDATION,
            description='Under-reaction',
            source='Empirical',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['Analyst recommendations', 'price reaction data']
        )
        
        self.inefficiencies['insider_trading'] = Inefficiency(
            id='insider_trading',
            name='Insider trading filings',
            time_scale=TimeScale.DAILY_WEEKLY,
            type=InefficiencyType.INSIDER_TRADING,
            description='Delayed public response',
            source='Empirical',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Years',
            difficulty='Low',
            data_requirements=['SEC EDGAR API', 'Form 4 filings']
        )
        
        self.inefficiencies['mutual_fund_flow'] = Inefficiency(
            id='mutual_fund_flow',
            name='Mutual fund flow-induced price pressure',
            time_scale=TimeScale.DAILY_WEEKLY,
            type=InefficiencyType.MUTUAL_FUND_FLOW,
            description='Reverses over weeks',
            source='Empirical',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Years',
            difficulty='Medium',
            data_requirements=['Mutual fund flows', 'price data']
        )
        
        # Monthly - Quarterly
        self.inefficiencies['momentum'] = Inefficiency(
            id='momentum',
            name='Momentum (12-1)',
            time_scale=TimeScale.MONTHLY_QUARTERLY,
            type=InefficiencyType.MOMENTUM,
            description='Classic factor with consistent premium',
            source='Fama & French 1992; 2025 factor book',
            expected_sharpe=0.4,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Monthly returns', '12-1 momentum calculation']
        )
        
        self.inefficiencies['low_volatility'] = Inefficiency(
            id='low_volatility',
            name='Low volatility anomaly',
            time_scale=TimeScale.MONTHLY_QUARTERLY,
            type=InefficiencyType.LOW_VOLATILITY,
            description='Low-volatility stocks outperform high-volatility on risk-adjusted basis',
            source='Ang et al. 2006; confirmed in emerging markets 2024',
            expected_sharpe=0.3,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Historical volatility (12 months)']
        )
        
        self.inefficiencies['value'] = Inefficiency(
            id='value',
            name='Value (B/M, E/P, CF/P)',
            time_scale=TimeScale.MONTHLY_QUARTERLY,
            type=InefficiencyType.VALUE,
            description='Cheap stocks outperform expensive stocks over long horizons',
            source='Fama & French 1992; 2025 factor ranking',
            expected_sharpe=0.3,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Book value', 'market cap', 'earnings']
        )
        
        self.inefficiencies['quality'] = Inefficiency(
            id='quality',
            name='Quality (ROE, accruals, leverage)',
            time_scale=TimeScale.MONTHLY_QUARTERLY,
            type=InefficiencyType.QUALITY,
            description='Most robust factor across regimes; especially in high-volatility periods',
            source='Novy-Marx 2013; recent factor ranking studies',
            expected_sharpe=0.4,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Low',
            data_requirements=['Financial statements (ROE, ROA, debt/equity, accruals)']
        )
        
        self.inefficiencies['seasonality'] = Inefficiency(
            id='seasonality',
            name='Seasonality (Sell in May, Halloween effect)',
            time_scale=TimeScale.MONTHLY_QUARTERLY,
            type=InefficiencyType.SEASONALITY,
            description='Seasonal patterns',
            source='Empirical',
            expected_sharpe=0.2,
            expected_capacity='Very High',
            decay='Persistent',
            difficulty='Very Low',
            data_requirements=['Calendar data', 'seasonal patterns']
        )
        
        # Annual+
        self.inefficiencies['option_term_structure'] = Inefficiency(
            id='option_term_structure',
            name='Option implied volatility term structure',
            time_scale=TimeScale.ANNUAL_PLUS,
            type=InefficiencyType.OPTION_TERM_STRUCTURE,
            description='Different maturities exhibit different premia; term structure slope predicts vol curve moves',
            source='Empirical',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Full option surface', 'correlation matrix']
        )
        
        self.inefficiencies['cross_asset_momentum'] = Inefficiency(
            id='cross_asset_momentum',
            name='Cross-asset momentum (Commodities leading equities)',
            time_scale=TimeScale.ANNUAL_PLUS,
            type=InefficiencyType.CROSS_ASSET_MOMENTUM,
            description='Commodities leading equities',
            source='2026 network study',
            expected_sharpe=0.3,
            expected_capacity='High',
            decay='Persistent',
            difficulty='Medium',
            data_requirements=['Commodity data', 'equity data']
        )
    
    def get_inefficiency(self, inefficiency_id: str) -> Optional[Inefficiency]:
        """Get an inefficiency by ID."""
        return self.inefficiencies.get(inefficiency_id)
    
    def get_inefficiencies_by_time_scale(self, time_scale: TimeScale) -> List[Inefficiency]:
        """Get inefficiencies by time scale."""
        return [i for i in self.inefficiencies.values() if i.time_scale == time_scale]
    
    def get_inefficiencies_by_type(self, inefficiency_type: InefficiencyType) -> List[Inefficiency]:
        """Get inefficiencies by type."""
        return [i for i in self.inefficiencies.values() if i.type == inefficiency_type]
    
    def get_highest_sharpe_inefficiencies(self, n: int = 10) -> List[Inefficiency]:
        """Get top N inefficiencies by expected Sharpe."""
        sorted_inefficiencies = sorted(
            self.inefficiencies.values(),
            key=lambda x: x.expected_sharpe,
            reverse=True
        )
        return sorted_inefficiencies[:n]
    
    def print_catalog_report(self) -> None:
        """Print catalog report."""
        print("\n" + "="*80)
        print("MARKET INEFFICIENCIES CATALOG REPORT")
        print("="*80)
        
        print(f"\nTotal Inefficiencies: {len(self.inefficiencies)}")
        
        print(f"\nBy Time Scale:")
        for scale in TimeScale:
            count = len(self.get_inefficiencies_by_time_scale(scale))
            if count > 0:
                print(f"  {scale.value}: {count}")
        
        print(f"\nBy Type:")
        for itype in InefficiencyType:
            count = len(self.get_inefficiencies_by_type(itype))
            if count > 0:
                print(f"  {itype.value}: {count}")
        
        print(f"\nTop 10 by Expected Sharpe:")
        top_10 = self.get_highest_sharpe_inefficiencies(10)
        print(f"{'ID':<25} {'Name':<40} {'Sharpe':<10} {'Capacity':<20}")
        print("-" * 100)
        for ineff in top_10:
            print(f"{ineff.id:<25} {ineff.name:<40} {ineff.expected_sharpe:<10.2f} {ineff.expected_capacity:<20}")
        
        print("\n" + "="*80)


def sample_market_inefficiencies_catalog():
    """Demonstrate market inefficiencies catalog."""
    print("=== Market Inefficiencies Catalog Demo ===\n")
    
    catalog = MarketInefficienciesCatalog()
    catalog.print_catalog_report()
    
    print("\n=== Market Inefficiencies Catalog Demo Complete ===")
    print("Key capabilities:")
    print("- Catalog of 20 market inefficiencies across time scales")
    print("- Classification by time scale (sub-second to annual+)")
    print("- Classification by type (queue priority, VPIN, momentum, etc.)")
    print("- Expected Sharpe, capacity, decay, and difficulty ratings")
    print("- Data requirements for each inefficiency")
    print("- Source attribution for each inefficiency")


if __name__ == "__main__":
    sample_market_inefficiencies_catalog()
