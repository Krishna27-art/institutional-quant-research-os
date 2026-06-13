"""
Research Agents for Institutional Quant System

This module implements AI-powered research agents for automated alpha discovery
and hypothesis generation as specified in the V4 Institutional Architecture.

Key Features:
- PaperReader: LLM-based paper ingestion and hypothesis extraction
- HypothesisGenerator: AI-driven alpha hypothesis generation
- Expected Research Output: 5 alphas/week (vs 1 alpha/month manual)
- Integration with feature factory and backtester

Based on V4 Blueprint - Institutional Architecture
Priority: Medium (Phase 2)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypothesisType(Enum):
    """Types of alpha hypotheses."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    MICROSTRUCTURE = "microstructure"
    FLOW = "flow"
    REGIME = "regime"
    CROSS_ASSET = "cross_asset"


@dataclass
class Hypothesis:
    """Alpha hypothesis from research."""
    name: str
    description: str
    hypothesis_type: HypothesisType
    features: List[str]
    expected_sharpe: float
    expected_capacity_cr: float
    source: str
    confidence: float
    created_at: datetime
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Paper:
    """Academic paper information."""
    title: str
    authors: List[str]
    year: int
    journal: str
    abstract: str
    url: str
    hypotheses: List[Hypothesis]


class PaperReader:
    """
    LLM-based paper ingestion and hypothesis extraction.
    
    This agent reads academic papers and extracts actionable alpha hypotheses.
    """
    
    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        self.hypotheses: List[Hypothesis] = []
    
    def ingest_paper(
        self,
        title: str,
        authors: List[str],
        year: int,
        journal: str,
        abstract: str,
        url: str = ""
    ) -> Paper:
        """
        Ingest a paper and extract hypotheses.
        
        Args:
            title: Paper title
            authors: List of authors
            year: Publication year
            journal: Journal name
            abstract: Paper abstract
            url: Paper URL
            
        Returns:
            Paper object with extracted hypotheses
        """
        paper = Paper(
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            abstract=abstract,
            url=url,
            hypotheses=[]
        )
        
        # Extract hypotheses from abstract (simplified - in production, use LLM)
        extracted_hypotheses = self._extract_hypotheses_from_abstract(abstract, title)
        paper.hypotheses = extracted_hypotheses
        self.hypotheses.extend(extracted_hypotheses)
        
        self.papers[title] = paper
        
        logger.info(f"Ingested paper: {title}, extracted {len(extracted_hypotheses)} hypotheses")
        
        return paper
    
    def _extract_hypotheses_from_abstract(self, abstract: str, title: str) -> List[Hypothesis]:
        """
        Extract hypotheses from abstract (simplified - in production, use LLM).
        
        Args:
            abstract: Paper abstract
            title: Paper title
            
        Returns:
            List of extracted hypotheses
        """
        hypotheses = []
        
        # Simplified hypothesis extraction based on keywords
        # In production, this would use an LLM to properly extract hypotheses
        
        abstract_lower = abstract.lower()
        
        # Momentum hypothesis
        if any(word in abstract_lower for word in ['momentum', 'trend', 'continuation']):
            hypothesis = Hypothesis(
                name=f"{title}_momentum",
                description=f"Momentum-based alpha from {title}",
                hypothesis_type=HypothesisType.MOMENTUM,
                features=['returns_5d', 'returns_20d', 'returns_60d'],
                expected_sharpe=0.8,
                expected_capacity_cr=500,
                source=title,
                confidence=0.7,
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)
        
        # Mean reversion hypothesis
        if any(word in abstract_lower for word in ['reversion', 'contrarian', 'reversal']):
            hypothesis = Hypothesis(
                name=f"{title}_reversion",
                description=f"Mean reversion alpha from {title}",
                hypothesis_type=HypothesisType.MEAN_REVERSION,
                features=['rsi', 'bollinger_position', 'z_score'],
                expected_sharpe=0.6,
                expected_capacity_cr=300,
                source=title,
                confidence=0.6,
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)
        
        # Volatility hypothesis
        if any(word in abstract_lower for word in ['volatility', 'variance', 'risk']):
            hypothesis = Hypothesis(
                name=f"{title}_volatility",
                description=f"Volatility-based alpha from {title}",
                hypothesis_type=HypothesisType.VOLATILITY,
                features=['realized_vol_5d', 'realized_vol_20d', 'parkinson_vol'],
                expected_sharpe=0.7,
                expected_capacity_cr=200,
                source=title,
                confidence=0.65,
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)
        
        # Liquidity hypothesis
        if any(word in abstract_lower for word in ['liquidity', 'amihud', 'turnover']):
            hypothesis = Hypothesis(
                name=f"{title}_liquidity",
                description=f"Liquidity-based alpha from {title}",
                hypothesis_type=HypothesisType.LIQUIDITY,
                features=['amihud', 'turnover', 'spread'],
                expected_sharpe=0.5,
                expected_capacity_cr=150,
                source=title,
                confidence=0.5,
                created_at=datetime.now()
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def get_hypotheses_by_type(self, hypothesis_type: HypothesisType) -> List[Hypothesis]:
        """Get hypotheses by type."""
        return [h for h in self.hypotheses if h.hypothesis_type == hypothesis_type]
    
    def get_top_hypotheses(self, n: int = 10) -> List[Hypothesis]:
        """Get top hypotheses by confidence."""
        return sorted(self.hypotheses, key=lambda x: x.confidence, reverse=True)[:n]


class HypothesisGenerator:
    """
    AI-driven alpha hypothesis generation.
    
    This agent generates new alpha hypotheses based on:
    - Existing features
    - Market regimes
    - Research literature
    - Cross-asset relationships
    """
    
    def __init__(self):
        self.generated_hypotheses: List[Hypothesis] = []
        self.feature_catalog: Dict[str, List[str]] = {
            'price': ['returns_1d', 'returns_5d', 'returns_20d', 'returns_60d', 'log_return_5d'],
            'volume': ['rv_5d', 'rv_20d', 'volume_impulse_5d'],
            'volatility': ['realized_vol_5d', 'realized_vol_20d', 'parkinson_vol', 'garman_klass_vol'],
            'microstructure': ['spread', 'ofi', 'vwap_deviation'],
            'behavioral': ['ibs', 'close_position'],
            'liquidity': ['amihud', 'turnover']
        }
    
    def generate_momentum_hypotheses(self) -> List[Hypothesis]:
        """Generate momentum-based hypotheses."""
        hypotheses = []
        
        # Time-series momentum
        hypothesis = Hypothesis(
            name="ts_momentum_20d",
            description="Time-series momentum over 20-day window",
            hypothesis_type=HypothesisType.MOMENTUM,
            features=['returns_20d', 'returns_60d', 'volatility_20d'],
            expected_sharpe=0.9,
            expected_capacity_cr=600,
            source="HypothesisGenerator",
            confidence=0.75,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        # Cross-sectional momentum
        hypothesis = Hypothesis(
            name="cross_sectional_momentum",
            description="Cross-sectional momentum across stocks",
            hypothesis_type=HypothesisType.MOMENTUM,
            features=['returns_5d', 'sector_returns_5d', 'relative_strength'],
            expected_sharpe=0.8,
            expected_capacity_cr=800,
            source="HypothesisGenerator",
            confidence=0.70,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        self.generated_hypotheses.extend(hypotheses)
        return hypotheses
    
    def generate_mean_reversion_hypotheses(self) -> List[Hypothesis]:
        """Generate mean reversion hypotheses."""
        hypotheses = []
        
        # Bollinger band reversion
        hypothesis = Hypothesis(
            name="bollinger_reversion",
            description="Mean reversion using Bollinger bands",
            hypothesis_type=HypothesisType.MEAN_REVERSION,
            features=['z_score_20d', 'bollinger_position', 'rsi'],
            expected_sharpe=0.7,
            expected_capacity_cr=400,
            source="HypothesisGenerator",
            confidence=0.65,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        # Overnight reversion
        hypothesis = Hypothesis(
            name="overnight_reversion",
            description="Overnight return reversion",
            hypothesis_type=HypothesisType.MEAN_REVERSION,
            features=['overnight_return', 'intraday_return', 'gap'],
            expected_sharpe=0.6,
            expected_capacity_cr=300,
            source="HypothesisGenerator",
            confidence=0.60,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        self.generated_hypotheses.extend(hypotheses)
        return hypotheses
    
    def generate_volatility_hypotheses(self) -> List[Hypothesis]:
        """Generate volatility-based hypotheses."""
        hypotheses = []
        
        # Volatility carry
        hypothesis = Hypothesis(
            name="volatility_carry",
            description="Volatility carry strategy",
            hypothesis_type=HypothesisType.VOLATILITY,
            features=['realized_vol_5d', 'realized_vol_20d', 'iv'],
            expected_sharpe=0.8,
            expected_capacity_cr=250,
            source="HypothesisGenerator",
            confidence=0.70,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        # Volatility mean reversion
        hypothesis = Hypothesis(
            name="volatility_reversion",
            description="Volatility mean reversion",
            hypothesis_type=HypothesisType.VOLATILITY,
            features=['realized_vol_5d', 'realized_vol_60d', 'vol_z_score'],
            expected_sharpe=0.6,
            expected_capacity_cr=200,
            source="HypothesisGenerator",
            confidence=0.55,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        self.generated_hypotheses.extend(hypotheses)
        return hypotheses
    
    def generate_liquidity_hypotheses(self) -> List[Hypothesis]:
        """Generate liquidity-based hypotheses."""
        hypotheses = []
        
        # Amihud illiquidity
        hypothesis = Hypothesis(
            name="amihud_illiquidity",
            description="Amihud illiquidity premium",
            hypothesis_type=HypothesisType.LIQUIDITY,
            features=['amihud', 'turnover', 'spread'],
            expected_sharpe=0.5,
            expected_capacity_cr=150,
            source="HypothesisGenerator",
            confidence=0.50,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        # Order flow imbalance
        hypothesis = Hypothesis(
            name="ofi_momentum",
            description="Order flow imbalance momentum",
            hypothesis_type=HypothesisType.LIQUIDITY,
            features=['ofi', 'vwap_deviation', 'volume_impulse'],
            expected_sharpe=0.6,
            expected_capacity_cr=200,
            source="HypothesisGenerator",
            confidence=0.55,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        self.generated_hypotheses.extend(hypotheses)
        return hypotheses
    
    def generate_regime_hypotheses(self) -> List[Hypothesis]:
        """Generate regime-based hypotheses."""
        hypotheses = []
        
        # Regime-aware momentum
        hypothesis = Hypothesis(
            name="regime_momentum",
            description="Regime-aware momentum strategy",
            hypothesis_type=HypothesisType.REGIME,
            features=['returns_20d', 'regime_state', 'regime_duration'],
            expected_sharpe=1.0,
            expected_capacity_cr=500,
            source="HypothesisGenerator",
            confidence=0.80,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        # Regime rotation
        hypothesis = Hypothesis(
            name="regime_rotation",
            description="Sector rotation based on regime",
            hypothesis_type=HypothesisType.REGIME,
            features=['regime_state', 'sector_momentum', 'sector_exposure'],
            expected_sharpe=0.7,
            expected_capacity_cr=400,
            source="HypothesisGenerator",
            confidence=0.65,
            created_at=datetime.now()
        )
        hypotheses.append(hypothesis)
        
        self.generated_hypotheses.extend(hypotheses)
        return hypotheses
    
    def generate_all_hypotheses(self) -> List[Hypothesis]:
        """Generate all hypothesis types."""
        all_hypotheses = []
        
        all_hypotheses.extend(self.generate_momentum_hypotheses())
        all_hypotheses.extend(self.generate_mean_reversion_hypotheses())
        all_hypotheses.extend(self.generate_volatility_hypotheses())
        all_hypotheses.extend(self.generate_liquidity_hypotheses())
        all_hypotheses.extend(self.generate_regime_hypotheses())
        
        logger.info(f"Generated {len(all_hypotheses)} hypotheses")
        
        return all_hypotheses
    
    def get_hypothesis_report(self) -> pd.DataFrame:
        """Generate hypothesis report as DataFrame."""
        data = []
        
        for h in self.generated_hypotheses:
            data.append({
                'name': h.name,
                'type': h.hypothesis_type.value,
                'expected_sharpe': h.expected_sharpe,
                'expected_capacity_cr': h.expected_capacity_cr,
                'confidence': h.confidence,
                'source': h.source,
                'created_at': h.created_at
            })
        
        return pd.DataFrame(data)


class ResearchAgentManager:
    """
    Manager for research agents.
    
    Coordinates PaperReader and HypothesisGenerator to produce
    a continuous stream of alpha hypotheses.
    """
    
    def __init__(self):
        self.paper_reader = PaperReader()
        self.hypothesis_generator = HypothesisGenerator()
        self.all_hypotheses: List[Hypothesis] = []
    
    def run_research_cycle(self) -> List[Hypothesis]:
        """
        Run a complete research cycle.
        
        Returns:
            List of new hypotheses
        """
        # Generate hypotheses from HypothesisGenerator
        generated = self.hypothesis_generator.generate_all_hypotheses()
        
        # Combine with paper-based hypotheses
        paper_hypotheses = self.paper_reader.hypotheses
        
        all_hypotheses = generated + paper_hypotheses
        self.all_hypotheses = all_hypotheses
        
        logger.info(f"Research cycle complete: {len(all_hypotheses)} total hypotheses")
        
        return all_hypotheses
    
    def get_research_summary(self) -> Dict:
        """Get research summary."""
        return {
            'total_hypotheses': len(self.all_hypotheses),
            'papers_ingested': len(self.paper_reader.papers),
            'by_type': {
                htype.value: len([h for h in self.all_hypotheses if h.hypothesis_type == htype])
                for htype in HypothesisType
            },
            'avg_confidence': np.mean([h.confidence for h in self.all_hypotheses]) if self.all_hypotheses else 0.0,
            'avg_expected_sharpe': np.mean([h.expected_sharpe for h in self.all_hypotheses]) if self.all_hypotheses else 0.0
        }


def sample_research_agents():
    """Demonstrate research agents."""
    print("=== Research Agents Demo ===\n")
    
    manager = ResearchAgentManager()
    
    # Ingest sample papers
    print("Ingesting sample papers...")
    manager.paper_reader.ingest_paper(
        title="Momentum in Indian Markets",
        authors=["A. Kumar", "B. Singh"],
        year=2023,
        journal="Journal of Financial Economics",
        abstract="This paper examines momentum effects in Indian equity markets. We find significant momentum over 20-day windows with Sharpe ratios of 0.8.",
        url="https://example.com/paper1"
    )
    
    manager.paper_reader.ingest_paper(
        title="Volatility Carry Strategies",
        authors=["C. Sharma", "D. Patel"],
        year=2022,
        journal="Quantitative Finance",
        abstract="We analyze volatility carry strategies using realized volatility and implied volatility. The strategy achieves Sharpe of 0.7 with capacity of ₹250 Cr.",
        url="https://example.com/paper2"
    )
    
    # Run research cycle
    print("\nRunning research cycle...")
    hypotheses = manager.run_research_cycle()
    
    # Print summary
    summary = manager.get_research_summary()
    print("\nResearch Summary:")
    print(f"  Total Hypotheses: {summary['total_hypotheses']}")
    print(f"  Papers Ingested: {summary['papers_ingested']}")
    print(f"  By Type: {summary['by_type']}")
    print(f"  Avg Confidence: {summary['avg_confidence']:.2f}")
    print(f"  Avg Expected Sharpe: {summary['avg_expected_sharpe']:.2f}")
    
    # Print top hypotheses
    print("\nTop 5 Hypotheses:")
    top_hypotheses = sorted(hypotheses, key=lambda x: x.confidence, reverse=True)[:5]
    for i, h in enumerate(top_hypotheses, 1):
        print(f"  {i}. {h.name}")
        print(f"     Type: {h.hypothesis_type.value}")
        print(f"     Expected Sharpe: {h.expected_sharpe:.2f}")
        print(f"     Confidence: {h.confidence:.2f}")
    
    # Generate hypothesis report
    print("\nHypothesis Report:")
    report = manager.hypothesis_generator.get_hypothesis_report()
    print(report.to_string(index=False))
    
    print("\n=== Research Agents Demo Complete ===")
    print("Expected Research Output: 5 alphas/week (vs 1 alpha/month manual)")


if __name__ == "__main__":
    sample_research_agents()
