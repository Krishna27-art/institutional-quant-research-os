"""
Automated Paper-to-Hypothesis Pipeline
Based on the critique: Extract hypotheses from papers, don't just collect them

Objective:
- Convert research papers into testable hypotheses
- Extract mechanisms and market inefficiencies
- Generate feature candidates
- Create signal candidates
- Automate the research workflow

Pipeline:
Paper → Extract Hypothesis → Generate Features → Backtest → Walk-forward → Paper Trade → Scale
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import re
    from collections import Counter
    TEXT_PROCESSING_AVAILABLE = True
except ImportError:
    TEXT_PROCESSING_AVAILABLE = False


class HypothesisType(Enum):
    """Types of hypotheses extracted from papers."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    STRUCTURAL = "structural"
    MICROSTRUCTURE = "microstructure"


class HypothesisStatus(Enum):
    """Status of hypothesis in pipeline."""
    EXTRACTED = "extracted"
    FEATURE_GENERATED = "feature_generated"
    SIGNAL_GENERATED = "signal_generated"
    BACKTESTED = "backtested"
    WALK_FORWARD = "walk_forward"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    KILLED = "killed"


@dataclass
class Paper:
    """Research paper metadata."""
    title: str
    authors: List[str]
    year: int
    journal: str
    url: str
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass
class ExtractedHypothesis:
    """Hypothesis extracted from a paper."""
    id: str
    paper_id: str
    hypothesis_text: str
    hypothesis_type: HypothesisType
    mechanism: str  # Why it should work
    market_inefficiency: str  # What inefficiency it exploits
    implementation_details: str
    required_data: List[str]
    status: HypothesisStatus = HypothesisStatus.EXTRACTED
    created_at: datetime = field(default_factory=datetime.now)
    
    # Generated components
    feature_candidates: List[str] = field(default_factory=list)
    signal_candidates: List[str] = field(default_factory=list)
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    information_coefficient: float = 0.0


class PaperToHypothesisPipeline:
    """
    Automated pipeline to convert research papers into testable hypotheses.
    
    Process:
    1. Extract paper metadata
    2. Identify hypotheses
    3. Extract mechanisms and inefficiencies
    4. Generate feature candidates
    5. Generate signal candidates
    6. Add to research queue
    """
    
    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        self.hypotheses: Dict[str, ExtractedHypothesis] = {}
        
        # Hypothesis templates
        self.hypothesis_templates = {
            HypothesisType.MOMENTUM: [
                "Assets with {metric} over {period} tend to continue in the same direction",
                "Past {period} returns predict future returns with {direction} persistence"
            ],
            HypothesisType.MEAN_REVERSION: [
                "Assets with extreme {metric} tend to revert to mean over {period}",
                "Deviations from {metric} predict reversions over {period}"
            ],
            HypothesisType.VOLATILITY: [
                "High volatility predicts {direction} returns over {period}",
                "Volatility clustering leads to {direction} price movements"
            ],
            HypothesisType.LIQUIDITY: [
                "Low liquidity stocks have {direction} returns due to {mechanism}",
                "Liquidity shocks predict {direction} price movements"
            ]
        }
    
    def add_paper(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        year: int,
        journal: str,
        url: str,
        abstract: str = "",
        keywords: List[str] = None
    ) -> None:
        """Add a paper to the database."""
        paper = Paper(
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            url=url,
            abstract=abstract,
            keywords=keywords or []
        )
        self.papers[paper_id] = paper
    
    def extract_hypotheses_from_paper(self, paper_id: str) -> List[ExtractedHypothesis]:
        """
        Extract hypotheses from a paper.
        
        In production, would use NLP to extract hypotheses from text.
        For now, uses keyword matching and templates.
        """
        if paper_id not in self.papers:
            return []
        
        paper = self.papers[paper_id]
        hypotheses = []
        
        # Analyze abstract for keywords
        abstract_lower = paper.abstract.lower()
        
        # Detect hypothesis type from keywords
        hypothesis_type = self._detect_hypothesis_type(abstract_lower)
        
        # Generate hypothesis text
        hypothesis_text = self._generate_hypothesis_text(hypothesis_type, abstract_lower)
        
        # Extract mechanism
        mechanism = self._extract_mechanism(abstract_lower)
        
        # Extract market inefficiency
        inefficiency = self._extract_inefficiency(abstract_lower)
        
        # Generate implementation details
        implementation = self._generate_implementation_details(hypothesis_type)
        
        # Determine required data
        required_data = self._determine_required_data(hypothesis_type)
        
        # Create hypothesis
        hypothesis_id = f"{paper_id}_{hypothesis_type.value}_{datetime.now().strftime('%Y%m%d')}"
        
        hypothesis = ExtractedHypothesis(
            id=hypothesis_id,
            paper_id=paper_id,
            hypothesis_text=hypothesis_text,
            hypothesis_type=hypothesis_type,
            mechanism=mechanism,
            market_inefficiency=inefficiency,
            implementation_details=implementation,
            required_data=required_data
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _detect_hypothesis_type(self, text: str) -> HypothesisType:
        """Detect hypothesis type from text keywords."""
        keywords = {
            HypothesisType.MOMENTUM: ['momentum', 'trend', 'continuation', 'persistence', 'trend-following'],
            HypothesisType.MEAN_REVERSION: ['reversion', 'mean', 'contrarian', 'revert', 'oversold', 'overbought'],
            HypothesisType.VOLATILITY: ['volatility', 'variance', 'risk', 'vol', 'uncertainty'],
            HypothesisType.LIQUIDITY: ['liquidity', 'volume', 'trading', 'turnover', 'illiquid'],
            HypothesisType.SENTIMENT: ['sentiment', 'news', 'social', 'media', 'investor'],
            HypothesisType.STRUCTURAL: ['structure', 'calendar', 'seasonal', 'cycle', 'pattern']
        }
        
        scores = {}
        for htype, hkeywords in keywords.items():
            score = sum(1 for kw in hkeywords if kw in text)
            scores[htype] = score
        
        # Return type with highest score
        if scores:
            return max(scores, key=scores.get)
        return HypothesisType.MOMENTUM  # Default
    
    def _generate_hypothesis_text(self, htype: HypothesisType, text: str) -> str:
        """Generate hypothesis text using templates."""
        templates = self.hypothesis_templates.get(htype, [])
        if templates:
            return templates[0]
        return f"Hypothesis of type {htype.value}"
    
    def _extract_mechanism(self, text: str) -> str:
        """Extract mechanism from text."""
        # Simplified extraction
        if 'behavioral' in text or 'bias' in text:
            return "Behavioral bias causes mispricing"
        elif 'risk' in text:
            return "Risk premium for bearing uncertainty"
        elif 'information' in text:
            return "Information diffusion causes delayed reaction"
        elif 'liquidity' in text:
            return "Liquidity constraints create price pressure"
        else:
            return "Market inefficiency due to structural factors"
    
    def _extract_inefficiency(self, text: str) -> str:
        """Extract market inefficiency from text."""
        if 'momentum' in text:
            return "Underreaction to information"
        elif 'reversion' in text:
            return "Overreaction to information"
        elif 'volatility' in text:
            return "Mispricing of risk"
        elif 'liquidity' in text:
            return "Liquidity premium"
        else:
            return "General market inefficiency"
    
    def _generate_implementation_details(self, htype: HypothesisType) -> str:
        """Generate implementation details for hypothesis type."""
        details = {
            HypothesisType.MOMENTUM: "Use past returns over lookback period, rank stocks, go long top decile",
            HypothesisType.MEAN_REVERSION: "Use deviation from mean, identify extremes, take contrarian position",
            HypothesisType.VOLATILITY: "Use realized volatility, identify high/low vol regimes, position accordingly",
            HypothesisType.LIQUIDITY: "Use trading volume or turnover, identify illiquid stocks, adjust position size"
        }
        return details.get(htype, "Standard implementation")
    
    def _determine_required_data(self, htype: HypothesisType) -> List[str]:
        """Determine required data for hypothesis type."""
        data_requirements = {
            HypothesisType.MOMENTUM: ['price', 'returns'],
            HypothesisType.MEAN_REVERSION: ['price', 'returns', 'moving averages'],
            HypothesisType.VOLATILITY: ['price', 'returns', 'volume'],
            HypothesisType.LIQUIDITY: ['price', 'volume', 'turnover'],
            HypothesisType.SENTIMENT: ['price', 'news', 'social media'],
            HypothesisType.STRUCTURAL: ['price', 'calendar', 'earnings dates']
        }
        return data_requirements.get(htype, ['price'])
    
    def generate_feature_candidates(self, hypothesis_id: str) -> List[str]:
        """Generate feature candidates for a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return []
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        # Generate features based on hypothesis type
        feature_generators = {
            HypothesisType.MOMENTUM: [
                'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
                'price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m'
            ],
            HypothesisType.MEAN_REVERSION: [
                'rsi_14', 'rsi_30', 'bollinger_position', 'deviation_from_mean',
                'z_score_20d', 'z_score_60d'
            ],
            HypothesisType.VOLATILITY: [
                'realized_volatility_5d', 'realized_volatility_20d', 'volatility_ratio',
                'atr', 'volatility_regime', 'vol_skew'
            ],
            HypothesisType.LIQUIDITY: [
                'volume_ratio', 'turnover', 'liquidity_ratio', 'bid_ask_spread',
                'depth_imbalance', 'participation_rate'
            ]
        }
        
        features = feature_generators.get(hypothesis.hypothesis_type, [])
        hypothesis.feature_candidates = features
        
        return features
    
    def generate_signal_candidates(self, hypothesis_id: str) -> List[str]:
        """Generate signal candidates for a hypothesis."""
        if hypothesis_id not in self.hypotheses:
            return []
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        # Generate signals based on hypothesis type
        signal_generators = {
            HypothesisType.MOMENTUM: [
                'cross_sectional_momentum', 'time_series_momentum', 'momentum_factor'
            ],
            HypothesisType.MEAN_REVERSION: [
                'mean_reversion_signal', 'contrarian_signal', 'bollinger_reversion'
            ],
            HypothesisType.VOLATILITY: [
                'volatility_breakout', 'volatility_mean_reversion', 'volatility_scaling'
            ],
            HypothesisType.LIQUIDITY: [
                'liquidity_premium_signal', 'volume_weighted_signal', 'turnover_signal'
            ]
        }
        
        signals = signal_generators.get(hypothesis.hypothesis_type, [])
        hypothesis.signal_candidates = signals
        
        return signals
    
    def get_hypothesis_summary(self) -> pd.DataFrame:
        """Get summary of all extracted hypotheses."""
        data = []
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            paper = self.papers.get(hypothesis.paper_id)
            
            data.append({
                'Hypothesis ID': hypothesis_id,
                'Paper': paper.title if paper else "Unknown",
                'Type': hypothesis.hypothesis_type.value,
                'Status': hypothesis.status.value,
                'Mechanism': hypothesis.mechanism,
                'Inefficiency': hypothesis.market_inefficiency,
                'Features': len(hypothesis.feature_candidates),
                'Signals': len(hypothesis.signal_candidates),
                'Sharpe': hypothesis.sharpe_ratio,
                'Created': hypothesis.created_at.strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(data)
    
    def get_research_queue(self) -> List[ExtractedHypothesis]:
        """Get hypotheses ready for research (EXTRACTED status)."""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.EXTRACTED
        ]
    
    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        sharpe: float = 0.0,
        max_dd: float = 0.0,
        ic: float = 0.0
    ) -> None:
        """Update hypothesis status after testing."""
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].status = status
            self.hypotheses[hypothesis_id].sharpe_ratio = sharpe
            self.hypotheses[hypothesis_id].max_drawdown = max_dd
            self.hypotheses[hypothesis_id].information_coefficient = ic


if __name__ == "__main__":
    # Test the Paper-to-Hypothesis Pipeline
    print("Testing Paper-to-Hypothesis Pipeline...")
    
    pipeline = PaperToHypothesisPipeline()
    
    # Add sample papers
    print("\nAdding sample papers...")
    pipeline.add_paper(
        paper_id="jegadeesh1993",
        title="Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency",
        authors=["Narasimhan Jegadeesh", "Sheridan Titman"],
        year=1993,
        journal="Journal of Finance",
        url="https://example.com/jegadeesh1993",
        abstract="This paper documents that strategies which buy stocks that have performed well in the past and sell stocks that have performed poorly earn significant abnormal returns over the subsequent 3 to 12 months.",
        keywords=["momentum", "returns", "anomaly"]
    )
    
    pipeline.add_paper(
        paper_id="de2009",
        title="The Cross-Section of Volatility and Expected Returns",
        authors=["Andrew Ang", "Robert Hodrick", "Yuhang Xing", "Xiaoyan Zhang"],
        year=2009,
        journal="Journal of Finance",
        url="https://example.com/de2009",
        abstract="We investigate the cross-sectional relation between volatility and expected returns. We find that stocks with high idiosyncratic volatility have low average returns.",
        keywords=["volatility", "returns", "cross-section"]
    )
    
    # Extract hypotheses
    print("\nExtracting hypotheses from papers...")
    for paper_id in pipeline.papers.keys():
        hypotheses = pipeline.extract_hypotheses_from_paper(paper_id)
        print(f"Extracted {len(hypotheses)} hypotheses from {paper_id}")
        
        for hypothesis in hypotheses:
            # Generate feature candidates
            features = pipeline.generate_feature_candidates(hypothesis.id)
            print(f"  Generated {len(features)} feature candidates")
            
            # Generate signal candidates
            signals = pipeline.generate_signal_candidates(hypothesis.id)
            print(f"  Generated {len(signals)} signal candidates")
    
    # Get hypothesis summary
    print("\nHypothesis Summary:")
    summary = pipeline.get_hypothesis_summary()
    print(summary.to_string(index=False))
    
    # Get research queue
    print("\nResearch Queue:")
    queue = pipeline.get_research_queue()
    print(f"Hypotheses ready for research: {len(queue)}")
    for h in queue:
        print(f"  - {h.id}: {h.hypothesis_text}")
