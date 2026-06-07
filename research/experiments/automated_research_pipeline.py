"""
Automated Research Pipeline
Paper crawler → hypothesis → test → registry

Critical for institutional-grade research velocity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re


class HypothesisStatus(Enum):
    """Status of hypothesis testing"""
    PENDING = "pending"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class Hypothesis:
    """Research hypothesis"""
    id: str
    name: str
    description: str
    source: str  # paper, arXiv, SSRN, etc.
    source_url: Optional[str]
    created_at: datetime
    status: HypothesisStatus
    sharpe_oos: float
    sharpe_is: float
    capacity_cr: float
    is_profitable: bool
    notes: str


@dataclass
class TestResult:
    """Result of hypothesis test"""
    hypothesis_id: str
    test_date: datetime
    sharpe_is: float
    sharpe_oos: float
    max_drawdown: float
    win_rate: float
    avg_trade_pnl: float
    transaction_cost_bps: float
    capacity_cr: float
    passed: bool
    failure_reason: Optional[str]


class HypothesisRegistry:
    """
    Registry for tracking all tested hypotheses.
    
    Prevents re-testing failed ideas and learns from negative results.
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.test_results: List[TestResult] = []
    
    def add_hypothesis(self, hypothesis: Hypothesis):
        """Add a new hypothesis to registry"""
        self.hypotheses[hypothesis.id] = hypothesis
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID"""
        return self.hypotheses.get(hypothesis_id)
    
    def update_status(self, hypothesis_id: str, status: HypothesisStatus,
                     sharpe_oos: float, sharpe_is: float, capacity_cr: float):
        """Update hypothesis status after testing"""
        if hypothesis_id in self.hypotheses:
            self.hypotheses[hypothesis_id].status = status
            self.hypotheses[hypothesis_id].sharpe_oos = sharpe_oos
            self.hypotheses[hypothesis_id].sharpe_is = sharpe_is
            self.hypotheses[hypothesis_id].capacity_cr = capacity_cr
            self.hypotheses[hypothesis_id].is_profitable = (sharpe_oos > 0.5)
    
    def add_test_result(self, result: TestResult):
        """Add test result"""
        self.test_results.append(result)
    
    def get_passed_hypotheses(self) -> List[Hypothesis]:
        """Get all passed hypotheses"""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.PASSED]
    
    def get_failed_hypotheses(self) -> List[Hypothesis]:
        """Get all failed hypotheses"""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.FAILED]
    
    def get_negative_results(self) -> List[Hypothesis]:
        """Get negative results (failed hypotheses) for learning"""
        return self.get_failed_hypotheses()
    
    def check_duplicate(self, description: str) -> Optional[str]:
        """Check if similar hypothesis already exists"""
        for h in self.hypotheses.values():
            if h.description.lower() == description.lower():
                return h.id
        return None
    
    def generate_report(self) -> str:
        """Generate registry report"""
        total = len(self.hypotheses)
        passed = len(self.get_passed_hypotheses())
        failed = len(self.get_failed_hypotheses())
        pending = len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.PENDING])
        
        report = f"""
Hypothesis Registry Report
{'=' * 50}
Total Hypotheses: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Failed: {failed} ({failed/total*100:.1f}%)
Pending: {pending} ({pending/total*100:.1f}%)
Total Tests: {len(self.test_results)}

Passed Hypotheses:
{'-' * 50}
"""
        
        for h in self.get_passed_hypotheses():
            report += f"- {h.name}: Sharpe OOS={h.sharpe_oos:.2f}, Capacity={h.capacity_cr:.0f}Cr\n"
        
        report += f"\nFailed Hypotheses (Negative Results):\n{'-' * 50}\n"
        for h in self.get_failed_hypotheses():
            report += f"- {h.name}: {h.notes}\n"
        
        return report


class AutomatedResearchPipeline:
    """
    Automated Research Pipeline
    
    Pipeline: Paper Crawler → Hypothesis Extraction → Feature Generation → 
    Backtesting → Validation → Registry
    
    Automates the entire research process from idea to deployment.
    """
    
    def __init__(self, registry: HypothesisRegistry):
        self.registry = registry
        
        # Pipeline components
        self.paper_sources = ["arXiv", "SSRN", "NBER"]
        self.backtest_func: Optional[Callable] = None
        self.validation_func: Optional[Callable] = None
    
    def set_backtest_function(self, func: Callable):
        """Set backtesting function"""
        self.backtest_func = func
    
    def set_validation_function(self, func: Callable):
        """Set validation function"""
        self.validation_func = func
    
    def crawl_papers(self, keywords: List[str], max_papers: int = 10) -> List[Dict]:
        """
        Crawl papers from sources (placeholder for actual implementation).
        
        In production, would use arXiv API, SSRN API, etc.
        """
        papers = []
        
        # Placeholder: simulate paper crawling
        for i in range(min(max_papers, 5)):
            papers.append({
                "title": f"Paper {i+1} on {keywords[0]}",
                "authors": ["Author A", "Author B"],
                "source": "arXiv",
                "url": f"https://arxiv.org/abs/240{i+1}.00001",
                "abstract": f"This paper studies {keywords[0]} in financial markets..."
            })
        
        return papers
    
    def extract_hypotheses(self, paper: Dict) -> List[Hypothesis]:
        """
        Extract hypotheses from paper abstract.
        
        In production, would use NLP/LLM to extract testable hypotheses.
        """
        hypotheses = []
        
        # Placeholder: extract simple hypothesis from paper
        hypothesis = Hypothesis(
            id=f"hypo_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=paper["title"][:50],
            description=paper["abstract"][:200],
            source=paper["source"],
            source_url=paper["url"],
            created_at=datetime.now(),
            status=HypothesisStatus.PENDING,
            sharpe_oos=0.0,
            sharpe_is=0.0,
            capacity_cr=0.0,
            is_profitable=False,
            notes=""
        )
        
        # Check for duplicates
        if not self.registry.check_duplicate(hypothesis.description):
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def generate_features(self, hypothesis: Hypothesis) -> pd.DataFrame:
        """
        Generate features for hypothesis.
        
        In production, would use feature factory based on hypothesis type.
        """
        # Placeholder: generate random features
        np.random.seed(42)
        n = 2520  # 10 years
        features = pd.DataFrame({
            f"feature_{i}": np.random.randn(n) for i in range(20)
        })
        
        return features
    
    def run_backtest(self, hypothesis: Hypothesis, features: pd.DataFrame) -> Dict:
        """
        Run backtest for hypothesis.
        
        In production, would use actual backtesting engine.
        """
        if self.backtest_func:
            return self.backtest_func(hypothesis, features)
        
        # Placeholder: simulate backtest results
        return {
            "sharpe_is": np.random.uniform(0.5, 2.0),
            "sharpe_oos": np.random.uniform(-0.5, 1.5),
            "max_drawdown": np.random.uniform(5, 20),
            "win_rate": np.random.uniform(0.4, 0.6),
            "avg_trade_pnl": np.random.uniform(-10, 50),
            "transaction_cost_bps": np.random.uniform(10, 30),
            "capacity_cr": np.random.uniform(50, 500)
        }
    
    def validate(self, hypothesis: Hypothesis, backtest_result: Dict) -> bool:
        """
        Validate hypothesis against institutional criteria.
        
        Criteria:
        - OOS Sharpe > 0.5
        - Max drawdown < 15%
        - Win rate > 45%
        - Average trade PnL > 2x transaction cost
        - Capacity > 50 Cr
        """
        if self.validation_func:
            return self.validation_func(hypothesis, backtest_result)
        
        # Default validation criteria
        passed = (
            backtest_result["sharpe_oos"] > 0.5 and
            backtest_result["max_drawdown"] < 15.0 and
            backtest_result["win_rate"] > 0.45 and
            backtest_result["avg_trade_pnl"] > 2 * backtest_result["transaction_cost_bps"] and
            backtest_result["capacity_cr"] > 50.0
        )
        
        return passed
    
    def process_hypothesis(self, hypothesis: Hypothesis) -> TestResult:
        """
        Process a hypothesis through the full pipeline.
        
        Returns:
            TestResult
        """
        # Update status to testing
        hypothesis.status = HypothesisStatus.TESTING
        
        # Generate features
        features = self.generate_features(hypothesis)
        
        # Run backtest
        backtest_result = self.run_backtest(hypothesis, features)
        
        # Validate
        passed = self.validate(hypothesis, backtest_result)
        
        # Create test result
        result = TestResult(
            hypothesis_id=hypothesis.id,
            test_date=datetime.now(),
            sharpe_is=backtest_result["sharpe_is"],
            sharpe_oos=backtest_result["sharpe_oos"],
            max_drawdown=backtest_result["max_drawdown"],
            win_rate=backtest_result["win_rate"],
            avg_trade_pnl=backtest_result["avg_trade_pnl"],
            transaction_cost_bps=backtest_result["transaction_cost_bps"],
            capacity_cr=backtest_result["capacity_cr"],
            passed=passed,
            failure_reason=None if passed else "Did not meet validation criteria"
        )
        
        # Update registry
        status = HypothesisStatus.PASSED if passed else HypothesisStatus.FAILED
        self.registry.update_status(
            hypothesis.id,
            status,
            backtest_result["sharpe_oos"],
            backtest_result["sharpe_is"],
            backtest_result["capacity_cr"]
        )
        
        if not passed:
            hypothesis.notes = result.failure_reason
        
        self.registry.add_test_result(result)
        
        return result
    
    def run_pipeline(self, keywords: List[str], max_papers: int = 10) -> List[TestResult]:
        """
        Run the full automated research pipeline.
        
        Args:
            keywords: Keywords to search for papers
            max_papers: Maximum number of papers to process
        
        Returns:
            List of test results
        """
        results = []
        
        # Crawl papers
        papers = self.crawl_papers(keywords, max_papers)
        
        # Extract hypotheses
        for paper in papers:
            hypotheses = self.extract_hypotheses(paper)
            
            for hypothesis in hypotheses:
                # Add to registry
                self.registry.add_hypothesis(hypothesis)
                
                # Process hypothesis
                result = self.process_hypothesis(hypothesis)
                results.append(result)
        
        return results


if __name__ == "__main__":
    # Example usage
    registry = HypothesisRegistry()
    pipeline = AutomatedResearchPipeline(registry)
    
    # Run pipeline
    print("Running automated research pipeline...")
    results = pipeline.run_pipeline(keywords=["momentum", "mean reversion"], max_papers=5)
    
    print(f"\nProcessed {len(results)} hypotheses")
    print(registry.generate_report())
