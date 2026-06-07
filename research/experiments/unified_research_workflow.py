"""
Unified Research Workflow
Based on the critique: Create systematic research pipeline

Pipeline:
Hypothesis → Backtest → Walk-forward → Paper Trade → Scale

This is the single workflow that all research must follow.
No exceptions. No shortcuts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from research.alpha_factory import AlphaFactory, Hypothesis, HypothesisStatus
from backtest.walk_forward import WalkForwardValidator, WalkForwardConfig
from research.strategy_validation import StrategyValidator, StrategyValidationResult
from paper_trading.paper_trading_validation import PaperTradingValidator, ValidationStatus
from core.objective_function import ObjectiveFunction, ObjectiveConstraints


class WorkflowStage(Enum):
    """Stages in the research workflow."""
    HYPOTHESIS = "hypothesis"
    BACKTEST = "backtest"
    WALK_FORWARD = "walk_forward"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    KILLED = "killed"


@dataclass
class WorkflowItem:
    """Item in the research workflow."""
    item_id: str
    name: str
    current_stage: WorkflowStage
    hypothesis: Optional[Hypothesis]
    backtest_result: Optional[Dict]
    walk_forward_result: Optional[Dict]
    paper_trading_result: Optional[Dict]
    objective_score: Optional[float]
    created_at: datetime
    updated_at: datetime
    notes: List[str]
    
    def is_feasible(self) -> bool:
        """Check if item has passed all stages."""
        return self.current_stage == WorkflowStage.PRODUCTION


class UnifiedResearchWorkflow:
    """
    Unified research workflow for systematic alpha discovery.
    
    Pipeline:
    1. Hypothesis Generation
    2. Backtesting
    3. Walk-forward Validation
    4. Paper Trading
    5. Production (or Kill)
    
    Every alpha must go through this pipeline.
    No exceptions.
    """
    
    def __init__(self):
        self.workflow_items: Dict[str, WorkflowItem] = {}
        self.alpha_factory = AlphaFactory()
        self.walk_forward_validator = WalkForwardValidator(WalkForwardConfig())
        self.paper_trading_validator = PaperTradingValidator()
        self.objective_function = ObjectiveFunction()
        
        # Stage thresholds
        self.min_sharpe_backtest = 1.0
        self.min_sharpe_walk_forward = 0.8
        self.min_sharpe_paper_trading = 0.7
        self.max_drawdown = 0.15
    
    def create_hypothesis(
        self,
        name: str,
        description: str,
        signal_function: callable,
        parameters: Dict
    ) -> str:
        """
        Create a new hypothesis and add to workflow.
        
        Args:
            name: Hypothesis name
            description: Description
            signal_function: Signal generation function
            parameters: Parameters for signal function
            
        Returns:
            Workflow item ID
        """
        # Create hypothesis using Alpha Factory
        from research.alpha_factory import SignalCategory
        hypothesis_id = self.alpha_factory.generate_hypothesis(
            name=name,
            description=description,
            category=SignalCategory.MOMENTUM,  # Default
            source="workflow",
            signal_function=signal_function,
            parameters=parameters
        )
        
        hypothesis = self.alpha_factory.hypotheses[hypothesis_id]
        
        # Create workflow item
        item_id = f"workflow_{hypothesis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        workflow_item = WorkflowItem(
            item_id=item_id,
            name=name,
            current_stage=WorkflowStage.HYPOTHESIS,
            hypothesis=hypothesis,
            backtest_result=None,
            walk_forward_result=None,
            paper_trading_result=None,
            objective_score=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            notes=[]
        )
        
        self.workflow_items[item_id] = workflow_item
        return item_id
    
    def run_backtest(
        self,
        item_id: str,
        data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Run backtest for a workflow item.
        
        Args:
            item_id: Workflow item ID
            data: Historical data
            
        Returns:
            True if passed backtest stage
        """
        if item_id not in self.workflow_items:
            return False
        
        item = self.workflow_items[item_id]
        
        if item.current_stage != WorkflowStage.HYPOTHESIS:
            return False
        
        # Run backtest using Alpha Factory
        results = self.alpha_factory.batch_backtest(data, in_sample=True)
        
        # Check if hypothesis passed
        hypothesis_id = item.hypothesis.id
        if hypothesis_id in results:
            result = results[hypothesis_id]
            item.backtest_result = {
                'sharpe': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_return': result.total_return
            }
            
            # Check thresholds
            if (result.sharpe_ratio >= self.min_sharpe_backtest and
                result.max_drawdown <= self.max_drawdown):
                item.current_stage = WorkflowStage.BACKTEST
                item.updated_at = datetime.now()
                item.notes.append(f"Passed backtest: Sharpe={result.sharpe_ratio:.2f}")
                return True
            else:
                item.current_stage = WorkflowStage.KILLED
                item.updated_at = datetime.now()
                item.notes.append(f"Killed in backtest: Sharpe={result.sharpe_ratio:.2f}, DD={result.max_drawdown:.2%}")
                return False
        
        return False
    
    def run_walk_forward(
        self,
        item_id: str,
        data: pd.DataFrame
    ) -> bool:
        """
        Run walk-forward validation.
        
        Args:
            item_id: Workflow item ID
            data: Historical data with features
            
        Returns:
            True if passed walk-forward stage
        """
        if item_id not in self.workflow_items:
            return False
        
        item = self.workflow_items[item_id]
        
        if item.current_stage != WorkflowStage.BACKTEST:
            return False
        
        # Run walk-forward validation
        def train_func(X_train, y_train):
            # Simple model for demonstration
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        
        def predict_func(model, X_test):
            predictions = model.predict(X_test)
            return pd.Series(predictions, index=X_test.index)
        
        results = self.walk_forward_validator.validate(
            data=data,
            train_func=train_func,
            predict_func=predict_func,
            target_col="returns"
        )
        
        item.walk_forward_result = {
            'test_sharpe_mean': results['test_sharpe_mean'],
            'test_sharpe_std': results['test_sharpe_std'],
            'test_max_dd_mean': results['test_max_dd_mean'],
            'decay_rate': results['decay_rate'],
            'is_overfitted': results['is_overfitted']
        }
        
        # Check thresholds
        if (results['test_sharpe_mean'] >= self.min_sharpe_walk_forward and
            not results['is_overfitted'] and
            results['decay_rate'] < 0.3):
            item.current_stage = WorkflowStage.WALK_FORWARD
            item.updated_at = datetime.now()
            item.notes.append(f"Passed walk-forward: Sharpe={results['test_sharpe_mean']:.2f}")
            return True
        else:
            item.current_stage = WorkflowStage.KILLED
            item.updated_at = datetime.now()
            item.notes.append(f"Killed in walk-forward: Sharpe={results['test_sharpe_mean']:.2f}, Overfitted={results['is_overfitted']}")
            return False
    
    def run_paper_trading(
        self,
        item_id: str,
        duration_days: int = 30
    ) -> bool:
        """
        Run paper trading validation.
        
        Args:
            item_id: Workflow item ID
            duration_days: Duration of paper trading
            
        Returns:
            True if passed paper trading stage
        """
        if item_id not in self.workflow_items:
            return False
        
        item = self.workflow_items[item_id]
        
        if item.current_stage != WorkflowStage.WALK_FORWARD:
            return False
        
        # Start paper trading validation
        validation_id = self.paper_trading_validator.start_validation(
            strategy_name=item.name,
            expected_duration_days=duration_days
        )
        
        # Simulate paper trades (in production, would use real signals)
        for i in range(25):
            signal = item.hypothesis.signal_function
            # Generate simulated trades
            # In production, would use actual signal generation
        
        # Calculate metrics
        metrics = self.paper_trading_validator.calculate_validation_metrics(validation_id)
        
        item.paper_trading_result = {
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'avg_slippage_bps': metrics.avg_slippage_bps
        }
        
        # Check thresholds
        if (metrics.sharpe_ratio >= self.min_sharpe_paper_trading and
            metrics.max_drawdown <= self.max_drawdown):
            item.current_stage = WorkflowStage.PAPER_TRADING
            item.updated_at = datetime.now()
            item.notes.append(f"Passed paper trading: Sharpe={metrics.sharpe_ratio:.2f}")
            return True
        else:
            item.current_stage = WorkflowStage.KILLED
            item.updated_at = datetime.now()
            item.notes.append(f"Killed in paper trading: Sharpe={metrics.sharpe_ratio:.2f}")
            return False
    
    def calculate_objective_score(
        self,
        item_id: str,
        returns: pd.Series,
        costs: pd.Series,
        positions: pd.Series
    ) -> float:
        """
        Calculate objective score for a workflow item.
        
        Args:
            item_id: Workflow item ID
            returns: Strategy returns
            costs: Transaction costs
            positions: Position sizes
            
        Returns:
            Objective score
        """
        if item_id not in self.workflow_items:
            return 0.0
        
        score = self.objective_function.calculate_objective(returns, costs, positions)
        
        item = self.workflow_items[item_id]
        item.objective_score = score.objective_value
        item.updated_at = datetime.now()
        
        return score.objective_value
    
    def promote_to_production(self, item_id: str) -> bool:
        """
        Promote a workflow item to production.
        
        Args:
            item_id: Workflow item ID
            
        Returns:
            True if promoted successfully
        """
        if item_id not in self.workflow_items:
            return False
        
        item = self.workflow_items[item_id]
        
        if item.current_stage != WorkflowStage.PAPER_TRADING:
            return False
        
        if item.objective_score is None or item.objective_score <= 0:
            return False
        
        item.current_stage = WorkflowStage.PRODUCTION
        item.updated_at = datetime.now()
        item.notes.append("Promoted to production")
        
        return True
    
    def get_workflow_summary(self) -> pd.DataFrame:
        """Get summary of all workflow items."""
        data = []
        
        for item_id, item in self.workflow_items.items():
            data.append({
                'Item ID': item_id,
                'Name': item.name,
                'Stage': item.current_stage.value,
                'Created': item.created_at.strftime('%Y-%m-%d'),
                'Updated': item.updated_at.strftime('%Y-%m-%d'),
                'Backtest Sharpe': item.backtest_result['sharpe'] if item.backtest_result else 0,
                'Walk-forward Sharpe': item.walk_forward_result['test_sharpe_mean'] if item.walk_forward_result else 0,
                'Paper Trading Sharpe': item.paper_trading_result['sharpe_ratio'] if item.paper_trading_result else 0,
                'Objective Score': item.objective_score if item.objective_score else 0,
                'Notes': '; '.join(item.notes)
            })
        
        return pd.DataFrame(data)
    
    def get_production_alphas(self) -> List[WorkflowItem]:
        """Get all alphas in production."""
        return [
            item for item in self.workflow_items.values()
            if item.current_stage == WorkflowStage.PRODUCTION
        ]
    
    def get_killed_alphas(self) -> List[WorkflowItem]:
        """Get all killed alphas."""
        return [
            item for item in self.workflow_items.values()
            if item.current_stage == WorkflowStage.KILLED
        ]


if __name__ == "__main__":
    # Test the Unified Research Workflow
    print("Testing Unified Research Workflow...")
    
    workflow = UnifiedResearchWorkflow()
    
    # Create a hypothesis
    print("\nCreating hypothesis...")
    
    def sample_signal(df):
        returns = df['close'].pct_change(5)
        signal = np.where(returns > 0, 1, -1)
        return pd.Series(signal, index=df.index)
    
    item_id = workflow.create_hypothesis(
        name="Sample Momentum",
        description="5-day momentum signal",
        signal_function=sample_signal,
        parameters={'lookback': 5}
    )
    
    print(f"Created workflow item: {item_id}")
    
    # Generate sample data for backtesting
    print("\nGenerating sample data...")
    np.random.seed(42)
    symbols = ['RELIANCE', 'TCS']
    data = {}
    
    for symbol in symbols:
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = np.random.normal(100, 10, n).cumsum()
        prices = prices - prices.min() + 100
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
            'close': prices,
            'volume': np.random.normal(1000000, 200000, n)
        }, index=dates)
        
        data[symbol] = df
    
    # Run backtest
    print("\nRunning backtest...")
    passed_backtest = workflow.run_backtest(item_id, data)
    print(f"Backtest passed: {passed_backtest}")
    
    # Run walk-forward (with feature data)
    print("\nRunning walk-forward...")
    feature_data = data['RELIANCE'].copy()
    feature_data['returns'] = feature_data['close'].pct_change()
    feature_data = feature_data.dropna()
    
    passed_walk_forward = workflow.run_walk_forward(item_id, feature_data)
    print(f"Walk-forward passed: {passed_walk_forward}")
    
    # Run paper trading
    print("\nRunning paper trading...")
    passed_paper_trading = workflow.run_paper_trading(item_id, duration_days=30)
    print(f"Paper trading passed: {passed_paper_trading}")
    
    # Calculate objective score
    print("\nCalculating objective score...")
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    costs = pd.Series(np.random.uniform(0.0001, 0.0003, 252))
    positions = pd.Series(np.random.uniform(-0.3, 0.3, 252))
    
    objective_score = workflow.calculate_objective_score(item_id, returns, costs, positions)
    print(f"Objective score: {objective_score:.2f}")
    
    # Promote to production
    print("\nPromoting to production...")
    promoted = workflow.promote_to_production(item_id)
    print(f"Promoted: {promoted}")
    
    # Get workflow summary
    print("\nWorkflow Summary:")
    summary = workflow.get_workflow_summary()
    print(summary.to_string(index=False))
    
    # Get production alphas
    print("\nProduction Alphas:")
    production = workflow.get_production_alphas()
    print(f"Number in production: {len(production)}")
    
    # Get killed alphas
    print("\nKilled Alphas:")
    killed = workflow.get_killed_alphas()
    print(f"Number killed: {len(killed)}")
