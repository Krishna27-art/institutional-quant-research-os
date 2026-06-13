"""
Alpha Research Factory - Systematic Hypothesis Generation and Testing
Based on the critique: Turn from feature-building to alpha-discovery machine

Architecture:
Generate 10,000 ideas → Test 10,000 ideas → Keep 10 → Deploy 3

Features:
- Hypothesis database with metadata
- Signal generation from multiple sources
- Automated backtesting pipeline
- Walk-forward validation
- Research ranking system
- Model graveyard
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class HypothesisStatus(Enum):
    """Status of hypothesis in research pipeline."""
    PROPOSED = "proposed"
    BACKTESTING = "backtesting"
    WALK_FORWARD = "walk_forward"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    KILLED = "killed"
    ARCHIVED = "archived"


class SignalCategory(Enum):
    """Categories of alpha signals."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    STRUCTURAL = "structural"
    OPTIONS = "options"


@dataclass
class Hypothesis:
    """Research hypothesis for alpha generation."""
    id: str
    name: str
    description: str
    category: SignalCategory
    source: str  # paper, github, reddit, custom, etc.
    signal_function: Callable
    parameters: Dict
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    created_at: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    
    # Validation metrics
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    walk_forward_sharpe: float = 0.0
    paper_trading_sharpe: float = 0.0
    
    # Ranking score (0-100)
    research_score: float = 0.0
    
    # Notes
    notes: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Result of backtesting a hypothesis."""
    hypothesis_id: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    turnover: float
    in_sample: bool = False


class AlphaFactory:
    """
    Alpha Research Factory for systematic hypothesis generation.
    
    Pipeline:
    1. Generate hypotheses from multiple sources
    2. Backtest all hypotheses
    3. Walk-forward validation on top performers
    4. Paper trading validation
    5. Deploy top performers to production
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.backtest_results: Dict[str, List[BacktestResult]] = {}
        self.production_alphas: List[str] = []
        self.graveyard: List[str] = []
        
        # Thresholds for progression
        self.sharpe_threshold = 1.0
        self.drawdown_threshold = 0.15
        self.min_trades = 100
        self.max_turnover = 5.0
    
    def generate_hypothesis(
        self,
        name: str,
        description: str,
        category: SignalCategory,
        source: str,
        signal_function: Callable,
        parameters: Dict
    ) -> str:
        """
        Generate a new hypothesis.
        
        Args:
            name: Hypothesis name
            description: Description of the hypothesis
            category: Signal category
            source: Source of the idea
            signal_function: Function that generates signals
            parameters: Parameters for the signal function
            
        Returns:
            Hypothesis ID
        """
        hypothesis_id = f"{category.value}_{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        
        hypothesis = Hypothesis(
            id=hypothesis_id,
            name=name,
            description=description,
            category=category,
            source=source,
            signal_function=signal_function,
            parameters=parameters
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        return hypothesis_id
    
    def batch_generate_momentum_hypotheses(self, lookback_periods: List[int] = None) -> List[str]:
        """Generate momentum hypotheses with different lookback periods."""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 40, 60]
        
        hypothesis_ids = []
        
        for lookback in lookback_periods:
            def momentum_signal(df, lb=lookback):
                returns = df['close'].pct_change(lb)
                signal = np.where(returns > 0, 1, -1)
                return pd.Series(signal, index=df.index)
            
            hypothesis_id = self.generate_hypothesis(
                name=f"Momentum {lookback}d",
                description=f"Momentum signal with {lookback}-day lookback",
                category=SignalCategory.MOMENTUM,
                source="factory",
                signal_function=momentum_signal,
                parameters={"lookback": lookback}
            )
            hypothesis_ids.append(hypothesis_id)
        
        return hypothesis_ids
    
    def batch_generate_mean_reversion_hypotheses(self, lookback_periods: List[int] = None) -> List[str]:
        """Generate mean reversion hypotheses with different lookback periods."""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20]
        
        hypothesis_ids = []
        
        for lookback in lookback_periods:
            def mean_reversion_signal(df, lb=lookback):
                returns = df['close'].pct_change(lb)
                z_score = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
                signal = np.where(z_score > 2, -1, np.where(z_score < -2, 1, 0))
                return pd.Series(signal, index=df.index)
            
            hypothesis_id = self.generate_hypothesis(
                name=f"Mean Reversion {lookback}d",
                description=f"Mean reversion signal with {lookback}-day lookback",
                category=SignalCategory.MEAN_REVERSION,
                source="factory",
                signal_function=mean_reversion_signal,
                parameters={"lookback": lookback}
            )
            hypothesis_ids.append(hypothesis_id)
        
        return hypothesis_ids
    
    def batch_generate_volatility_hypotheses(self) -> List[str]:
        """Generate volatility-based hypotheses."""
        hypothesis_ids = []
        
        # Volatility breakout
        def vol_breakout_signal(df):
            vol = df['close'].pct_change().rolling(20).std()
            vol_ma = vol.rolling(50).mean()
            signal = np.where(vol > vol_ma * 1.5, 1, 0)
            return pd.Series(signal, index=df.index)
        
        hypothesis_id = self.generate_hypothesis(
            name="Volatility Breakout",
            description="Enter when volatility exceeds 1.5x average",
            category=SignalCategory.VOLATILITY,
            source="factory",
            signal_function=vol_breakout_signal,
            parameters={}
        )
        hypothesis_ids.append(hypothesis_id)
        
        # Low vol entry
        def low_vol_signal(df):
            vol = df['close'].pct_change().rolling(20).std()
            vol_ma = vol.rolling(50).mean()
            signal = np.where(vol < vol_ma * 0.7, 1, 0)
            return pd.Series(signal, index=df.index)
        
        hypothesis_id = self.generate_hypothesis(
            name="Low Volatility Entry",
            description="Enter when volatility is below 0.7x average",
            category=SignalCategory.VOLATILITY,
            source="factory",
            signal_function=low_vol_signal,
            parameters={}
        )
        hypothesis_ids.append(hypothesis_id)
        
        return hypothesis_ids
    
    def backtest_hypothesis(
        self,
        hypothesis_id: str,
        data: Dict[str, pd.DataFrame],
        in_sample: bool = True
    ) -> Optional[BacktestResult]:
        """
        Backtest a single hypothesis.
        
        Args:
            hypothesis_id: ID of hypothesis to backtest
            data: Dictionary of symbol -> OHLCV DataFrame
            in_sample: Whether this is in-sample or out-of-sample
            
        Returns:
            BacktestResult or None if failed
        """
        if hypothesis_id not in self.hypotheses:
            return None
        
        hypothesis = self.hypotheses[hypothesis_id]
        all_returns = []
        total_trades = 0
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            # Generate signals
            signals = hypothesis.signal_function(df)
            
            # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
            # Calculate returns
            returns = df['close'].pct_change().shift(1)
            strategy_returns = signals * returns
            
            all_returns.extend(strategy_returns.dropna().tolist())
            total_trades += len(signals[signals != 0])
        
        if len(all_returns) < self.min_trades:
            hypothesis.status = HypothesisStatus.KILLED
            hypothesis.notes.append(f"Insufficient trades: {len(all_returns)}")
            self.graveyard.append(hypothesis_id)
            return None
        
        returns_array = np.array(all_returns)
        
        # Calculate metrics
        total_return = (1 + returns_array).prod() - 1
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Drawdown
        cum_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = returns_array[returns_array > 0]
        win_rate = len(winning_trades) / len(returns_array) if returns_array.size > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum()
        gross_loss = abs(returns_array[returns_array < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Turnover (approximate)
        turnover = total_trades / len(returns_array) if returns_array.size > 0 else 0
        
        result = BacktestResult(
            hypothesis_id=hypothesis_id,
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=0,  # Would calculate if needed
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=mean_return,
            total_trades=total_trades,
            turnover=turnover,
            in_sample=in_sample
        )
        
        # Update hypothesis metrics
        if in_sample:
            hypothesis.in_sample_sharpe = sharpe
        else:
            hypothesis.out_of_sample_sharpe = sharpe
        
        hypothesis.sharpe_ratio = sharpe
        hypothesis.max_drawdown = abs(max_drawdown)
        hypothesis.win_rate = win_rate
        hypothesis.profit_factor = profit_factor
        hypothesis.turnover = turnover
        
        # Store result
        if hypothesis_id not in self.backtest_results:
            self.backtest_results[hypothesis_id] = []
        self.backtest_results[hypothesis_id].append(result)
        
        return result
    
    def batch_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        in_sample: bool = True
    ) -> Dict[str, BacktestResult]:
        """Backtest all proposed hypotheses."""
        results = {}
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis.status == HypothesisStatus.PROPOSED:
                result = self.backtest_hypothesis(hypothesis_id, data, in_sample)
                if result:
                    results[hypothesis_id] = result
                    hypothesis.status = HypothesisStatus.BACKTESTING
        
        return results
    
    def filter_hypotheses(self) -> List[str]:
        """Filter hypotheses based on performance thresholds."""
        passed = []
        
        for hypothesis_id, hypothesis in self.hypotheses.items():
            if hypothesis.status == HypothesisStatus.KILLED:
                continue
            
            # Check thresholds
            if (hypothesis.sharpe_ratio >= self.sharpe_threshold and
                hypothesis.max_drawdown <= self.drawdown_threshold and
                hypothesis.turnover <= self.max_turnover):
                passed.append(hypothesis_id)
                hypothesis.status = HypothesisStatus.WALK_FORWARD
            else:
                hypothesis.status = HypothesisStatus.KILLED
                hypothesis.notes.append(
                    f"Failed thresholds: Sharpe={hypothesis.sharpe_ratio:.2f}, "
                    f"DD={hypothesis.max_drawdown:.2%}, Turnover={hypothesis.turnover:.2f}"
                )
                self.graveyard.append(hypothesis_id)
        
        return passed
    
    def calculate_research_score(self, hypothesis_id: str) -> float:
        """
        Calculate research score for ranking.
        
        Score = 0.4 * Sharpe + 0.3 * (1 - DD) + 0.2 * WinRate + 0.1 * (1 - Turnover/MaxTurnover)
        """
        if hypothesis_id not in self.hypotheses:
            return 0.0
        
        hypothesis = self.hypotheses[hypothesis_id]
        
        # Normalize components
        sharpe_score = min(hypothesis.sharpe_ratio / 3.0, 1.0)  # Cap at 3.0
        dd_score = 1 - min(hypothesis.max_drawdown / 0.3, 1.0)  # Cap at 30%
        win_rate_score = hypothesis.win_rate
        turnover_score = 1 - min(hypothesis.turnover / self.max_turnover, 1.0)
        
        research_score = (
            0.4 * sharpe_score +
            0.3 * dd_score +
            0.2 * win_rate_score +
            0.1 * turnover_score
        ) * 100
        
        hypothesis.research_score = research_score
        return research_score
    
    def rank_hypotheses(self) -> List[Tuple[str, float]]:
        """Rank hypotheses by research score."""
        rankings = []
        
        for hypothesis_id in self.hypotheses:
            if self.hypotheses[hypothesis_id].status != HypothesisStatus.KILLED:
                score = self.calculate_research_score(hypothesis_id)
                rankings.append((hypothesis_id, score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_research_ranking_table(self) -> pd.DataFrame:
        """Get research ranking table."""
        rankings = self.rank_hypotheses()
        
        data = []
        for hypothesis_id, score in rankings:
            hypothesis = self.hypotheses[hypothesis_id]
            data.append({
                'ID': hypothesis_id,
                'Name': hypothesis.name,
                'Category': hypothesis.category.value,
                'Source': hypothesis.source,
                'Status': hypothesis.status.value,
                'Sharpe': hypothesis.sharpe_ratio,
                'Max DD': hypothesis.max_drawdown,
                'Win Rate': hypothesis.win_rate,
                'Profit Factor': hypothesis.profit_factor,
                'Turnover': hypothesis.turnover,
                'Research Score': score
            })
        
        return pd.DataFrame(data)
    
    def promote_to_production(self, top_n: int = 3) -> List[str]:
        """Promote top N hypotheses to production."""
        rankings = self.rank_hypotheses()
        
        promoted = []
        for hypothesis_id, score in rankings[:top_n]:
            self.hypotheses[hypothesis_id].status = HypothesisStatus.PRODUCTION
            self.production_alphas.append(hypothesis_id)
            promoted.append(hypothesis_id)
        
        return promoted
    
    def get_graveyard(self) -> pd.DataFrame:
        """Get graveyard of killed hypotheses."""
        data = []
        
        for hypothesis_id in self.graveyard:
            hypothesis = self.hypotheses[hypothesis_id]
            data.append({
                'ID': hypothesis_id,
                'Name': hypothesis.name,
                'Category': hypothesis.category.value,
                'Source': hypothesis.source,
                'Sharpe': hypothesis.sharpe_ratio,
                'Max DD': hypothesis.max_drawdown,
                'Win Rate': hypothesis.win_rate,
                'Notes': '; '.join(hypothesis.notes)
            })
        
        return pd.DataFrame(data)
    
    def get_production_alphas(self) -> List[Hypothesis]:
        """Get production alphas."""
        return [self.hypotheses[hid] for hid in self.production_alphas]


if __name__ == "__main__":
    # Test the Alpha Factory
    print("Testing Alpha Research Factory...")
    
    factory = AlphaFactory()
    
    # Generate hypotheses
    print("\nGenerating hypotheses...")
    momentum_ids = factory.batch_generate_momentum_hypotheses([5, 10, 20])
    mean_rev_ids = factory.batch_generate_mean_reversion_hypotheses([5, 10])
    vol_ids = factory.batch_generate_volatility_hypotheses()
    
    print(f"Generated {len(momentum_ids)} momentum hypotheses")
    print(f"Generated {len(mean_rev_ids)} mean reversion hypotheses")
    print(f"Generated {len(vol_ids)} volatility hypotheses")
    print(f"Total hypotheses: {len(factory.hypotheses)}")
    
    # Generate sample data for backtesting
    print("\nGenerating sample data...")
    np.random.seed(42)
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
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
    
    # Backtest all hypotheses
    print("\nBacktesting hypotheses...")
    results = factory.batch_backtest(data, in_sample=True)
    print(f"Backtested {len(results)} hypotheses")
    
    # Filter hypotheses
    print("\nFiltering hypotheses...")
    passed = factory.filter_hypotheses()
    print(f"Passed thresholds: {len(passed)}")
    
    # Rank hypotheses
    print("\nRanking hypotheses...")
    rankings = factory.rank_hypotheses()
    print(f"Top 5 hypotheses:")
    for hypothesis_id, score in rankings[:5]:
        hypothesis = factory.hypotheses[hypothesis_id]
        print(f"  {hypothesis.name}: Score={score:.1f}, Sharpe={hypothesis.sharpe_ratio:.2f}")
    
    # Get research ranking table
    print("\nResearch Ranking Table:")
    ranking_table = factory.get_research_ranking_table()
    print(ranking_table.to_string(index=False))
    
    # Get graveyard
    print("\nGraveyard:")
    graveyard = factory.get_graveyard()
    if not graveyard.empty:
        print(graveyard.to_string(index=False))
    else:
        print("No hypotheses in graveyard yet")
