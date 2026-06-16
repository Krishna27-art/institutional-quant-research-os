import pandas as pd
import numpy as np
from typing import Dict, List, Any

class CorrelationManager:
    """
    Manages and analyzes correlations across the portfolio to prevent disguised risk.
    Ensures that 10 different strategies aren't just placing the exact same trade.
    """
    def __init__(self, correlation_threshold: float = 0.75):
        self.correlation_threshold = correlation_threshold
        # Stores historical signals by strategy (strategy_id -> list of signals)
        self.strategy_signals: Dict[str, List[float]] = {}
        self.strategy_returns: Dict[str, List[float]] = {}

    def log_strategy_state(self, strategy_id: str, signal_value: float, daily_return: float = 0.0):
        """Record the daily signal and return for a given strategy."""
        if strategy_id not in self.strategy_signals:
            self.strategy_signals[strategy_id] = []
            self.strategy_returns[strategy_id] = []
            
        self.strategy_signals[strategy_id].append(signal_value)
        self.strategy_returns[strategy_id].append(daily_return)

    def calculate_signal_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculates the correlation matrix of the raw signals emitted by strategies.
        High correlation means strategies are firing together.
        """
        if not self.strategy_signals:
            return pd.DataFrame()
            
        # Ensure all lengths are the same by taking the minimum length
        min_len = min(len(sigs) for sigs in self.strategy_signals.values())
        if min_len < 2:
            return pd.DataFrame() # Not enough data
            
        data = {k: v[-min_len:] for k, v in self.strategy_signals.items()}
        df = pd.DataFrame(data)
        return df.corr()

    def calculate_strategy_return_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculates the correlation matrix of the realized returns of strategies.
        """
        if not self.strategy_returns:
            return pd.DataFrame()
            
        min_len = min(len(rets) for rets in self.strategy_returns.values())
        if min_len < 2:
            return pd.DataFrame()
            
        data = {k: v[-min_len:] for k, v in self.strategy_returns.items()}
        df = pd.DataFrame(data)
        return df.corr()

    def get_highly_correlated_pairs(self) -> List[tuple]:
        """
        Identifies pairs of strategies that have a signal correlation exceeding the threshold.
        These are prime candidates for capital reduction or deprecation.
        """
        corr_matrix = self.calculate_signal_correlation_matrix()
        if corr_matrix.empty:
            return []
            
        highly_correlated = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                    highly_correlated.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
                    
        return highly_correlated
        
    def check_trade_redundancy(self, pending_signals: Dict[str, float]) -> Dict[str, float]:
        """
        If two highly correlated strategies are about to place the same trade,
        we can reduce their weights to prevent double exposure.
        """
        highly_correlated = self.get_highly_correlated_pairs()
        adjusted_signals = pending_signals.copy()
        
        for strat_a, strat_b, corr in highly_correlated:
            if strat_a in adjusted_signals and strat_b in adjusted_signals:
                # If they are both trying to trade in the same direction
                if np.sign(adjusted_signals[strat_a]) == np.sign(adjusted_signals[strat_b]):
                    # Reduce both by a factor of 1 - (corr / 2)
                    penalty = corr / 2.0
                    adjusted_signals[strat_a] *= (1.0 - penalty)
                    adjusted_signals[strat_b] *= (1.0 - penalty)
                    
        return adjusted_signals
