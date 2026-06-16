"""
Alpha Attribution Engine
Computes granular performance metrics (PnL, Hit Rate, Sharpe) by Alpha, Regime, and Feature.
Tracks exactly where PnL comes from.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AlphaAttributionEngine:
    def __init__(self, trade_logger=None):
        self.trade_logger = trade_logger
        self._attributions = []

    def load_trades(self):
        """Loads closed trades with alpha/regime metadata from the trade logger."""
        if not self.trade_logger:
            return []
        trades = []
        for trade in self.trade_logger.trades.values():
            if trade.exit_price is not None:
                meta = trade.metadata or {}
                trades.append({
                    'trade_id': trade.trade_id,
                    'alpha_id': meta.get('alpha_id', 'unknown_alpha'),
                    'regime_id': meta.get('regime_id', 'unknown_regime'),
                    'feature_hash': meta.get('feature_hash', 'unknown_feature'),
                    'pnl': trade.pnl,
                    'is_win': trade.is_win(),
                    'entry_time': trade.entry_time
                })
        return pd.DataFrame(trades)

    def calculate_attribution(self, group_by: str) -> pd.DataFrame:
        """Calculate aggregate performance grouped by a specific metadata key (e.g., 'alpha_id')."""
        df = self.load_trades()
        if df.empty:
            return pd.DataFrame()
            
        grouped = df.groupby(group_by)
        
        result = []
        for name, group in grouped:
            total_trades = len(group)
            win_rate = group['is_win'].mean()
            total_pnl = group['pnl'].sum()
            
            # Simple Sharpe calculation (annualized) assuming roughly daily trades per group
            returns = group['pnl']
            std_dev = returns.std()
            sharpe = (returns.mean() / std_dev * np.sqrt(252)) if std_dev > 0 else 0.0
            
            result.append({
                group_by: name,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'sharpe_ratio': sharpe
            })
            
        return pd.DataFrame(result).set_index(group_by)

    def get_alpha_performance_by_regime(self, alpha_id: str) -> pd.DataFrame:
        """Get performance of a specific alpha across all regimes."""
        df = self.load_trades()
        if df.empty:
            return pd.DataFrame()
            
        alpha_df = df[df['alpha_id'] == alpha_id]
        if alpha_df.empty:
            return pd.DataFrame()
            
        grouped = alpha_df.groupby('regime_id')
        
        result = []
        for regime, group in grouped:
            result.append({
                'regime': regime,
                'pnl': group['pnl'].sum(),
                'hit_rate': group['is_win'].mean(),
                'trades': len(group)
            })
            
        return pd.DataFrame(result).set_index('regime')
