"""
Regime Persistence - Store regime history for smoothing and analysis
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque


class RegimePersistence:
    """Store and manage regime history"""
    
    def __init__(self, max_history: int = 252):
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
    
    def add(self, regime: str, timestamp: datetime, probabilities: Optional[Dict[str, float]] = None) -> None:
        """Add a regime observation"""
        self.history.append({
            'regime': regime,
            'timestamp': timestamp,
            'probabilities': probabilities
        })
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get n most recent regime observations"""
        return list(self.history)[-n:]
    
    def get_regime_sequence(self, n: int = 10) -> List[str]:
        """Get sequence of last n regimes"""
        return [obs['regime'] for obs in list(self.history)[-n:]]
    
    def get_current_regime(self) -> Optional[str]:
        """Get current regime"""
        if not self.history:
            return None
        return self.history[-1]['regime']
    
    def get_regime_duration(self, regime: str) -> int:
        """Get how long current regime has persisted"""
        if not self.history:
            return 0
        
        count = 0
        for obs in reversed(self.history):
            if obs['regime'] == regime:
                count += 1
            else:
                break
        
        return count
    
    def get_regime_distribution(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get distribution of regimes in history window"""
        if window is None:
            observations = list(self.history)
        else:
            observations = list(self.history)[-window:]
        
        if not observations:
            return {}
        
        regime_counts = {}
        for obs in observations:
            regime = obs['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(observations)
        return {r: count / total for r, count in regime_counts.items()}
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame"""
        data = []
        for obs in self.history:
            row = {
                'timestamp': obs['timestamp'],
                'regime': obs['regime']
            }
            if obs['probabilities']:
                row.update(obs['probabilities'])
            data.append(row)
        
        return pd.DataFrame(data)
    
    def clear(self) -> None:
        """Clear all history"""
        self.history.clear()
