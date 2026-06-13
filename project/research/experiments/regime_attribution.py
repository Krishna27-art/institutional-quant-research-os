"""
Regime Attribution for Trade Explainability
Based on the critique: Every trade should be explainable

Objective:
- Understand why a strategy entered/exited
- Determine what regime was active
- Identify which features mattered
- Enable decision replay for analysis

Features:
- Trade-level attribution
- Regime impact analysis
- Feature importance per trade
- Decision replay visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class RegimeType(Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


@dataclass
class TradeAttribution:
    """Attribution information for a single trade."""
    trade_id: str
    symbol: str
    entry_date: datetime
    exit_date: datetime
    direction: str
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    
    # Attribution
    regime_at_entry: RegimeType
    regime_at_exit: RegimeType
    feature_importance: Dict[str, float]
    primary_feature: str
    secondary_features: List[str]
    
    # Decision context
    signal_strength: float
    confidence: float
    risk_limit_hit: bool
    execution_quality: str


@dataclass
class RegimeImpact:
    """Impact of regime on strategy performance."""
    regime: RegimeType
    num_trades: int
    total_pnl: float
    avg_pnl: float
    win_rate: float
    sharpe: float
    max_drawdown: float


class RegimeAttributionEngine:
    """
    Regime Attribution Engine for trade explainability.
    
    Provides:
    - Trade-level attribution
    - Regime impact analysis
    - Feature importance per trade
    - Decision replay
    """
    
    def __init__(self):
        self.trade_history: List[TradeAttribution] = []
        self.regime_impacts: Dict[RegimeType, RegimeImpact] = {}
        
        # Feature importance cache
        self.feature_importance_cache: Dict[str, Dict[str, float]] = {}
    
    def detect_regime(
        self,
        data: pd.DataFrame,
        timestamp: datetime,
        lookback: int = 20
    ) -> RegimeType:
        """
        Detect market regime at a specific timestamp.
        
        Simple regime detection based on volatility and trend.
        """
        historical = data.loc[:timestamp].tail(lookback)
        
        if len(historical) < lookback:
            return RegimeType.SIDEWAYS
        
        # Calculate metrics
        returns = historical['close'].pct_change().dropna()
        volatility = returns.std()
        trend = (historical['close'].iloc[-1] / historical['close'].iloc[0] - 1)
        
        # Classify regime
        if volatility > 0.02:
            return RegimeType.HIGH_VOL
        elif volatility < 0.005:
            return RegimeType.LOW_VOL
        elif trend > 0.02:
            return RegimeType.BULL_TREND
        elif trend < -0.02:
            return RegimeType.BEAR_TREND
        else:
            return RegimeType.SIDEWAYS
    
    def calculate_feature_importance(
        self,
        features: Dict[str, float],
        model=None
    ) -> Dict[str, float]:
        """
        Calculate feature importance for a trade.
        
        In production, would use SHAP values from the model.
        For now, uses absolute feature values as proxy.
        """
        importance = {}
        
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                importance[feature_name] = abs(value)
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def attribute_trade(
        self,
        symbol: str,
        entry_date: datetime,
        exit_date: datetime,
        direction: str,
        entry_price: float,
        exit_price: float,
        features: Dict[str, float],
        signal_strength: float = 0.5,
        confidence: float = 0.7,
        risk_limit_hit: bool = False,
        execution_quality: str = "good"
    ) -> TradeAttribution:
        """
        Attribute a trade with explainability information.
        
        Args:
            symbol: Trading symbol
            entry_date: Entry timestamp
            exit_date: Exit timestamp
            direction: Trade direction
            entry_price: Entry price
            exit_price: Exit price
            features: Feature values at entry
            signal_strength: Strength of the signal
            confidence: Confidence in the signal
            risk_limit_hit: Whether risk limit was hit
            execution_quality: Quality of execution
            
        Returns:
            TradeAttribution with full attribution
        """
        # Calculate PnL
        if direction == "long":
            pnl = (exit_price - entry_price)
        else:
            pnl = (entry_price - exit_price)
        
        return_pct = pnl / entry_price
        
        # Detect regimes
        # Note: In production, would pass actual market data
        regime_at_entry = RegimeType.SIDEWAYS  # Placeholder
        regime_at_exit = RegimeType.SIDEWAYS  # Placeholder
        
        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(features)
        
        # Identify primary and secondary features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        primary_feature = sorted_features[0][0] if sorted_features else "unknown"
        secondary_features = [f[0] for f in sorted_features[1:4]] if len(sorted_features) > 1 else []
        
        # Create attribution
        trade_id = f"{symbol}_{entry_date.strftime('%Y%m%d_%H%M%S')}_{direction}"
        
        attribution = TradeAttribution(
            trade_id=trade_id,
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            return_pct=return_pct,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            feature_importance=feature_importance,
            primary_feature=primary_feature,
            secondary_features=secondary_features,
            signal_strength=signal_strength,
            confidence=confidence,
            risk_limit_hit=risk_limit_hit,
            execution_quality=execution_quality
        )
        
        self.trade_history.append(attribution)
        return attribution
    
    def analyze_regime_impact(self) -> Dict[RegimeType, RegimeImpact]:
        """
        Analyze the impact of different regimes on strategy performance.
        
        Returns:
            Dictionary of regime -> impact statistics
        """
        regime_trades: Dict[RegimeType, List[TradeAttribution]] = {}
        
        # Group trades by entry regime
        for trade in self.trade_history:
            regime = trade.regime_at_entry
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)
        
        # Calculate statistics for each regime
        for regime, trades in regime_trades.items():
            if not trades:
                continue
            
            pnls = [t.pnl for t in trades]
            returns = [t.return_pct for t in trades]
            
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)
            win_rate = len([p for p in pnls if p > 0]) / len(pnls)
            
            # Sharpe
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Max drawdown
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            self.regime_impacts[regime] = RegimeImpact(
                regime=regime,
                num_trades=len(trades),
                total_pnl=total_pnl,
                avg_pnl=avg_pnl,
                win_rate=win_rate,
                sharpe=sharpe,
                max_drawdown=max_drawdown
            )
        
        return self.regime_impacts
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of regime impacts."""
        self.analyze_regime_impact()
        
        data = []
        for regime, impact in self.regime_impacts.items():
            data.append({
                'Regime': regime.value,
                'Trades': impact.num_trades,
                'Total PnL': f"{impact.total_pnl:.2f}",
                'Avg PnL': f"{impact.avg_pnl:.2f}",
                'Win Rate': f"{impact.win_rate:.2%}",
                'Sharpe': f"{impact.sharpe:.2f}",
                'Max DD': f"{impact.max_drawdown:.2%}"
            })
        
        return pd.DataFrame(data)
    
    def explain_trade(self, trade_id: str) -> Optional[Dict]:
        """
        Explain a specific trade.
        
        Returns detailed explanation of why the trade was taken.
        """
        for trade in self.trade_history:
            if trade.trade_id == trade_id:
                return {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'direction': trade.direction,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'pnl': trade.pnl,
                    'return_pct': f"{trade.return_pct:.2%}",
                    'regime_at_entry': trade.regime_at_entry.value,
                    'regime_at_exit': trade.regime_at_exit.value,
                    'primary_feature': trade.primary_feature,
                    'secondary_features': trade.secondary_features,
                    'signal_strength': f"{trade.signal_strength:.2f}",
                    'confidence': f"{trade.confidence:.2%}",
                    'risk_limit_hit': trade.risk_limit_hit,
                    'execution_quality': trade.execution_quality,
                    'feature_importance': trade.feature_importance
                }
        
        return None
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get summary of feature importance across all trades."""
        feature_counts: Dict[str, int] = {}
        feature_as_primary: Dict[str, int] = {}
        
        for trade in self.trade_history:
            for feature in trade.feature_importance.keys():
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            if trade.primary_feature:
                feature_as_primary[trade.primary_feature] = feature_as_primary.get(trade.primary_feature, 0) + 1
        
        data = []
        for feature in sorted(feature_counts.keys()):
            data.append({
                'Feature': feature,
                'Usage Count': feature_counts[feature],
                'Primary Count': feature_as_primary.get(feature, 0),
                'Primary Rate': f"{feature_as_primary.get(feature, 0) / feature_counts[feature]:.2%}" if feature_counts[feature] > 0 else "0%"
            })
        
        return pd.DataFrame(data).sort_values('Usage Count', ascending=False)
    
    def decision_replay(self, trade_id: str) -> str:
        """
        Generate a narrative explanation of the decision process.
        
        Returns a human-readable explanation of why the trade was taken.
        """
        trade = next((t for t in self.trade_history if t.trade_id == trade_id), None)
        
        if not trade:
            return "Trade not found"
        
        narrative = f"""
DECISION REPLAY: {trade.trade_id}

Context:
- Symbol: {trade.symbol}
- Direction: {trade.direction.upper()}
- Entry: {trade.entry_date} @ {trade.entry_price:.2f}
- Exit: {trade.exit_date} @ {trade.exit_price:.2f}
- PnL: {trade.pnl:.2f} ({trade.return_pct:.2%})

Market Regime:
- At Entry: {trade.regime_at_entry.value}
- At Exit: {trade.regime_at_exit.value}

Signal Analysis:
- Signal Strength: {trade.signal_strength:.2f}
- Confidence: {trade.confidence:.2%}
- Risk Limit Hit: {trade.risk_limit_hit}

Key Features:
- Primary: {trade.primary_feature} (importance: {trade.feature_importance.get(trade.primary_feature, 0):.2%})
- Secondary: {', '.join(trade.secondary_features)}

Execution:
- Quality: {trade.execution_quality}

Rationale:
The {trade.direction} signal was generated based primarily on {trade.primary_feature}.
The strategy entered during a {trade.regime_at_entry.value} regime with {trade.confidence:.0%} confidence.
"""
        return narrative.strip()


if __name__ == "__main__":
    # Test the Regime Attribution Engine
    print("Testing Regime Attribution Engine...")
    
    engine = RegimeAttributionEngine()
    
    # Simulate some trades
    print("\nSimulating trades...")
    
    for i in range(10):
        entry_date = datetime(2024, 1, 1) + timedelta(days=i)
        exit_date = entry_date + timedelta(days=5)
        
        features = {
            'momentum_5d': np.random.uniform(-0.05, 0.05),
            'volatility_20d': np.random.uniform(0.01, 0.03),
            'rsi_14': np.random.uniform(30, 70),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'price_to_ma20': np.random.uniform(0.95, 1.05)
        }
        
        engine.attribute_trade(
            symbol="RELIANCE",
            entry_date=entry_date,
            exit_date=exit_date,
            direction="long" if i % 2 == 0 else "short",
            entry_price=2500 + np.random.uniform(-50, 50),
            exit_price=2500 + np.random.uniform(-100, 100),
            features=features,
            signal_strength=np.random.uniform(0.3, 0.9),
            confidence=np.random.uniform(0.5, 0.9)
        )
    
    print(f"Attributed {len(engine.trade_history)} trades")
    
    # Analyze regime impact
    print("\nRegime Impact Analysis:")
    regime_summary = engine.get_regime_summary()
    print(regime_summary.to_string(index=False))
    
    # Feature importance summary
    print("\nFeature Importance Summary:")
    feature_summary = engine.get_feature_importance_summary()
    print(feature_summary.to_string(index=False))
    
    # Explain a specific trade
    print("\nTrade Explanation:")
    trade_id = engine.trade_history[0].trade_id
    explanation = engine.explain_trade(trade_id)
    for key, value in explanation.items():
        print(f"  {key}: {value}")
    
    # Decision replay
    print("\nDecision Replay:")
    narrative = engine.decision_replay(trade_id)
    print(narrative)
