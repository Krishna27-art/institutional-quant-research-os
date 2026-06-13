"""
Continuous Regime Discovery Engine
Automatically detects new market regimes and updates HMM/HSMM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from hmmlearn import hmm
from scipy import stats

from time_machine_simulator import TimeMachineSimulator, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Regime types"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class RegimeState:
    """Regime state"""
    regime_id: str
    regime_type: RegimeType
    name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime]
    probability: float
    characteristics: Dict[str, float]
    alpha_weights: Dict[str, float]
    status: str = "active"


@dataclass
class RegimeDiscoveryResult:
    """Result of regime discovery"""
    run_id: str
    timestamp: datetime
    num_regimes: int
    new_regimes: List[RegimeState]
    regime_transitions: List[Dict]
    stability_score: float
    likelihood_improvement: float


class RegimeDiscoveryEngine:
    """
    Continuous Regime Discovery Engine
    """
    
    def __init__(
        self,
        time_machine: TimeMachineSimulator,
        existing_regimes: Optional[List[RegimeState]] = None
    ):
        self.time_machine = time_machine
        self.existing_regimes = existing_regimes or []
        self.discovered_regimes: List[RegimeState] = []
        
        # HMM parameters
        self.n_components = 4
        self.n_iter = 100
        self.covariance_type = "full"
        
        # Feature set for regime detection
        self.regime_features = [
            'realized_vol_5d',
            'iv',
            'nifty_return',
            'turnover',
            'fii_flow',
            'spread'
        ]
        
        logger.info("Regime Discovery Engine initialized")
    
    def discover_regimes(
        self,
        start_date: datetime,
        end_date: datetime,
        lookback_years: int = 2
    ) -> RegimeDiscoveryResult:
        """
        Discover market regimes using HMM
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            lookback_years: Lookback window for training
            
        Returns:
            RegimeDiscoveryResult
        """
        run_id = f"REGIME_DISCOVERY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting regime discovery {run_id}")
        
        # Get training data
        training_start = start_date - timedelta(days=lookback_years * 365)
        
        snapshots = self.time_machine.get_snapshot_range(
            start_date=training_start,
            end_date=end_date,
            frequency='1D',
            symbols=['NIFTY'],
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        features = self.time_machine.get_feature_matrix(snapshots)
        
        # Prepare regime features
        regime_features_df = self._prepare_regime_features(features)
        
        if regime_features_df.empty:
            logger.error("No regime features available")
            return RegimeDiscoveryResult(
                run_id=run_id,
                timestamp=datetime.now(),
                num_regimes=0,
                new_regimes=[],
                regime_transitions=[],
                stability_score=0.0,
                likelihood_improvement=0.0
            )
        
        # Train HMM
        hmm_model = self._train_hmm(regime_features_df)
        
        # Predict regimes
        hidden_states = hmm_model.predict(regime_features_df.values)
        
        # Analyze regimes
        regimes = self._analyze_regimes(
            hidden_states,
            regime_features_df,
            training_start,
            end_date
        )
        
        # Detect new regimes
        new_regimes = self._detect_new_regimes(regimes)
        
        # Calculate regime transitions
        transitions = self._calculate_regime_transitions(hidden_states)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(regimes)
        
        # Calculate likelihood improvement
        likelihood_improvement = self._calculate_likelihood_improvement(hmm_model)
        
        # Store discovered regimes
        self.discovered_regimes.extend(new_regimes)
        
        result = RegimeDiscoveryResult(
            run_id=run_id,
            timestamp=datetime.now(),
            num_regimes=len(regimes),
            new_regimes=new_regimes,
            regime_transitions=transitions,
            stability_score=stability_score,
            likelihood_improvement=likelihood_improvement
        )
        
        logger.info(
            f"Regime discovery complete: {len(regimes)} regimes, "
            f"{len(new_regimes)} new regimes detected"
        )
        
        return result
    
    def _prepare_regime_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for regime detection"""
        # In production, this would compute actual regime features
        # For simulation, generate synthetic features
        
        regime_features = pd.DataFrame(index=features.index)
        
        # Simulate regime features
        np.random.seed(42)
        regime_features['realized_vol_5d'] = np.random.uniform(0.01, 0.05, len(features))
        regime_features['iv'] = np.random.uniform(0.1, 0.3, len(features))
        regime_features['nifty_return'] = np.random.normal(0.001, 0.02, len(features))
        regime_features['turnover'] = np.random.uniform(0.5, 2.0, len(features))
        regime_features['fii_flow'] = np.random.normal(0, 100, len(features))
        regime_features['spread'] = np.random.uniform(0.001, 0.01, len(features))
        
        return regime_features
    
    def _train_hmm(self, features: pd.DataFrame) -> hmm.GaussianHMM:
        """Train Hidden Markov Model"""
        model = hmm.GaussianHMM(
            n_components=self.n_components,
            n_iter=self.n_iter,
            covariance_type=self.covariance_type,
            random_state=42
        )
        
        model.fit(features.values)
        
        return model
    
    def _analyze_regimes(
        self,
        hidden_states: np.ndarray,
        features: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> List[RegimeState]:
        """Analyze discovered regimes"""
        regimes = []
        
        for state in range(self.n_components):
            # Get data points for this state
            state_mask = hidden_states == state
            state_features = features[state_mask]
            
            if len(state_features) == 0:
                continue
            
            # Calculate regime characteristics
            characteristics = {
                'mean_volatility': state_features['realized_vol_5d'].mean(),
                'mean_iv': state_features['iv'].mean(),
                'mean_return': state_features['nifty_return'].mean(),
                'mean_turnover': state_features['turnover'].mean(),
                'mean_fii_flow': state_features['fii_flow'].mean(),
            }
            
            # Determine regime type
            regime_type = self._classify_regime(characteristics)
            
            # Calculate regime-specific alpha weights
            alpha_weights = self._calculate_regime_alpha_weights(regime_type)
            
            # Find start and end dates
            state_indices = np.where(state_mask)[0]
            if len(state_indices) > 0:
                regime_start = start_date + timedelta(days=state_indices[0])
                regime_end = start_date + timedelta(days=state_indices[-1])
            else:
                regime_start = start_date
                regime_end = None
            
            regime = RegimeState(
                regime_id=f"REGIME_{state}_{datetime.now().strftime('%Y%m%d')}",
                regime_type=regime_type,
                name=f"{regime_type.value}_{state}",
                description=f"Regime {state}: {regime_type.value}",
                start_date=regime_start,
                end_date=regime_end,
                probability=state_mask.mean(),
                characteristics=characteristics,
                alpha_weights=alpha_weights,
                status="active"
            )
            
            regimes.append(regime)
        
        return regimes
    
    def _classify_regime(self, characteristics: Dict[str, float]) -> RegimeType:
        """Classify regime based on characteristics"""
        vol = characteristics['mean_volatility']
        ret = characteristics['mean_return']
        
        if vol > 0.04:
            return RegimeType.HIGH_VOLATILITY
        elif vol < 0.015:
            return RegimeType.LOW_VOLATILITY
        elif ret > 0.01:
            return RegimeType.BULL_TREND
        elif ret < -0.01:
            return RegimeType.BEAR_TREND
        elif abs(ret) < 0.002:
            return RegimeType.SIDEWAYS
        else:
            return RegimeType.SIDEWAYS
    
    def _calculate_regime_alpha_weights(self, regime_type: RegimeType) -> Dict[str, float]:
        """Calculate optimal alpha weights for regime"""
        # Simplified regime-specific weights
        weights = {
            'ORB': 0.3,
            'VWAP': 0.2,
            'MEAN_REVERSION': 0.2,
            'MOMENTUM': 0.3,
        }
        
        # Adjust based on regime type
        if regime_type == RegimeType.BULL_TREND:
            weights['MOMENTUM'] = 0.5
            weights['MEAN_REVERSION'] = 0.1
        elif regime_type == RegimeType.BEAR_TREND:
            weights['MEAN_REVERSION'] = 0.5
            weights['MOMENTUM'] = 0.1
        elif regime_type == RegimeType.HIGH_VOLATILITY:
            weights['ORB'] = 0.1
            weights['VWAP'] = 0.4
            weights['MEAN_REVERSION'] = 0.3
            weights['MOMENTUM'] = 0.2
        
        return weights
    
    def _detect_new_regimes(self, regimes: List[RegimeState]) -> List[RegimeState]:
        """Detect new regimes not seen before"""
        new_regimes = []
        
        for regime in regimes:
            # Check if similar regime exists
            is_new = True
            for existing in self.existing_regimes:
                if regime.regime_type == existing.regime_type:
                    is_new = False
                    break
            
            if is_new:
                new_regimes.append(regime)
        
        return new_regimes
    
    def _calculate_regime_transitions(self, hidden_states: np.ndarray) -> List[Dict]:
        """Calculate regime transition matrix"""
        transitions = []
        
        for i in range(len(hidden_states) - 1):
            transitions.append({
                'from_state': int(hidden_states[i]),
                'to_state': int(hidden_states[i + 1]),
            })
        
        # Calculate transition probabilities
        transition_counts = {}
        for t in transitions:
            key = (t['from_state'], t['to_state'])
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        transition_probs = {}
        for (from_state, to_state), count in transition_counts.items():
            total_from = sum(c for (fs, ts), c in transition_counts.items() if fs == from_state)
            if total_from > 0:
                transition_probs[(from_state, to_state)] = count / total_from
        
        return [
            {
                'from_state': from_state,
                'to_state': to_state,
                'probability': prob
            }
            for (from_state, to_state), prob in transition_probs.items()
        ]
    
    def _calculate_stability_score(self, regimes: List[RegimeState]) -> float:
        """Calculate regime stability score"""
        # Regimes that persist >5 days are stable
        stable_regimes = 0
        for regime in regimes:
            if regime.end_date:
                duration = (regime.end_date - regime.start_date).days
                if duration >= 5:
                    stable_regimes += 1
        
        stability_score = stable_regimes / len(regimes) if regimes else 0.0
        
        return stability_score
    
    def _calculate_likelihood_improvement(self, hmm_model: hmm.GaussianHMM) -> float:
        """Calculate likelihood improvement over baseline"""
        # Simplified: use log-likelihood
        likelihood = hmm_model.score(np.random.randn(100, 6))
        baseline_likelihood = -1000.0
        
        improvement = (likelihood - baseline_likelihood) / abs(baseline_likelihood)
        
        return improvement
    
    def test_regime_stability(
        self,
        regimes: List[RegimeState],
        min_persistence_days: int = 5
    ) -> Dict[str, Any]:
        """
        Test regime stability
        
        Args:
            regimes: List of regimes to test
            min_persistence_days: Minimum persistence threshold
            
        Returns:
            Stability test results
        """
        transient_regimes = []
        stable_regimes = []
        
        for regime in regimes:
            if regime.end_date:
                duration = (regime.end_date - regime.start_date).days
                if duration < min_persistence_days:
                    transient_regimes.append(regime)
                else:
                    stable_regimes.append(regime)
        
        # Merge transient regimes with nearest neighbor
        merged_regimes = []
        for transient in transient_regimes:
            # Find nearest stable regime
            nearest = min(
                stable_regimes,
                key=lambda r: abs(r.characteristics['mean_volatility'] - transient.characteristics['mean_volatility'])
            )
            merged_regimes.append({
                'transient_regime': transient.regime_id,
                'merged_with': nearest.regime_id,
            })
        
        return {
            'total_regimes': len(regimes),
            'stable_regimes': len(stable_regimes),
            'transient_regimes': len(transient_regimes),
            'merged_regimes': merged_regimes,
            'stability_percentage': len(stable_regimes) / len(regimes) if regimes else 0.0,
        }
    
    def update_production_regime_model(
        self,
        new_regimes: List[RegimeState],
        likelihood_improvement: float,
        threshold: float = 0.05
    ) -> bool:
        """
        Update production regime model if improvement is significant
        
        Args:
            new_regimes: New regimes discovered
            likelihood_improvement: Likelihood improvement
            threshold: Improvement threshold
            
        Returns:
            True if model updated
        """
        if likelihood_improvement > threshold:
            # Update production model
            self.existing_regimes.extend(new_regimes)
            logger.info(f"Updated production regime model (improvement: {likelihood_improvement:.2%})")
            return True
        else:
            logger.info(f"Regime model not updated (improvement: {likelihood_improvement:.2%} < {threshold:.2%})")
            return False
    
    def get_regime_weights(self, current_regime: RegimeType) -> Dict[str, float]:
        """Get alpha weights for current regime"""
        for regime in self.existing_regimes:
            if regime.regime_type == current_regime:
                return regime.alpha_weights
        
        # Default weights if regime not found
        return {
            'ORB': 0.25,
            'VWAP': 0.25,
            'MEAN_REVERSION': 0.25,
            'MOMENTUM': 0.25,
        }


def simulate_regime_discovery():
    """Simulate regime discovery"""
    
    print("="*60)
    print("REGIME DISCOVERY ENGINE SIMULATION")
    print("="*60)
    
    # Initialize time machine
    time_machine = TimeMachineSimulator()
    
    # Initialize discovery engine
    discovery_engine = RegimeDiscoveryEngine(time_machine)
    
    # Run discovery
    print("\n1. Running regime discovery...")
    result = discovery_engine.discover_regimes(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2024, 1, 1),
        lookback_years=2
    )
    
    print(f"  Run ID: {result.run_id}")
    print(f"  Total regimes: {result.num_regimes}")
    print(f"  New regimes: {len(result.new_regimes)}")
    print(f"  Stability score: {result.stability_score:.2%}")
    print(f"  Likelihood improvement: {result.likelihood_improvement:.2%}")
    
    # Show regimes
    print("\n2. Discovered regimes:")
    for regime in discovery_engine.discovered_regimes:
        print(f"  {regime.name}:")
        print(f"    Type: {regime.regime_type.value}")
        print(f"    Probability: {regime.probability:.2%}")
        print(f"    Duration: {(regime.end_date - regime.start_date).days if regime.end_date else 0} days")
        print(f"    Alpha weights: {regime.alpha_weights}")
    
    # Show transitions
    print("\n3. Regime transitions:")
    for transition in result.regime_transitions[:5]:
        print(f"  {transition['from_state']} → {transition['to_state']}: {transition['probability']:.2%}")
    
    # Test stability
    print("\n4. Testing regime stability...")
    stability_test = discovery_engine.test_regime_stability(discovery_engine.discovered_regimes)
    print(f"  Stable regimes: {stability_test['stable_regimes']}")
    print(f"  Transient regimes: {stability_test['transient_regimes']}")
    print(f"  Stability percentage: {stability_test['stability_percentage']:.2%}")
    
    # Update production model
    print("\n5. Updating production regime model...")
    updated = discovery_engine.update_production_regime_model(
        result.new_regimes,
        result.likelihood_improvement
    )
    print(f"  Updated: {updated}")
    
    # Get regime weights
    print("\n6. Getting regime weights for current regime...")
    weights = discovery_engine.get_regime_weights(RegimeType.BULL_TREND)
    print(f"  Alpha weights: {weights}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_regime_discovery()
