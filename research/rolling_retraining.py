"""
Rolling Retraining Framework
Keep production models fresh without introducing lookahead
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import hashlib
import lightgbm as lgb

from time_machine_simulator import TimeMachineSimulator, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Triggers for retraining"""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"


@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: int
    created_at: datetime
    training_data_start: datetime
    training_data_end: datetime
    model_hash: str
    data_hash: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: str  # "candidate", "shadow", "production", "deprecated"
    trigger: RetrainingTrigger


@dataclass
class RetrainingResult:
    """Result of retraining process"""
    model_id: str
    previous_model_id: Optional[str]
    trigger: RetrainingTrigger
    training_start: datetime
    training_end: datetime
    training_duration_seconds: float
    new_model_version: ModelVersion
    performance_comparison: Dict[str, float]
    promoted: bool
    timestamp: datetime


class RollingRetrainingFramework:
    """
    Rolling retraining framework for keeping models fresh
    """
    
    def __init__(
        self,
        time_machine: TimeMachineSimulator,
        training_window_years: int = 3,
        validation_window_years: int = 1,
        retrain_interval_days: int = 30,
        performance_threshold: float = 0.8,
        canary_duration_days: int = 5
    ):
        self.time_machine = time_machine
        self.training_window_years = training_window_years
        self.validation_window_years = validation_window_years
        self.retrain_interval_days = retrain_interval_days
        self.performance_threshold = performance_threshold
        self.canary_duration_days = canary_duration_days
        
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.current_production_models: Dict[str, str] = {}
        
        logger.info(
            f"Rolling Retraining Framework initialized: "
            f"train={training_window_years}y, interval={retrain_interval_days}d"
        )
    
    def check_retraining_trigger(
        self,
        model_id: str,
        current_performance: float,
        historical_performance: float
    ) -> Optional[RetrainingTrigger]:
        """
        Check if retraining is needed
        
        Args:
            model_id: Model identifier
            current_performance: Current model performance (Sharpe)
            historical_performance: Historical average performance
            
        Returns:
            Trigger type if retraining needed, None otherwise
        """
        # Performance degradation check
        if current_performance < historical_performance * self.performance_threshold:
            logger.warning(
                f"Model {model_id} performance degraded: "
                f"{current_performance:.2f} < {historical_performance * self.performance_threshold:.2f}"
            )
            return RetrainingTrigger.PERFORMANCE_DEGRADATION
        
        return None
    
    def retrain_model(
        self,
        model_id: str,
        symbols: List[str],
        base_model: Any,
        train_func: Callable,
        predict_func: Callable,
        evaluate_func: Callable,
        trigger: RetrainingTrigger = RetrainingTrigger.SCHEDULED
    ) -> RetrainingResult:
        """
        Retrain a model with fresh data
        
        Args:
            model_id: Model identifier
            symbols: Symbols to train on
            base_model: Base model architecture
            train_func: Training function
            predict_func: Prediction function
            evaluate_func: Evaluation function
            trigger: Retraining trigger
            
        Returns:
            RetrainingResult
        """
        training_start = datetime.now()
        
        logger.info(f"Retraining model {model_id} (trigger: {trigger.value})")
        
        # Calculate training window
        training_end = datetime.now()
        training_start_date = training_end - timedelta(days=self.training_window_years * 365)
        validation_end = training_end
        validation_start = training_end - timedelta(days=self.validation_window_years * 365)
        
        # Get training data (point-in-time)
        train_snapshots = self.time_machine.get_snapshot_range(
            start_date=training_start_date,
            end_date=training_end - timedelta(days=self.validation_window_years * 365),
            frequency='1D',
            symbols=symbols,
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        train_features = self.time_machine.get_feature_matrix(train_snapshots)
        train_labels = self.time_machine.get_labels(train_snapshots, forward_periods=1)
        
        # Get validation data
        val_snapshots = self.time_machine.get_snapshot_range(
            start_date=validation_start,
            end_date=validation_end,
            frequency='1D',
            symbols=symbols,
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        val_features = self.time_machine.get_feature_matrix(val_snapshots)
        val_labels = self.time_machine.get_labels(val_snapshots, forward_periods=1)
        
        # Train new model
        trained_model = train_func(base_model, train_features, train_labels)
        
        # Evaluate on validation set
        val_preds = predict_func(trained_model, val_features)
        val_metrics = evaluate_func(val_labels, val_preds)
        
        # Generate model version
        version = len(self.model_versions.get(model_id, [])) + 1
        model_hash = self._generate_model_hash(trained_model)
        data_hash = self._generate_data_hash(train_features)
        
        new_model_version = ModelVersion(
            model_id=model_id,
            version=version,
            created_at=datetime.now(),
            training_data_start=training_start_date,
            training_data_end=training_end - timedelta(days=self.validation_window_years * 365),
            model_hash=model_hash,
            data_hash=data_hash,
            hyperparameters={},
            performance_metrics=val_metrics,
            status="candidate",
            trigger=trigger
        )
        
        # Add to model versions
        if model_id not in self.model_versions:
            self.model_versions[model_id] = []
        self.model_versions[model_id].append(new_model_version)
        
        # Compare with previous model
        previous_model_id = self.current_production_models.get(model_id)
        performance_comparison = {}
        
        if previous_model_id:
            previous_version = self._get_model_version(model_id, previous_model_id)
            if previous_version:
                performance_comparison = {
                    'previous_sharpe': previous_version.performance_metrics.get('sharpe', 0),
                    'new_sharpe': val_metrics.get('sharpe', 0),
                    'improvement': val_metrics.get('sharpe', 0) - previous_version.performance_metrics.get('sharpe', 0),
                }
        
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start).total_seconds()
        
        # Auto-promote if improvement
        promoted = False
        if performance_comparison.get('improvement', 0) > 0:
            promoted = self.promote_model(model_id, new_model_version.model_id)
        
        result = RetrainingResult(
            model_id=model_id,
            previous_model_id=previous_model_id,
            trigger=trigger,
            training_start=training_start,
            training_end=training_end_time,
            training_duration_seconds=training_duration,
            new_model_version=new_model_version,
            performance_comparison=performance_comparison,
            promoted=promoted,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Retraining complete: {model_id} v{version}, "
            f"Sharpe={val_metrics.get('sharpe', 0):.2f}, promoted={promoted}"
        )
        
        return result
    
    def promote_model(
        self,
        model_id: str,
        version_id: str
    ) -> bool:
        """
        Promote a model version to production
        
        Args:
            model_id: Model identifier
            version_id: Version identifier
            
        Returns:
            True if promoted successfully
        """
        if model_id not in self.model_versions:
            logger.error(f"Model {model_id} not found")
            return False
        
        model_version = self._get_model_version(model_id, version_id)
        if not model_version:
            logger.error(f"Version {version_id} not found for model {model_id}")
            return False
        
        # Demote previous production model
        previous_production_id = self.current_production_models.get(model_id)
        if previous_production_id:
            previous_version = self._get_model_version(model_id, previous_production_id)
            if previous_version:
                previous_version.status = "deprecated"
        
        # Promote new model
        model_version.status = "production"
        self.current_production_models[model_id] = version_id
        
        logger.info(f"Promoted {model_id} {version_id} to production")
        
        return True
    
    def canary_deployment(
        self,
        model_id: str,
        candidate_version_id: str,
        duration_days: int = 5
    ) -> bool:
        """
        Deploy model in canary mode before full promotion
        
        Args:
            model_id: Model identifier
            candidate_version_id: Candidate version identifier
            duration_days: Canary duration in days
            
        Returns:
            True if canary passed and should be promoted
        """
        logger.info(
            f"Starting canary deployment for {model_id} {candidate_version_id} "
            f"for {duration_days} days"
        )
        
        # Update status to shadow
        model_version = self._get_model_version(model_id, candidate_version_id)
        if model_version:
            model_version.status = "shadow"
        
        # In production, this would:
        # 1. Deploy model to shadow environment
        # 2. Monitor performance for duration_days
        # 3. Compare with production model
        # 4. Promote if performance is better
        
        # For simulation, assume canary passes
        logger.info(f"Canary deployment passed for {model_id} {candidate_version_id}")
        
        return True
    
    def rollback_model(
        self,
        model_id: str
    ) -> bool:
        """
        Rollback to previous production model
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if rollback successful
        """
        current_version_id = self.current_production_models.get(model_id)
        if not current_version_id:
            logger.error(f"No production model found for {model_id}")
            return False
        
        # Find previous production model
        versions = self.model_versions.get(model_id, [])
        production_versions = [v for v in versions if v.status == "production"]
        
        if len(production_versions) < 2:
            logger.error(f"No previous production model to rollback to for {model_id}")
            return False
        
        # Rollback to previous version
        previous_version = production_versions[-2]
        return self.promote_model(model_id, previous_version.model_id)
    
    def get_model_version(
        self,
        model_id: str,
        version_id: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get model version
        
        Args:
            model_id: Model identifier
            version_id: Version identifier (None = current production)
            
        Returns:
            ModelVersion or None
        """
        if version_id is None:
            version_id = self.current_production_models.get(model_id)
        
        return self._get_model_version(model_id, version_id)
    
    def _get_model_version(
        self,
        model_id: str,
        version_id: str
    ) -> Optional[ModelVersion]:
        """Get specific model version"""
        if model_id not in self.model_versions:
            return None
        
        for version in self.model_versions[model_id]:
            if version.model_id == version_id:
                return version
        
        return None
    
    def _generate_model_hash(self, model: Any) -> str:
        """Generate hash for model"""
        if hasattr(model, 'save_model'):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                model.save_model(f.name)
                with open(f.name, 'rb') as rf:
                    model_bytes = rf.read()
                Path(f.name).unlink()
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        else:
            return hashlib.sha256(str(model).encode()).hexdigest()[:16]
    
    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data"""
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def get_retraining_history(
        self,
        model_id: str
    ) -> List[ModelVersion]:
        """Get retraining history for a model"""
        return self.model_versions.get(model_id, [])
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get current status of a model"""
        current_version_id = self.current_production_models.get(model_id)
        current_version = self._get_model_version(model_id, current_version_id)
        
        versions = self.model_versions.get(model_id, [])
        
        return {
            'model_id': model_id,
            'current_version': current_version_id,
            'current_version_number': current_version.version if current_version else None,
            'current_status': current_version.status if current_version else None,
            'current_performance': current_version.performance_metrics if current_version else {},
            'total_versions': len(versions),
            'last_retrained': current_version.created_at if current_version else None,
        }


def simulate_rolling_retraining():
    """Simulate rolling retraining"""
    
    print("="*60)
    print("ROLLING RETRAINING FRAMEWORK SIMULATION")
    print("="*60)
    
    # Initialize time machine
    time_machine = TimeMachineSimulator()
    
    # Initialize retraining framework
    framework = RollingRetrainingFramework(
        time_machine=time_machine,
        training_window_years=1,  # Reduced for simulation
        validation_window_years=0.5,
        retrain_interval_days=30,
        performance_threshold=0.8
    )
    
    # Create base model
    base_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Define training functions (simplified for simulation)
    def train_func(model, features, labels):
        return model
    
    def predict_func(model, features):
        return pd.Series(np.random.normal(0, 0.01, len(features)), index=features.index)
    
    def evaluate_func(labels, predictions):
        return {
            'sharpe': np.random.uniform(0.8, 1.5),
            'max_drawdown': -np.random.uniform(0.1, 0.2),
            'mean_return': np.random.normal(0.001, 0.01),
        }
    
    # Retrain model
    print("\n1. Retraining model...")
    result = framework.retrain_model(
        model_id="orb_classifier",
        symbols=['NIFTY', 'BANKNIFTY'],
        base_model=base_model,
        train_func=train_func,
        predict_func=predict_func,
        evaluate_func=evaluate_func,
        trigger=RetrainingTrigger.SCHEDULED
    )
    
    print(f"  Model ID: {result.model_id}")
    print(f"  Version: {result.new_model_version.version}")
    print(f"  Training duration: {result.training_duration_seconds:.2f}s")
    print(f"  Promoted: {result.promoted}")
    
    # Check retraining trigger
    print("\n2. Checking retraining trigger...")
    trigger = framework.check_retraining_trigger(
        model_id="orb_classifier",
        current_performance=0.6,
        historical_performance=1.2
    )
    if trigger:
        print(f"  Trigger detected: {trigger.value}")
    else:
        print("  No trigger detected")
    
    # Get model status
    print("\n3. Getting model status...")
    status = framework.get_model_status("orb_classifier")
    print(f"  Current version: {status['current_version_number']}")
    print(f"  Status: {status['current_status']}")
    print(f"  Total versions: {status['total_versions']}")
    
    # Canary deployment
    print("\n4. Testing canary deployment...")
    canary_passed = framework.canary_deployment(
        model_id="orb_classifier",
        candidate_version_id=result.new_model_version.model_id,
        duration_days=5
    )
    print(f"  Canary passed: {canary_passed}")
    
    # Rollback test
    print("\n5. Testing rollback...")
    rollback_success = framework.rollback_model("orb_classifier")
    print(f"  Rollback successful: {rollback_success}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_rolling_retraining()
