"""
Model Registry (MLflow Compatible)

This module implements a model registry with MLflow-compatible API for
model versioning, tracking, and deployment management.

Key Features:
- Model versioning and lineage tracking
- Training metrics and parameters logging
- Model artifact storage
- Deployment stage management (Staging, Production, Archived)
- Model comparison and selection
- Rollback capabilities
- MLflow-compatible API

Based on Audit Report Priority 0: Critical - Week 1-2
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import json
import uuid
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model deployment stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelType(Enum):
    """Model types."""
    ALPHA = "alpha"
    REGIME = "regime"
    VOLATILITY = "volatility"
    EXECUTION = "execution"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    sharpe_ratio: float
    accuracy: float
    win_rate: float
    max_drawdown: float
    information_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    hit_rate: float
    avg_return: float
    std_return: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    name: str
    version: int
    model_type: ModelType
    stage: ModelStage
    created_at: datetime
    created_by: str
    parameters: Dict[str, Any]
    metrics: ModelMetrics
    training_start: datetime
    training_end: datetime
    artifact_path: str
    run_id: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    parent_model_id: Optional[str] = None


class ModelRegistry:
    """
    Model registry with MLflow-compatible API.
    
    This class provides model versioning, tracking, and deployment management
    capabilities for institutional-grade ML systems.
    """
    
    def __init__(self, registry_path: str = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store model registry data
        """
        self.registry_path = Path(registry_path) if registry_path else Path(__file__).parent / "registry"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.models: Dict[str, Dict[int, ModelVersion]] = {}
        self.model_artifacts: Dict[str, Any] = {}
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def log_model(
        self,
        name: str,
        model: Any,
        model_type: ModelType = ModelType.ALPHA,
        parameters: Dict[str, Any] = None,
        metrics: ModelMetrics = None,
        description: str = "",
        tags: Dict[str, str] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT
    ) -> str:
        """
        Log a model to the registry.
        
        Args:
            name: Model name
            model: Model object (pickle-serializable)
            model_type: Type of model
            parameters: Model hyperparameters
            metrics: Model performance metrics
            description: Model description
            tags: Tags for the model
            stage: Initial deployment stage
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Get next version
        if name not in self.models:
            self.models[name] = {}
        next_version = len(self.models[name]) + 1
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            name=name,
            version=next_version,
            model_type=model_type,
            stage=stage,
            created_at=datetime.now(),
            created_by="system",
            parameters=parameters or {},
            metrics=metrics or ModelMetrics(
                sharpe_ratio=0.0, accuracy=0.0, win_rate=0.0,
                max_drawdown=0.0, information_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, hit_rate=0.0, avg_return=0.0, std_return=0.0
            ),
            training_start=datetime.now(),
            training_end=datetime.now(),
            artifact_path=str(self.registry_path / name / f"v{next_version}"),
            run_id=str(uuid.uuid4()),
            description=description,
            tags=tags or {}
        )
        
        # Store model version
        self.models[name][next_version] = model_version
        
        # Store model artifact
        artifact_path = Path(model_version.artifact_path)
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        with open(artifact_path / "model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        self.model_artifacts[model_id] = model
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Logged model {name} v{next_version} with ID {model_id}")
        return model_id
    
    def load_model(self, name: str, version: Optional[int] = None, stage: Optional[ModelStage] = None) -> Any:
        """
        Load a model from the registry.
        
        Args:
            name: Model name
            version: Specific version (None for latest)
            stage: Load model with specific stage (None for any)
            
        Returns:
            Model object
        """
        # Get version to load
        if version is None:
            version = self._get_latest_version(name, stage)
            if version is None:
                raise ValueError(f"No model found for {name}")
        
        # Get model version info
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name} v{version} not found")
        
        model_version = self.models[name][version]
        
        # Load artifact
        artifact_path = Path(model_version.artifact_path) / "model.pkl"
        
        if not artifact_path.exists():
            raise ValueError(f"Model artifact not found at {artifact_path}")
        
        with open(artifact_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model {name} v{version}")
        return model
    
    def transition_model_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage
    ) -> None:
        """
        Transition a model to a different stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage
        """
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name} v{version} not found")
        
        # Archive current production model if transitioning to production
        if stage == ModelStage.PRODUCTION:
            self._archive_current_production(name)
        
        # Update stage
        self.models[name][version].stage = stage
        self.models[name][version].created_at = datetime.now()
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Transitioned {name} v{version} to {stage.value}")
    
    def _archive_current_production(self, name: str) -> None:
        """Archive current production model."""
        if name not in self.models:
            return
        
        for version, model_version in self.models[name].items():
            if model_version.stage == ModelStage.PRODUCTION:
                model_version.stage = ModelStage.ARCHIVED
                logger.info(f"Archived {name} v{version}")
    
    def get_model_info(self, name: str, version: Optional[int] = None) -> Optional[ModelVersion]:
        """
        Get model information.
        
        Args:
            name: Model name
            version: Model version (None for latest)
            
        Returns:
            ModelVersion
        """
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                return None
        
        if name not in self.models or version not in self.models[name]:
            return None
        
        return self.models[name][version]
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[Dict]:
        """
        List all models in the registry.
        
        Args:
            model_type: Filter by model type
            
        Returns:
            List of model information
        """
        models_list = []
        
        for name, versions in self.models.items():
            for version, model_version in versions.items():
                if model_type is None or model_version.model_type == model_type:
                    models_list.append({
                        'name': name,
                        'version': version,
                        'model_type': model_version.model_type.value,
                        'stage': model_version.stage.value,
                        'created_at': model_version.created_at.isoformat(),
                        'sharpe_ratio': model_version.metrics.sharpe_ratio,
                        'description': model_version.description
                    })
        
        return models_list
    
    def compare_models(
        self,
        name: str,
        version1: int,
        version2: int
    ) -> Dict:
        """
        Compare two model versions.
        
        Args:
            name: Model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        if version1 not in self.models[name] or version2 not in self.models[name]:
            raise ValueError("One or both versions not found")
        
        model1 = self.models[name][version1]
        model2 = self.models[name][version2]
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'sharpe_diff': model1.metrics.sharpe_ratio - model2.metrics.sharpe_ratio,
            'accuracy_diff': model1.metrics.accuracy - model2.metrics.accuracy,
            'win_rate_diff': model1.metrics.win_rate - model2.metrics.win_rate,
            'drawdown_diff': model1.metrics.max_drawdown - model2.metrics.max_drawdown,
            'better_sharpe': version1 if model1.metrics.sharpe_ratio > model2.metrics.sharpe_ratio else version2,
            'better_accuracy': version1 if model1.metrics.accuracy > model2.metrics.accuracy else version2
        }
        
        return comparison
    
    def rollback_model(self, name: str, target_version: int) -> None:
        """
        Rollback to a previous model version.
        
        Args:
            name: Model name
            target_version: Version to rollback to
        """
        if name not in self.models or target_version not in self.models[name]:
            raise ValueError(f"Model {name} v{target_version} not found")
        
        # Archive current production
        self._archive_current_production(name)
        
        # Transition target to production
        self.transition_model_stage(name, target_version, ModelStage.PRODUCTION)
        
        logger.info(f"Rolled back {name} to v{target_version}")
    
    def delete_model(self, name: str, version: int) -> None:
        """
        Delete a model version.
        
        Args:
            name: Model name
            version: Version to delete
        """
        if name not in self.models or version not in self.models[name]:
            raise ValueError(f"Model {name} v{version} not found")
        
        model_version = self.models[name][version]
        
        # Delete artifacts
        artifact_path = Path(model_version.artifact_path)
        if artifact_path.exists():
            import shutil
            shutil.rmtree(artifact_path)
        
        # Remove from registry
        del self.models[name][version]
        
        # Clean up if no versions left
        if not self.models[name]:
            del self.models[name]
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted {name} v{version}")
    
    def _get_latest_version(self, name: str, stage: Optional[ModelStage] = None) -> Optional[int]:
        """Get the latest version of a model."""
        if name not in self.models:
            return None
        
        versions = list(self.models[name].keys())
        
        if stage is not None:
            # Filter by stage
            versions = [
                v for v in versions
                if self.models[name][v].stage == stage
            ]
        
        return max(versions) if versions else None
    
    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
            
            for name, versions_data in data.items():
                self.models[name] = {}
                for version_str, model_data in versions_data.items():
                    version = int(version_str)
                    metrics = ModelMetrics(**model_data['metrics'])
                    
                    model_version = ModelVersion(
                        model_id=model_data['model_id'],
                        name=model_data['name'],
                        version=version,
                        model_type=ModelType(model_data['model_type']),
                        stage=ModelStage(model_data['stage']),
                        created_at=pd.to_datetime(model_data['created_at']),
                        created_by=model_data['created_by'],
                        parameters=model_data['parameters'],
                        metrics=metrics,
                        training_start=pd.to_datetime(model_data['training_start']),
                        training_end=pd.to_datetime(model_data['training_end']),
                        artifact_path=model_data['artifact_path'],
                        run_id=model_data['run_id'],
                        description=model_data.get('description', ''),
                        tags=model_data.get('tags', {}),
                        parent_model_id=model_data.get('parent_model_id')
                    )
                    
                    self.models[name][version] = model_version
            
            logger.info(f"Loaded registry with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"
        
        try:
            data = {}
            for name, versions in self.models.items():
                data[name] = {}
                for version, model_version in versions.items():
                    data[name][str(version)] = {
                        'model_id': model_version.model_id,
                        'name': model_version.name,
                        'version': model_version.version,
                        'model_type': model_version.model_type.value,
                        'stage': model_version.stage.value,
                        'created_at': model_version.created_at.isoformat(),
                        'created_by': model_version.created_by,
                        'parameters': model_version.parameters,
                        'metrics': {
                            'sharpe_ratio': model_version.metrics.sharpe_ratio,
                            'accuracy': model_version.metrics.accuracy,
                            'win_rate': model_version.metrics.win_rate,
                            'max_drawdown': model_version.metrics.max_drawdown,
                            'information_ratio': model_version.metrics.information_ratio,
                            'sortino_ratio': model_version.metrics.sortino_ratio,
                            'calmar_ratio': model_version.metrics.calmar_ratio,
                            'hit_rate': model_version.metrics.hit_rate,
                            'avg_return': model_version.metrics.avg_return,
                            'std_return': model_version.metrics.std_return,
                            'custom_metrics': model_version.metrics.custom_metrics
                        },
                        'training_start': model_version.training_start.isoformat(),
                        'training_end': model_version.training_end.isoformat(),
                        'artifact_path': model_version.artifact_path,
                        'run_id': model_version.run_id,
                        'description': model_version.description,
                        'tags': model_version.tags,
                        'parent_model_id': model_version.parent_model_id
                    }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Saved registry to disk")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def print_registry_summary(self) -> None:
        """Print registry summary."""
        models_list = self.list_models()
        
        print("\n" + "="*60)
        print("MODEL REGISTRY SUMMARY")
        print("="*60)
        print(f"\nTotal Models: {len(models_list)}")
        
        if models_list:
            print(f"\nModels:")
            for model in models_list:
                print(f"  {model['name']} v{model['version']} - {model['stage']} "
                      f"(Sharpe: {model['sharpe_ratio']:.2f})")
        
        print("\n" + "="*60)


# Singleton instance
_model_registry = None

def get_model_registry() -> ModelRegistry:
    """Get the singleton model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


if __name__ == "__main__":
    # Test the model registry
    print("Testing Model Registry...")
    
    registry = ModelRegistry()
    
    # Create a mock model
    class MockModel:
        def predict(self, X):
            return np.random.randn(len(X))
    
    model = MockModel()
    
    # Log model
    metrics = ModelMetrics(
        sharpe_ratio=1.2,
        accuracy=0.75,
        win_rate=0.65,
        max_drawdown=-0.15,
        information_ratio=0.8,
        sortino_ratio=1.5,
        calmar_ratio=2.0,
        hit_rate=0.60,
        avg_return=0.001,
        std_return=0.02
    )
    
    model_id = registry.log_model(
        name="test_alpha",
        model=model,
        model_type=ModelType.ALPHA,
        parameters={'learning_rate': 0.01, 'n_estimators': 100},
        metrics=metrics,
        description="Test alpha model"
    )
    
    print(f"Logged model with ID: {model_id}")
    
    # List models
    models = registry.list_models()
    print(f"Total models in registry: {len(models)}")
    
    # Print summary
    registry.print_registry_summary()
