"""
Model Integration Module

This module integrates the model registry into production for
model versioning, deployment, and rollback capabilities.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Import model registry
try:
    from models.model_registry import get_model_registry, ModelType, ModelStage
    model_registry = get_model_registry()
except Exception:
    model_registry = None

# Import ensemble model
try:
    from models.ensemble_model import create_ensemble
    ensemble_model = create_ensemble(ensemble_method="weighted_average")
except Exception:
    ensemble_model = None


class ModelIntegrator:
    """
    Model integrator for production deployment.
    
    This class manages model deployment, versioning, and rollback.
    """
    
    def __init__(self):
        """Initialize model integrator."""
        self.current_models: Dict[str, str] = {}  # strategy -> model_id
        self.model_versions: Dict[str, int] = {}  # strategy -> version
        
        logger.info("ModelIntegrator initialized")
    
    def deploy_model(
        self,
        strategy: str,
        model: any,
        model_type: ModelType = ModelType.ALPHA,
        metrics: Dict = None
    ) -> str:
        """
        Deploy a model to production.
        
        Args:
            strategy: Strategy name
            model: Model object
            model_type: Type of model
            metrics: Performance metrics
            
        Returns:
            Model ID
        """
        if not model_registry:
            logger.warning("Model registry not available")
            return "mock_model_id"
        
        try:
            # Log model to registry
            model_id = model_registry.log_model(
                name=strategy,
                model=model,
                model_type=model_type,
                parameters={'deployed_at': datetime.now().isoformat()},
                metrics=metrics
            )
            
            # Transition to production
            model_registry.transition_model_stage(
                name=strategy,
                version=int(model_id.split('_')[-1]) if '_' in model_id else 1,
                stage=ModelStage.PRODUCTION
            )
            
            # Update current models
            self.current_models[strategy] = model_id
            
            logger.info(f"Deployed model {model_id} for strategy {strategy}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to deploy model for {strategy}: {e}")
            return "mock_model_id"
    
    def get_production_model(self, strategy: str) -> Optional[any]:
        """
        Get the production model for a strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Model object
        """
        if not model_registry:
            return None
        
        try:
            model_info = model_registry.get_model_info(strategy)
            if model_info and model_info.stage == ModelStage.PRODUCTION:
                return model_registry.load_model(strategy, version=model_info.version)
        except Exception as e:
            logger.error(f"Failed to load production model for {strategy}: {e}")
        
        return None
    
    def rollback_model(self, strategy: str, target_version: int) -> bool:
        """
        Rollback to a previous model version.
        
        Args:
            strategy: Strategy name
            target_version: Target version to rollback to
            
        Returns:
            Success status
        """
        if not model_registry:
            return False
        
        try:
            model_registry.rollback_model(strategy, target_version)
            logger.info(f"Rolled back {strategy} to version {target_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback {strategy}: {e}")
            return False
    
    def get_model_performance(self, strategy: str) -> Dict:
        """
        Get performance metrics for a strategy's model.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Performance metrics
        """
        if not model_registry:
            return {}
        
        try:
            return model_registry.get_model_performance(strategy)
        except Exception as e:
            logger.error(f"Failed to get model performance for {strategy}: {e}")
            return {}


# Singleton instance
_model_integrator = None

def get_model_integrator() -> ModelIntegrator:
    """Get the singleton model integrator instance."""
    global _model_integrator
    if _model_integrator is None:
        _model_integrator = ModelIntegrator()
    return _model_integrator


if __name__ == "__main__":
    # Test model integrator
    print("Testing Model Integrator...")
    
    integrator = ModelIntegrator()
    
    # Deploy a mock model
    class MockModel:
        def predict(self, X):
            return [0.5]
    
    model_id = integrator.deploy_model("test_strategy", MockModel())
    print(f"Deployed model: {model_id}")
