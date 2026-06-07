"""
Feature Computer - Main computation engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..definitions.base import FeatureRegistry, FeatureDefinition
from .price import PriceFeatures
from .volume import VolumeFeatures
from .volatility_features import VolatilityFeatures
from .microstructure import BreadthFeatures
from .longmemory import ResearchFeatures


class FeatureComputer:
    """Main feature computation engine"""
    
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self.registry = registry or FeatureRegistry()
        self.price_features = PriceFeatures()
        self.volume_features = VolumeFeatures()
        self.volatility_features = VolatilityFeatures()
        self.breadth_features = BreadthFeatures()
        self.research_features = ResearchFeatures()
    
    def compute_feature(self, feature_name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a single feature"""
        feature_def = self.registry.get(feature_name)
        if not feature_def:
            raise ValueError(f"Feature {feature_name} not found in registry")
        
        category = feature_def.category
        
        if category.value == "price":
            try:
                return self.price_features.compute(feature_name, data)
            except ValueError:
                return self.research_features.compute(feature_name, data)
        elif category.value == "volume":
            return self.volume_features.compute(feature_name, data)
        elif category.value == "volatility":
            try:
                return self.volatility_features.compute(feature_name, data)
            except ValueError:
                return self.research_features.compute(feature_name, data)
        elif category.value == "breadth":
            return self.breadth_features.compute(feature_name, data)
        else:
            raise NotImplementedError(f"Category {category} not yet implemented")
    
    def compute_all(self, data: pd.DataFrame, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute multiple features"""
        if feature_names is None:
            feature_names = [f.name for f in self.registry.list_all()]
        
        results = {}
        for feature_name in feature_names:
            try:
                results[feature_name] = self.compute_feature(feature_name, data)
            except Exception as e:
                print(f"Warning: Failed to compute {feature_name}: {e}")
                results[feature_name] = np.nan
        
        return pd.DataFrame(results, index=data.index)
    
    def compute_batch(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame], 
                     feature_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Compute features for multiple symbols"""
        results = {}
        for symbol in symbols:
            if symbol in data_dict:
                results[symbol] = self.compute_all(data_dict[symbol], feature_names)
        return results
