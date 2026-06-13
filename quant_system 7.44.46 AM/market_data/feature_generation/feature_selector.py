"""
Feature Selector using Mutual Information
Reduces overfitting by selecting only the most informative features.

Critical for institutional-grade model building.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FeatureSelectionResult:
    """Result of feature selection"""
    selected_features: List[str]
    removed_features: List[str]
    feature_scores: Dict[str, float]
    selection_ratio: float  # selected / total


class MutualInformationSelector:
    """
    Feature Selector using Mutual Information
    
    Selects top N features based on mutual information with target.
    Reduces overfitting by eliminating low-information features.
    
    Rules:
    1. Use mutual information (not correlation) for feature importance
    2. Select top 20 features (or configurable N)
    3. Remove features with MI < threshold
    4. Remove highly correlated features (>0.95)
    """
    
    def __init__(self, n_features: int = 20, mi_threshold: float = 0.01,
                 correlation_threshold: float = 0.95):
        self.n_features = n_features
        self.mi_threshold = mi_threshold
        self.correlation_threshold = correlation_threshold
        self.feature_scores: Dict[str, float] = {}
    
    def select_features(self, features_df: pd.DataFrame, target: pd.Series,
                       task_type: str = "regression") -> FeatureSelectionResult:
        """
        Select top features using mutual information.
        
        Args:
            features_df: DataFrame of features (columns = features)
            target: Target variable
            task_type: "regression" or "classification"
        
        Returns:
            FeatureSelectionResult
        """
        # Calculate mutual information
        if task_type == "regression":
            mi_scores = mutual_info_regression(features_df, target, random_state=42)
        else:
            mi_scores = mutual_info_classif(features_df, target, random_state=42)
        
        # Store scores
        self.feature_scores = dict(zip(features_df.columns, mi_scores))
        
        # Filter by threshold
        above_threshold = {f: s for f, s in self.feature_scores.items() 
                          if s >= self.mi_threshold}
        
        # Sort by MI score
        sorted_features = sorted(above_threshold.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N
        top_features = [f for f, s in sorted_features[:self.n_features]]
        
        # Remove highly correlated features
        final_features = self._remove_correlated_features(features_df[top_features])
        
        # Get removed features
        all_features = set(features_df.columns)
        selected_features = set(final_features)
        removed_features = list(all_features - selected_features)
        
        result = FeatureSelectionResult(
            selected_features=final_features,
            removed_features=removed_features,
            feature_scores=self.feature_scores,
            selection_ratio=len(final_features) / len(features_df.columns)
        )
        
        return result
    
    def _remove_correlated_features(self, features_df: pd.DataFrame) -> List[str]:
        """Remove features with correlation > threshold"""
        if len(features_df.columns) <= 1:
            return features_df.columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs
        to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    # Remove the one with lower MI score
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]
                    
                    if self.feature_scores[feat_i] < self.feature_scores[feat_j]:
                        to_remove.add(feat_i)
                    else:
                        to_remove.add(feat_j)
        
        # Return features not marked for removal
        final_features = [f for f in features_df.columns if f not in to_remove]
        
        return final_features
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top N features by mutual information"""
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]
    
    def generate_report(self) -> str:
        """Generate feature selection report"""
        if not self.feature_scores:
            return "No feature scores available"
        
        sorted_features = sorted(self.feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        report = f"""
Feature Selection Report
{'=' * 50}
Total Features: {len(self.feature_scores)}
Selection Threshold: {self.mi_threshold}
Correlation Threshold: {self.correlation_threshold}
Target Features: {self.n_features}

Top Features by Mutual Information:
{'-' * 50}
"""
        
        for i, (feature, score) in enumerate(sorted_features[:20], 1):
            report += f"{i}. {feature}: {score:.4f}\n"
        
        return report


def reduce_feature_count(features_df: pd.DataFrame, target: pd.Series,
                        n_features: int = 20) -> Tuple[pd.DataFrame, FeatureSelectionResult]:
    """
    Reduce feature count to top N using mutual information.
    
    Args:
        features_df: DataFrame of features
        target: Target variable
        n_features: Number of features to keep
    
    Returns:
        Tuple of (reduced features DataFrame, selection result)
    """
    selector = MutualInformationSelector(n_features=n_features)
    result = selector.select_features(features_df, target)
    
    reduced_features = features_df[result.selected_features]
    
    return reduced_features, result


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample features
    np.random.seed(42)
    n = 1000
    n_features = 50
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(np.random.randn(n, n_features), columns=feature_names)
    
    # Create target with signal in first 5 features
    target = (features_df.iloc[:, 0] + features_df.iloc[:, 1] + 
              features_df.iloc[:, 2] + features_df.iloc[:, 3] + 
              features_df.iloc[:, 4]) * 0.2 + np.random.randn(n) * 0.5
    
    # Select features
    selector = MutualInformationSelector(n_features=20)
    result = selector.select_features(features_df, target)
    
    print(selector.generate_report())
    
    print(f"\nSelected {len(result.selected_features)} features")
    print(f"Removed {len(result.removed_features)} features")
    print(f"Selection ratio: {result.selection_ratio:.2%}")
