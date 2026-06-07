"""
Feature Version Manager - Track feature versions and lineage
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class FeatureVersion:
    """Represents a version of a feature"""
    name: str
    version: str
    formula: str
    parameters: Dict[str, Any]
    created_at: datetime
    hash: str
    dependencies: List[str] = field(default_factory=list)
    deprecated: bool = False
    parent_version: Optional[str] = None


class FeatureVersionManager:
    """Manage feature versions and lineage"""
    
    def __init__(self):
        self.versions: Dict[str, List[FeatureVersion]] = {}
    
    def register_version(self, name: str, version: str, formula: str, 
                         parameters: Dict[str, Any], 
                         dependencies: Optional[List[str]] = None,
                         parent_version: Optional[str] = None) -> FeatureVersion:
        """Register a new feature version"""
        version_hash = self._compute_hash(formula, parameters)
        
        feature_version = FeatureVersion(
            name=name,
            version=version,
            formula=formula,
            parameters=parameters,
            created_at=datetime.now(),
            hash=version_hash,
            dependencies=dependencies or [],
            parent_version=parent_version
        )
        
        if name not in self.versions:
            self.versions[name] = []
        
        self.versions[name].append(feature_version)
        return feature_version
    
    def get_latest_version(self, name: str) -> Optional[FeatureVersion]:
        """Get the latest version of a feature"""
        if name not in self.versions or not self.versions[name]:
            return None
        return self.versions[name][-1]
    
    def get_version(self, name: str, version: str) -> Optional[FeatureVersion]:
        """Get a specific version of a feature"""
        if name not in self.versions:
            return None
        for fv in self.versions[name]:
            if fv.version == version:
                return fv
        return None
    
    def list_versions(self, name: str) -> List[FeatureVersion]:
        """List all versions of a feature"""
        return self.versions.get(name, [])
    
    def deprecate_version(self, name: str, version: str) -> bool:
        """Deprecate a feature version"""
        feature_version = self.get_version(name, version)
        if feature_version:
            feature_version.deprecated = True
            return True
        return False
    
    def get_lineage(self, name: str) -> List[FeatureVersion]:
        """Get the full lineage of a feature"""
        return self.versions.get(name, [])
    
    def check_hash_collision(self, name: str, formula: str, 
                           parameters: Dict[str, Any]) -> bool:
        """Check if a feature with the same hash already exists"""
        version_hash = self._compute_hash(formula, parameters)
        if name not in self.versions:
            return False
        return any(fv.hash == version_hash for fv in self.versions[name])
    
    def _compute_hash(self, formula: str, parameters: Dict[str, Any]) -> str:
        """Compute hash of formula and parameters"""
        content = f"{formula}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def export_registry(self) -> Dict[str, List[Dict]]:
        """Export version registry as dict"""
        export = {}
        for name, versions in self.versions.items():
            export[name] = [
                {
                    'version': fv.version,
                    'formula': fv.formula,
                    'parameters': fv.parameters,
                    'created_at': fv.created_at.isoformat(),
                    'hash': fv.hash,
                    'dependencies': fv.dependencies,
                    'deprecated': fv.deprecated,
                    'parent_version': fv.parent_version
                }
                for fv in versions
            ]
        return export
