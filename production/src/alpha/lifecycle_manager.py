"""
Alpha Lifecycle Management
Tracks alphas across their lifecycle: INCUBATION -> ACTIVE -> PROBATION -> RETIRED -> DEAD.
Includes YAML logging to build an Alpha Graveyard.
"""

import os
import yaml
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

class AlphaState(Enum):
    INCUBATION = "incubation"
    ACTIVE = "active"
    PROBATION = "probation"
    RETIRED = "retired"
    DEAD = "dead"

@dataclass
class AlphaGraveyardEntry:
    alpha_id: str
    created: str
    author: str
    hypothesis: str
    live_start: Optional[str] = None
    live_end: Optional[str] = None
    death_reason: Optional[str] = None
    max_sharpe: Optional[float] = None
    avg_ic: Optional[float] = None

class LifecycleManager:
    """Manages alpha states and writes graveyard data."""
    def __init__(self, registry_dir: str = "alpha_registry"):
        self.registry_dir = registry_dir
        self._ensure_dirs()

    def _ensure_dirs(self):
        for state in [s.value for s in AlphaState if s != AlphaState.INCUBATION]:
            os.makedirs(os.path.join(self.registry_dir, state), exist_ok=True)

    def transition_alpha(self, alpha_id: str, new_state: AlphaState, 
                         metadata: Optional[Dict[str, Any]] = None):
        """Move an alpha to a new state and store metadata."""
        if new_state == AlphaState.DEAD and metadata:
            self._write_graveyard_entry(alpha_id, metadata)
            
    def _write_graveyard_entry(self, alpha_id: str, metadata: Dict[str, Any]):
        """Write a YAML file to the dead folder."""
        entry = AlphaGraveyardEntry(
            alpha_id=alpha_id,
            created=metadata.get('created', datetime.now().isoformat()),
            author=metadata.get('author', 'system'),
            hypothesis=metadata.get('hypothesis', 'Unknown hypothesis'),
            live_start=metadata.get('live_start'),
            live_end=datetime.now().isoformat(),
            death_reason=metadata.get('death_reason', 'Unspecified decay'),
            max_sharpe=metadata.get('max_sharpe'),
            avg_ic=metadata.get('avg_ic')
        )
        
        filepath = os.path.join(self.registry_dir, AlphaState.DEAD.value, f"{alpha_id}.yaml")
        with open(filepath, 'w') as f:
            yaml.dump(asdict(entry), f, default_flow_style=False)
