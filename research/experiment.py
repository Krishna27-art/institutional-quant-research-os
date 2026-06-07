"""
Research experiment module.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentRecord:
    """Experiment record."""
    name: str
    results: dict
    timestamp: str


class ExperimentStore:
    """Experiment store."""
    
    def __init__(self):
        self.experiments = []
    
    def save(self, experiment: ExperimentRecord) -> None:
        """Save experiment."""
        self.experiments.append(experiment)
    
    def load(self, name: str) -> ExperimentRecord | None:
        """Load experiment."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None
