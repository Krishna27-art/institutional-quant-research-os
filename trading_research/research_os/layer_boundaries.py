"""Layer boundaries to eliminate conceptual redundancy."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


class SystemLayer(Enum):
    RAW_STATE = "raw_state"
    MARKET_REGIME = "market_regime"
    PARTICIPANT_STATE = "participant_state"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    STRATEGY_CONTEXT = "strategy_context"


class LayerBoundaryViolation:
    def __init__(self, file_path: str, violation_type: str, description: str, severity: str):
        self.file_path = file_path
        self.violation_type = violation_type
        self.description = description
        self.severity = severity

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "violation_type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
        }


class BoundaryEnforcer:
    MODULE_LAYER_MAPPING = {
        "trading_research/research/feature_set.py": SystemLayer.RAW_STATE,
        "trading_research/data/market_breadth.py": SystemLayer.RAW_STATE,
        "trading_research/research/regime_detection.py": SystemLayer.MARKET_REGIME,
        "trading_research/core/participant_state.py": SystemLayer.PARTICIPANT_STATE,
        "trading_research/intelligence/participant_model/participant_inference.py": SystemLayer.PARTICIPANT_STATE,
        "trading_research/participants/gap_participant_models.py": SystemLayer.PARTICIPANT_STATE,
        "trading_research/intelligence/behavioral_activation/behavioral_engine.py": SystemLayer.BEHAVIORAL_ACTIVATION,
        "trading_research/intelligence/flow_analysis/gap_behavior_extractor.py": SystemLayer.BEHAVIORAL_ACTIVATION,
        "trading_research/core/market_context.py": SystemLayer.STRATEGY_CONTEXT,
    }

    BOUNDARY_VIOLATIONS = [
        LayerBoundaryViolation(
            file_path="trading_research/core/market_context.py",
            violation_type="DUPLICATE_REGIME_LOGIC",
            description="LocalRegime enum duplicates Regime enum from regime_detection.py",
            severity="HIGH",
        ),
        LayerBoundaryViolation(
            file_path="trading_research/research/enhanced_regime.py",
            violation_type="DUPLICATE_REGIME_LOGIC",
            description="Enhanced regime detection duplicates base regime_detection.py",
            severity="HIGH",
        ),
        LayerBoundaryViolation(
            file_path="trading_research/research/enhanced_regime_v3.py",
            violation_type="DUPLICATE_REGIME_LOGIC",
            description="Third version of regime detection - consolidation needed",
            severity="HIGH",
        ),
    ]

    @classmethod
    def get_violations(cls) -> List[LayerBoundaryViolation]:
        return cls.BOUNDARY_VIOLATIONS

