"""src/data — unified data layer package."""

from src.data.quality_gate import DataQualityGate, GateResult, get_quality_gate

__all__ = ["DataQualityGate", "GateResult", "get_quality_gate"]
