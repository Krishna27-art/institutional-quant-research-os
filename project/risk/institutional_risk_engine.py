"""
Compatibility facade for InstitutionalRiskEngine.
"""

from src.risk.institutional_risk_engine import (
    InstitutionalRiskEngine,
    Position,
    RiskMetrics
)

__all__ = [
    "InstitutionalRiskEngine",
    "Position",
    "RiskMetrics"
]
