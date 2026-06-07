"""Risk management engine for the quantitative trading system."""

from .institutional_risk_engine import InstitutionalRiskEngine

RiskEngine = InstitutionalRiskEngine

__all__ = [
    "RiskEngine",
    "InstitutionalRiskEngine",
]
