"""
Alpha Registry - Track and manage alpha strategies
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class AlphaType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    OPTIONS = "options"
    FACTOR = "factor"


class AlphaStatus(Enum):
    PHASE1_RESEARCH = "phase1_research"
    PHASE2_PAPER_TRADING = "phase2_paper_trading"
    PHASE3_LIVE_SMALL = "phase3_live_small"
    PHASE4_SCALE = "phase4_scale"
    REJECTED = "rejected"


@dataclass
class AlphaDefinition:
    """Definition of an alpha strategy"""
    alpha_id: str
    name: str
    version: int
    alpha_type: AlphaType
    logic: str
    parameters: Dict[str, Any]
    expected_sharpe: float
    capacity_cr: float  # Capacity in Crores
    decay_months: int
    confidence: float
    status: AlphaStatus
    priority: int  # 1 = highest
    training_start: Optional[datetime] = None
    training_end: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    regime_dependencies: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alpha_id': self.alpha_id,
            'name': self.name,
            'version': self.version,
            'alpha_type': self.alpha_type.value,
            'logic': self.logic,
            'parameters': self.parameters,
            'expected_sharpe': self.expected_sharpe,
            'capacity_cr': self.capacity_cr,
            'decay_months': self.decay_months,
            'confidence': self.confidence,
            'status': self.status.value,
            'priority': self.priority,
            'training_start': self.training_start.isoformat() if self.training_start else None,
            'training_end': self.training_end.isoformat() if self.training_end else None,
            'created_at': self.created_at.isoformat(),
            'regime_dependencies': self.regime_dependencies
        }


class AlphaRegistry:
    """Registry for alpha strategies"""
    
    def __init__(self):
        self._alphas: Dict[str, AlphaDefinition] = {}
        self._initialize_default_alphas()
    
    def register(self, alpha: AlphaDefinition) -> None:
        """Register a new alpha"""
        self._alphas[alpha.alpha_id] = alpha
    
    def get(self, alpha_id: str) -> Optional[AlphaDefinition]:
        """Get an alpha by ID"""
        return self._alphas.get(alpha_id)
    
    def get_by_name(self, name: str) -> Optional[AlphaDefinition]:
        """Get an alpha by name"""
        for alpha in self._alphas.values():
            if alpha.name == name:
                return alpha
        return None
    
    def list_by_type(self, alpha_type: AlphaType) -> List[AlphaDefinition]:
        """List alphas by type"""
        return [a for a in self._alphas.values() if a.alpha_type == alpha_type]
    
    def list_by_status(self, status: AlphaStatus) -> List[AlphaDefinition]:
        """List alphas by status"""
        return [a for a in self._alphas.values() if a.status == status]
    
    def list_all(self) -> List[AlphaDefinition]:
        """List all alphas"""
        return list(self._alphas.values())
    
    def update_status(self, alpha_id: str, status: AlphaStatus) -> bool:
        """Update alpha status"""
        alpha = self.get(alpha_id)
        if alpha:
            alpha.status = status
            return True
        return False
    
    def _initialize_default_alphas(self) -> None:
        """Initialize default alpha strategies from blueprint"""
        
        # Momentum Alphas
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="TSMOM_12_1",
            version=1,
            alpha_type=AlphaType.MOMENTUM,
            logic="Buy past 12m winners, skip 1m",
            parameters={'lookback': 12, 'skip': 1},
            expected_sharpe=0.7,
            capacity_cr=500,
            decay_months=12,
            confidence=0.6,
            status=AlphaStatus.PHASE1_RESEARCH,
            priority=1,
            regime_dependencies=['bull_trend', 'low_vol']
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Dual_Momentum",
            version=1,
            alpha_type=AlphaType.MOMENTUM,
            logic="Absolute (200d SMA) + relative momentum",
            parameters={'sma_period': 200},
            expected_sharpe=0.8,
            capacity_cr=400,
            decay_months=12,
            confidence=0.7,
            status=AlphaStatus.PHASE1_RESEARCH,
            priority=2,
            regime_dependencies=['bull_trend']
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Sector_Momentum",
            version=1,
            alpha_type=AlphaType.MOMENTUM,
            logic="Long winning sectors, short losing",
            parameters={},
            expected_sharpe=0.6,
            capacity_cr=300,
            decay_months=6,
            confidence=0.5,
            status=AlphaStatus.PHASE2_PAPER_TRADING,
            priority=3,
            regime_dependencies=['bull_trend']
        ))
        
        # Mean Reversion Alphas
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="ORB_5min",
            version=1,
            alpha_type=AlphaType.MEAN_REVERSION,
            logic="5-min opening range breakout",
            parameters={'lookback': 5},
            expected_sharpe=0.9,
            capacity_cr=100,
            decay_months=6,
            confidence=0.8,
            status=AlphaStatus.PHASE1_RESEARCH,
            priority=1,
            regime_dependencies=['sideways']
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="VWAP_Reversion",
            version=1,
            alpha_type=AlphaType.MEAN_REVERSION,
            logic="Price crossing VWAP",
            parameters={},
            expected_sharpe=0.7,
            capacity_cr=200,
            decay_months=9,
            confidence=0.6,
            status=AlphaStatus.PHASE1_RESEARCH,
            priority=2,
            regime_dependencies=['sideways']
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="IBS",
            version=1,
            alpha_type=AlphaType.MEAN_REVERSION,
            logic="Internal Bar Strength - close near low buy, near high sell",
            parameters={},
            expected_sharpe=0.5,
            capacity_cr=300,
            decay_months=12,
            confidence=0.4,
            status=AlphaStatus.PHASE2_PAPER_TRADING,
            priority=4,
            regime_dependencies=['sideways', 'low_vol']
        ))
        
        # Volatility Alphas
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="VRP_Short_Straddle",
            version=1,
            alpha_type=AlphaType.VOLATILITY,
            logic="Short ATM straddle, delta hedge",
            parameters={},
            expected_sharpe=1.0,
            capacity_cr=200,
            decay_months=6,
            confidence=0.8,
            status=AlphaStatus.REJECTED,
            priority=99,
            regime_dependencies=['low_vol', 'contango']
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Volatility_Targeting",
            version=1,
            alpha_type=AlphaType.VOLATILITY,
            logic="Scale position to target volatility",
            parameters={'target_vol': 0.15},
            expected_sharpe=0.6,
            capacity_cr=500,
            decay_months=12,
            confidence=0.5,
            status=AlphaStatus.PHASE3_LIVE_SMALL,
            priority=3,
            regime_dependencies=[]
        ))
        
        # Options Alphas
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Put_Call_Carry",
            version=1,
            alpha_type=AlphaType.OPTIONS,
            logic="Long/short based on carry gap",
            parameters={},
            expected_sharpe=0.0,
            capacity_cr=0,
            decay_months=0,
            confidence=0.0,
            status=AlphaStatus.REJECTED,
            priority=99,
            regime_dependencies=[]
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Tail_Hedging",
            version=1,
            alpha_type=AlphaType.OPTIONS,
            logic="Long OTM puts for tail protection",
            parameters={'delta': 0.1},
            expected_sharpe=-0.2,
            capacity_cr=1000,
            decay_months=0,
            confidence=0.9,
            status=AlphaStatus.PHASE3_LIVE_SMALL,
            priority=3,
            regime_dependencies=['high_vol', 'panic']
        ))
        
        # Factor Alphas
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Low_Volatility",
            version=1,
            alpha_type=AlphaType.FACTOR,
            logic="Buy low vol stocks, short high vol",
            parameters={},
            expected_sharpe=0.5,
            capacity_cr=500,
            decay_months=12,
            confidence=0.4,
            status=AlphaStatus.PHASE4_SCALE,
            priority=5,
            regime_dependencies=[]
        ))
        
        self.register(AlphaDefinition(
            alpha_id=str(uuid.uuid4()),
            name="Value",
            version=1,
            alpha_type=AlphaType.FACTOR,
            logic="Buy cheap (B/P, E/P), short expensive",
            parameters={},
            expected_sharpe=0.4,
            capacity_cr=500,
            decay_months=18,
            confidence=0.3,
            status=AlphaStatus.PHASE4_SCALE,
            priority=6,
            regime_dependencies=[]
        ))


# Global registry instance
alpha_registry = AlphaRegistry()
