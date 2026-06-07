"""
Promotion Pipeline with Validation Gates
Research → Paper Trading → Shadow Live → Production
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromotionStage(Enum):
    """Promotion stages"""
    RESEARCH = "research"
    PAPER_TRADING = "paper_trading"
    SHADOW_LIVE = "shadow_live"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class GateStatus(Enum):
    """Gate status"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationGate:
    """Validation gate definition"""
    gate_id: str
    name: str
    description: str
    stage: PromotionStage
    criteria: Dict[str, Any]
    gatekeeper: str  # "automated", "quant_analyst", "risk_manager", "quant_lead"
    status: GateStatus = GateStatus.PENDING
    passed_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionRequest:
    """Promotion request for an alpha/model"""
    request_id: str
    alpha_id: str
    alpha_name: str
    current_stage: PromotionStage
    target_stage: PromotionStage
    requested_by: str
    requested_at: datetime
    metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    model_id: Optional[str] = None
    gates: List[ValidationGate] = field(default_factory=list)
    status: str = "pending"  # "pending", "approved", "rejected"
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


class PromotionPipeline:
    """
    Promotion Pipeline with validation gates
    """
    
    def __init__(self):
        self.requests: Dict[str, PromotionRequest] = {}
        self.gate_definitions: List[Dict] = self._initialize_gate_definitions()
        
        logger.info("Promotion Pipeline initialized")
    
    def _initialize_gate_definitions(self) -> List[Dict]:
        """Initialize gate definitions for each stage"""
        return [
            # Research → Paper Trading gates
            {
                'stage': PromotionStage.RESEARCH,
                'gates': [
                    {
                        'gate_id': 'research_oos_sharpe',
                        'name': 'Out-of-Sample Sharpe',
                        'description': 'OOS Sharpe must be > 1.2',
                        'criteria': {'min_sharpe': 1.2},
                        'gatekeeper': 'automated'
                    },
                    {
                        'gate_id': 'research_max_dd',
                        'name': 'Maximum Drawdown',
                        'description': 'Max DD must be < 15%',
                        'criteria': {'max_drawdown': 0.15},
                        'gatekeeper': 'automated'
                    },
                    {
                        'gate_id': 'research_p_value',
                        'name': 'Statistical Significance',
                        'description': 'Bootstrap p-value must be < 0.05',
                        'criteria': {'max_p_value': 0.05},
                        'gatekeeper': 'automated'
                    },
                ]
            },
            # Paper Trading → Shadow Live gates
            {
                'stage': PromotionStage.PAPER_TRADING,
                'gates': [
                    {
                        'gate_id': 'paper_sharpe',
                        'name': 'Paper Trading Sharpe',
                        'description': 'Paper Sharpe (net of costs) must be > 1.0',
                        'criteria': {'min_sharpe': 1.0},
                        'gatekeeper': 'quant_analyst'
                    },
                    {
                        'gate_id': 'paper_duration',
                        'name': 'Paper Trading Duration',
                        'description': 'Must run for 30 trading days',
                        'criteria': {'min_days': 30},
                        'gatekeeper': 'automated'
                    },
                    {
                        'gate_id': 'paper_technical',
                        'name': 'Technical Review',
                        'description': 'No technical issues identified',
                        'criteria': {},
                        'gatekeeper': 'quant_analyst'
                    },
                ]
            },
            # Shadow Live → Production gates
            {
                'stage': PromotionStage.SHADOW_LIVE,
                'gates': [
                    {
                        'gate_id': 'shadow_sharpe',
                        'name': 'Shadow Live Sharpe',
                        'description': 'Real Sharpe (after costs) must be > 0.8',
                        'criteria': {'min_sharpe': 0.8},
                        'gatekeeper': 'risk_manager'
                    },
                    {
                        'gate_id': 'shadow_max_loss',
                        'name': 'Maximum Daily Loss',
                        'description': 'Max daily loss must be < 2% of allocated capital',
                        'criteria': {'max_daily_loss_pct': 0.02},
                        'gatekeeper': 'risk_manager'
                    },
                    {
                        'gate_id': 'shadow_duration',
                        'name': 'Shadow Duration',
                        'description': 'Must run for 20 trading days',
                        'criteria': {'min_days': 20},
                        'gatekeeper': 'automated'
                    },
                    {
                        'gate_id': 'shadow_approval',
                        'name': 'Risk Manager Approval',
                        'description': 'Risk manager and quant lead approval',
                        'criteria': {},
                        'gatekeeper': 'quant_lead'
                    },
                ]
            },
        ]
    
    def create_promotion_request(
        self,
        alpha_id: str,
        alpha_name: str,
        current_stage: PromotionStage,
        target_stage: PromotionStage,
        requested_by: str,
        metrics: Dict[str, float],
        backtest_results: Dict[str, Any],
        model_id: Optional[str] = None
    ) -> PromotionRequest:
        """
        Create a promotion request
        
        Args:
            alpha_id: Alpha identifier
            alpha_name: Alpha name
            current_stage: Current stage
            target_stage: Target stage
            requested_by: User requesting promotion
            metrics: Performance metrics
            backtest_results: Backtest results
            model_id: Model identifier (if applicable)
            
        Returns:
            PromotionRequest
        """
        request_id = f"PROMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alpha_id}"
        
        # Get gates for target stage
        gates = []
        stage_defs = next(
            (s for s in self.gate_definitions if s['stage'] == target_stage),
            None
        )
        
        if stage_defs:
            for gate_def in stage_defs['gates']:
                gate = ValidationGate(
                    gate_id=gate_def['gate_id'],
                    name=gate_def['name'],
                    description=gate_def['description'],
                    stage=target_stage,
                    criteria=gate_def['criteria'],
                    gatekeeper=gate_def['gatekeeper']
                )
                gates.append(gate)
        
        request = PromotionRequest(
            request_id=request_id,
            alpha_id=alpha_id,
            alpha_name=alpha_name,
            current_stage=current_stage,
            target_stage=target_stage,
            requested_by=requested_by,
            requested_at=datetime.now(),
            metrics=metrics,
            backtest_results=backtest_results,
            model_id=model_id,
            gates=gates,
            status="pending"
        )
        
        self.requests[request_id] = request
        
        logger.info(f"Created promotion request {request_id} for {alpha_name}")
        
        return request
    
    def evaluate_gate(
        self,
        request_id: str,
        gate_id: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> GateStatus:
        """
        Evaluate a validation gate
        
        Args:
            request_id: Promotion request ID
            gate_id: Gate ID to evaluate
            additional_data: Additional data for evaluation
            
        Returns:
            GateStatus
        """
        request = self.requests.get(request_id)
        if not request:
            logger.error(f"Request {request_id} not found")
            return GateStatus.FAILED
        
        gate = next((g for g in request.gates if g.gate_id == gate_id), None)
        if not gate:
            logger.error(f"Gate {gate_id} not found")
            return GateStatus.FAILED
        
        # Evaluate based on gate type
        if gate.gatekeeper == "automated":
            passed = self._evaluate_automated_gate(request, gate, additional_data)
        else:
            # Manual gates require human approval
            passed = False  # Default to failed until approved
        
        gate.status = GateStatus.PASSED if passed else GateStatus.FAILED
        
        if passed:
            gate.passed_at = datetime.now()
        else:
            gate.failure_reason = "Criteria not met"
        
        logger.info(f"Gate {gate_id} evaluated: {gate.status.value}")
        
        return gate.status
    
    def _evaluate_automated_gate(
        self,
        request: PromotionRequest,
        gate: ValidationGate,
        additional_data: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate automated gate"""
        criteria = gate.criteria
        metrics = request.metrics
        
        if gate.gate_id == 'research_oos_sharpe':
            return metrics.get('oos_sharpe', 0) >= criteria['min_sharpe']
        
        elif gate.gate_id == 'research_max_dd':
            return abs(metrics.get('max_drawdown', 1)) <= criteria['max_drawdown']
        
        elif gate.gate_id == 'research_p_value':
            return metrics.get('p_value', 1) <= criteria['max_p_value']
        
        elif gate.gate_id == 'paper_sharpe':
            return metrics.get('paper_sharpe', 0) >= criteria['min_sharpe']
        
        elif gate.gate_id == 'paper_duration':
            return additional_data.get('duration_days', 0) >= criteria['min_days']
        
        elif gate.gate_id == 'shadow_sharpe':
            return metrics.get('shadow_sharpe', 0) >= criteria['min_sharpe']
        
        elif gate.gate_id == 'shadow_max_loss':
            return abs(metrics.get('max_daily_loss_pct', 1)) <= criteria['max_daily_loss_pct']
        
        elif gate.gate_id == 'shadow_duration':
            return additional_data.get('duration_days', 0) >= criteria['min_days']
        
        return True  # Default pass for unknown gates
    
    def approve_gate(
        self,
        request_id: str,
        gate_id: str,
        approved_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Manually approve a gate (for manual gatekeepers)
        
        Args:
            request_id: Promotion request ID
            gate_id: Gate ID to approve
            approved_by: User approving
            notes: Approval notes
            
        Returns:
            True if approved successfully
        """
        request = self.requests.get(request_id)
        if not request:
            logger.error(f"Request {request_id} not found")
            return False
        
        gate = next((g for g in request.gates if g.gate_id == gate_id), None)
        if not gate:
            logger.error(f"Gate {gate_id} not found")
            return False
        
        gate.status = GateStatus.PASSED
        gate.passed_at = datetime.now()
        gate.metadata['approved_by'] = approved_by
        gate.metadata['notes'] = notes
        
        logger.info(f"Gate {gate_id} approved by {approved_by}")
        
        return True
    
    def reject_gate(
        self,
        request_id: str,
        gate_id: str,
        rejected_by: str,
        reason: str
    ) -> bool:
        """
        Reject a gate
        
        Args:
            request_id: Promotion request ID
            gate_id: Gate ID to reject
            rejected_by: User rejecting
            reason: Rejection reason
            
        Returns:
            True if rejected successfully
        """
        request = self.requests.get(request_id)
        if not request:
            logger.error(f"Request {request_id} not found")
            return False
        
        gate = next((g for g in request.gates if g.gate_id == gate_id), None)
        if not gate:
            logger.error(f"Gate {gate_id} not found")
            return False
        
        gate.status = GateStatus.FAILED
        gate.failure_reason = reason
        gate.metadata['rejected_by'] = rejected_by
        
        logger.info(f"Gate {gate_id} rejected by {rejected_by}: {reason}")
        
        return True
    
    def check_all_gates_passed(self, request_id: str) -> bool:
        """Check if all gates have passed"""
        request = self.requests.get(request_id)
        if not request:
            return False
        
        return all(g.status == GateStatus.PASSED for g in request.gates)
    
    def approve_promotion(self, request_id: str, approved_by: str) -> bool:
        """
        Approve promotion request (all gates must pass)
        
        Args:
            request_id: Promotion request ID
            approved_by: User approving promotion
            
        Returns:
            True if approved successfully
        """
        request = self.requests.get(request_id)
        if not request:
            logger.error(f"Request {request_id} not found")
            return False
        
        # Check all gates passed
        if not self.check_all_gates_passed(request_id):
            logger.error(f"Not all gates passed for request {request_id}")
            return False
        
        request.status = "approved"
        request.approved_at = datetime.now()
        request.metadata['approved_by'] = approved_by
        
        logger.info(f"Promotion request {request_id} approved by {approved_by}")
        
        return True
    
    def reject_promotion(
        self,
        request_id: str,
        rejected_by: str,
        reason: str
    ) -> bool:
        """
        Reject promotion request
        
        Args:
            request_id: Promotion request ID
            rejected_by: User rejecting
            reason: Rejection reason
            
        Returns:
            True if rejected successfully
        """
        request = self.requests.get(request_id)
        if not request:
            logger.error(f"Request {request_id} not found")
            return False
        
        request.status = "rejected"
        request.rejected_at = datetime.now()
        request.rejection_reason = reason
        request.metadata['rejected_by'] = rejected_by
        
        logger.info(f"Promotion request {request_id} rejected by {rejected_by}: {reason}")
        
        return True
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of promotion request"""
        request = self.requests.get(request_id)
        if not request:
            return {"error": "Request not found"}
        
        gates_status = [
            {
                'gate_id': g.gate_id,
                'name': g.name,
                'status': g.status.value,
                'gatekeeper': g.gatekeeper,
                'passed_at': g.passed_at.isoformat() if g.passed_at else None,
                'failure_reason': g.failure_reason,
            }
            for g in request.gates
        ]
        
        return {
            'request_id': request.request_id,
            'alpha_id': request.alpha_id,
            'alpha_name': request.alpha_name,
            'current_stage': request.current_stage.value,
            'target_stage': request.target_stage.value,
            'status': request.status,
            'requested_by': request.requested_by,
            'requested_at': request.requested_at.isoformat(),
            'approved_at': request.approved_at.isoformat() if request.approved_at else None,
            'rejected_at': request.rejected_at.isoformat() if request.rejected_at else None,
            'rejection_reason': request.rejection_reason,
            'gates': gates_status,
            'all_gates_passed': self.check_all_gates_passed(request_id),
        }
    
    def get_pending_requests(self, stage: Optional[PromotionStage] = None) -> List[PromotionRequest]:
        """Get pending promotion requests"""
        pending = []
        
        for request in self.requests.values():
            if request.status == "pending":
                if stage is None or request.target_stage == stage:
                    pending.append(request)
        
        return pending
    
    def get_promotion_history(self, alpha_id: str) -> List[PromotionRequest]:
        """Get promotion history for an alpha"""
        history = [
            r for r in self.requests.values()
            if r.alpha_id == alpha_id
        ]
        
        return sorted(history, key=lambda x: x.requested_at, reverse=True)


def simulate_promotion_pipeline():
    """Simulate promotion pipeline"""
    
    print("="*60)
    print("PROMOTION PIPELINE SIMULATION")
    print("="*60)
    
    # Initialize pipeline
    pipeline = PromotionPipeline()
    
    # Create promotion request: Research → Paper Trading
    print("\n1. Creating promotion request (Research → Paper Trading)...")
    request1 = pipeline.create_promotion_request(
        alpha_id="ORB_v3",
        alpha_name="Opening Range Breakout v3",
        current_stage=PromotionStage.RESEARCH,
        target_stage=PromotionStage.PAPER_TRADING,
        requested_by="quant_researcher",
        metrics={
            'oos_sharpe': 1.35,
            'max_drawdown': -0.12,
            'p_value': 0.03,
        },
        backtest_results={
            'total_return': 0.45,
            'win_rate': 0.58,
        }
    )
    
    print(f"  Request ID: {request1.request_id}")
    print(f"  Gates: {len(request1.gates)}")
    
    # Evaluate automated gates
    print("\n2. Evaluating automated gates...")
    for gate in request1.gates:
        if gate.gatekeeper == "automated":
            status = pipeline.evaluate_gate(request1.request_id, gate.gate_id)
            print(f"  {gate.name}: {status.value}")
    
    # Approve manual gate
    print("\n3. Approving manual gate (Technical Review)...")
    pipeline.approve_gate(
        request1.request_id,
        "paper_technical",
        "quant_analyst",
        "No technical issues found"
    )
    
    # Check status
    print("\n4. Checking request status...")
    status = pipeline.get_request_status(request1.request_id)
    print(f"  All gates passed: {status['all_gates_passed']}")
    
    # Approve promotion
    print("\n5. Approving promotion...")
    approved = pipeline.approve_promotion(request1.request_id, "quant_lead")
    print(f"  Approved: {approved}")
    
    # Create second request: Paper Trading → Shadow Live
    print("\n6. Creating promotion request (Paper Trading → Shadow Live)...")
    request2 = pipeline.create_promotion_request(
        alpha_id="ORB_v3",
        alpha_name="Opening Range Breakout v3",
        current_stage=PromotionStage.PAPER_TRADING,
        target_stage=PromotionStage.SHADOW_LIVE,
        requested_by="quant_researcher",
        metrics={
            'paper_sharpe': 1.15,
            'max_daily_loss_pct': -0.015,
        },
        backtest_results={
            'paper_return': 0.08,
        },
        additional_data={'duration_days': 30}
    )
    
    print(f"  Request ID: {request2.request_id}")
    
    # Evaluate gates
    print("\n7. Evaluating gates for shadow live...")
    for gate in request2.gates:
        if gate.gatekeeper == "automated":
            status = pipeline.evaluate_gate(request2.request_id, gate.gate_id, {'duration_days': 25})
            print(f"  {gate.name}: {status.value}")
    
    # Get pending requests
    print("\n8. Getting pending requests...")
    pending = pipeline.get_pending_requests()
    print(f"  Pending requests: {len(pending)}")
    
    # Get promotion history
    print("\n9. Getting promotion history...")
    history = pipeline.get_promotion_history("ORB_v3")
    print(f"  Total requests: {len(history)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_promotion_pipeline()
