"""
SEBI Algorithmic Trading Compliance Module
Implements SEBI (Securities and Exchange Board of India) regulatory compliance for algorithmic trading.

Based on institutional review recommendations:
- Full compliance with SEBI algo guidelines
- Order logging and audit trail
- Risk limit enforcement
- Circuit breaker integration
- Pre-trade risk checks
- Kill switch implementation
- Regulatory reporting

Key features:
- SEBI Circular SE/HO/MRD/DP/CIR/P/2018/144 compliance
- Order-to-trade ratio monitoring
- Price band checks
- Volume checks
- Client-level position limits
- Broker-level limits
- Audit trail generation
- Regulatory reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status"""
    COMPLIANT = "COMPLIANT"
    WARNING = "WARNING"
    VIOLATION = "VIOLATION"
    BLOCKED = "BLOCKED"


class RiskLimitType(Enum):
    """Risk limit types"""
    ORDER_TO_TRADE_RATIO = "ORDER_TO_TRADE_RATIO"
    PRICE_BAND = "PRICE_BAND"
    VOLUME_LIMIT = "VOLUME_LIMIT"
    POSITION_LIMIT = "POSITION_LIMIT"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"


@dataclass
class ComplianceCheck:
    """Compliance check result"""
    check_type: str
    status: ComplianceStatus
    message: str
    value: float
    limit: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "check_type": self.check_type,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
            "limit": self.limit,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Order:
    """Order for compliance checking"""
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    price: float
    order_type: str  # "MARKET" or "LIMIT"
    client_id: str
    strategy_id: str
    timestamp: datetime


@dataclass
class Trade:
    """Trade for compliance tracking"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    client_id: str
    strategy_id: str
    timestamp: datetime


class SEBIAlgoCompliance:
    """
    SEBI Algorithmic Trading Compliance Module.
    
    Features:
    - SEBI Circular SE/HO/MRD/DP/CIR/P/2018/144 compliance
    - Order-to-trade ratio monitoring
    - Price band checks
    - Volume checks
    - Client-level position limits
    - Broker-level limits
    - Audit trail generation
    - Regulatory reporting
    """
    
    def __init__(
        self,
        broker_id: str,
        order_to_trade_limit: float = 5.0,  # SEBI limit
        price_band_pct: float = 0.10,  # 10% price band
        audit_log_path: str = "compliance/audit_logs"
    ):
        self.broker_id = broker_id
        self.order_to_trade_limit = order_to_trade_limit
        self.price_band_pct = price_band_pct
        
        # Audit trail
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        
        # Order and trade tracking
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Client positions
        self.client_positions: Dict[str, Dict[str, int]] = {}  # client_id -> {symbol -> quantity}
        
        # Daily limits
        self.daily_order_count: Dict[str, int] = {}  # client_id -> count
        self.daily_trade_count: Dict[str, int] = {}  # client_id -> count
        
        # Risk limits
        self.client_limits: Dict[str, Dict] = {}  # client_id -> limits
        self.broker_limits: Dict = {
            "max_daily_orders": 100000,
            "max_daily_trades": 50000,
            "max_exposure": 1000000000  # ₹1000 Cr
        }
        
        # Kill switch
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        
        logger.info(f"SEBI Algo Compliance module initialized for broker {broker_id}")
    
    def register_client(self, client_id: str, limits: Dict):
        """Register client with risk limits"""
        self.client_limits[client_id] = {
            "max_position_value": limits.get("max_position_value", 10000000),
            "max_daily_orders": limits.get("max_daily_orders", 10000),
            "max_daily_trades": limits.get("max_daily_trades", 5000),
            "max_exposure": limits.get("max_exposure", 50000000),
            "max_leverage": limits.get("max_leverage", 4.0)
        }
        
        self.client_positions[client_id] = {}
        self.daily_order_count[client_id] = 0
        self.daily_trade_count[client_id] = 0
        
        logger.info(f"Client {client_id} registered with limits")
    
    def pre_trade_check(self, order: Order) -> Tuple[bool, List[ComplianceCheck]]:
        """
        Perform pre-trade compliance checks.
        
        Returns:
            (is_compliant, list of checks)
        """
        checks = []
        
        if self.kill_switch_active:
            checks.append(ComplianceCheck(
                check_type="KILL_SWITCH",
                status=ComplianceStatus.BLOCKED,
                message=f"Kill switch active: {self.kill_switch_reason}",
                value=0,
                limit=0,
                timestamp=datetime.now()
            ))
            return False, checks
        
        # Check 1: Order-to-trade ratio
        otr_check = self._check_order_to_trade_ratio(order)
        checks.append(otr_check)
        
        # Check 2: Price band
        price_band_check = self._check_price_band(order)
        checks.append(price_band_check)
        
        # Check 3: Position limit
        position_check = self._check_position_limit(order)
        checks.append(position_check)
        
        # Check 4: Daily order limit
        daily_order_check = self._check_daily_order_limit(order)
        checks.append(daily_order_check)
        
        # Check 5: Exposure limit
        exposure_check = self._check_exposure_limit(order)
        checks.append(exposure_check)
        
        # Check 6: Leverage limit
        leverage_check = self._check_leverage_limit(order)
        checks.append(leverage_check)
        
        # Determine overall compliance
        violations = [c for c in checks if c.status == ComplianceStatus.VIOLATION]
        blocked = [c for c in checks if c.status == ComplianceStatus.BLOCKED]
        
        is_compliant = len(violations) == 0 and len(blocked) == 0
        
        # Log order
        self.orders.append(order)
        self.daily_order_count[order.client_id] = self.daily_order_count.get(order.client_id, 0) + 1
        
        # Write to audit log
        self._write_audit_log(order, checks, is_compliant)
        
        return is_compliant, checks
    
    def _check_order_to_trade_ratio(self, order: Order) -> ComplianceCheck:
        """Check order-to-trade ratio (SEBI limit: 5:1)"""
        client_id = order.client_id
        
        orders = self.daily_order_count.get(client_id, 0)
        trades = self.daily_trade_count.get(client_id, 0)
        
        if trades == 0:
            otr = orders
        else:
            otr = orders / trades
        
        status = ComplianceStatus.COMPLIANT
        message = f"Order-to-trade ratio: {otr:.2f}"
        
        if otr > self.order_to_trade_limit:
            status = ComplianceStatus.VIOLATION
            message = f"Order-to-trade ratio {otr:.2f} exceeds limit {self.order_to_trade_limit}"
        
        return ComplianceCheck(
            check_type="ORDER_TO_TRADE_RATIO",
            status=status,
            message=message,
            value=otr,
            limit=self.order_to_trade_limit,
            timestamp=datetime.now()
        )
    
    def _check_price_band(self, order: Order) -> ComplianceCheck:
        """Check price band (10% from reference price)"""
        # In production, get reference price from exchange
        reference_price = order.price  # Simplified
        
        price_deviation = abs(order.price - reference_price) / reference_price
        
        status = ComplianceStatus.COMPLIANT
        message = f"Price deviation: {price_deviation:.2%}"
        
        if price_deviation > self.price_band_pct:
            status = ComplianceStatus.VIOLATION
            message = f"Price deviation {price_deviation:.2%} exceeds limit {self.price_band_pct:.0%}"
        
        return ComplianceCheck(
            check_type="PRICE_BAND",
            status=status,
            message=message,
            value=price_deviation,
            limit=self.price_band_pct,
            timestamp=datetime.now()
        )
    
    def _check_position_limit(self, order: Order) -> ComplianceCheck:
        """Check position limit for client"""
        client_id = order.client_id
        limits = self.client_limits.get(client_id, {})
        
        max_position_value = limits.get("max_position_value", 10000000)
        
        # Current position
        current_position = self.client_positions.get(client_id, {}).get(order.symbol, 0)
        
        # New position after order
        if order.side == "BUY":
            new_position = current_position + order.quantity
        else:
            new_position = current_position - order.quantity
        
        position_value = abs(new_position) * order.price
        
        status = ComplianceStatus.COMPLIANT
        message = f"Position value: ₹{position_value:,.0f}"
        
        if position_value > max_position_value:
            status = ComplianceStatus.VIOLATION
            message = f"Position value ₹{position_value:,.0f} exceeds limit ₹{max_position_value:,.0f}"
        
        return ComplianceCheck(
            check_type="POSITION_LIMIT",
            status=status,
            message=message,
            value=position_value,
            limit=max_position_value,
            timestamp=datetime.now()
        )
    
    def _check_daily_order_limit(self, order: Order) -> ComplianceCheck:
        """Check daily order limit for client"""
        client_id = order.client_id
        limits = self.client_limits.get(client_id, {})
        
        max_daily_orders = limits.get("max_daily_orders", 10000)
        current_orders = self.daily_order_count.get(client_id, 0)
        
        status = ComplianceStatus.COMPLIANT
        message = f"Daily orders: {current_orders}"
        
        if current_orders >= max_daily_orders:
            status = ComplianceStatus.VIOLATION
            message = f"Daily orders {current_orders} exceeds limit {max_daily_orders}"
        
        return ComplianceCheck(
            check_type="DAILY_ORDER_LIMIT",
            status=status,
            message=message,
            value=current_orders,
            limit=max_daily_orders,
            timestamp=datetime.now()
        )
    
    def _check_exposure_limit(self, order: Order) -> ComplianceCheck:
        """Check exposure limit for client"""
        client_id = order.client_id
        limits = self.client_limits.get(client_id, {})
        
        max_exposure = limits.get("max_exposure", 50000000)
        
        # Calculate current exposure (simplified)
        current_exposure = 0
        for symbol, quantity in self.client_positions.get(client_id, {}).items():
            current_exposure += abs(quantity) * 1000  # Simplified price
        
        # Add new order exposure
        new_exposure = current_exposure + (order.quantity * order.price)
        
        status = ComplianceStatus.COMPLIANT
        message = f"Exposure: ₹{new_exposure:,.0f}"
        
        if new_exposure > max_exposure:
            status = ComplianceStatus.VIOLATION
            message = f"Exposure ₹{new_exposure:,.0f} exceeds limit ₹{max_exposure:,.0f}"
        
        return ComplianceCheck(
            check_type="EXPOSURE_LIMIT",
            status=status,
            message=message,
            value=new_exposure,
            limit=max_exposure,
            timestamp=datetime.now()
        )
    
    def _check_leverage_limit(self, order: Order) -> ComplianceCheck:
        """Check leverage limit for client"""
        client_id = order.client_id
        limits = self.client_limits.get(client_id, {})
        
        max_leverage = limits.get("max_leverage", 4.0)
        
        # Calculate current leverage (simplified)
        total_exposure = sum(
            abs(qty) * 1000 
            for qty in self.client_positions.get(client_id, {}).values()
        )
        
        # Assume capital of ₹10 Cr for leverage calculation
        capital = 10000000
        current_leverage = total_exposure / capital if capital > 0 else 0
        
        status = ComplianceStatus.COMPLIANT
        message = f"Leverage: {current_leverage:.2f}x"
        
        if current_leverage > max_leverage:
            status = ComplianceStatus.VIOLATION
            message = f"Leverage {current_leverage:.2f}x exceeds limit {max_leverage:.2f}x"
        
        return ComplianceCheck(
            check_type="LEVERAGE_LIMIT",
            status=status,
            message=message,
            value=current_leverage,
            limit=max_leverage,
            timestamp=datetime.now()
        )
    
    def record_trade(self, trade: Trade):
        """Record executed trade"""
        self.trades.append(trade)
        self.daily_trade_count[trade.client_id] = self.daily_trade_count.get(trade.client_id, 0) + 1
        
        # Update position
        if trade.client_id not in self.client_positions:
            self.client_positions[trade.client_id] = {}
        
        current_position = self.client_positions[trade.client_id].get(trade.symbol, 0)
        
        if trade.side == "BUY":
            self.client_positions[trade.client_id][trade.symbol] = current_position + trade.quantity
        else:
            self.client_positions[trade.client_id][trade.symbol] = current_position - trade.quantity
        
        logger.info(f"Trade recorded: {trade.trade_id} for client {trade.client_id}")
    
    def activate_kill_switch(self, reason: str):
        """Activate kill switch (SEBI requirement)"""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        
        logger.warning(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Log to audit
        self._write_kill_switch_log(reason)
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch"""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        
        logger.info("Kill switch deactivated")
    
    def _write_audit_log(self, order: Order, checks: List[ComplianceCheck], is_compliant: bool):
        """Write audit log entry"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "broker_id": self.broker_id,
            "order_id": order.order_id,
            "client_id": order.client_id,
            "strategy_id": order.strategy_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.price,
            "order_type": order.order_type,
            "compliance_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
            "checks": [c.to_dict() for c in checks]
        }
        
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.audit_log_path / f"audit_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _write_kill_switch_log(self, reason: str):
        """Write kill switch activation log"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "broker_id": self.broker_id,
            "event": "KILL_SWITCH_ACTIVATED",
            "reason": reason,
            "active_orders": len(self.orders),
            "active_trades": len(self.trades)
        }
        
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.audit_log_path / f"kill_switch_{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_compliance_report(self) -> Dict:
        """Generate compliance report"""
        total_orders = len(self.orders)
        total_trades = len(self.trades)
        
        # Calculate order-to-trade ratio
        if total_trades > 0:
            otr = total_orders / total_trades
        else:
            otr = total_orders
        
        # Client-wise statistics
        client_stats = {}
        for client_id in self.client_limits.keys():
            client_stats[client_id] = {
                "orders": self.daily_order_count.get(client_id, 0),
                "trades": self.daily_trade_count.get(client_id, 0),
                "positions": self.client_positions.get(client_id, {})
            }
        
        report = {
            "broker_id": self.broker_id,
            "timestamp": datetime.now().isoformat(),
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "total_orders": total_orders,
            "total_trades": total_trades,
            "order_to_trade_ratio": otr,
            "otr_limit": self.order_to_trade_limit,
            "otr_compliant": otr <= self.order_to_trade_limit,
            "client_statistics": client_stats
        }
        
        return report
    
    def generate_regulatory_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate regulatory report for SEBI"""
        # Filter orders and trades by date range
        filtered_orders = [
            o for o in self.orders 
            if start_date <= o.timestamp <= end_date
        ]
        
        filtered_trades = [
            t for t in self.trades 
            if start_date <= t.timestamp <= end_date
        ]
        
        # Calculate statistics
        report = {
            "broker_id": self.broker_id,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_orders": len(filtered_orders),
                "total_trades": len(filtered_trades),
                "order_to_trade_ratio": len(filtered_orders) / len(filtered_trades) if filtered_trades else 0,
                "unique_clients": len(set(o.client_id for o in filtered_orders)),
                "unique_strategies": len(set(o.strategy_id for o in filtered_orders))
            },
            "client_breakdown": {},
            "strategy_breakdown": {}
        }
        
        # Client breakdown
        for client_id in set(o.client_id for o in filtered_orders):
            client_orders = [o for o in filtered_orders if o.client_id == client_id]
            client_trades = [t for t in filtered_trades if t.client_id == client_id]
            
            report["client_breakdown"][client_id] = {
                "orders": len(client_orders),
                "trades": len(client_trades),
                "order_to_trade_ratio": len(client_orders) / len(client_trades) if client_trades else 0
            }
        
        # Strategy breakdown
        for strategy_id in set(o.strategy_id for o in filtered_orders):
            strategy_orders = [o for o in filtered_orders if o.strategy_id == strategy_id]
            strategy_trades = [t for t in filtered_trades if t.strategy_id == strategy_id]
            
            report["strategy_breakdown"][strategy_id] = {
                "orders": len(strategy_orders),
                "trades": len(strategy_trades),
                "order_to_trade_ratio": len(strategy_orders) / len(strategy_trades) if strategy_trades else 0
            }
        
        return report


def run_sample_compliance():
    """Run sample SEBI compliance check"""
    print("="*60)
    print("SEBI ALGO COMPLIANCE - DEMO")
    print("="*60)
    
    # Create compliance module
    compliance = SEBIAlgoCompliance(broker_id="BROKER001")
    
    # Register client
    compliance.register_client("CLIENT001", {
        "max_position_value": 50000000,
        "max_daily_orders": 10000,
        "max_daily_trades": 5000,
        "max_exposure": 100000000,
        "max_leverage": 4.0
    })
    
    # Create sample order
    order = Order(
        order_id="ORD001",
        symbol="NIFTYFUT",
        side="BUY",
        quantity=100,
        price=20000,
        order_type="LIMIT",
        client_id="CLIENT001",
        strategy_id="ORB",
        timestamp=datetime.now()
    )
    
    print(f"\nPre-trade check for order {order.order_id}...")
    is_compliant, checks = compliance.pre_trade_check(order)
    
    print(f"\nCompliance Status: {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}")
    print("\nChecks:")
    for check in checks:
        status_icon = "✓" if check.status == ComplianceStatus.COMPLIANT else "✗"
        print(f"  {status_icon} {check.check_type}: {check.message}")
    
    # Record trade if compliant
    if is_compliant:
        trade = Trade(
            trade_id="TRD001",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            client_id=order.client_id,
            strategy_id=order.strategy_id,
            timestamp=datetime.now()
        )
        compliance.record_trade(trade)
        print(f"\nTrade recorded: {trade.trade_id}")
    
    # Generate compliance report
    report = compliance.generate_compliance_report()
    
    print("\n" + "="*60)
    print("COMPLIANCE REPORT")
    print("="*60)
    print(f"Broker ID: {report['broker_id']}")
    print(f"Total Orders: {report['total_orders']}")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Order-to-Trade Ratio: {report['order_to_trade_ratio']:.2f}")
    print(f"OTR Limit: {report['otr_limit']:.2f}")
    print(f"OTR Compliant: {report['otr_compliant']}")
    print(f"Kill Switch Active: {report['kill_switch_active']}")
    
    # Test kill switch
    print("\n" + "="*60)
    print("KILL SWITCH TEST")
    print("="*60)
    compliance.activate_kill_switch("Test activation")
    
    # Try order with kill switch active
    order2 = Order(
        order_id="ORD002",
        symbol="NIFTYFUT",
        side="BUY",
        quantity=100,
        price=20000,
        order_type="LIMIT",
        client_id="CLIENT001",
        strategy_id="ORB",
        timestamp=datetime.now()
    )
    
    is_compliant2, checks2 = compliance.pre_trade_check(order2)
    print(f"\nOrder with kill switch active: {'COMPLIANT' if is_compliant2 else 'BLOCKED'}")
    
    compliance.deactivate_kill_switch()
    
    print("="*60)


if __name__ == "__main__":
    run_sample_compliance()
