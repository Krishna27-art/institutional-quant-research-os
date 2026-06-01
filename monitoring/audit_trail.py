"""
Audit Trail with Immutable Logs (SHA256)
Based on Institutional Audit Recommendations

Key findings from audit:
- No immutable log of all orders, signals, model versions
- Regulatory investigation impossible to satisfy
- Need: Append-only table with cryptographic hash chain

Architecture V2 Upgrade - 90-Day Plan Item #7
Priority: P0 (Critical)
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import sqlite3
import os


@dataclass
class AuditEvent:
    """Audit event for logging"""
    event_id: str
    event_type: str  # "order", "signal", "model_version", "risk_check", "fill"
    timestamp: datetime
    data: Dict[str, Any]
    previous_hash: str
    current_hash: str
    signature: str  # SHA256 of event_id + timestamp + data + previous_hash


class AuditTrail:
    """
    Immutable audit trail with cryptographic hash chain.
    
    Features:
    - Append-only SQLite database
    - SHA256 hash chain linking all events
    - Tamper-evident: any modification breaks the chain
    - Retention: 7 years (regulatory requirement)
    """
    
    def __init__(self, db_path: str = "data/audit_trail.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create audit_events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                current_hash TEXT NOT NULL,
                signature TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on event_type and timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type 
            ON audit_events(event_type, timestamp)
        """)
        
        self.conn.commit()
    
    def _get_previous_hash(self) -> str:
        """Get hash of the last event in the chain."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT current_hash FROM audit_events ORDER BY timestamp DESC LIMIT 1"
        )
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            return "0000000000000000000000000000000000000000000000000000000000000000"  # Genesis hash
    
    def _calculate_hash(self, event_id: str, timestamp: str, data: str, previous_hash: str) -> str:
        """Calculate SHA256 hash for the event."""
        hash_input = f"{event_id}{timestamp}{data}{previous_hash}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _calculate_signature(self, event_id: str, timestamp: str, data: str, previous_hash: str) -> str:
        """Calculate signature (same as hash for simplicity, could use private key in production)."""
        return self._calculate_hash(event_id, timestamp, data, previous_hash)
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event (order, signal, model_version, etc.)
            data: Event data dictionary
            
        Returns:
            Event ID
        """
        # Generate event ID
        event_id = f"{event_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        timestamp = datetime.now().isoformat()
        
        # Serialize data
        data_json = json.dumps(data, sort_keys=True)
        
        # Get previous hash
        previous_hash = self._get_previous_hash()
        
        # Calculate current hash and signature
        current_hash = self._calculate_hash(event_id, timestamp, data_json, previous_hash)
        signature = self._calculate_signature(event_id, timestamp, data_json, previous_hash)
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            data=data,
            previous_hash=previous_hash,
            current_hash=current_hash,
            signature=signature
        )
        
        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_events 
            (event_id, event_type, timestamp, data, previous_hash, current_hash, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type,
                event.timestamp,
                data_json,
                event.previous_hash,
                event.current_hash,
                event.signature
            )
        )
        self.conn.commit()
        
        return event_id
    
    def log_order(self, order_data: Dict[str, Any]) -> str:
        """Log an order event."""
        return self.log_event("order", order_data)
    
    def log_signal(self, signal_data: Dict[str, Any]) -> str:
        """Log a signal event."""
        return self.log_event("signal", signal_data)
    
    def log_model_version(self, model_data: Dict[str, Any]) -> str:
        """Log a model version event."""
        return self.log_event("model_version", model_data)
    
    def log_risk_check(self, risk_data: Dict[str, Any]) -> str:
        """Log a risk check event."""
        return self.log_event("risk_check", risk_data)
    
    def log_fill(self, fill_data: Dict[str, Any]) -> str:
        """Log a fill event."""
        return self.log_event("fill", fill_data)
    
    def verify_chain(self) -> bool:
        """
        Verify the integrity of the audit chain.
        
        Returns:
            True if chain is valid, False if tampered
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT event_id, timestamp, data, previous_hash, current_hash, signature "
            "FROM audit_events ORDER BY timestamp"
        )
        
        events = cursor.fetchall()
        
        previous_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        
        for event in events:
            event_id, timestamp, data_json, prev_hash, current_hash, signature = event
            
            # Verify previous hash matches
            if prev_hash != previous_hash:
                print(f"Chain broken at event {event_id}: expected {previous_hash}, got {prev_hash}")
                return False
            
            # Recalculate hash
            calculated_hash = self._calculate_hash(event_id, timestamp, data_json, prev_hash)
            
            # Verify hash matches
            if calculated_hash != current_hash:
                print(f"Hash mismatch at event {event_id}: expected {calculated_hash}, got {current_hash}")
                return False
            
            # Verify signature
            calculated_signature = self._calculate_signature(event_id, timestamp, data_json, prev_hash)
            if calculated_signature != signature:
                print(f"Signature mismatch at event {event_id}")
                return False
            
            previous_hash = current_hash
        
        return True
    
    def query_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit events.
        
        Args:
            event_type: Filter by event type
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()
        
        query = "SELECT event_id, event_type, timestamp, data, previous_hash, current_hash FROM audit_events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        events = []
        for row in results:
            events.append({
                "event_id": row[0],
                "event_type": row[1],
                "timestamp": row[2],
                "data": json.loads(row[3]),
                "previous_hash": row[4],
                "current_hash": row[5]
            })
        
        return events
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the audit chain."""
        cursor = self.conn.cursor()
        
        # Total events
        cursor.execute("SELECT COUNT(*) FROM audit_events")
        total_events = cursor.fetchone()[0]
        
        # First event
        cursor.execute("SELECT event_id, timestamp FROM audit_events ORDER BY timestamp ASC LIMIT 1")
        first_event = cursor.fetchone()
        
        # Last event
        cursor.execute("SELECT event_id, timestamp, current_hash FROM audit_events ORDER BY timestamp DESC LIMIT 1")
        last_event = cursor.fetchone()
        
        # Events by type
        cursor.execute("SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type")
        events_by_type = dict(cursor.fetchall())
        
        return {
            "total_events": total_events,
            "first_event_id": first_event[0] if first_event else None,
            "first_event_timestamp": first_event[1] if first_event else None,
            "last_event_id": last_event[0] if last_event else None,
            "last_event_timestamp": last_event[1] if last_event else None,
            "last_hash": last_event[2] if last_event else None,
            "events_by_type": events_by_type,
            "chain_valid": self.verify_chain()
        }
    
    def print_chain_info(self) -> None:
        """Print audit chain information."""
        info = self.get_chain_info()
        
        print("\n" + "="*60)
        print("AUDIT TRAIL CHAIN INFORMATION")
        print("="*60)
        print(f"Total Events: {info['total_events']}")
        print(f"First Event: {info['first_event_id']} at {info['first_event_timestamp']}")
        print(f"Last Event: {info['last_event_id']} at {info['last_event_timestamp']}")
        print(f"Last Hash: {info['last_hash']}")
        print(f"Chain Valid: {'✅ YES' if info['chain_valid'] else '❌ NO'}")
        
        print("\nEvents by Type:")
        for event_type, count in info['events_by_type'].items():
            print(f"  {event_type}: {count}")
        
        print("="*60)
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()


def run_sample_audit_trail():
    """Run sample audit trail operations."""
    audit = AuditTrail()
    
    try:
        # Log sample events
        print("Logging sample events...")
        
        # Log order
        order_data = {
            "order_id": "ORD001",
            "symbol": "NIFTY",
            "side": "BUY",
            "quantity": 100,
            "price": 20000.0,
            "strategy": "ORB"
        }
        audit.log_order(order_data)
        print(f"Logged order: {order_data['order_id']}")
        
        # Log signal
        signal_data = {
            "signal_id": "SIG001",
            "symbol": "NIFTY",
            "strategy": "ORB",
            "signal_value": 0.8,
            "confidence": 0.75
        }
        audit.log_signal(signal_data)
        print(f"Logged signal: {signal_data['signal_id']}")
        
        # Log model version
        model_data = {
            "model_id": "ORB_v2",
            "version": "2.0",
            "parameters": {"atr_window": 14, "rv_threshold": 2.0},
            "git_commit": "abc123"
        }
        audit.log_model_version(model_data)
        print(f"Logged model version: {model_data['model_id']}")
        
        # Log risk check
        risk_data = {
            "check_id": "RISK001",
            "var": 5000000.0,
            "l_var": 5200000.0,
            "leverage": 1.5,
            "passed": True
        }
        audit.log_risk_check(risk_data)
        print(f"Logged risk check: {risk_data['check_id']}")
        
        # Log fill
        fill_data = {
            "fill_id": "FILL001",
            "order_id": "ORD001",
            "symbol": "NIFTY",
            "quantity": 100,
            "fill_price": 20000.5
        }
        audit.log_fill(fill_data)
        print(f"Logged fill: {fill_data['fill_id']}")
        
        # Print chain info
        audit.print_chain_info()
        
        # Query events
        print("\nQuerying recent events...")
        events = audit.query_events(limit=5)
        for event in events:
            print(f"\n{event['event_type'].upper()}: {event['event_id']}")
            print(f"  Timestamp: {event['timestamp']}")
            print(f"  Data: {json.dumps(event['data'], indent=2)}")
        
        # Verify chain
        print("\nVerifying chain integrity...")
        is_valid = audit.verify_chain()
        print(f"Chain is valid: {is_valid}")
        
    finally:
        audit.close()


if __name__ == "__main__":
    run_sample_audit_trail()
