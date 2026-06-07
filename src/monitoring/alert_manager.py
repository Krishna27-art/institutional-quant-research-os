"""
Alert Manager - Alerting rules and notification system
"""

import smtplib
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    ALPHA_DECAY = "alpha_decay"
    FEATURE_DRIFT = "feature_drift"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    LATENCY_SPIKE = "latency_spike"
    DATA_GAP = "data_gap"
    REGIME_INSTABILITY = "regime_instability"
    CIRCUIT_BREAKER = "circuit_breaker"


class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, email_config: Optional[Dict] = None,
                 slack_webhook: Optional[str] = None):
        self.email_config = email_config
        self.slack_webhook = slack_webhook
        self.alert_history: List[Dict] = []
        self.alert_rules: Dict[AlertType, Dict] = self._initialize_rules()
        self.alert_handlers: Dict[AlertType, List[Callable]] = {}
    
    def _initialize_rules(self) -> Dict[AlertType, Dict]:
        """Initialize default alert rules"""
        return {
            AlertType.ALPHA_DECAY: {
                'enabled': True,
                'threshold_sharpe': 0.5,
                'threshold_drawdown': 0.10,
                'cooldown_minutes': 60
            },
            AlertType.FEATURE_DRIFT: {
                'enabled': True,
                'threshold_psi': 0.1,
                'cooldown_minutes': 30
            },
            AlertType.DRAWDOWN: {
                'enabled': True,
                'threshold': 0.10,
                'cooldown_minutes': 15
            },
            AlertType.VAR_BREACH: {
                'enabled': True,
                'threshold': 0.015,
                'cooldown_minutes': 5
            },
            AlertType.LATENCY_SPIKE: {
                'enabled': True,
                'threshold_ms': 500,
                'cooldown_minutes': 5
            },
            AlertType.DATA_GAP: {
                'enabled': True,
                'threshold_minutes': 5,
                'cooldown_minutes': 10
            },
            AlertType.REGIME_INSTABILITY: {
                'enabled': True,
                'threshold_persistence': 0.5,
                'cooldown_minutes': 30
            },
            AlertType.CIRCUIT_BREAKER: {
                'enabled': True,
                'cooldown_minutes': 1
            }
        }
    
    def register_handler(self, alert_type: AlertType, handler: Callable) -> None:
        """Register a custom alert handler"""
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        self.alert_handlers[alert_type].append(handler)
    
    def trigger_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     message: str, metadata: Optional[Dict] = None) -> bool:
        """
        Trigger an alert
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        # Check if alert type is enabled
        if not self.alert_rules.get(alert_type, {}).get('enabled', True):
            return False
        
        # Check cooldown
        if self._is_on_cooldown(alert_type):
            logger.info(f"Alert {alert_type.value} is on cooldown, skipping")
            return False
        
        # Create alert record
        alert = {
            'alert_type': alert_type.value,
            'severity': severity.value,
            'message': message,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"ALERT [{severity.value.upper()}] {alert_type.value}: {message}")
        
        # Execute handlers
        if alert_type in self.alert_handlers:
            for handler in self.alert_handlers[alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
        
        # Send notifications based on severity
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            self._send_email_alert(alert)
            self._send_slack_alert(alert)
        
        return True
    
    def check_alpha_decay(self, alpha_id: str, sharpe: float, 
                        drawdown: float) -> Optional[bool]:
        """Check alpha decay conditions and trigger alert if needed"""
        rule = self.alert_rules[AlertType.ALPHA_DECAY]
        
        if sharpe < rule['threshold_sharpe'] or drawdown > rule['threshold_drawdown']:
            return self.trigger_alert(
                AlertType.ALPHA_DECAY,
                AlertSeverity.WARNING if sharpe > 0 else AlertSeverity.ERROR,
                f"Alpha {alpha_id} decay detected: Sharpe={sharpe:.2f}, Drawdown={drawdown:.2%}",
                {'alpha_id': alpha_id, 'sharpe': sharpe, 'drawdown': drawdown}
            )
        
        return None
    
    def check_drawdown(self, current_drawdown: float) -> Optional[bool]:
        """Check drawdown and trigger alert if needed"""
        rule = self.alert_rules[AlertType.DRAWDOWN]
        
        if current_drawdown > rule['threshold']:
            severity = AlertSeverity.CRITICAL if current_drawdown > 0.15 else AlertSeverity.ERROR
            return self.trigger_alert(
                AlertType.DRAWDOWN,
                severity,
                f"Drawdown breach: {current_drawdown:.2%}",
                {'drawdown': current_drawdown}
            )
        
        return None
    
    def check_var_breach(self, current_var: float, limit: float) -> Optional[bool]:
        """Check VaR breach and trigger alert if needed"""
        rule = self.alert_rules[AlertType.VAR_BREACH]
        
        if current_var > limit:
            return self.trigger_alert(
                AlertType.VAR_BREACH,
                AlertSeverity.ERROR,
                f"VaR breach: {current_var:.2%} > {limit:.2%}",
                {'var': current_var, 'limit': limit}
            )
        
        return None
    
    def check_latency(self, latency_ms: float) -> Optional[bool]:
        """Check latency and trigger alert if needed"""
        rule = self.alert_rules[AlertType.LATENCY_SPIKE]
        
        if latency_ms > rule['threshold_ms']:
            return self.trigger_alert(
                AlertType.LATENCY_SPIKE,
                AlertSeverity.WARNING,
                f"Latency spike: {latency_ms:.0f}ms",
                {'latency_ms': latency_ms}
            )
        
        return None
    
    def _is_on_cooldown(self, alert_type: AlertType) -> bool:
        """Check if alert type is on cooldown"""
        rule = self.alert_rules.get(alert_type, {})
        cooldown_minutes = rule.get('cooldown_minutes', 0)
        
        if cooldown_minutes == 0:
            return False
        
        # Get last alert of this type
        recent_alerts = [
            a for a in self.alert_history 
            if a['alert_type'] == alert_type.value
        ]
        
        if not recent_alerts:
            return False
        
        last_alert = recent_alerts[-1]
        last_time = datetime.fromisoformat(last_alert['timestamp'])
        elapsed = (datetime.now() - last_time).total_seconds() / 60
        
        return elapsed < cooldown_minutes
    
    def _send_email_alert(self, alert: Dict) -> bool:
        """Send email alert"""
        if not self.email_config:
            return False
        
        try:
            msg = MIMEText(f"""
Alert: {alert['alert_type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}
Message: {alert['message']}
Metadata: {alert['metadata']}
            """)
            
            msg['Subject'] = f"[{alert['severity'].upper()}] {alert['alert_type']}"
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            
            with smtplib.SMTP(
                self.email_config['host'],
                self.email_config['port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['username'],
                    self.email_config['password']
                )
                server.send_message(msg)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack_alert(self, alert: Dict) -> bool:
        """Send Slack alert via webhook"""
        if not self.slack_webhook:
            return False
        
        # Placeholder for Slack webhook implementation
        logger.info(f"Slack alert would be sent: {alert}")
        return True
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return self.alert_history[-n:]
