"""
Smart Order Router - Route orders to best broker
"""

import numpy as np
from typing import Dict, List, Optional
from ..brokers.broker_adapter import BrokerAdapter, Order, Quote


class SmartOrderRouter:
    """Smart order router for multi-broker execution"""
    
    def __init__(self, brokers: List[BrokerAdapter]):
        self.brokers = brokers
        self.broker_metrics: Dict[str, Dict] = {
            broker.__class__.__name__: {
                'fill_rate': 0.95,
                'latency_ms': 100,
                'cost_bps': 5.0
            } for broker in brokers
        }
    
    def route(self, order: Order) -> tuple:
        """
        Route order to best broker
        
        Args:
            order: Order to route
            
        Returns:
            (broker, order_id)
        """
        if not self.brokers:
            raise RuntimeError("No brokers available")
        
        # Get quotes from all brokers
        quotes = {}
        for broker in self.brokers:
            try:
                quotes[broker] = broker.get_quote(order.symbol)
            except Exception:
                quotes[broker] = None
        
        # Score each broker
        scores = []
        for broker, quote in quotes.items():
            if quote is None:
                continue
            
            score = self._score_broker(broker, quote, order)
            scores.append((score, broker))
        
        if not scores:
            # Fallback to first broker
            broker = self.brokers[0]
        else:
            # Select best broker
            scores.sort(key=lambda x: x[0], reverse=True)
            broker = scores[0][1]
        
        # Place order
        order_id = broker.place_order(order)
        
        return broker, order_id
    
    def _score_broker(self, broker: BrokerAdapter, quote: Quote, 
                     order: Order) -> float:
        """Score broker based on multiple factors"""
        metrics = self.broker_metrics.get(broker.__class__.__name__, {})
        
        score = 0.0
        
        # Best bid/ask (10 points)
        if order.side.value == 'buy':
            if quote.bid > 0:
                score += 10
        else:
            if quote.ask > 0:
                score += 10
        
        # Historical fill rate (5 points)
        score += metrics.get('fill_rate', 0.95) * 5
        
        # Latency penalty (up to 3 points)
        latency = metrics.get('latency_ms', 100)
        score -= min(latency / 100, 3)
        
        # Cost penalty (up to 2 points)
        cost = metrics.get('cost_bps', 5.0)
        score -= min(cost / 5, 2)
        
        return score
    
    def update_broker_metrics(self, broker_name: str, fill_rate: Optional[float] = None,
                            latency_ms: Optional[float] = None,
                            cost_bps: Optional[float] = None) -> None:
        """Update broker metrics"""
        if broker_name not in self.broker_metrics:
            self.broker_metrics[broker_name] = {}
        
        if fill_rate is not None:
            self.broker_metrics[broker_name]['fill_rate'] = fill_rate
        if latency_ms is not None:
            self.broker_metrics[broker_name]['latency_ms'] = latency_ms
        if cost_bps is not None:
            self.broker_metrics[broker_name]['cost_bps'] = cost_bps
    
    def add_broker(self, broker: BrokerAdapter) -> None:
        """Add a new broker"""
        self.brokers.append(broker)
        self.broker_metrics[broker.__class__.__name__] = {
            'fill_rate': 0.95,
            'latency_ms': 100,
            'cost_bps': 5.0
        }
    
    def remove_broker(self, broker: BrokerAdapter) -> None:
        """Remove a broker"""
        if broker in self.brokers:
            self.brokers.remove(broker)
            broker_name = broker.__class__.__name__
            if broker_name in self.broker_metrics:
                del self.broker_metrics[broker_name]
