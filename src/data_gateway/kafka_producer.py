"""
Kafka Data Producer - Publish market data to Kafka topics
"""

import json
from typing import Dict, Optional
from datetime import datetime
from kafka import KafkaProducer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KafkaDataProducer:
    """Kafka producer for market data streaming"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[KafkaProducer] = None
        self.topics = {
            'raw_market_tick': 'raw.market.tick',
            'raw_market_bar': 'raw.market.bar_1min',
            'features_computed': 'features.computed',
            'regime_state': 'regime.state',
            'alpha_signal': 'alpha.signal',
            'portfolio_target': 'portfolio.target',
            'risk_check': 'risk.check',
            'order_new': 'order.new',
            'order_fill': 'order.fill',
            'pnl_realized': 'pnl.realized',
            'metrics_performance': 'metrics.performance'
        }
    
    def connect(self) -> bool:
        """Connect to Kafka cluster"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Kafka"""
        if self.producer:
            self.producer.close()
            self.producer = None
    
    def publish_tick(self, tick_data: Dict) -> bool:
        """
        Publish tick data
        
        Args:
            tick_data: Dict with tick information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['raw_market_tick'],
                key=tick_data.get('symbol'),
                value=tick_data
            )
            self.producer.flush(timeout=1.0)
            return True
        except Exception as e:
            logger.error(f"Failed to publish tick: {e}")
            return False
    
    def publish_bar(self, bar_data: Dict) -> bool:
        """
        Publish bar data
        
        Args:
            bar_data: Dict with OHLCV bar information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['raw_market_bar'],
                key=bar_data.get('symbol'),
                value=bar_data
            )
            self.producer.flush(timeout=1.0)
            return True
        except Exception as e:
            logger.error(f"Failed to publish bar: {e}")
            return False
    
    def publish_feature(self, feature_data: Dict) -> bool:
        """
        Publish computed feature
        
        Args:
            feature_data: Dict with feature information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['features_computed'],
                key=feature_data.get('symbol'),
                value=feature_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish feature: {e}")
            return False
    
    def publish_regime(self, regime_data: Dict) -> bool:
        """
        Publish regime state
        
        Args:
            regime_data: Dict with regime information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['regime_state'],
                value=regime_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish regime: {e}")
            return False
    
    def publish_alpha_signal(self, signal_data: Dict) -> bool:
        """
        Publish alpha signal
        
        Args:
            signal_data: Dict with signal information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['alpha_signal'],
                key=signal_data.get('alpha_id'),
                value=signal_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish alpha signal: {e}")
            return False
    
    def publish_order(self, order_data: Dict) -> bool:
        """
        Publish new order
        
        Args:
            order_data: Dict with order information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['order_new'],
                key=order_data.get('order_id'),
                value=order_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish order: {e}")
            return False
    
    def publish_fill(self, fill_data: Dict) -> bool:
        """
        Publish order fill
        
        Args:
            fill_data: Dict with fill information
            
        Returns:
            Success status
        """
        if not self.producer:
            return False
        
        try:
            self.producer.send(
                self.topics['order_fill'],
                key=fill_data.get('order_id'),
                value=fill_data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish fill: {e}")
            return False
