"""
Redis Streams for Feature Pipeline
Based on Architecture V2 agent debate consensus

Key findings from research:
- Redis Streams for feature pipeline (simpler than Kafka for our scale)
- One stream per symbol, consumer group for features
- Real-time ingest: WebSocket → Go routine → Redis Streams
- Batch insert into ClickHouse every minute
- Redis Hash for latest features (hot cache)

Architecture V2 - Quantitative Trading System for Indian Markets
Phase 1: Redis Streams for feature pipeline, Redis Hash for hot cache
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import redis


@dataclass
class StreamConfig:
    """Configuration for Redis Streams"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    stream_prefix: str = "market_data"
    features_prefix: str = "features"
    consumer_group: str = "feature_processor"


class RedisStreamsPipeline:
    """
    Redis Streams Pipeline for feature processing.
    
    Architecture V2 Data Flow:
    - Data feed: WebSocket → Go routine → Redis Streams
    - Feature calculation: Python + Polars (vectorized) → Redis Hash
    - Signal generation: LightGBM (C API via Python) → Redis Pub/Sub
    - Risk & order generation: Python (single thread) → Redis Queue
    - Execution: FastAPI (order submission to broker) → HTTP/2
    
    Redis Streams Benefits:
    - Simpler than Kafka for our scale
    - Built-in consumer groups
    - Automatic message acknowledgment
    - Persistent if needed
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.redis_client = None
    
    def connect(self) -> None:
        """Establish connection to Redis."""
        self.redis_client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            decode_responses=True
        )
        print(f"Connected to Redis at {self.config.host}:{self.config.port}")
    
    def disconnect(self) -> None:
        """Close connection to Redis."""
        if self.redis_client:
            self.redis_client.close()
            print("Disconnected from Redis")
    
    def create_stream(self, symbol: str) -> None:
        """
        Create a stream for a symbol.
        
        Args:
            symbol: Stock symbol
        """
        stream_name = f"{self.config.stream_prefix}:{symbol}"
        try:
            self.redis_client.xadd(stream_name, {"test": "init"})
            self.redis_client.xdel(stream_name, self.redis_client.xrange(stream_name, "-", "+")[0][0])
            print(f"Stream {stream_name} created")
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                print(f"Stream {stream_name} already exists or error: {e}")
    
    def create_consumer_group(self, symbol: str) -> None:
        """
        Create consumer group for a stream.
        
        Args:
            symbol: Stock symbol
        """
        stream_name = f"{self.config.stream_prefix}:{symbol}"
        try:
            self.redis_client.xgroup_create(
                stream_name,
                self.config.consumer_group,
                id='0',
                mkstream=True
            )
            print(f"Consumer group {self.config.consumer_group} created for {stream_name}")
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                print(f"Consumer group already exists or error: {e}")
    
    def publish_tick(self, symbol: str, tick_data: Dict) -> str:
        """
        Publish tick data to stream.
        
        Args:
            symbol: Stock symbol
            tick_data: Dictionary with tick data
            
        Returns:
            Message ID
        """
        stream_name = f"{self.config.stream_prefix}:{symbol}"
        
        # Add timestamp
        tick_data['timestamp'] = datetime.now().isoformat()
        
        message_id = self.redis_client.xadd(stream_name, tick_data)
        return message_id
    
    def consume_ticks(
        self,
        symbol: str,
        consumer_name: str,
        count: int = 10
    ) -> List[Dict]:
        """
        Consume ticks from stream.
        
        Args:
            symbol: Stock symbol
            consumer_name: Consumer name
            count: Number of messages to read
            
        Returns:
            List of tick data dictionaries
        """
        stream_name = f"{self.config.stream_prefix}:{symbol}"
        
        # Read messages
        messages = self.redis_client.xreadgroup(
            groupname=self.config.consumer_group,
            consumername=consumer_name,
            streams={stream_name: '>'},
            count=count,
            block=1000  # 1 second timeout
        )
        
        if not messages:
            return []
        
        # Parse messages
        ticks = []
        for stream, message_list in messages:
            for message_id, data in message_list:
                ticks.append(data)
                # Acknowledge message
                self.redis_client.xack(stream_name, self.config.consumer_group, message_id)
        
        return ticks
    
    def cache_features(self, symbol: str, features: Dict[str, float]) -> None:
        """
        Cache latest features in Redis Hash.
        
        Args:
            symbol: Stock symbol
            features: Dictionary of feature name -> value
        """
        hash_name = f"{self.config.features_prefix}:{symbol}"
        
        # Add timestamp
        features['timestamp'] = datetime.now().isoformat()
        
        # Store in hash
        self.redis_client.hset(hash_name, mapping=features)
        
        # Set expiry (24 hours)
        self.redis_client.expire(hash_name, 86400)
    
    def get_cached_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get cached features from Redis Hash.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of features or None
        """
        hash_name = f"{self.config.features_prefix}:{symbol}"
        
        features = self.redis_client.hgetall(hash_name)
        
        if not features:
            return None
        
        # Convert string values to float
        for key, value in features.items():
            if key != 'timestamp':
                try:
                    features[key] = float(value)
                except ValueError:
                    pass
        
        return features
    
    def publish_signal(self, symbol: str, signal_data: Dict) -> None:
        """
        Publish signal to Redis Pub/Sub.
        
        Args:
            symbol: Stock symbol
            signal_data: Dictionary with signal data
        """
        channel = f"signals:{symbol}"
        
        # Add timestamp
        signal_data['timestamp'] = datetime.now().isoformat()
        
        # Publish
        self.redis_client.publish(channel, json.dumps(signal_data))
    
    def subscribe_signals(self, symbol: str) -> redis.client.PubSub:
        """
        Subscribe to signals for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            PubSub object
        """
        channel = f"signals:{symbol}"
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(channel)
        return pubsub
    
    def add_to_order_queue(self, order_data: Dict) -> None:
        """
        Add order to Redis Queue.
        
        Args:
            order_data: Dictionary with order data
        """
        queue_name = "orders"
        
        # Add timestamp
        order_data['timestamp'] = datetime.now().isoformat()
        
        # Push to queue
        self.redis_client.lpush(queue_name, json.dumps(order_data))
    
    def get_from_order_queue(self) -> Optional[Dict]:
        """
        Get order from Redis Queue.
        
        Returns:
            Order data dictionary or None
        """
        queue_name = "orders"
        
        # Pop from queue (blocking with timeout)
        result = self.redis_client.brpop(queue_name, timeout=1)
        
        if result:
            _, order_json = result
            return json.loads(order_json)
        
        return None
    
    def get_stream_info(self, symbol: str) -> Dict:
        """
        Get stream information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stream info
        """
        stream_name = f"{self.config.stream_prefix}:{symbol}"
        
        try:
            info = self.redis_client.xinfo_stream(stream_name)
            return {
                'length': info[0][1]['length'],
                'groups': info[0][1]['groups'],
                'first_entry': info[0][1].get('first-entry'),
                'last_entry': info[0][1].get('last-entry')
            }
        except redis.ResponseError:
            return {'error': 'Stream does not exist'}
    
    def print_stream_info(self, symbol: str) -> None:
        """Print stream information."""
        info = self.get_stream_info(symbol)
        
        print("\n" + "="*60)
        print(f"REDIS STREAM INFO: {symbol}")
        print("="*60)
        if 'error' in info:
            print(f"Error: {info['error']}")
        else:
            print(f"Stream Length: {info['length']}")
            print(f"Consumer Groups: {info['groups']}")
            print(f"First Entry: {info['first_entry']}")
            print(f"Last Entry: {info['last_entry']}")
        print("="*60)


def run_sample_pipeline():
    """Run sample Redis Streams pipeline."""
    config = StreamConfig(
        host="localhost",
        port=6379,
        db=0,
        stream_prefix="market_data",
        features_prefix="features",
        consumer_group="feature_processor"
    )
    
    pipeline = RedisStreamsPipeline(config)
    
    try:
        # Connect
        pipeline.connect()
        
        # Create stream and consumer group for NIFTY
        pipeline.create_stream("NIFTY")
        pipeline.create_consumer_group("NIFTY")
        
        # Publish some ticks
        for i in range(5):
            tick_data = {
                'price': 20000 + i * 10,
                'volume': 100000 + i * 1000,
                'bid': 20000 + i * 10 - 1,
                'ask': 20000 + i * 10 + 1
            }
            message_id = pipeline.publish_tick("NIFTY", tick_data)
            print(f"Published tick {i+1}: {message_id}")
        
        # Cache features
        features = {
            'relative_volume': 2.5,
            'vwap_distance': 0.01,
            'realized_volatility': 0.15,
            'rsi': 55.0
        }
        pipeline.cache_features("NIFTY", features)
        
        # Get cached features
        cached = pipeline.get_cached_features("NIFTY")
        print(f"\nCached features: {cached}")
        
        # Publish signal
        signal_data = {
            'strategy': 'ORB',
            'signal': 0.8,
            'confidence': 0.75
        }
        pipeline.publish_signal("NIFTY", signal_data)
        print(f"\nPublished signal: {signal_data}")
        
        # Add to order queue
        order_data = {
            'symbol': 'NIFTY',
            'side': 'BUY',
            'quantity': 100,
            'price': 20000
        }
        pipeline.add_to_order_queue(order_data)
        print(f"\nAdded to order queue: {order_data}")
        
        # Get from order queue
        order = pipeline.get_from_order_queue()
        print(f"Retrieved from order queue: {order}")
        
        # Print stream info
        pipeline.print_stream_info("NIFTY")
        
    finally:
        pipeline.disconnect()


if __name__ == "__main__":
    run_sample_pipeline()
