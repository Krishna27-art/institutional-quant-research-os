"""
Redis Manager - Cache and pub/sub operations
"""

import redis
import json
from typing import Optional, Dict, List, Any
from datetime import timedelta


class RedisManager:
    """Manage Redis operations"""
    
    def __init__(self, client: redis.Redis):
        self.client = client
    
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store JSON value with optional TTL"""
        try:
            serialized = json.dumps(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
            return True
        except Exception:
            return False
    
    def get_json(self, key: str) -> Optional[Any]:
        """Retrieve and deserialize JSON value"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception:
            return None
    
    def set_series(self, key: str, series: Dict[str, float], ttl: Optional[int] = None) -> bool:
        """Store time series as hash"""
        try:
            self.client.hset(key, mapping=series)
            if ttl:
                self.client.expire(key, ttl)
            return True
        except Exception:
            return False
    
    def get_series(self, key: str) -> Dict[str, float]:
        """Retrieve time series hash"""
        try:
            return {k: float(v) for k, v in self.client.hgetall(key).items()}
        except Exception:
            return {}
    
    def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        try:
            serialized = json.dumps(message)
            return self.client.publish(channel, serialized)
        except Exception:
            return 0
    
    def subscribe(self, channels: List[str]):
        """Subscribe to channels (returns pubsub object)"""
        return self.client.pubsub()
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        try:
            deleted_count = 0
            keys_batch = []
            for key in self.client.scan_iter(match=pattern, count=100):
                keys_batch.append(key)
                if len(keys_batch) >= 100:
                    deleted_count += self.client.delete(*keys_batch)
                    keys_batch = []
            if keys_batch:
                deleted_count += self.client.delete(*keys_batch)
            return deleted_count
        except Exception:
            return 0
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        return self.client.incrby(key, amount)
    
    def get_counter(self, key: str) -> int:
        """Get counter value"""
        value = self.client.get(key)
        return int(value) if value else 0
