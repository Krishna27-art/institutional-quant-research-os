"""
Feature Cache - Redis-based caching for features
"""

import redis
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta


class FeatureCache:
    """Redis-based feature cache"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, 
                 db: int = 0, password: Optional[str] = None):
        self.redis_client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            password=password,
            decode_responses=True
        )
        self.default_ttl = 300  # 5 minutes default
    
    def get(self, symbol: str, feature_name: str, timestamp: Optional[datetime] = None) -> Optional[pd.Series]:
        """Get a cached feature"""
        try:
            key = self._make_key(symbol, feature_name, timestamp)
            cached = self.redis_client.get(key)
            
            if cached is None:
                return None
            
            data = json.loads(cached)
            return pd.Series(data['values'], index=pd.to_datetime(data['index']))
        except Exception as e:
            print(f"Error retrieving cached feature: {e}")
            return None
    
    def set(self, symbol: str, feature_name: str, feature_series: pd.Series, 
            ttl: Optional[int] = None) -> bool:
        """Cache a feature"""
        key = self._make_key(symbol, feature_name)
        ttl = ttl or self.default_ttl
        
        try:
            data = {
                'values': feature_series.tolist(),
                'index': feature_series.index.astype(str).tolist()
            }
            self.redis_client.setex(key, ttl, json.dumps(data))
            return True
        except Exception as e:
            print(f"Error caching feature: {e}")
            return False
    
    def get_batch(self, symbol: str, feature_names: List[str], 
                  timestamp: Optional[datetime] = None) -> Dict[str, Optional[pd.Series]]:
        """Get multiple cached features for a symbol"""
        results = {}
        for feature_name in feature_names:
            results[feature_name] = self.get(symbol, feature_name, timestamp)
        return results
    
    def set_batch(self, symbol: str, features: Dict[str, pd.Series], 
                  ttl: Optional[int] = None) -> bool:
        """Cache multiple features for a symbol"""
        success = True
        for feature_name, feature_series in features.items():
            if not self.set(symbol, feature_name, feature_series, ttl):
                success = False
        return success
    
    def invalidate(self, symbol: str, feature_name: Optional[str] = None) -> bool:
        """Invalidate cached features"""
        try:
            if feature_name:
                key = self._make_key(symbol, feature_name)
                return self.redis_client.delete(key) > 0
            else:
                # Invalidate all features for symbol
                pattern = self._make_key(symbol, "*")
                deleted = 0
                keys_batch = []
                for key in self.redis_client.scan_iter(match=pattern, count=100):
                    keys_batch.append(key)
                    if len(keys_batch) >= 100:
                        deleted += self.redis_client.delete(*keys_batch)
                        keys_batch = []
                if keys_batch:
                    deleted += self.redis_client.delete(*keys_batch)
                return deleted > 0
        except Exception as e:
            print(f"Error invalidating cache: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all cached features"""
        try:
            pattern = "feature:*"
            deleted = 0
            keys_batch = []
            for key in self.redis_client.scan_iter(match=pattern, count=100):
                keys_batch.append(key)
                if len(keys_batch) >= 100:
                    deleted += self.redis_client.delete(*keys_batch)
                    keys_batch = []
            if keys_batch:
                deleted += self.redis_client.delete(*keys_batch)
            return deleted > 0
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def _make_key(self, symbol: str, feature_name: str = "*", 
                  timestamp: Optional[datetime] = None) -> str:
        """Generate cache key"""
        if timestamp:
            return f"feature:{symbol}:{feature_name}:{timestamp.isoformat()}"
        return f"feature:{symbol}:{feature_name}"
    
    def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
