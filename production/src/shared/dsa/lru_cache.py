"""
LRU Cache - O(1) get/set with bounded memory
"""

from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """Least Recently Used Cache with O(1) operations"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value by key, move to front (most recently used)"""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, key: Any, value: Any) -> None:
        """Set value, evict least recently used if at capacity"""
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def delete(self, key: Any) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries"""
        self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def __contains__(self, key: Any) -> bool:
        return key in self.cache
