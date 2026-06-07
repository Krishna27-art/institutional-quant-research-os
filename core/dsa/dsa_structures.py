"""
DSA (Data Structures & Algorithms) for High-Performance Trading
Architecture V2 - Quantitative Trading System for Indian Markets

Implements:
- Fenwick Tree: Cumulative volume for VWAP (O(log N) update, O(log N) query)
- Heap: Top-20 stocks by RV (O(log N) insert, O(1) top)
- Ring Buffer: Tick stream (O(1) append, O(1) pop)
- Segment Tree: Range min/max for OHLC aggregates (O(log N) query)
- Bloom Filter: Duplicate tick detection (O(k) with 1% false positive)
- Sparse Table: Pre-computed volatility ranges (O(1) query after O(N log N) build)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import heapq
from dataclasses import dataclass
import hashlib
import mmh3


class FenwickTree:
    """
    Fenwick Tree (Binary Indexed Tree) for cumulative volume calculations.
    
    Use case: VWAP calculation - Σ(Price × Volume) / Σ(Volume)
    Complexity: O(log N) for both update and query operations.
    """
    
    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.float64)
    
    def update(self, index: int, delta: float) -> None:
        """Add delta to element at index (1-based)."""
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)
    
    def query(self, index: int) -> float:
        """Query sum from 1 to index (1-based)."""
        result = 0.0
        while index > 0:
            result += self.tree[index]
            index -= index & (-index)
        return result
    
    def range_query(self, left: int, right: int) -> float:
        """Query sum from left to right (1-based)."""
        return self.query(right) - self.query(left - 1)
    
    def reset(self) -> None:
        """Reset tree to zeros."""
        self.tree.fill(0.0)


class MaxHeap:
    """
    Max-Heap for maintaining top-N stocks by Relative Volume.
    
    Use case: Select top 20 stocks with highest RV for ORB strategy.
    Complexity: O(log N) insert, O(1) get max, O(log N) extract max.
    """
    
    def __init__(self, max_size: Optional[int] = None):
        self.heap: List[Tuple[float, str]] = []  # (value, symbol)
        self.max_size = max_size
        self.symbol_map: Dict[str, float] = {}  # For O(1) lookups
    
    def push(self, value: float, symbol: str) -> None:
        """Push (value, symbol) onto heap."""
        heapq.heappush(self.heap, (-value, symbol))  # Negative for max-heap
        self.symbol_map[symbol] = value
        
        # Maintain max size
        if self.max_size and len(self.heap) > self.max_size:
            heapq.heappop(self.heap)
    
    def pop(self) -> Tuple[float, str]:
        """Pop and return (value, symbol) with max value."""
        neg_value, symbol = heapq.heappop(self.heap)
        value = -neg_value
        del self.symbol_map[symbol]
        return value, symbol
    
    def peek(self) -> Optional[Tuple[float, str]]:
        """Return (value, symbol) with max value without popping."""
        if not self.heap:
            return None
        neg_value, symbol = self.heap[0]
        return -neg_value, symbol
    
    def get_top_n(self, n: int) -> List[Tuple[float, str]]:
        """Return top n items sorted by value descending."""
        sorted_heap = sorted(self.heap, key=lambda x: -x[0])  # Sort by negative value
        return [(-value, symbol) for value, symbol in sorted_heap[:n]]
    
    def __len__(self) -> int:
        return len(self.heap)
    
    def clear(self) -> None:
        """Clear heap."""
        self.heap.clear()
        self.symbol_map.clear()


class RingBuffer:
    """
    Ring Buffer (Circular Buffer) for tick stream processing.
    
    Use case: Store recent ticks for real-time processing.
    Complexity: O(1) append, O(1) pop, O(1) access.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Any] = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
    
    def append(self, item: Any) -> None:
        """Append item to buffer. Overwrites oldest if full."""
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        
        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity
    
    def pop_left(self) -> Optional[Any]:
        """Pop and return oldest item."""
        if self.size == 0:
            return None
        
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item
    
    def get_latest(self, n: int) -> List[Any]:
        """Get n most recent items."""
        if n > self.size:
            n = self.size
        
        result = []
        for i in range(n):
            idx = (self.tail - 1 - i) % self.capacity
            result.append(self.buffer[idx])
        
        return result
    
    def __len__(self) -> int:
        return self.size
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size == self.capacity
    
    def clear(self) -> None:
        """Clear buffer."""
        self.head = 0
        self.tail = 0
        self.size = 0


class SegmentTree:
    """
    Segment Tree for range minimum/maximum queries on OHLC data.
    
    Use case: Efficient range queries for high/low prices over time windows.
    Complexity: O(N) build, O(log N) query, O(log N) update.
    """
    
    def __init__(self, data: np.ndarray, mode: str = "max"):
        """
        Initialize segment tree.
        
        Args:
            data: Input array
            mode: "min" or "max" for query type
        """
        self.data = data
        self.mode = mode
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        
        # Build tree
        self.tree = np.zeros(2 * self.size)
        self._build()
    
    def _build(self) -> None:
        """Build segment tree."""
        # Fill leaves
        for i in range(self.n):
            self.tree[self.size + i] = self.data[i]
        
        # Build internal nodes
        for i in range(self.size - 1, 0, -1):
            if self.mode == "max":
                self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])
            else:
                self.tree[i] = min(self.tree[2 * i], self.tree[2 * i + 1])
    
    def query(self, left: int, right: int) -> float:
        """
        Query range [left, right] (inclusive).
        
        Returns:
            Minimum or maximum value in range.
        """
        left += self.size
        right += self.size
        
        if self.mode == "max":
            result = -np.inf
        else:
            result = np.inf
        
        while left <= right:
            if left % 2 == 1:
                if self.mode == "max":
                    result = max(result, self.tree[left])
                else:
                    result = min(result, self.tree[left])
                left += 1
            
            if right % 2 == 0:
                if self.mode == "max":
                    result = max(result, self.tree[right])
                else:
                    result = min(result, self.tree[right])
                right -= 1
            
            left //= 2
            right //= 2
        
        return result
    
    def update(self, index: int, value: float) -> None:
        """Update element at index to value."""
        index += self.size
        self.tree[index] = value
        
        index //= 2
        while index >= 1:
            if self.mode == "max":
                self.tree[index] = max(self.tree[2 * index], self.tree[2 * index + 1])
            else:
                self.tree[index] = min(self.tree[2 * index], self.tree[2 * index + 1])
            index //= 2


class BloomFilter:
    """
    Bloom Filter for duplicate tick detection.
    
    Use case: Detect duplicate market data ticks to avoid double-processing.
    Complexity: O(k) for insert/query where k is number of hash functions.
    False positive rate: ~1% with proper sizing.
    """
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Initialize bloom filter.
        
        Args:
            expected_items: Expected number of items to store
            false_positive_rate: Desired false positive rate
        """
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal size and hash functions
        self.size = self._calculate_size(expected_items, false_positive_rate)
        self.hash_count = self._calculate_hash_count(self.size, expected_items)
        
        self.bit_array = np.zeros(self.size, dtype=bool)
    
    def _calculate_size(self, n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        m = -(n * np.log(p)) / (np.log(2) ** 2)
        return int(m)
    
    def _calculate_hash_count(self, m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * np.log(2)
        return int(k)
    
    def _hash(self, item: str, seed: int) -> int:
        """Hash item with seed using MurmurHash3."""
        hash_value = mmh3.hash(item, seed)
        return hash_value % self.size
    
    def add(self, item: str) -> None:
        """Add item to bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
    
    def __contains__(self, item: str) -> bool:
        """Check if item might be in bloom filter."""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True
    
    def clear(self) -> None:
        """Clear bloom filter."""
        self.bit_array.fill(False)


class SparseTable:
    """
    Sparse Table for range minimum queries with O(1) query time.
    
    Use case: Pre-computed volatility ranges for fast regime detection.
    Complexity: O(N log N) build, O(1) query, no updates.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Initialize sparse table.
        
        Args:
            data: Input array (static, no updates)
        """
        self.data = data
        self.n = len(data)
        self.log = np.zeros(self.n + 1, dtype=int)
        
        # Precompute logarithms
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
        
        # Build sparse table
        k = self.log[self.n] + 1
        self.st = np.zeros((self.n, k), dtype=np.float64)
        
        for i in range(self.n):
            self.st[i, 0] = data[i]
        
        for j in range(1, k):
            for i in range(self.n - (1 << j) + 1):
                self.st[i, j] = min(self.st[i, j - 1], self.st[i + (1 << (j - 1)), j - 1])
    
    def query(self, left: int, right: int) -> float:
        """
        Query minimum in range [left, right] (inclusive).
        
        Returns:
            Minimum value in range.
        """
        j = self.log[right - left + 1]
        return min(self.st[left, j], self.st[right - (1 << j) + 1, j])


class SymbolCache:
    """
    Hash Map-based symbol cache for O(1) lookups.
    
    Use case: Fast access to latest market data per symbol.
    Complexity: O(1) average for insert, delete, lookup.
    """
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_order: List[str] = []
    
    def set(self, symbol: str, data: Dict[str, Any]) -> None:
        """Set data for symbol."""
        if symbol in self.cache:
            self.access_order.remove(symbol)
        elif len(self.cache) >= self.max_size:
            # LRU eviction
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[symbol] = data
        self.access_order.append(symbol)
    
    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data for symbol."""
        if symbol in self.cache:
            # Update access order
            self.access_order.remove(symbol)
            self.access_order.append(symbol)
            return self.cache[symbol]
        return None
    
    def delete(self, symbol: str) -> bool:
        """Delete symbol from cache."""
        if symbol in self.cache:
            del self.cache[symbol]
            self.access_order.remove(symbol)
            return True
        return False
    
    def __contains__(self, symbol: str) -> bool:
        return symbol in self.cache
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_order.clear()


class OrderBook:
    """
    Priority Queue-based order book for bid/ask management.
    
    Use case: Simulate order book for execution logic.
    Complexity: O(log N) insert/delete, O(1) peek.
    """
    
    def __init__(self):
        self.bids: List[Tuple[float, int, str]] = []  # (price, quantity, order_id)
        self.asks: List[Tuple[float, int, str]] = []  # (price, quantity, order_id)
    
    def add_bid(self, price: float, quantity: int, order_id: str) -> None:
        """Add bid order (max-heap)."""
        heapq.heappush(self.bids, (-price, quantity, order_id))
    
    def add_ask(self, price: float, quantity: int, order_id: str) -> None:
        """Add ask order (min-heap)."""
        heapq.heappush(self.asks, (price, quantity, order_id))
    
    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        """Get best bid (highest price)."""
        if not self.bids:
            return None
        neg_price, quantity, _ = self.bids[0]
        return -neg_price, quantity
    
    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        """Get best ask (lowest price)."""
        if not self.asks:
            return None
        price, quantity, _ = self.asks[0]
        return price, quantity
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask[0] - best_bid[0]
    
    def remove_bid(self) -> Optional[Tuple[float, int, str]]:
        """Remove and return best bid."""
        if not self.bids:
            return None
        neg_price, quantity, order_id = heapq.heappop(self.bids)
        return -neg_price, quantity, order_id
    
    def remove_ask(self) -> Optional[Tuple[float, int, str]]:
        """Remove and return best ask."""
        if not self.asks:
            return None
        return heapq.heappop(self.asks)
    
    def clear(self) -> None:
        """Clear order book."""
        self.bids.clear()
        self.asks.clear()
