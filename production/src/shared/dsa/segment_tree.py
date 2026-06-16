"""
Segment Tree - O(log N) range queries for rolling max/min/drawdown
"""

import numpy as np
from typing import List, Optional


class SegmentTree:
    """Segment tree for range minimum/maximum queries in O(log N)"""
    
    def __init__(self, data: List[float], operation: str = 'max'):
        """
        Args:
            data: Initial data
            operation: 'max', 'min', or 'sum'
        """
        self.operation = operation
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        
        # Build tree
        self.tree = [float('-inf') if operation == 'max' else 
                    float('inf') if operation == 'min' else 
                    0.0] * (2 * self.size)
        
        # Fill leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        
        # Build internal nodes
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self._combine(self.tree[2 * i], self.tree[2 * i + 1])
    
    def _combine(self, a: float, b: float) -> float:
        """Combine two values based on operation"""
        if self.operation == 'max':
            return max(a, b)
        elif self.operation == 'min':
            return min(a, b)
        elif self.operation == 'sum':
            return a + b
        else:
            raise ValueError(f"Unknown operation: {self.operation}")
    
    def query(self, left: int, right: int) -> float:
        """Query range [left, right] in O(log N)"""
        left += self.size
        right += self.size
        
        result = float('-inf') if self.operation == 'max' else \
                 float('inf') if self.operation == 'min' else \
                 0.0
        
        while left <= right:
            if left % 2 == 1:
                result = self._combine(result, self.tree[left])
                left += 1
            if right % 2 == 0:
                result = self._combine(result, self.tree[right])
                right -= 1
            left //= 2
            right //= 2
        
        return result
    
    def update(self, index: int, value: float) -> None:
        """Update value at index in O(log N)"""
        index += self.size
        self.tree[index] = value
        
        index //= 2
        while index >= 1:
            self.tree[index] = self._combine(self.tree[2 * index], self.tree[2 * index + 1])
            index //= 2
    
    def rolling_max(self, window: int) -> List[float]:
        """Compute rolling maximum using segment tree"""
        result = []
        for i in range(self.n):
            left = max(0, i - window + 1)
            right = i
            result.append(self.query(left, right))
        return result
    
    def rolling_min(self, window: int) -> List[float]:
        """Compute rolling minimum using segment tree"""
        result = []
        for i in range(self.n):
            left = max(0, i - window + 1)
            right = i
            result.append(self.query(left, right))
        return result
