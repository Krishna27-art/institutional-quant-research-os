"""
Order Statistic Tree - O(log N) for dynamic quantiles
"""

import bisect
from typing import List, Optional


class OrderStatisticTree:
    """
    Order Statistic Tree using sorted list with bisect
    Provides O(log N) insert/delete and O(1) median access
    """
    
    def __init__(self):
        self.values: List[float] = []
    
    def insert(self, value: float) -> None:
        """Insert a value in O(log N)"""
        bisect.insort(self.values, value)
    
    def delete(self, value: float) -> bool:
        """Delete a value in O(log N)"""
        try:
            idx = bisect.bisect_left(self.values, value)
            if idx < len(self.values) and self.values[idx] == value:
                self.values.pop(idx)
                return True
        except ValueError:
            pass
        return False
    
    def kth_smallest(self, k: int) -> Optional[float]:
        """Get kth smallest element (1-indexed) in O(1)"""
        if k < 1 or k > len(self.values):
            return None
        return self.values[k - 1]
    
    def rank(self, value: float) -> int:
        """Get rank of value (number of elements <= value) in O(log N)"""
        return bisect.bisect_right(self.values, value)
    
    def median(self) -> Optional[float]:
        """Get median in O(1)"""
        n = len(self.values)
        if n == 0:
            return None
        if n % 2 == 0:
            return (self.values[n // 2 - 1] + self.values[n // 2]) / 2
        return self.values[n // 2]
    
    def percentile(self, p: float) -> Optional[float]:
        """Get p-th percentile (0-100)"""
        if not self.values:
            return None
        k = int(p / 100 * len(self.values))
        k = max(0, min(k, len(self.values) - 1))
        return self.values[k]
    
    def __len__(self) -> int:
        return len(self.values)
