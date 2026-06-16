"""
Priority Queue - O(log N) push/pop for time-ordered events
"""

import heapq
from typing import List, Tuple, Any


class PriorityQueue:
    """Priority queue wrapper around heapq"""
    
    def __init__(self):
        self.heap: List[Tuple[float, Any]] = []
        self.counter = 0  # For tie-breaking
    
    def push(self, priority: float, item: Any) -> None:
        """Push item with priority in O(log N)"""
        heapq.heappush(self.heap, (priority, self.counter, item))
        self.counter += 1
    
    def pop(self) -> Any:
        """Pop item with lowest priority in O(log N)"""
        if not self.heap:
            raise IndexError("pop from empty priority queue")
        _, _, item = heapq.heappop(self.heap)
        return item
    
    def peek(self) -> Any:
        """Peek at item with lowest priority in O(1)"""
        if not self.heap:
            raise IndexError("peek from empty priority queue")
        return self.heap[0][2]
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.heap) == 0
    
    def __len__(self) -> int:
        return len(self.heap)
