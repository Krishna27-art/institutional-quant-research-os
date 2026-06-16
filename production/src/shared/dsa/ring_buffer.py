"""
Ring Buffer - O(1) append, O(1) average for rolling windows
"""

import numpy as np
from typing import Optional, List
from collections import deque


class RingBuffer:
    """Fixed-size ring buffer for rolling window operations"""
    
    def __init__(self, size: int, dtype: type = float):
        self.size = size
        self.dtype = dtype
        self.buffer = np.zeros(size, dtype=dtype)
        self.index = 0
        self.count = 0
        self.is_full = False
    
    def append(self, value: float) -> None:
        """Append a value to the buffer"""
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        
        if not self.is_full:
            self.count += 1
            if self.count == self.size:
                self.is_full = True
    
    def get(self, i: int) -> float:
        """Get value at index i (0 = oldest, -1 = newest)"""
        if i < 0:
            i = self.count + i
        
        if i >= self.count:
            raise IndexError("Index out of range")
        
        actual_index = (self.index - self.count + i) % self.size
        return self.buffer[actual_index]
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in chronological order"""
        if not self.is_full:
            return self.buffer[:self.count]
        
        return np.concatenate([self.buffer[self.index:], self.buffer[:self.index]])
    
    def mean(self) -> float:
        """Compute mean of all values"""
        if self.count == 0:
            return 0.0
        return np.mean(self.to_array())
    
    def std(self) -> float:
        """Compute standard deviation"""
        if self.count < 2:
            return 0.0
        return np.std(self.to_array())
    
    def sum(self) -> float:
        """Compute sum"""
        return np.sum(self.to_array())
    
    def reset(self) -> None:
        """Reset buffer"""
        self.buffer.fill(0)
        self.index = 0
        self.count = 0
        self.is_full = False
    
    def __len__(self) -> int:
        return self.count
    
    def __getitem__(self, i: int) -> float:
        return self.get(i)
