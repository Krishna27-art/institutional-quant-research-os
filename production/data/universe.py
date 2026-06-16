"""
Universe module.
"""

from typing import Any


class UniverseRegistry:
    """Universe registry."""
    
    def __init__(self):
        self.universe = []
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to universe."""
        self.universe.append(symbol)
    
    def get_universe(self) -> list:
        """Get universe."""
        return self.universe
