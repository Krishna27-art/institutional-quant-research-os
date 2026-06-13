"""
Point-in-Time Universe Tracker
Manages index constituents over time to prevent survivorship bias in historical backtests.
"""

import logging
from datetime import datetime
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)


class UniverseTracker:
    """
    Tracks constituent changes in indices (e.g. NIFTY 50) over time,
    providing point-in-time active symbols to backtesters.
    """

    def __init__(self) -> None:
        # Map of index_name -> sorted list of tuples (date, added_set, removed_set)
        self._history: Dict[str, List[tuple[datetime, Set[str], Set[str]]]] = {}
        # Initial state: index_name -> set of initial constituents
        self._initial_universe: Dict[str, Set[str]] = {}

    def set_initial_universe(self, index_name: str, symbols: List[str]) -> None:
        """Set the base universe for an index at the beginning of tracking."""
        self._initial_universe[index_name] = set(symbols)
        if index_name not in self._history:
            self._history[index_name] = []
        logger.info(f"Initialized base universe for {index_name} with {len(symbols)} symbols.")

    def add_change(
        self,
        index_name: str,
        date: datetime,
        added: List[str],
        removed: List[str]
    ) -> None:
        """Record a change (addition/removal) to index constituents at a specific date."""
        if index_name not in self._history:
            self._history[index_name] = []
            
        self._history[index_name].append((date, set(added), set(removed)))
        # Keep changes sorted by date
        self._history[index_name].sort(key=lambda x: x[0])
        logger.info(f"Recorded change for {index_name} on {date.strftime('%Y-%m-%d')}: +{len(added)}, -{len(removed)}")

    def get_universe(self, index_name: str, timestamp: datetime) -> List[str]:
        """
        Get the list of active constituents for an index at a specific point in time.
        """
        universe = set(self._initial_universe.get(index_name, []))

        # Replay changes up to the given timestamp
        changes = self._history.get(index_name, [])
        for change_date, added, removed in changes:
            if change_date > timestamp:
                # Changes are sorted, so we can stop replaying future changes
                break
            universe.update(added)
            universe.difference_update(removed)

        return sorted(list(universe))
