"""Compatibility wrapper for the NSE data loader.

This routes imports to the unified `src/data/` package.
"""

from src.data.data_loader import NSEDataLoader, CorporateActionAdjuster

__all__ = ["NSEDataLoader", "CorporateActionAdjuster"]
