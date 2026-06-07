"""
Corporate actions module.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CorporateAction:
    """Corporate action event."""
    symbol: str
    action_type: str
    date: datetime
    details: dict
