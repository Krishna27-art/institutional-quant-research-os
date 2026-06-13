"""Live trading components for the quantitative trading system."""

try:
    from .server import app, manager, start_server
except ModuleNotFoundError:
    app = None
    manager = None

    def start_server(*args, **kwargs):
        raise RuntimeError("execution.live.server is disabled or unavailable")

from .broker_api import BrokerAPI

__all__ = [
    "app",
    "manager",
    "start_server",
    "BrokerAPI",
]
