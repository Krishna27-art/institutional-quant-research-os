"""
Layer Interfaces - Clear Contracts Between Architectural Layers

This module defines the interfaces that each layer must implement.
These interfaces enforce the architectural boundaries and ensure
proper separation of concerns.

Each layer implements a specific interface:
- Infrastructure Layer: InfrastructureProvider
- Data Layer: DataProvider
- Feature Layer: FeatureProvider
- Research Layer: SignalProvider
- Portfolio Layer: PortfolioManager
- Execution Layer: ExecutionEngine
- Presentation Layer: PresentationLayer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
from enum import Enum


# ============================================================================
# Data Models
# ============================================================================

class DataType(Enum):
    """Types of data."""
    TICK = "tick"
    OHLCV = "ohlcv"
    ORDER_BOOK = "order_book"
    NEWS = "news"


class SignalType(Enum):
    """Types of trading signals."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Tick:
    """Tick data."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    signal_type: SignalType
    confidence: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    timestamp: datetime
    strategy: str
    metadata: Dict[str, Any]


@dataclass
class Order:
    """Trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]
    stop_price: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class ExecutionResult:
    """Order execution result."""
    order_id: str
    success: bool
    filled_quantity: int
    filled_price: float
    timestamp: datetime
    error_message: Optional[str]


@dataclass
class Position:
    """Trading position."""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime


@dataclass
class Portfolio:
    """Portfolio state."""
    positions: Dict[str, Position]
    cash: float
    total_value: float
    timestamp: datetime


# ============================================================================
# Layer Interfaces
# ============================================================================

class InfrastructureProvider(ABC):
    """Infrastructure layer interface."""
    
    @abstractmethod
    def publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to the event bus."""
        pass
    
    @abstractmethod
    def subscribe_event(self, event_type: str, callback: Callable) -> None:
        """Subscribe to events from the event bus."""
        pass
    
    @abstractmethod
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def emit_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Emit a metric for monitoring."""
        pass


class DataProvider(ABC):
    """Data layer interface."""
    
    @abstractmethod
    def get_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1min"
    ) -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            symbol: Stock/index symbol
            start: Start datetime
            end: End datetime
            interval: Data interval (1min, 5min, 15min, 1hour, 1day)
        
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get the latest tick for a symbol."""
        pass
    
    @abstractmethod
    def subscribe_ticks(self, symbols: List[str], callback: Callable) -> None:
        """Subscribe to real-time tick data."""
        pass
    
    @abstractmethod
    def unsubscribe_ticks(self, symbols: List[str]) -> None:
        """Unsubscribe from tick data."""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
    
    @abstractmethod
    def is_data_valid(self, symbol: str) -> bool:
        """Check if data for symbol is valid and fresh."""
        pass


class FeatureProvider(ABC):
    """Feature layer interface."""
    
    @abstractmethod
    def compute_features(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Compute features for a symbol.
        
        Args:
            symbol: Stock/index symbol
            data: OHLCV data
            feature_names: List of feature names to compute
        
        Returns:
            DataFrame with computed features
        """
        pass
    
    @abstractmethod
    def get_features(
        self,
        symbol: str,
        feature_names: List[str],
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Get pre-computed features from feature store.
        
        Args:
            symbol: Stock/index symbol
            feature_names: List of feature names
            start: Start datetime
            end: End datetime
        
        Returns:
            DataFrame with features
        """
        pass
    
    @abstractmethod
    def store_features(
        self,
        symbol: str,
        features: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store computed features in feature store."""
        pass
    
    @abstractmethod
    def get_available_features(self) -> List[str]:
        """Get list of available feature names."""
        pass
    
    @abstractmethod
    def validate_features(self, features: pd.DataFrame) -> bool:
        """Validate features for correctness and completeness."""
        pass


class SignalProvider(ABC):
    """Research layer interface."""
    
    @abstractmethod
    def generate_signals(
        self,
        symbols: List[str],
        features: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Generate trading signals from features.
        
        Args:
            symbols: List of symbols to generate signals for
            features: Dictionary of symbol -> features DataFrame
        
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def get_signal_confidence(self, signal: Signal) -> float:
        """Get confidence score for a signal."""
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Signal) -> bool:
        """Validate signal for correctness and risk."""
        pass
    
    @abstractmethod
    def get_active_strategies(self) -> List[str]:
        """Get list of active trading strategies."""
        pass
    
    @abstractmethod
    def train_model(
        self,
        strategy: str,
        features: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train a model for a strategy.
        
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def get_model_performance(self, strategy: str) -> Dict[str, Any]:
        """Get performance metrics for a strategy's model."""
        pass


class PortfolioManager(ABC):
    """Portfolio layer interface."""
    
    @abstractmethod
    def construct_portfolio(
        self,
        signals: List[Signal],
        current_portfolio: Portfolio
    ) -> Portfolio:
        """
        Construct portfolio from signals.
        
        Args:
            signals: List of trading signals
            current_portfolio: Current portfolio state
        
        Returns:
            New portfolio state
        """
        pass
    
    @abstractmethod
    def manage_risk(self, portfolio: Portfolio) -> Portfolio:
        """
        Apply risk management to portfolio.
        
        Args:
            portfolio: Current portfolio state
        
        Returns:
            Risk-adjusted portfolio
        """
        pass
    
    @abstractmethod
    def allocate_capital(
        self,
        signals: List[Signal],
        total_capital: float
    ) -> Dict[str, float]:
        """
        Allocate capital across signals.
        
        Args:
            signals: List of trading signals
            total_capital: Total capital to allocate
        
        Returns:
            Dictionary of symbol -> allocated capital
        """
        pass
    
    @abstractmethod
    def get_portfolio_metrics(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get portfolio metrics (returns, risk, etc.)."""
        pass
    
    @abstractmethod
    def validate_portfolio(self, portfolio: Portfolio) -> bool:
        """Validate portfolio for constraints and limits."""
        pass


class ExecutionEngine(ABC):
    """Execution layer interface."""
    
    @abstractmethod
    def execute_order(self, order: Order) -> ExecutionResult:
        """
        Execute a trading order.
        
        Args:
            order: Order to execute
        
        Returns:
            Execution result
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders, optionally filtered by status."""
        pass
    
    @abstractmethod
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics (latency, fill rate, etc.)."""
        pass
    
    @abstractmethod
    def validate_order(self, order: Order) -> bool:
        """Validate order for correctness and risk."""
        pass


class PresentationLayer(ABC):
    """Presentation layer interface."""
    
    @abstractmethod
    def display_dashboard(self) -> None:
        """Display the main dashboard."""
        pass
    
    @abstractmethod
    def handle_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an API request.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
        
        Returns:
            API response
        """
        pass
    
    @abstractmethod
    def execute_command(self, command: str, args: List[str]) -> Dict[str, Any]:
        """
        Execute a CLI command.
        
        Args:
            command: Command name
            args: Command arguments
        
        Returns:
            Command result
        """
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        pass
    
    @abstractmethod
    def send_alert(self, alert_type: str, message: str) -> None:
        """Send an alert to the user."""
        pass


# ============================================================================
# Layer Registry
# ============================================================================

class LayerRegistry:
    """
    Registry for layer implementations.
    
    This ensures that each layer is properly registered and
    can be retrieved by its interface type.
    """
    
    def __init__(self):
        self._layers: Dict[type, Any] = {}
    
    def register_layer(self, layer_interface: type, implementation: Any) -> None:
        """
        Register a layer implementation.
        
        Args:
            layer_interface: The interface class (e.g., DataProvider)
            implementation: The implementation instance
        """
        if not isinstance(implementation, layer_interface):
            raise TypeError(
                f"Implementation must implement {layer_interface.__name__}"
            )
        self._layers[layer_interface] = implementation
    
    def get_layer(self, layer_interface: type) -> Any:
        """
        Get a layer implementation.
        
        Args:
            layer_interface: The interface class
        
        Returns:
            The registered implementation
        """
        if layer_interface not in self._layers:
            raise ValueError(
                f"No implementation registered for {layer_interface.__name__}"
            )
        return self._layers[layer_interface]
    
    def is_registered(self, layer_interface: type) -> bool:
        """Check if a layer is registered."""
        return layer_interface in self._layers
    
    def get_registered_layers(self) -> List[type]:
        """Get list of registered layer interfaces."""
        return list(self._layers.keys())


# Singleton instance
_layer_registry: Optional[LayerRegistry] = None


def get_layer_registry() -> LayerRegistry:
    """Get the singleton layer registry instance."""
    global _layer_registry
    if _layer_registry is None:
        _layer_registry = LayerRegistry()
    return _layer_registry


# ============================================================================
# Layer Validation
# ============================================================================

def validate_layer_implementation(layer_interface: type, implementation: Any) -> bool:
    """
    Validate that an implementation properly implements an interface.
    
    Args:
        layer_interface: The interface class
        implementation: The implementation to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(implementation, layer_interface):
        return False
    
    # Check that all abstract methods are implemented
    for method_name in dir(layer_interface):
        if not method_name.startswith('_'):
            attr = getattr(layer_interface, method_name)
            if callable(attr) and hasattr(attr, '__isabstractmethod__'):
                if not hasattr(implementation, method_name):
                    return False
    
    return True


if __name__ == "__main__":
    # Test the layer interfaces
    print("Testing Layer Interfaces...")
    
    # Create registry
    registry = LayerRegistry()
    
    # Test that registry works
    print(f"Registered layers: {registry.get_registered_layers()}")
    print(f"Data Provider registered: {registry.is_registered(DataProvider)}")
    
    print("Layer interfaces test completed successfully")
