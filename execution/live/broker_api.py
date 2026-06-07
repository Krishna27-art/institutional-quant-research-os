"""
Broker API adapter for Indian markets.
Supports Zerodha Kite and other brokers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    ZERODHA = "zerodha"
    GROWW = "groww"
    UPSTOX = "upstox"
    ANGELONE = "angelone"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_MARKET = "stop_loss_market"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    last_price: float
    pnl: float
    pnl_pct: float


class BrokerAPI(ABC):
    """Abstract base class for broker APIs."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to broker API."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_holdings(self) -> List[Position]:
        """Get holdings (overnight positions)."""
        pass

    @abstractmethod
    async def get_account_balance(self) -> Dict:
        """Get account balance and margin."""
        pass


class ZerodhaAPI(BrokerAPI):
    """
    Zerodha Kite Connect API adapter.
    """

    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self._kite = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Zerodha Kite Connect."""
        try:
            from kiteconnect import KiteConnect

            self._kite = KiteConnect(api_key=self.api_key)
            self._kite.set_access_token(self.access_token)

            # Test connection
            profile = self._kite.profile()
            logger.info(f"Connected to Zerodha as {profile.get('user_name', 'unknown')}")
            self._connected = True

        except ImportError:
            raise ImportError("kiteconnect not installed. Run: pip install kiteconnect")
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Zerodha."""
        self._kite = None
        self._connected = False
        logger.info("Disconnected from Zerodha")

    async def place_order(self, order: Order) -> Order:
        """Place an order with Zerodha."""
        if not self._connected:
            raise RuntimeError("Not connected to broker")

        try:
            kite_order = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=self._kite.EXCHANGE_NSE,
                tradingsymbol=order.symbol,
                transaction_type=self._kite.TRANSACTION_TYPE_BUY
                if order.side == OrderSide.BUY
                else self._kite.TRANSACTION_TYPE_SELL,
                quantity=order.quantity,
                order_type=self._kite.ORDER_TYPE_MARKET
                if order.order_type == OrderType.MARKET
                else self._kite.ORDER_TYPE_LIMIT,
                price=order.price if order.order_type == OrderType.LIMIT else None,
                trigger_price=order.trigger_price
                if order.order_type == OrderType.STOP_LOSS
                else None,
                product=self._kite.PRODUCT_MIS,
                exchange_token=None,
                validity=self._kite.VALIDITY_DAY,
                disclosed_quantity=None,
                tag=None,
            )

            order.order_id = str(kite_order["order_id"])
            order.status = OrderStatus.OPEN
            order.updated_at = datetime.now()

            logger.info(f"Order placed: {order.order_id} for {order.symbol}")
            return order

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._connected:
            raise RuntimeError("Not connected to broker")

        try:
            self._kite.cancel_order(order_id=order_id, variety=self._kite.VARIETY_REGULAR)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str) -> Order:
        """Get order status."""
        if not self._connected:
            raise RuntimeError("Not connected to broker")

        try:
            order_info = self._kite.order_history(order_id)
            latest = order_info[-1] if order_info else None

            if latest:
                status_map = {
                    "COMPLETE": OrderStatus.FILLED,
                    "CANCELLED": OrderStatus.CANCELLED,
                    "REJECTED": OrderStatus.REJECTED,
                    "OPEN": OrderStatus.OPEN,
                    "PARTIALLY FILLED": OrderStatus.PARTIALLY_FILLED,
                }

                order = Order(
                    order_id=order_id,
                    symbol=latest.get("tradingsymbol", ""),
                    side=OrderSide.BUY
                    if latest.get("transaction_type") == "BUY"
                    else OrderSide.SELL,
                    order_type=OrderType.MARKET,  # Simplified
                    quantity=latest.get("quantity", 0),
                    status=status_map.get(latest.get("status"), OrderStatus.PENDING),
                    filled_quantity=latest.get("filled_quantity", 0),
                    avg_fill_price=latest.get("average_price"),
                )
                return order

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")

        # Return empty order if failed
        return Order(
            order_id=order_id,
            symbol="",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=OrderStatus.PENDING,
        )

    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        if not self._connected:
            raise RuntimeError("Not connected to broker")

        try:
            positions = self._kite.positions()
            position_list = []

            for pos in positions.get("day", []):
                if pos.get("quantity", 0) != 0:
                    pnl = pos.get("pnl", 0.0)
                    position = Position(
                        symbol=pos.get("tradingsymbol", ""),
                        quantity=pos.get("quantity", 0),
                        avg_price=pos.get("average_price", 0.0),
                        last_price=pos.get("last_price", 0.0),
                        pnl=pnl,
                        pnl_pct=pos.get("pnl_percentage", 0.0),
                    )
                    position_list.append(position)

            return position_list

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_holdings(self) -> List[Position]:
        """Get holdings (overnight positions)."""
        if not self._connected:
            raise RuntimeError("Not connected to broker")

        try:
            holdings = self._kite.holdings()
            holding_list = []

            for h in holdings:
                holding = Position(
                    symbol=h.get("tradingsymbol", ""),
                    quantity=h.get("quantity", 0),
                    avg_price=h.get("average_price", 0.0),
                    last_price=h.get("last_price", 0.0),
                    pnl=h.get("pnl", 0.0),
                    pnl_pct=h.get("pnl_percentage", 0.0),
                )
                holding_list.append(holding)

            return holding_list

        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return []

    async def get_account_balance(self) -> Dict:
        """Get account balance and margin."""
        if not self._connected:
            raise RuntimeError("Not connected to broker")

        try:
            margins = self._kite.margins(segment="equity")

            return {
                "available_cash": margins.get("equity", {}).get("available", {}).get("live_balance", 0.0),
                "used_margin": margins.get("equity", {}).get("utilised", {}).get("debit", 0.0),
                "total_balance": margins.get("equity", {}).get("net", 0.0),
            }

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}


class BrokerAPIFactory:
    """Factory for creating broker API instances."""

    @staticmethod
    def create(broker_type: BrokerType, **kwargs) -> BrokerAPI:
        """Create a broker API instance."""
        if broker_type == BrokerType.ZERODHA:
            api_key = kwargs.get("api_key")
            access_token = kwargs.get("access_token")
            if not api_key or not access_token:
                raise ValueError("api_key and access_token required for Zerodha")
            return ZerodhaAPI(api_key, access_token)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
