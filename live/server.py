"""
Live trading server for the quantitative trading system.
Handles WebSocket connections, order management, and real-time monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pydantic

logger = logging.getLogger(__name__)


app = FastAPI(title="NiftyQuant Live Trading Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OrderRequest(pydantic.BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: str = "market"  # "market", "limit", "vwap"
    price: Optional[float] = None


class OrderResponse(pydantic.BaseModel):
    order_id: str
    status: str
    message: str


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")


manager = ConnectionManager()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "system": "NiftyQuant",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/orders", response_model=OrderResponse)
async def submit_order(order: OrderRequest):
    """
    Submit a new order.

    Args:
        order: Order request with symbol, side, quantity, etc.

    Returns:
        Order response with order ID and status
    """
    # In production, this would interface with the execution engine
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{order.symbol}"
    
    logger.info(
        f"Order received: {order.side.upper()} {order.quantity} {order.symbol} "
        f"@ {order.price if order.price else 'MARKET'}"
    )
    
    # Broadcast order submission
    await manager.broadcast({
        "type": "order_submitted",
        "order_id": order_id,
        "symbol": order.symbol,
        "side": order.side,
        "quantity": order.quantity,
        "price": order.price,
        "timestamp": datetime.now().isoformat(),
    })
    
    return OrderResponse(
        order_id=order_id,
        status="submitted",
        message="Order submitted successfully"
    )


@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order status by ID."""
    # In production, this would query the order database
    return {
        "order_id": order_id,
        "status": "filled",
        "filled_quantity": 100,
        "avg_price": 2500.50,
    }


@app.get("/positions")
async def get_positions():
    """Get current positions."""
    # In production, this would query the position database
    return {
        "positions": [
            {
                "symbol": "RELIANCE",
                "quantity": 100,
                "avg_price": 2500.50,
                "current_price": 2510.00,
                "pnl": 950.00,
            }
        ]
    }


@app.get("/portfolio")
async def get_portfolio():
    """Get portfolio summary."""
    # In production, this would calculate from positions
    return {
        "total_value": 10_500_000.00,
        "cash": 500_000.00,
        "invested": 10_000_000.00,
        "daily_pnl": 50_000.00,
        "daily_pnl_pct": 0.48,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            logger.debug(f"WebSocket received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the live trading server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
