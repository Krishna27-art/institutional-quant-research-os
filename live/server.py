"""
Live trading server for the quantitative trading system.
Handles WebSocket connections, order management, and real-time monitoring.
"""

import asyncio
import os
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LiveState:
    """In-memory state for dashboard and live endpoints."""

    def __init__(self) -> None:
        self.positions: list[dict] = [
            {"symbol": "RELIANCE", "quantity": 100, "avg_price": 2500.50, "current_price": 2510.00, "pnl": 950.00}
        ]
        self.alpha_signals: list[dict] = [
            {"strategy": "orb", "symbol": "RELIANCE", "direction": 1.0, "strength": 0.7, "confidence": 0.8, "timestamp": datetime.now().isoformat()}
        ]
        self.pnl: dict[str, float] = {
            "total_value": 10_500_000.00,
            "cash": 500_000.00,
            "invested": 10_000_000.00,
            "daily_pnl": 50_000.00,
            "daily_pnl_pct": 0.48,
        }
        self.risk_metrics: dict[str, float | bool] = {
            "var": 2.14,
            "cvar": 3.82,
            "gross_exposure": 142,
            "max_drawdown": -4.21,
            "circuit_breaker_active": False,
        }


state = LiveState()


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


def _validate_runtime_env() -> None:
    """Validate live credentials before startup."""
    live_mode = os.getenv("LIVE_TRADING_MODE", "false").lower() in {"1", "true", "yes"}
    required = ["ZERODHA_API_KEY", "ZERODHA_ACCESS_TOKEN"]
    missing = [key for key in required if not os.getenv(key)]
    if missing and live_mode:
        raise RuntimeError(f"Missing live trading credentials: {', '.join(missing)}")
    if missing:
        logger.warning("Missing live trading credentials: %s", ", ".join(missing))


@app.on_event("startup")
async def _startup() -> None:
    _validate_runtime_env()


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


@app.get("/api/portfolio/positions")
async def api_portfolio_positions():
    """Return live portfolio positions."""
    return {"positions": state.positions}


@app.get("/api/portfolio/pnl")
async def api_portfolio_pnl():
    """Return live PnL summary."""
    return state.pnl


@app.get("/api/alpha/signals")
async def api_alpha_signals():
    """Return current alpha signals."""
    return {"signals": state.alpha_signals}


@app.get("/api/risk/metrics")
async def api_risk_metrics():
    """Return current risk metrics."""
    return state.risk_metrics


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

    state.alpha_signals.append({
        "strategy": "execution",
        "symbol": order.symbol,
        "direction": 1.0 if order.side.lower() == "buy" else -1.0,
        "strength": 0.0,
        "confidence": 1.0,
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


# Dashboard API endpoints
@app.get("/api/market")
async def get_market_data():
    """Get market data for dashboard."""
    return {
        "nifty": {"value": 24550, "change": 73.2, "change_pct": 0.30},
        "banknifty": {"value": 52120, "change": 260.5, "change_pct": 0.50},
        "vix": {"value": 14.20, "change": -0.18, "change_pct": -1.25},
        "usdinr": {"value": 83.47, "change": 0.12, "change_pct": 0.14},
    }


@app.get("/api/regime")
async def get_regime_data():
    """Get regime data for dashboard."""
    return {
        "current_regime": {"name": "BULL TREND", "state": 0, "confidence": 87},
        "duration": 34,
        "transition_prob": 12.3,
        "regime_sharpe": 2.14,
    }


@app.get("/api/alpha")
async def get_alpha_data():
    """Get alpha data for dashboard."""
    return {
        "live_alphas": 8,
        "avg_sharpe": 1.87,
        "best_sharpe": 3.21,
        "alpha_correlation": 0.18,
    }


@app.get("/api/risk")
async def get_risk_data():
    """Get risk data for dashboard."""
    return {
        "var": 2.14,
        "cvar": 3.82,
        "gross_exposure": 142,
        "max_drawdown": -4.21,
    }


@app.get("/api/portfolio")
async def get_portfolio_dashboard():
    """Get portfolio data for dashboard."""
    return {"aum": 500, "daily_pnl": 1.42, "mtd_pnl": 8.74, "net_exposure": 62}


@app.get("/api/options")
async def get_options_data():
    """Get options data for dashboard."""
    return {
        "atm_iv": 14.80,
        "iv_rank": 28,
        "pcr_oi": 1.24,
        "max_pain": 24500,
    }


@app.get("/api/signals")
async def get_signals_data():
    """Get signals data for dashboard."""
    return {"active_signals": len(state.alpha_signals), "hit_rate": 62.3, "avg_r": 0.82, "signal_strength": "HIGH"}


@app.get("/api/alerts")
async def get_alerts_data():
    """Get alerts data for dashboard."""
    return {
        "alerts_today": 7,
        "critical": 2,
        "warning": 3,
        "false_positive": 12,
    }


class CopilotRequest(pydantic.BaseModel):
    message: str


@app.post("/api/copilot/chat")
async def copilot_chat(request: CopilotRequest):
    """AI Copilot chat endpoint."""
    # In production, this would call an AI service
    responses = [
        "Analyzing current market conditions... VaR is within safe zone at 2.14%. Regime confidence is strong at 87%.",
        "Based on feature drift analysis, VWAP-distance and relative volume show the strongest predictive power this week.",
        "I recommend monitoring the BANKNIFTY position — PSI drift detected on key features. Consider reducing exposure by 15%.",
        "Rolling Sharpe on ORB Momentum has declined to 1.2 from 2.1 over 30 days. Investigate regime-conditional performance.",
        "Stress test suggests COVID-like scenario would draw down portfolio by 12.4%. Current hedges cover 40% of downside.",
    ]
    import random
    return {"reply": random.choice(responses)}


@app.websocket("/ws")
@app.websocket("/ws/live")
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


@app.on_event("startup")
async def _initial_broadcast() -> None:
    """Warm the websocket feed with a first snapshot."""
    await manager.broadcast({
        "type": "snapshot",
        "positions": state.positions,
        "pnl": state.pnl,
        "risk": state.risk_metrics,
        "timestamp": datetime.now().isoformat(),
    })


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the live trading server."""
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except OSError as e:
        if "address already in use" in str(e):
            print(f"Error: Port {port} is already in use.")
            print(f"Try killing the process using: lsof -ti:{port} | xargs kill -9")
            print(f"Or use a different port: python3 live/server.py --port 8001")
        raise


if __name__ == "__main__":
    import sys
    port = 8000
    if len(sys.argv) > 1:
        if sys.argv[1] == "--port" and len(sys.argv) > 2:
            port = int(sys.argv[2])
    start_server(port=port)
