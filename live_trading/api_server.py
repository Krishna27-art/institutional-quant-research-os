"""
Live Trading Architecture with FastAPI and WebSocket
Architecture V2 - Quantitative Trading System for Indian Markets

Components:
- FastAPI REST API for order management
- WebSocket for real-time market data
- Signal generation and execution
- Risk checks and position management
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass

from config_v2 import ArchitectureV2Config
from alpha_engines import create_alpha_engine, AlphaSignal
from regime import HMMRegimeEngine
from portfolio.alpha_combination import AlphaCombinationEngine
from portfolio.risk import RiskManagerV2
from features.feature_pipeline import FeaturePipeline, FeatureConfig
from database import DatabaseManager, DatabaseConfig


# Pydantic models for API
class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # "buy" or "sell"
    order_type: str = "limit"
    price: Optional[float] = None


class SignalResponse(BaseModel):
    symbol: str
    direction: str
    confidence: float
    expected_return: float
    timestamp: datetime


class PositionResponse(BaseModel):
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float


class PortfolioSummary(BaseModel):
    total_pnl: float
    daily_pnl: float
    positions: List[PositionResponse]
    leverage: float


@dataclass
class TradingState:
    """Live trading state."""
    positions: Dict[str, Dict]  # symbol -> position info
    orders: List[Dict]
    signals: List[Dict]
    daily_pnl: float = 0.0
    is_trading: bool = False


class LiveTradingEngine:
    """
    Live trading engine coordinating all components.
    
    Flow:
    1. Receive market data via WebSocket
    2. Compute features
    3. Detect regime
    4. Generate alpha signals
    5. Combine signals
    6. Risk check
    7. Execute orders
    """
    
    def __init__(self, config: ArchitectureV2Config):
        self.config = config
        
        # Initialize components
        self.db = DatabaseManager(DatabaseConfig(
            redis_host=config.database.redis_host,
            redis_port=config.database.redis_port,
            clickhouse_host=config.database.clickhouse_host,
            clickhouse_port=config.database.clickhouse_port,
            postgres_host=config.database.postgres_host,
            postgres_port=config.database.postgres_port
        ))
        
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.regime_engine = HMMRegimeEngine({
            "n_states": config.regime_engine.n_states,
            "features": config.regime_engine.features,
            "training_window_days": config.regime_engine.training_window_days
        })
        
        self.alpha_combination = AlphaCombinationEngine({
            "method": config.alpha_combination.method,
            "kelly_fraction": config.alpha_combination.kelly_fraction,
            "regime_weights": config.alpha_combination.regime_weights
        })
        
        self.risk_manager = RiskManagerV2(
            max_position_pct=config.risk_engine.max_position_size_pct,
            max_sector_exposure_pct=config.risk_engine.max_sector_exposure_pct,
            var_99_1day_cap_pct=config.risk_engine.var_99_1day_cap_pct,
            correlation_heat_threshold=config.risk_engine.correlation_heat_threshold,
            daily_circuit_breaker_pct=config.risk_engine.daily_circuit_breaker_pct
        )
        
        # Initialize alpha engines
        self.alpha_engines = {}
        for alpha_config in config.alpha_ranking.alphas:
            if alpha_config.status == "Must Build":
                self.alpha_engines[alpha_config.name] = create_alpha_engine(
                    alpha_config.name,
                    {"enabled": True}
                )
        
        # Trading state
        self.state = TradingState(
            positions={},
            orders=[],
            signals=[]
        )
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
    
    async def initialize(self):
        """Initialize trading engine."""
        print("Initializing live trading engine...")
        self.db.initialize()
        print("Live trading engine initialized")
    
    async def process_market_data(self, market_data: Dict):
        """
        Process incoming market data.
        
        Args:
            market_data: Dictionary with symbol, OHLCV data
        """
        symbol = market_data.get("symbol")
        if not symbol:
            return
        
        # Store in database
        self.db.store_market_data(symbol, market_data, persist_to_clickhouse=True)
        
        # Compute features
        # (In production, this would use historical data from database)
        # For now, skip feature computation
        features = {}
        
        # Store features
        self.db.store_features(symbol, features)
        
        # Detect regime
        regime_state = self.regime_engine.predict_regime(features, datetime.now())
        
        # Generate alpha signals
        all_signals = {}
        for alpha_name, alpha_engine in self.alpha_engines.items():
            signals = alpha_engine.generate_signals(
                {symbol: market_data},
                {symbol: features},
                datetime.now()
            )
            if signals:
                all_signals[alpha_name] = signals
        
        # Combine signals
        if all_signals:
            combined = self.alpha_combination.combine_signals(
                all_signals,
                regime_state.regime.value if regime_state else None
            )
            
            # Risk check
            # (Implement risk check logic)
            
            # Execute orders
            # (Implement order execution logic)
    
    async def broadcast_update(self, message: Dict):
        """Broadcast update to all WebSocket connections."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


# FastAPI application
app = FastAPI(
    title="Quantitative Trading System API",
    description="Architecture V2 - Indian Markets",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trading engine
trading_engine: Optional[LiveTradingEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize trading engine on startup."""
    global trading_engine
    config = ArchitectureV2Config()
    trading_engine = LiveTradingEngine(config)
    await trading_engine.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global trading_engine
    if trading_engine:
        trading_engine.db.close()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Quantitative Trading System API",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/orders")
async def create_order(order: OrderRequest):
    """
    Create a new order.
    
    Args:
        order: Order request with symbol, quantity, side, type, price
        
    Returns:
        Order confirmation
    """
    # Validate order
    if order.side not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Invalid side")
    
    if order.order_type == "limit" and order.price is None:
        raise HTTPException(status_code=400, detail="Price required for limit orders")
    
    # Risk check
    # (Implement risk check logic)
    
    # Create order
    order_response = {
        "order_id": f"ORD-{datetime.now().timestamp()}",
        "symbol": order.symbol,
        "quantity": order.quantity,
        "side": order.side,
        "order_type": order.order_type,
        "price": order.price,
        "status": "pending",
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to trading state
    if trading_engine:
        trading_engine.state.orders.append(order_response)
    
    return order_response


@app.get("/orders")
async def get_orders():
    """Get all orders."""
    if trading_engine:
        return trading_engine.state.orders
    return []


@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order by ID."""
    if trading_engine:
        for order in trading_engine.state.orders:
            if order.get("order_id") == order_id:
                return order
    raise HTTPException(status_code=404, detail="Order not found")


@app.get("/positions")
async def get_positions():
    """Get current positions."""
    if trading_engine:
        positions = []
        for symbol, pos in trading_engine.state.positions.items():
            positions.append({
                "symbol": symbol,
                "quantity": pos.get("quantity", 0),
                "avg_price": pos.get("avg_price", 0),
                "unrealized_pnl": pos.get("unrealized_pnl", 0)
            })
        return positions
    return []


@app.get("/portfolio")
async def get_portfolio_summary():
    """Get portfolio summary."""
    if trading_engine:
        positions = await get_positions()
        return PortfolioSummary(
            total_pnl=trading_engine.state.daily_pnl,
            daily_pnl=trading_engine.state.daily_pnl,
            positions=positions,
            leverage=1.0  # Calculate actual leverage
        )
    return PortfolioSummary(
        total_pnl=0.0,
        daily_pnl=0.0,
        positions=[],
        leverage=0.0
    )


@app.get("/signals")
async def get_signals():
    """Get recent signals."""
    if trading_engine:
        return trading_engine.state.signals[-100:]  # Last 100 signals
    return []


@app.get("/regime")
async def get_current_regime():
    """Get current market regime."""
    if trading_engine:
        regime = trading_engine.regime_engine.get_current_regime()
        if regime:
            return {
                "regime": regime.regime.value,
                "probability": regime.probability,
                "timestamp": regime.timestamp.isoformat(),
                "features": regime.features
            }
    return {"regime": "unknown", "probability": 0.0}


@app.post("/trading/start")
async def start_trading():
    """Start live trading."""
    if trading_engine:
        trading_engine.state.is_trading = True
        return {"status": "trading_started", "timestamp": datetime.now().isoformat()}
    raise HTTPException(status_code=500, detail="Trading engine not initialized")


@app.post("/trading/stop")
async def stop_trading():
    """Stop live trading."""
    if trading_engine:
        trading_engine.state.is_trading = False
        return {"status": "trading_stopped", "timestamp": datetime.now().isoformat()}
    raise HTTPException(status_code=500, detail="Trading engine not initialized")


@app.get("/trading/status")
async def get_trading_status():
    """Get trading status."""
    if trading_engine:
        return {
            "is_trading": trading_engine.state.is_trading,
            "daily_pnl": trading_engine.state.daily_pnl,
            "position_count": len(trading_engine.state.positions),
            "order_count": len(trading_engine.state.orders)
        }
    return {"is_trading": False, "daily_pnl": 0.0, "position_count": 0, "order_count": 0}


@app.websocket("/ws/market")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data."""
    await websocket.accept()
    if trading_engine:
        trading_engine.active_connections.append(websocket)
    
    try:
        while True:
            # Receive market data
            data = await websocket.receive_json()
            
            # Process market data
            if trading_engine:
                await trading_engine.process_market_data(data)
            
            # Broadcast update
            if trading_engine:
                await trading_engine.broadcast_update({
                    "type": "market_update",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        if trading_engine:
            trading_engine.active_connections.remove(websocket)


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signals."""
    await websocket.accept()
    if trading_engine:
        trading_engine.active_connections.append(websocket)
    
    try:
        while True:
            # Send recent signals
            if trading_engine:
                signals = trading_engine.state.signals[-10:]
                await websocket.send_json({
                    "type": "signals",
                    "data": signals,
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(1)  # Send every second
    
    except WebSocketDisconnect:
        if trading_engine:
            trading_engine.active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
