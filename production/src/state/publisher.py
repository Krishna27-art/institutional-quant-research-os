import asyncio
from fastapi import WebSocket
from src.core.config.settings import TRADING_CAPITAL

class StatePublisher:
    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.state = {
            'nav': TRADING_CAPITAL,
            'last_pnl': 0.0,
            'daily_pnl': 0.0,
            'positions': [],
            'risk': {'var': 0.0, 'cvar': 0.0, 'tail_risk': 0.0},
            'regime': 'unknown',
            'regime_confidence': 0.0,
            'signals': [],
            'pnl': {'daily': 0.0},
            'indices': [],
            'market_status': {},
            'updated_at': None,
        }

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)
        await websocket.send_json(self.state)

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, data: dict) -> None:
        self.state = {**self.state, **data}
        stale = []
        for websocket in self.connections:
            try:
                await websocket.send_json(self.state)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)

publisher = StatePublisher()
state_lock = asyncio.Lock()
