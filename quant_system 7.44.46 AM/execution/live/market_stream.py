import asyncio
import inspect
import json
import os
import websockets
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NSEWebSocketStream:
    def __init__(self, symbols: list, on_bar_callback):
        self.symbols = symbols
        self.on_bar = on_bar_callback
        self.current_bars = {}
        self.last_cumulative_volume = {}
        self.ws = None
        
    async def connect(self):
        uri = os.getenv("NSE_WEBSOCKET_URL") or os.getenv("KITE_WEBSOCKET_URL")
        if not uri:
            logger.error("Set NSE_WEBSOCKET_URL or KITE_WEBSOCKET_URL to connect a real market stream")
            return
        
        try:
            self.ws = await websockets.connect(uri)
            # Subscribe to symbols
            subscribe_msg = {"action": "subscribe", "symbols": self.symbols}
            await self.ws.send(json.dumps(subscribe_msg))
            
            async for message in self.ws:
                data = json.loads(message)
                result = self._process_tick(data)
                if inspect.isawaitable(result):
                    await result
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
    
    def _process_tick(self, tick):
        if isinstance(tick, list):
            for item in tick:
                self._process_tick(item)
            return

        if 'tick' in tick and isinstance(tick['tick'], dict):
            tick = tick['tick']

        symbol = tick.get('symbol')
        if not symbol:
            return

        price = tick.get('price', tick.get('last_price', tick.get('ltp')))
        if price is None:
            return
        price = float(price)

        raw_ts = tick.get('timestamp', tick.get('time', tick.get('exchange_timestamp')))
        ts = self._parse_timestamp(raw_ts)
        minute = ts.replace(second=0, microsecond=0)

        volume_delta = self._volume_delta(symbol, tick)
        key = (symbol, minute)

        previous_keys = [bar_key for bar_key in self.current_bars if bar_key[0] == symbol and bar_key[1] < minute]
        for previous_key in sorted(previous_keys, key=lambda item: item[1]):
            result = self._emit_bar(self.current_bars.pop(previous_key))
            if inspect.isawaitable(result):
                return result

        if key not in self.current_bars:
            self.current_bars[key] = {
                "symbol": symbol,
                "timestamp": minute,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume_delta,
            }
            return

        bar = self.current_bars[key]
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += volume_delta

    def _parse_timestamp(self, raw_ts):
        if raw_ts is None:
            return datetime.now()
        if isinstance(raw_ts, datetime):
            return raw_ts
        if isinstance(raw_ts, (int, float)):
            value = raw_ts / 1000 if raw_ts > 10_000_000_000 else raw_ts
            return datetime.fromtimestamp(value)
        return datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00")).replace(tzinfo=None)

    def _volume_delta(self, symbol, tick):
        volume = tick.get('volume', tick.get('last_traded_quantity', tick.get('last_quantity', 0)))
        volume = float(volume or 0)
        if tick.get('volume_is_cumulative', False):
            last = self.last_cumulative_volume.get(symbol, volume)
            self.last_cumulative_volume[symbol] = volume
            return max(0.0, volume - last)
        return max(0.0, volume)

    def _emit_bar(self, bar):
        result = self.on_bar(bar)
        if inspect.isawaitable(result):
            return result
