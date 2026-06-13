import asyncio
import inspect
import json
import os
import websockets
from datetime import datetime
import logging
import time
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

class NSEWebSocketStream:
    def __init__(self, symbols: list, on_bar_callback):
        self.symbols = symbols
        self.on_bar = on_bar_callback
        self.current_bars = {}
        self.last_cumulative_volume = {}
        self.ws = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 60.0  # Maximum delay in seconds
        self.is_connected = False
        self.last_message_time = None
        self.message_timeout = 30.0  # Seconds before considering connection dead
        
    async def connect(self):
        """Connect to WebSocket with auto-reconnect and exponential backoff."""
        uri = os.getenv("NSE_WEBSOCKET_URL") or os.getenv("KITE_WEBSOCKET_URL")
        if not uri:
            logger.error("Set NSE_WEBSOCKET_URL or KITE_WEBSOCKET_URL to connect a real market stream")
            return False
        
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                logger.info(f"WebSocket connection attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts}")
                self.ws = await websockets.connect(uri, ping_interval=20, ping_timeout=10)
                self.is_connected = True
                self.reconnect_attempts = 0
                self.reconnect_delay = 1.0
                self.last_message_time = datetime.now()
                
                # Subscribe to symbols
                subscribe_msg = {"action": "subscribe", "symbols": self.symbols}
                await self.ws.send(json.dumps(subscribe_msg))
                logger.info(f"Subscribed to {len(self.symbols)} symbols")
                
                # Process messages with timeout check
                try:
                    async for message in self.ws:
                        self.last_message_time = datetime.now()
                        data = json.loads(message)
                        result = self._process_tick(data)
                        if inspect.isawaitable(result):
                            await result
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed, attempting reconnect...")
                except Exception as e:
                    logger.error(f"Error processing WebSocket messages: {e}")
                    
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                self.is_connected = False
                
            # Exponential backoff for reconnection
            if self.reconnect_attempts < self.max_reconnect_attempts:
                delay = min(self.reconnect_delay * (2 ** self.reconnect_attempts), self.max_reconnect_delay)
                logger.info(f"Reconnecting in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
                self.reconnect_attempts += 1
            else:
                logger.error("Max reconnection attempts reached. Giving up.")
                return False
                
        return False
    
    async def start(self):
        """Start the WebSocket connection with automatic reconnection."""
        logger.info("Starting WebSocket stream...")
        await self.connect()
        
    async def stop(self):
        """Stop the WebSocket connection."""
        if self.ws:
            await self.ws.close()
        self.is_connected = False
        logger.info("WebSocket stream stopped")
    
    def is_alive(self) -> bool:
        """Check if connection is alive and receiving messages."""
        if not self.is_connected or not self.ws:
            return False
        
        # Check if we've received messages recently
        if self.last_message_time:
            time_since_last = (datetime.now() - self.last_message_time).total_seconds()
            if time_since_last > self.message_timeout:
                logger.warning(f"No messages for {time_since_last:.1f} seconds, connection may be dead")
                return False
        
        return True
    
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
