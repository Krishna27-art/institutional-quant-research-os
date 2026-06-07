"""
Zerodha Kite Connect Integration
Live market data feed and order execution

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, time
from dataclasses import dataclass
from enum import Enum
import aiohttp
import hmac
import hashlib
import base64


class BrokerType(Enum):
    ZERODHA = "zerodha"
    UPSTOX = "upstox"


@dataclass
class BrokerConfig:
    """Broker API configuration"""
    broker_type: BrokerType = BrokerType.ZERODHA
    
    # Zerodha Kite Connect
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_request_token: str = ""
    kite_access_token: str = ""
    
    # Upstox
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_access_token: str = ""
    
    # WebSocket
    ws_host: str = "wss://ws.kite.trade"  # Zerodha
    ws_port: int = 443
    
    # Rate limiting
    requests_per_second: int = 10
    requests_per_minute: int = 200


@dataclass
class Tick:
    """Market data tick"""
    instrument_token: str
    timestamp: datetime
    last_price: float
    last_quantity: int
    average_price: float
    volume: int
    buy_quantity: int
    sell_quantity: int
    ohlc: Dict[str, float]  # open, high, low, close


@dataclass
class Order:
    """Order representation"""
    order_id: str
    instrument_token: str
    exchange: str
    symbol: str
    transaction_type: str  # "BUY" or "SELL"
    quantity: int
    price: float
    trigger_price: Optional[float]
    order_type: str  # "MARKET", "LIMIT", "SL"
    status: str
    timestamp: datetime


class KiteConnectClient:
    """
    Zerodha Kite Connect API client.
    
    Features:
    - Authentication (request token -> access token)
    - Market data quotes
    - Historical data
    - Order placement
    - WebSocket for live ticks
    """
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.user_id: Optional[str] = None
        
        # API endpoints
        self.base_url = "https://api.kite.trade"
        
        # WebSocket
        self.ws_connection = None
        self.tick_callbacks: List[Callable] = []
        
        # Rate limiting
        self.request_timestamps: List[float] = []
    
    async def initialize(self) -> bool:
        """Initialize the client and authenticate."""
        self.session = aiohttp.ClientSession()
        
        # Get access token if not provided
        if not self.config.kite_access_token:
            success = await self._generate_access_token()
            if not success:
                return False
        else:
            self.access_token = self.config.kite_access_token
        
        # Get user profile
        await self._get_user_profile()
        
        return True
    
    async def _generate_access_token(self) -> bool:
        """Generate access token from request token."""
        try:
            # Kite Connect requires manual authorization flow
            # For now, assume access token is provided
            print("Access token not provided. Please complete manual authorization flow.")
            print("Visit: https://kite.trade/connect/login?v=3&api_key={}".format(
                self.config.kite_api_key
            ))
            return False
        
        except Exception as e:
            logging.error(f"Error generating access token: {e}")
            return False
    
    async def _get_user_profile(self) -> bool:
        """Get user profile."""
        try:
            url = f"{self.base_url}/user/profile"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.user_id = data['data']['user_id']
                    return True
                else:
                    logging.error(f"Failed to get user profile: {response.status}")
                    return False
        
        except Exception as e:
            logging.error(f"Error getting user profile: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "X-Kite-Version": "3",
            "Authorization": f"token {self.config.kite_api_key}:{self.access_token}"
        }
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        import time
        
        now = time.time()
        
        # Remove timestamps older than 1 second
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 1.0]
        
        # Check per-second limit
        if len(self.request_timestamps) >= self.config.requests_per_second:
            await asyncio.sleep(1.0)
            self.request_timestamps.clear()
        
        self.request_timestamps.append(now)
    
    async def get_quote(self, instrument_token: str) -> Optional[Dict]:
        """
        Get quote for an instrument.
        
        Args:
            instrument_token: Instrument token
            
        Returns:
            Quote data dictionary
        """
        await self._check_rate_limit()
        
        try:
            url = f"{self.base_url}/quote/{instrument_token}"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['data'][instrument_token]
                else:
                    logging.error(f"Failed to get quote: {response.status}")
                    return None
        
        except Exception as e:
            logging.error(f"Error getting quote: {e}")
            return None
    
    async def get_quotes(self, instrument_tokens: List[str]) -> Optional[Dict]:
        """
        Get quotes for multiple instruments.
        
        Args:
            instrument_tokens: List of instrument tokens
            
        Returns:
            Dictionary mapping tokens to quote data
        """
        await self._check_rate_limit()
        
        try:
            tokens_str = ",".join(instrument_tokens)
            url = f"{self.base_url}/quote/{tokens_str}"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['data']
                else:
                    logging.error(f"Failed to get quotes: {response.status}")
                    return None
        
        except Exception as e:
            logging.error(f"Error getting quotes: {e}")
            return None
    
    async def get_historical_data(
        self,
        instrument_token: str,
        from_date: str,
        to_date: str,
        interval: str = "day"
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for an instrument.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            interval: Interval (minute, day, week, month)
            
        Returns:
            DataFrame with OHLCV data
        """
        await self._check_rate_limit()
        
        try:
            import pandas as pd
            
            url = f"{self.base_url}/instruments/historical/{instrument_token}/{interval}/{from_date}/{to_date}"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    candles = data['data']['candles']
                    
                    df = pd.DataFrame(candles, columns=[
                        'date', 'open', 'high', 'low', 'close', 'volume'
                    ])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    return df
                else:
                    logging.error(f"Failed to get historical data: {response.status}")
                    return None
        
        except Exception as e:
            logging.error(f"Error getting historical data: {e}")
            return None
    
    async def place_order(
        self,
        exchange: str,
        symbol: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product: str = "MIS",
        variety: str = "regular"
    ) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            exchange: Exchange (NSE, BSE, NFO)
            symbol: Trading symbol
            transaction_type: BUY or SELL
            quantity: Quantity
            order_type: MARKET, LIMIT, SL
            price: Limit price (required for LIMIT orders)
            trigger_price: Trigger price (required for SL orders)
            product: Product type (MIS, CNC, NRML)
            variety: Order variety (regular, amo, bo, co)
            
        Returns:
            Order response dictionary
        """
        await self._check_rate_limit()
        
        try:
            url = f"{self.base_url}/orders/{variety}"
            headers = self._get_headers()
            
            params = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "variety": variety
            }
            
            if price is not None:
                params["price"] = price
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
            
            async with self.session.post(url, headers=headers, json=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['data']
                else:
                    error_text = await response.text()
                    logging.error(f"Failed to place order: {response.status} - {error_text}")
                    return None
        
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful
        """
        await self._check_rate_limit()
        
        try:
            url = f"{self.base_url}/orders/regular/{order_id}"
            headers = self._get_headers()
            
            async with self.session.delete(url, headers=headers) as response:
                return response.status == 200
        
        except Exception as e:
            logging.error(f"Error cancelling order: {e}")
            return False
    
    async def get_positions(self) -> Optional[List[Dict]]:
        """Get current positions."""
        await self._check_rate_limit()
        
        try:
            url = f"{self.base_url}/portfolio/positions"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['data']['net']
                else:
                    logging.error(f"Failed to get positions: {response.status}")
                    return None
        
        except Exception as e:
            logging.error(f"Error getting positions: {e}")
            return None
    
    async def get_holdings(self) -> Optional[List[Dict]]:
        """Get current holdings."""
        await self._check_rate_limit()
        
        try:
            url = f"{self.base_url}/portfolio/holdings"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['data']
                else:
                    logging.error(f"Failed to get holdings: {response.status}")
                    return None
        
        except Exception as e:
            logging.error(f"Error getting holdings: {e}")
            return None
    
    async def connect_websocket(self, instrument_tokens: List[str]) -> bool:
        """
        Connect to WebSocket for live ticks.
        
        Args:
            instrument_tokens: List of instrument tokens to subscribe
            
        Returns:
            True if connection successful
        """
        try:
            import websockets
            
            # Kite WebSocket requires a specific format
            ws_url = f"{self.config.ws_host}/?api_key={self.config.kite_api_key}&access_token={self.access_token}"
            
            self.ws_connection = await websockets.connect(ws_url)
            
            # Subscribe to tokens
            subscribe_message = {
                "a": "subscribe",
                "v": instrument_tokens,
                "mode": "full"
            }
            
            await self.ws_connection.send(json.dumps(subscribe_message))
            
            # Start listening for ticks
            asyncio.create_task(self._listen_to_websocket())
            
            return True
        
        except ImportError:
            logging.error("websockets not installed. Install with: pip install websockets")
            return False
        except Exception as e:
            logging.error(f"Error connecting to WebSocket: {e}")
            return False
    
    async def _listen_to_websocket(self):
        """Listen for WebSocket messages."""
        try:
            while self.ws_connection:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                
                # Process tick data
                if isinstance(data, dict) and 'tick' in data:
                    tick = self._parse_tick(data['tick'])
                    
                    # Call registered callbacks
                    for callback in self.tick_callbacks:
                        await callback(tick)
        
        except Exception as e:
            logging.error(f"Error in WebSocket listener: {e}")
        finally:
            await self.disconnect_websocket()
    
    def _parse_tick(self, tick_data: Dict) -> Tick:
        """Parse tick data from WebSocket."""
        return Tick(
            instrument_token=str(tick_data.get('instrument_token', '')),
            timestamp=datetime.now(),
            last_price=tick_data.get('last_price', 0.0),
            last_quantity=tick_data.get('last_quantity', 0),
            average_price=tick_data.get('average_price', 0.0),
            volume=tick_data.get('volume', 0),
            buy_quantity=tick_data.get('buy_quantity', 0),
            sell_quantity=tick_data.get('sell_quantity', 0),
            ohlc=tick_data.get('ohlc', {})
        )
    
    def register_tick_callback(self, callback: Callable):
        """Register a callback for tick updates."""
        self.tick_callbacks.append(callback)
    
    async def disconnect_websocket(self):
        """Disconnect from WebSocket."""
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
        
        await self.disconnect_websocket()


class BrokerManager:
    """
    Unified broker manager supporting multiple brokers.
    
    Currently supports:
    - Zerodha Kite Connect
    """
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.client: Optional[KiteConnectClient] = None
    
    async def initialize(self) -> bool:
        """Initialize the broker connection."""
        if self.config.broker_type == BrokerType.ZERODHA:
            self.client = KiteConnectClient(self.config)
            return await self.client.initialize()
        else:
            logging.error(f"Unsupported broker type: {self.config.broker_type}")
            return False
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current market data for symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbols to quote data
        """
        if not self.client:
            return {}
        
        # Map symbols to instrument tokens (simplified)
        # In production, would use instrument lookup API
        instrument_tokens = [symbol for symbol in symbols]
        
        quotes_data = await self.client.get_quotes(instrument_tokens)
        
        if quotes_data:
            return {symbol: quotes_data.get(symbol, {}) for symbol in symbols}
        
        return {}
    
    async def subscribe_live_data(self, symbols: List[str], callback: Callable) -> bool:
        """
        Subscribe to live market data via WebSocket.
        
        Args:
            symbols: List of symbols to subscribe
            callback: Callback function for tick updates
            
        Returns:
            True if subscription successful
        """
        if not self.client:
            return False
        
        # Register callback
        self.client.register_tick_callback(callback)
        
        # Map symbols to instrument tokens
        instrument_tokens = [symbol for symbol in symbols]
        
        # Connect to WebSocket
        return await self.client.connect_websocket(instrument_tokens)
    
    async def place_order(self, order_params: Dict) -> Optional[Dict]:
        """
        Place an order.
        
        Args:
            order_params: Order parameters dictionary
            
        Returns:
            Order response
        """
        if not self.client:
            return None
        
        return await self.client.place_order(
            exchange=order_params.get('exchange', 'NSE'),
            symbol=order_params.get('symbol', ''),
            transaction_type=order_params.get('transaction_type', 'BUY'),
            quantity=order_params.get('quantity', 0),
            order_type=order_params.get('order_type', 'MARKET'),
            price=order_params.get('price'),
            trigger_price=order_params.get('trigger_price'),
            product=order_params.get('product', 'MIS'),
            variety=order_params.get('variety', 'regular')
        )
    
    async def get_positions(self) -> Optional[List[Dict]]:
        """Get current positions."""
        if not self.client:
            return None
        
        return await self.client.get_positions()
    
    async def close(self):
        """Close broker connection."""
        if self.client:
            await self.client.close()


async def test_broker_connection():
    """Test broker connection with example config."""
    config = BrokerConfig(
        broker_type=BrokerType.ZERODHA,
        kite_api_key="your_api_key",
        kite_api_secret="your_api_secret",
        kite_access_token="your_access_token"
    )
    
    manager = BrokerManager(config)
    
    # Note: This will fail without valid credentials
    # success = await manager.initialize()
    # if success:
    #     print("Broker connection successful")
    #     quotes = await manager.get_market_data(["RELIANCE", "HDFCBANK"])
    #     print(f"Quotes: {quotes}")
    # else:
    #     print("Broker connection failed")
    
    print("Broker integration module loaded. Configure credentials in .env file to use.")


if __name__ == "__main__":
    asyncio.run(test_broker_connection())
