"""
Institutional-grade data layer for Indian markets.
Supports Zerodha Kite, Yahoo Finance, and Arctic tick database.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis
import yaml
from arctic import Arctic

logger = logging.getLogger(__name__)


class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"


class AssetClass(Enum):
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    INDEX = "INDEX"


@dataclass
class Instrument:
    """Represents a tradeable instrument on NSE/BSE."""
    symbol: str
    exchange: Exchange
    asset_class: AssetClass
    lot_size: int = 1
    tick_size: float = 0.05
    isin: str = ""
    sector: str = ""
    token: int = 0

    @property
    def key(self) -> str:
        return f"{self.exchange.value}:{self.symbol}"


@dataclass
class OHLCV:
    """Candlestick data with microstructure fields."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    trades: int = 0
    oi: int = 0              # Open Interest (F&O)
    bid_volume: int = 0
    ask_volume: int = 0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class Tick:
    """Level-1 tick data."""
    timestamp: datetime
    symbol: str
    last_price: float
    last_quantity: int
    bid_price: float
    bid_quantity: int
    ask_price: float
    ask_quantity: int
    volume: int
    oi: int = 0


class DataFeed(ABC):
    """Abstract base class for market data feeds."""

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def get_historical(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: str = "5m"
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    async def subscribe(self, instruments: List[Instrument], callback) -> None:
        pass


class ZerodhaFeed(DataFeed):
    """
    Zerodha Kite Connect feed for Indian markets.
    Handles WebSocket streaming and REST historical data.
    """

    INSTRUMENT_CACHE_TTL = 86400  # 24 hours

    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self._kite = None
        self._ws = None
        self._redis = redis.Redis(host="localhost", port=6379, db=0)
        self._instrument_cache: Dict[str, Instrument] = {}
        self._subscribers = {}

    async def connect(self) -> None:
        try:
            from kiteconnect import KiteConnect, KiteTicker
            self._kite = KiteConnect(api_key=self.api_key)
            self._kite.set_access_token(self.access_token)
            logger.info("Zerodha Kite connected successfully")
        except ImportError:
            raise ImportError(
                "kiteconnect not installed. "
                "Run: pip install kiteconnect"
            )
        except Exception as e:
            logger.error(f"Zerodha connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        if self._ws:
            self._ws.close()
        logger.info("Zerodha disconnected")

    def _load_instruments(self) -> None:
        """Load and cache all NSE instruments."""
        cache_key = "nse_instruments"
        cached = self._redis.get(cache_key)

        if cached:
            import pickle
            self._instrument_cache = pickle.loads(cached)
            return

        instruments = self._kite.instruments("NSE")
        for inst in instruments:
            symbol = inst.get("tradingsymbol", "")
            if not symbol:
                continue

            instrument = Instrument(
                symbol=symbol,
                exchange=Exchange.NSE,
                asset_class=AssetClass.EQUITY,
                lot_size=inst.get("lot_size", 1),
                tick_size=inst.get("tick_size", 0.05),
                isin=inst.get("isin", ""),
                sector="",  # Will be populated from sector mapping
                token=inst.get("instrument_token", 0),
            )
            self._instrument_cache[instrument.key] = instrument

        import pickle
        self._redis.setex(
            cache_key,
            self.INSTRUMENT_CACHE_TTL,
            pickle.dumps(self._instrument_cache)
        )
        logger.info(
            f"Cached {len(self._instrument_cache)} NSE instruments"
        )

    async def get_historical(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Zerodha.

        Args:
            instrument: Instrument to fetch
            start: Start datetime
            end: End datetime
            interval: Candle interval (1m, 5m, 15m, 60m, day)

        Returns:
            DataFrame with OHLCV columns
        """
        if not self._kite:
            await self.connect()

        all_data = []
        current = start

        # Zerodha limits: max 100 days for minute data
        chunk_days = 95 if "m" in interval else 365

        while current < end:
            chunk_end = min(
                current + timedelta(days=chunk_days),
                end
            )

            try:
                data = self._kite.historical_data(
                    instrument.token,
                    current,
                    chunk_end,
                    interval
                )

                if data:
                    df = pd.DataFrame(data)
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)

                    df.rename(columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }, inplace=True)

                    all_data.append(df)

            except Exception as e:
                logger.error(
                    f"Historical data fetch failed for "
                    f"{instrument.symbol}: {e}"
                )

            current = chunk_end + timedelta(seconds=1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data).sort_index()
        result = result[~result.index.duplicated(keep="first")]
        return result

    async def subscribe(
        self,
        instruments: List[Instrument],
        callback
    ) -> None:
        """Subscribe to real-time tick data via WebSocket."""
        from kiteconnect import KiteTicker

        tokens = [inst.token for inst in instruments]

        self._ws = KiteTicker(
            self.api_key,
            self.access_token
        )

        self._ws.on_ticks = callback
        self._ws.on_connect = lambda ws: ws.subscribe(tokens)
        self._ws.on_reconnect = lambda ws, attempts: logger.warning(
            f"WebSocket reconnecting (attempt {attempts})"
        )

        self._ws.thread()


class YahooFeed(DataFeed):
    """Yahoo Finance fallback feed (free tier, 15-min delayed)."""

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "60m": "60m",
        "day": "1d",
    }

    def __init__(self):
        self._redis = redis.Redis(host="localhost", port=6379, db=0)

    async def connect(self) -> None:
        logger.info("Yahoo Finance feed ready (no connection needed)")

    async def disconnect(self) -> None:
        pass

    def _symbol_to_yahoo(self, instrument: Instrument) -> str:
        """Convert NSE symbol to Yahoo Finance format."""
        return f"{instrument.symbol}.NS"

    async def get_historical(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        interval: str = "5m"
    ) -> pd.DataFrame:
        import yfinance as yf

        yahoo_symbol = self._symbol_to_yahoo(instrument)
        yahoo_interval = self.INTERVAL_MAP.get(interval, "5m")

        # Yahoo limits intraday to last 60 days
        if "m" in interval:
            max_start = datetime.now(timezone.utc) - timedelta(days=58)
            start = max(start, max_start.replace(tzinfo=None))

        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=yahoo_interval
            )

            if df.empty:
                return df

            df.columns = [c.title() for c in df.columns]

            if "Vwap" not in df.columns:
                typical = (df["High"] + df["Low"] + df["Close"]) / 3
                cum_vol = df["Volume"].cumsum()
                cum_vol_price = (typical * df["Volume"]).cumsum()
                df["Vwap"] = cum_vol_price / cum_vol.replace(0, np.nan)

            return df

        except Exception as e:
            logger.error(f"Yahoo fetch failed for {yahoo_symbol}: {e}")
            return pd.DataFrame()

    async def subscribe(
        self,
        instruments: List[Instrument],
        callback
    ) -> None:
        """Polling-based pseudo-subscription for Yahoo."""
        while True:
            for inst in instruments:
                df = await self.get_historical(
                    inst,
                    datetime.now() - timedelta(minutes=5),
                    datetime.now(),
                    "1m"
                )
                if not df.empty:
                    latest = df.iloc[-1]
                    tick = Tick(
                        timestamp=df.index[-1].to_pydatetime(),
                        symbol=inst.symbol,
                        last_price=latest["Close"],
                        last_quantity=int(latest.get("Volume", 0)),
                        bid_price=latest["Close"],
                        bid_quantity=0,
                        ask_price=latest["Close"],
                        ask_quantity=0,
                        volume=int(latest.get("Volume", 0)),
                    )
                    await callback(None, [tick])

            await asyncio.sleep(15)  # 15-sec polling


class ArcticStore:
    """
    Arctic tick-level database for fast backtesting.
    Stores minute-level and tick-level data in MongoDB.
    """

    def __init__(self, mongo_host: str = "localhost", mongo_port: int = 27017):
        self._arctic = Arctic(f"{mongo_host}:{mongo_port}")
        self._libraries = {}

    def _get_library(self, lib_name: str = "nse_5min"):
        if lib_name not in self._libraries:
            self._libraries[lib_name] = self._arctic.get_library(
                lib_name,
                create=True
            )
        return self._libraries[lib_name]

    def write(
        self,
        symbol: str,
        data: pd.DataFrame,
        lib_name: str = "nse_5min",
        metadata: dict = None
    ):
        """Write OHLCV data to Arctic."""
        lib = self._get_library(lib_name)
        lib.write(symbol, data, metadata=metadata)
        logger.info(f"Wrote {len(data)} rows for {symbol}")

    def read(
        self,
        symbol: str,
        start: datetime = None,
        end: datetime = None,
        lib_name: str = "nse_5min"
    ) -> pd.DataFrame:
        """Read OHLCV data from Arctic."""
        lib = self._get_library(lib_name)

        try:
            item = lib.read(
                symbol,
                date_range=(start, end) if start or end else None
            )
            return item.data
        except Exception as e:
            logger.error(f"Arctic read failed for {symbol}: {e}")
            return pd.DataFrame()

    def list_symbols(self, lib_name: str = "nse_5min") -> List[str]:
        lib = self._get_library(lib_name)
        return lib.list_symbols()


class DataManager:
    """
    Central data manager that orchestrates all feeds,
    caching, and database operations.
    """

    def __init__(self, config: dict):
        self.config = config
        self._feeds: Dict[str, DataFeed] = {}
        self._arctic = ArcticStore()
        self._redis = redis.Redis(
            host=config.get("data", {}).get("redis_host", "localhost"),
            port=config.get("data", {}).get("redis_port", 6379),
            decode_responses=True
        )
        self._instrument_registry: Dict[str, Instrument] = {}
        self._latest_ticks: Dict[str, Tick] = {}

        # NIFTY 50 sector mapping (subset)
        self.SECTOR_MAP = {
            "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking",
            "INFY": "IT", "ICICIBANK": "Banking", "HINDUNILVR": "FMCG",
            "ITC": "FMCG", "SBIN": "Banking", "BHARTIARTL": "Telecom",
            "LT": "Capital Goods", "KOTAKBANK": "Banking",
            "AXISBANK": "Banking", "BAJFINANCE": "Finance",
            "MARUTI": "Auto", "SUNPHARMA": "Pharma",
            "TITAN": "Consumer", "WIPRO": "IT", "ULTRACEMCO": "Cement",
            "ADANIENT": "Infrastructure", "NTPC": "Power",
            "POWERGRID": "Power", "ONGC": "Energy",
            "TATAMOTORS": "Auto", "TATASTEEL": "Metals",
            "HCLTECH": "IT", "BAJAJFINSV": "Finance",
            "ASIANPAINT": "Consumer", "COALINDIA": "Mining",
            "TECHM": "IT", "JSWSTEEL": "Metals",
            "HINDALCO": "Metals", "DRREDDY": "Pharma",
            "CIPLA": "Pharma", "BPCL": "Energy",
            "M_M": "Auto", "EICHERMOT": "Auto",
            "GRASIM": "Cement", "INDUSINDBK": "Banking",
            "HEROMOTOCO": "Auto", "TATACONSUM": "FMCG",
            "BRITANNIA": "FMCG", "DIVISLAB": "Pharma",
            "APOLLOHOSP": "Healthcare", "HDFCLIFE": "Insurance",
            "SBILIFE": "Insurance", "TATAMTRDVR": "Auto",
        }

    def get_instrument(self, symbol: str) -> Instrument:
        """Get or create instrument with sector mapping."""
        if symbol not in self._instrument_registry:
            inst = Instrument(
                symbol=symbol,
                exchange=Exchange.NSE,
                asset_class=AssetClass.EQUITY,
                sector=self.SECTOR_MAP.get(symbol, "Unknown"),
            )
            self._instrument_registry[symbol] = inst
        return self._instrument_registry[symbol]

    async def initialize_feeds(self) -> None:
        """Initialize all configured data feeds."""
        primary = self.config.get("data", {}).get("primary_feed", "yahoo")

        if primary == "zerodha":
            from os import environ
            self._feeds["zerodha"] = ZerodhaFeed(
                api_key=environ.get("ZERODHA_API_KEY", ""),
                access_token=environ.get("ZERODHA_ACCESS_TOKEN", ""),
            )
            await self._feeds["zerodha"].connect()

        self._feeds["yahoo"] = YahooFeed()
        await self._feeds["yahoo"].connect()

        logger.info(f"Initialized {len(self._feeds)} data feeds")

    async def get_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "5m",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical data with Arctic caching.

        Flow: Check Arctic → Check Redis → Fetch from API → Cache
        """
        instrument = self.get_instrument(symbol)

        # 1. Check Arctic
        if use_cache:
            arctic_data = self._arctic.read(symbol, start, end, f"nse_{interval}")
            if not arctic_data.empty:
                logger.debug(f"Arctic cache hit for {symbol}")
                return arctic_data

        # 2. Fetch from primary feed
        feed_name = self.config.get("data", {}).get("primary_feed", "yahoo")
        feed = self._feeds.get(feed_name, self._feeds.get("yahoo"))

        if not feed:
            logger.error(f"No feed available for {symbol}")
            return pd.DataFrame()

        data = await feed.get_historical(instrument, start, end, interval)

        # 3. Cache to Arctic
        if use_cache and not data.empty:
            self._arctic.write(symbol, data, f"nse_{interval}")

        return data

    async def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        return self._latest_ticks.get(symbol)

    async def on_tick(self, ws, ticks) -> None:
        """Callback for real-time WebSocket ticks."""
        for tick_data in ticks:
            symbol = tick_data.get("tradingsymbol", "")
            if not symbol:
                continue

            tick = Tick(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                last_price=tick_data.get("last_price", 0),
                last_quantity=tick_data.get("last_quantity", 0),
                bid_price=tick_data.get("bid_price", 0),
                bid_quantity=tick_data.get("bid_quantity", 0),
                ask_price=tick_data.get("ask_price", 0),
                ask_quantity=tick_data.get("ask_quantity", 0),
                volume=tick_data.get("volume", 0),
                oi=tick_data.get("oi", 0),
            )

            self._latest_ticks[symbol] = tick

            # Publish to Redis for other processes
            import json
            self._redis.publish(
                f"ticks:{symbol}",
                json.dumps({
                    "ts": tick.timestamp.isoformat(),
                    "price": tick.last_price,
                    "vol": tick.last_quantity,
                    "bid": tick.bid_price,
                    "ask": tick.ask_price,
                })
            )
