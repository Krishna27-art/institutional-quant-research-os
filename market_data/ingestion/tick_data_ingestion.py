"""
Tick Data Ingestion for NIFTY Futures
Implements high-frequency data ingestion for institutional-grade microstructure analysis.

Based on institutional review recommendations:
- Tick data ingestion (NIFTY futures)
- Order flow imbalance (OFI) computation
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Real-time data quality monitoring
- Point-in-time data storage

Key features:
- WebSocket-based real-time data feed
- Tick-by-tick order book reconstruction
- OFI and VPIN computation
- Data quality validation
- Integration with Prometheus metrics
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Single tick data point"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_volume: int
    ask_volume: int
    trade_type: str  # "BUY" or "SELL"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "price": self.price,
            "volume": self.volume,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "trade_type": self.trade_type
        }


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # (price, volume)
    asks: List[Tuple[float, int]]  # (price, volume)
    mid_price: float
    spread: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "bids": self.bids,
            "asks": self.asks,
            "mid_price": self.mid_price,
            "spread": self.spread
        }


@dataclass
class MicrostructureFeatures:
    """Microstructure features computed from tick data"""
    timestamp: datetime
    symbol: str
    ofi: float  # Order Flow Imbalance
    vpin: float  # Volume-Synchronized Probability of Informed Trading
    spread: float
    depth_imbalance: float
    price_impact: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "ofi": self.ofi,
            "vpin": self.vpin,
            "spread": self.spread,
            "depth_imbalance": self.depth_imbalance,
            "price_impact": self.price_impact
        }


class TickDataIngestion:
    """
    Tick data ingestion for NIFTY futures.
    
    Features:
    - WebSocket-based real-time data feed
    - Order book reconstruction
    - OFI computation
    - VPIN computation
    - Data quality monitoring
    - Integration with Prometheus metrics
    """
    
    def __init__(
        self,
        symbol: str = "NIFTYFUT",
        websocket_url: Optional[str] = None,
        buffer_size: int = 10000
    ):
        self.symbol = symbol
        self.websocket_url = websocket_url
        self.buffer_size = buffer_size
        
        # Data buffers
        self.tick_buffer: deque = deque(maxlen=buffer_size)
        self.order_book: Dict[str, List[Tuple[float, int]]] = {
            "bids": [],
            "asks": []
        }
        
        # Feature computation
        self.ofi_window: deque = deque(maxlen=100)
        self.vpin_window: deque = deque(maxlen=1000)
        self.volume_bucket: float = 0.0
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0
        
        # Data quality
        self.last_tick_time: Optional[datetime] = None
        self.tick_count: int = 0
        self.gap_count: int = 0
        
        # Storage
        self.storage_path = Path("data/tick_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics (optional - if prometheus_client is available)
        self.metrics = None
        self._init_metrics()
        
        logger.info(f"Tick data ingestion initialized for {symbol}")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics if available"""
        try:
            from monitoring.prometheus_metrics import PrometheusMetrics
            self.metrics = PrometheusMetrics(port=8001)
            logger.info("Prometheus metrics initialized for tick data")
        except ImportError:
            logger.warning("Prometheus metrics not available")
    
    async def connect_websocket(self):
        """Connect to WebSocket data feed"""
        if not self.websocket_url:
            logger.warning("No WebSocket URL provided - using mock data")
            return
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                logger.info(f"Connected to WebSocket: {self.websocket_url}")
                
                # Subscribe to data
                subscribe_msg = {
                    "symbol": self.symbol,
                    "type": "subscribe",
                    "data": ["ticks", "order_book"]
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Receive data
                async for message in websocket:
                    await self.process_message(message)
                    
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            logger.info("Falling back to mock data generation")
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "tick":
                await self.process_tick(data)
            elif data.get("type") == "order_book":
                await self.process_order_book(data)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def process_tick(self, data: Dict):
        """Process tick data"""
        try:
            tick = Tick(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                symbol=data["symbol"],
                price=float(data["price"]),
                volume=int(data["volume"]),
                bid_price=float(data.get("bid_price", 0)),
                ask_price=float(data.get("ask_price", 0)),
                bid_volume=int(data.get("bid_volume", 0)),
                ask_volume=int(data.get("ask_volume", 0)),
                trade_type=data.get("trade_type", "UNKNOWN")
            )
            
            # Add to buffer
            self.tick_buffer.append(tick)
            
            # Update order book
            self._update_order_book(tick)
            
            # Compute microstructure features
            features = self._compute_microstructure_features(tick)
            
            # Record metrics
            if self.metrics:
                self.metrics.record_data_point("tick", self.symbol)
            
            # Store data
            self._store_tick(tick)
            if features:
                self._store_features(features)
            
            self.tick_count += 1
            
            # Check for gaps
            if self.last_tick_time:
                gap = (tick.timestamp - self.last_tick_time).total_seconds()
                if gap > 5:  # 5 second gap threshold
                    self.gap_count += 1
                    if self.metrics:
                        self.metrics.record_data_gap("tick", self.symbol)
            
            self.last_tick_time = tick.timestamp
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    async def process_order_book(self, data: Dict):
        """Process order book snapshot"""
        try:
            snapshot = OrderBookSnapshot(
                timestamp=datetime.fromisoformat(data["timestamp"]),
                symbol=data["symbol"],
                bids=[(float(p), int(v)) for p, v in data.get("bids", [])],
                asks=[(float(p), int(v)) for p, v in data.get("asks", [])],
                mid_price=float(data.get("mid_price", 0)),
                spread=float(data.get("spread", 0))
            )
            
            # Update order book
            self.order_book["bids"] = snapshot.bids
            self.order_book["asks"] = snapshot.asks
            
            # Store snapshot
            self._store_order_book(snapshot)
            
        except Exception as e:
            logger.error(f"Error processing order book: {e}")
    
    def _update_order_book(self, tick: Tick):
        """Update order book from tick data"""
        # Simple order book update based on tick
        if tick.trade_type == "BUY":
            # Buy trade - likely at ask price
            self.order_book["asks"] = [
                (p, v) for p, v in self.order_book["asks"]
                if p != tick.ask_price
            ]
        elif tick.trade_type == "SELL":
            # Sell trade - likely at bid price
            self.order_book["bids"] = [
                (p, v) for p, v in self.order_book["bids"]
                if p != tick.bid_price
            ]
    
    def _compute_microstructure_features(self, tick: Tick) -> Optional[MicrostructureFeatures]:
        """Compute microstructure features from tick data"""
        try:
            # Order Flow Imbalance (OFI)
            ofi = self._compute_ofi(tick)
            
            # VPIN
            vpin = self._compute_vpin(tick)
            
            # Spread
            spread = tick.ask_price - tick.bid_price if tick.ask_price > 0 and tick.bid_price > 0 else 0
            
            # Depth imbalance
            total_bid_volume = sum(v for p, v in self.order_book["bids"])
            total_ask_volume = sum(v for p, v in self.order_book["asks"])
            total_volume = total_bid_volume + total_ask_volume
            depth_imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            # Price impact (simplified)
            price_impact = abs(tick.price - ((tick.bid_price + tick.ask_price) / 2)) / tick.price if tick.price > 0 else 0
            
            features = MicrostructureFeatures(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                ofi=ofi,
                vpin=vpin,
                spread=spread,
                depth_imbalance=depth_imbalance,
                price_impact=price_impact
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing microstructure features: {e}")
            return None
    
    def _compute_ofi(self, tick: Tick) -> float:
        """
        Compute Order Flow Imbalance (OFI).
        
        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        """
        if tick.bid_volume + tick.ask_volume == 0:
            return 0.0
        
        ofi = (tick.bid_volume - tick.ask_volume) / (tick.bid_volume + tick.ask_volume)
        self.ofi_window.append(ofi)
        
        return ofi
    
    def _compute_vpin(self, tick: Tick, bucket_size: float = 1000000.0) -> float:
        """
        Compute Volume-Synchronized Probability of Informed Trading (VPIN).
        
        VPIN = |buy_volume - sell_volume| / total_volume
        Computed over volume buckets.
        """
        # Add to current bucket
        self.volume_bucket += tick.volume
        
        if tick.trade_type == "BUY":
            self.buy_volume += tick.volume
        elif tick.trade_type == "SELL":
            self.sell_volume += tick.volume
        
        # Check if bucket is full
        if self.volume_bucket >= bucket_size:
            # Compute VPIN for this bucket
            vpin = abs(self.buy_volume - self.sell_volume) / self.volume_bucket
            self.vpin_window.append(vpin)
            
            # Reset bucket
            self.volume_bucket = 0.0
            self.buy_volume = 0.0
            self.sell_volume = 0.0
            
            return vpin
        
        # Return average VPIN from window
        if self.vpin_window:
            return np.mean(self.vpin_window)
        
        return 0.0
    
    def _store_tick(self, tick: Tick):
        """Store tick data to file"""
        date_str = tick.timestamp.strftime("%Y%m%d")
        file_path = self.storage_path / f"ticks_{self.symbol}_{date_str}.jsonl"
        
        with open(file_path, "a") as f:
            f.write(json.dumps(tick.to_dict()) + "\n")
    
    def _store_features(self, features: MicrostructureFeatures):
        """Store microstructure features to file"""
        date_str = features.timestamp.strftime("%Y%m%d")
        file_path = self.storage_path / f"features_{self.symbol}_{date_str}.jsonl"
        
        with open(file_path, "a") as f:
            f.write(json.dumps(features.to_dict()) + "\n")
    
    def _store_order_book(self, snapshot: OrderBookSnapshot):
        """Store order book snapshot to file"""
        date_str = snapshot.timestamp.strftime("%Y%m%d")
        file_path = self.storage_path / f"orderbook_{self.symbol}_{date_str}.jsonl"
        
        with open(file_path, "a") as f:
            f.write(json.dumps(snapshot.to_dict()) + "\n")
    
    async def generate_mock_data(self, duration_seconds: int = 60):
        """Generate mock tick data for testing"""
        logger.info(f"Generating mock data for {duration_seconds} seconds...")
        
        start_time = datetime.now()
        price = 20000.0
        
        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            # Generate mock tick
            timestamp = datetime.now()
            
            # Random price movement
            price_change = np.random.normal(0, 0.5)
            price = max(price + price_change, 19000)
            
            # Random volume
            volume = np.random.randint(10, 100)
            
            # Random bid/ask
            bid_price = price - np.random.uniform(0.5, 2.0)
            ask_price = price + np.random.uniform(0.5, 2.0)
            
            # Random trade type
            trade_type = np.random.choice(["BUY", "SELL"])
            
            tick_data = {
                "timestamp": timestamp.isoformat(),
                "symbol": self.symbol,
                "price": price,
                "volume": volume,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_volume": np.random.randint(50, 200),
                "ask_volume": np.random.randint(50, 200),
                "trade_type": trade_type
            }
            
            await self.process_tick(json.dumps(tick_data))
            
            # Sleep to simulate real-time
            await asyncio.sleep(0.1)
        
        logger.info(f"Mock data generation complete. Total ticks: {self.tick_count}")
    
    def get_statistics(self) -> Dict:
        """Get ingestion statistics"""
        return {
            "symbol": self.symbol,
            "total_ticks": self.tick_count,
            "gap_count": self.gap_count,
            "buffer_size": len(self.tick_buffer),
            "ofi_window_size": len(self.ofi_window),
            "vpin_window_size": len(self.vpin_window),
            "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None
        }
    
    async def start(self, use_mock: bool = True, duration_seconds: int = 60):
        """Start tick data ingestion"""
        logger.info(f"Starting tick data ingestion for {self.symbol}")
        
        if use_mock:
            await self.generate_mock_data(duration_seconds)
        else:
            await self.connect_websocket()
        
        stats = self.get_statistics()
        logger.info(f"Ingestion statistics: {stats}")
        
        return stats


async def run_tick_ingestion():
    """Run tick data ingestion demo"""
    print("="*60)
    print("TICK DATA INGESTION - DEMO")
    print("="*60)
    
    # Create ingestion instance
    ingestion = TickDataIngestion(symbol="NIFTYFUT")
    
    # Run with mock data
    stats = await ingestion.start(use_mock=True, duration_seconds=30)
    
    print(f"\nIngestion Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_tick_ingestion())
