"""
Alternative Data Pipeline
India-specific alternative data sources: FII/DII, delivery, OI, PCR, VIX.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np


class DataSource(Enum):
    """Alternative data sources for Indian markets"""
    NSE_WEBSITE = "NSE_WEBSITE"
    BSE_WEBSITE = "BSE_WEBSITE"
    NSELIB = "NSELIB"
    MANUAL = "MANUAL"


@dataclass
class DataPoint:
    """Generic data point with timestamp"""
    timestamp: datetime
    value: float
    source: DataSource
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "source": self.source.value,
            "metadata": self.metadata,
        }


@dataclass
class FII_DIIData:
    """FII/DII net flow data"""
    date: date
    fii_net_flow_cr: float  # FII net flow in crores
    dii_net_flow_cr: float  # DII net flow in crores
    fii_buy_cr: float = 0.0
    fii_sell_cr: float = 0.0
    dii_buy_cr: float = 0.0
    dii_sell_cr: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "fii_net_flow_cr": self.fii_net_flow_cr,
            "dii_net_flow_cr": self.dii_net_flow_cr,
            "fii_buy_cr": self.fii_buy_cr,
            "fii_sell_cr": self.fii_sell_cr,
            "dii_buy_cr": self.dii_buy_cr,
            "dii_sell_cr": self.dii_sell_cr,
        }
    
    def calculate_flow_change_1d(self, previous: Optional['FII_DIIData']) -> float:
        """Calculate 1-day change in FII net flow"""
        if previous is None:
            return 0.0
        return self.fii_net_flow_cr - previous.fii_net_flow_cr
    
    def calculate_flow_change_5d(self, history: List['FII_DIIData']) -> float:
        """Calculate 5-day change in FII net flow"""
        if len(history) < 5:
            return 0.0
        return self.fii_net_flow_cr - history[-5].fii_net_flow_cr


@dataclass
class DeliveryData:
    """NSE delivery percentage data"""
    date: date
    symbol: str
    delivery_percentage: float
    total_traded_quantity: float = 0.0
    delivery_quantity: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "delivery_percentage": self.delivery_percentage,
            "total_traded_quantity": self.total_traded_quantity,
            "delivery_quantity": self.delivery_quantity,
        }
    
    def calculate_delivery_ratio_trend(self, history: List['DeliveryData'], days: int = 5) -> float:
        """Calculate delivery ratio trend over N days"""
        if len(history) < days:
            return 0.0
        
        recent = [d.delivery_percentage for d in history[-days:]]
        return np.polyfit(range(days), recent, 1)[0]  # Slope


@dataclass
class OIData:
    """Open Interest data for NIFTY & BANKNIFTY"""
    date: date
    symbol: str  # "NIFTY" or "BANKNIFTY"
    timestamp: datetime
    open_interest: float
    change_in_oi: float = 0.0
    oi_change_pct: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open_interest": self.open_interest,
            "change_in_oi": self.change_in_oi,
            "oi_change_pct": self.oi_change_pct,
        }
    
    def calculate_oi_trend(self, history: List['OIData'], hours: int = 24) -> float:
        """Calculate OI trend over N hours"""
        if len(history) < 2:
            return 0.0
        
        # Find data point N hours ago
        cutoff_time = self.timestamp - timedelta(hours=hours)
        old_oi = next((d.open_interest for d in history if d.timestamp <= cutoff_time), None)
        
        if old_oi is None:
            return 0.0
        
        return (self.open_interest - old_oi) / old_ii if old_oi > 0 else 0.0


@dataclass
class PCRData:
    """Put-Call Ratio data (OI and volume)"""
    date: date
    symbol: str
    timestamp: datetime
    pcr_oi: float  # Put-Call Ratio based on Open Interest
    pcr_volume: float  # Put-Call Ratio based on Volume
    put_oi: float = 0.0
    call_oi: float = 0.0
    put_volume: float = 0.0
    call_volume: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "pcr_oi": self.pcr_oi,
            "pcr_volume": self.pcr_volume,
            "put_oi": self.put_oi,
            "call_oi": self.call_oi,
            "put_volume": self.put_volume,
            "call_volume": self.call_volume,
        }


@dataclass
class VIXData:
    """India VIX data"""
    date: date
    timestamp: datetime
    vix_value: float
    vix_change: float = 0.0
    vix_change_pct: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "vix_value": self.vix_value,
            "vix_change": self.vix_change,
            "vix_change_pct": self.vix_change_pct,
        }


class AlternativeDataPipeline:
    """
    Pipeline for ingesting and storing India-specific alternative data.
    Includes scrapers for NSE/BSE websites and feature engineering.
    """
    
    def __init__(self):
        self.fii_dii_history: List[FII_DIIData] = []
        self.delivery_history: Dict[str, List[DeliveryData]] = {}  # symbol -> history
        self.oi_history: Dict[str, List[OIData]] = {}  # symbol -> history
        self.pcr_history: Dict[str, List[PCRData]] = {}  # symbol -> history
        self.vix_history: List[VIXData] = []
        
        # Data fetchers
        self.fii_dii_fetcher: Optional[Callable] = None
        self.delivery_fetcher: Optional[Callable] = None
        self.oi_fetcher: Optional[Callable] = None
        self.pcr_fetcher: Optional[Callable] = None
        self.vix_fetcher: Optional[Callable] = None
    
    def set_fii_dii_fetcher(self, fetcher: Callable) -> None:
        """Set FII/DII data fetcher function"""
        self.fii_dii_fetcher = fetcher
    
    def set_delivery_fetcher(self, fetcher: Callable) -> None:
        """Set delivery data fetcher function"""
        self.delivery_fetcher = fetcher
    
    def set_oi_fetcher(self, fetcher: Callable) -> None:
        """Set OI data fetcher function"""
        self.oi_fetcher = fetcher
    
    def set_pcr_fetcher(self, fetcher: Callable) -> None:
        """Set PCR data fetcher function"""
        self.pcr_fetcher = fetcher
    
    def set_vix_fetcher(self, fetcher: Callable) -> None:
        """Set VIX data fetcher function"""
        self.vix_fetcher = fetcher
    
    def fetch_fii_dii(self, date: date) -> Optional[FII_DIIData]:
        """
        Fetch FII/DII data for a given date.
        
        Args:
            date: Date to fetch data for
        
        Returns:
            FII_DIIData or None if fetcher not set
        """
        if self.fii_dii_fetcher is None:
            return None
        
        try:
            data = self.fii_dii_fetcher(date)
            self.fii_dii_history.append(data)
            return data
        except Exception as e:
            print(f"Error fetching FII/DII data: {e}")
            return None
    
    def fetch_delivery(self, date: date, symbol: str) -> Optional[DeliveryData]:
        """
        Fetch delivery data for a given date and symbol.
        
        Args:
            date: Date to fetch data for
            symbol: Trading symbol
        
        Returns:
            DeliveryData or None if fetcher not set
        """
        if self.delivery_fetcher is None:
            return None
        
        try:
            data = self.delivery_fetcher(date, symbol)
            if symbol not in self.delivery_history:
                self.delivery_history[symbol] = []
            self.delivery_history[symbol].append(data)
            return data
        except Exception as e:
            print(f"Error fetching delivery data: {e}")
            return None
    
    def fetch_oi(self, date: date, symbol: str) -> Optional[OIData]:
        """
        Fetch OI data for a given date and symbol.
        
        Args:
            date: Date to fetch data for
            symbol: Trading symbol (NIFTY or BANKNIFTY)
        
        Returns:
            OIData or None if fetcher not set
        """
        if self.oi_fetcher is None:
            return None
        
        try:
            data = self.oi_fetcher(date, symbol)
            if symbol not in self.oi_history:
                self.oi_history[symbol] = []
            self.oi_history[symbol].append(data)
            return data
        except Exception as e:
            print(f"Error fetching OI data: {e}")
            return None
    
    def fetch_pcr(self, date: date, symbol: str) -> Optional[PCRData]:
        """
        Fetch PCR data for a given date and symbol.
        
        Args:
            date: Date to fetch data for
            symbol: Trading symbol
        
        Returns:
            PCRData or None if fetcher not set
        """
        if self.pcr_fetcher is None:
            return None
        
        try:
            data = self.pcr_fetcher(date, symbol)
            if symbol not in self.pcr_history:
                self.pcr_history[symbol] = []
            self.pcr_history[symbol].append(data)
            return data
        except Exception as e:
            print(f"Error fetching PCR data: {e}")
            return None
    
    def fetch_vix(self, date: date) -> Optional[VIXData]:
        """
        Fetch VIX data for a given date.
        
        Args:
            date: Date to fetch data for
        
        Returns:
            VIXData or None if fetcher not set
        """
        if self.vix_fetcher is None:
            return None
        
        try:
            data = self.vix_fetcher(date)
            self.vix_history.append(data)
            return data
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return None
    
    def calculate_features(self, date: date) -> Dict[str, float]:
        """
        Calculate features from alternative data for a given date.
        
        Args:
            date: Date to calculate features for
        
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # FII/DII features
        fii_dii_data = next((d for d in self.fii_dii_history if d.date == date), None)
        if fii_dii_data:
            features["fii_net_flow_1d"] = fii_dii_data.calculate_flow_change_1d(
                next((d for d in reversed(self.fii_dii_history) if d.date < date), None)
            )
            features["fii_net_flow_5d"] = fii_dii_data.calculate_flow_change_5d(
                [d for d in self.fii_dii_history if d.date <= date]
            )
            features["dii_net_flow_1d"] = fii_dii_data.dii_net_flow_cr
        
        # Delivery features (for NIFTY 50)
        nifty_symbols = ["NIFTY 50"]  # Add more as needed
        for symbol in nifty_symbols:
            if symbol in self.delivery_history:
                delivery_data = next((d for d in self.delivery_history[symbol] if d.date == date), None)
                if delivery_data:
                    features[f"{symbol}_delivery_ratio_trend"] = delivery_data.calculate_delivery_ratio_trend(
                        self.delivery_history[symbol]
                    )
        
        # OI features
        for symbol in ["NIFTY", "BANKNIFTY"]:
            if symbol in self.oi_history:
                oi_data = next((d for d in self.oi_history[symbol] if d.date == date), None)
                if oi_data:
                    features[f"{symbol}_oi_trend"] = oi_data.calculate_oi_trend(self.oi_history[symbol])
        
        # PCR features
        for symbol in ["NIFTY", "BANKNIFTY"]:
            if symbol in self.pcr_history:
                pcr_data = next((d for d in self.pcr_history[symbol] if d.date == date), None)
                if pcr_data:
                    features[f"{symbol}_pcr_oi"] = pcr_data.pcr_oi
                    features[f"{symbol}_pcr_volume"] = pcr_data.pcr_volume
        
        # VIX features
        vix_data = next((d for d in self.vix_history if d.date == date), None)
        if vix_data:
            features["vix_value"] = vix_data.vix_value
            features["vix_change_pct"] = vix_data.vix_change_pct
        
        return features
    
    def get_latest_data(self, data_type: str, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Get latest data point for a given data type.
        
        Args:
            data_type: Type of data ("fii_dii", "delivery", "oi", "pcr", "vix")
            symbol: Symbol for delivery, oi, pcr data
        
        Returns:
            Latest data point as dict or None
        """
        if data_type == "fii_dii":
            if self.fii_dii_history:
                return self.fii_dii_history[-1].to_dict()
        elif data_type == "delivery":
            if symbol and symbol in self.delivery_history and self.delivery_history[symbol]:
                return self.delivery_history[symbol][-1].to_dict()
        elif data_type == "oi":
            if symbol and symbol in self.oi_history and self.oi_history[symbol]:
                return self.oi_history[symbol][-1].to_dict()
        elif data_type == "pcr":
            if symbol and symbol in self.pcr_history and self.pcr_history[symbol]:
                return self.pcr_history[symbol][-1].to_dict()
        elif data_type == "vix":
            if self.vix_history:
                return self.vix_history[-1].to_dict()
        
        return None
    
    def get_data_history(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        days: int = 30
    ) -> List[Dict]:
        """
        Get historical data for a given data type.
        
        Args:
            data_type: Type of data
            symbol: Symbol for delivery, oi, pcr data
            days: Number of days of history
        
        Returns:
            List of data points as dicts
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        if data_type == "fii_dii":
            return [d.to_dict() for d in self.fii_dii_history if d.date >= cutoff_date]
        elif data_type == "delivery":
            if symbol and symbol in self.delivery_history:
                return [d.to_dict() for d in self.delivery_history[symbol] if d.date >= cutoff_date]
        elif data_type == "oi":
            if symbol and symbol in self.oi_history:
                return [d.to_dict() for d in self.oi_history[symbol] if d.date >= cutoff_date]
        elif data_type == "pcr":
            if symbol and symbol in self.pcr_history:
                return [d.to_dict() for d in self.pcr_history[symbol] if d.date >= cutoff_date]
        elif data_type == "vix":
            return [d.to_dict() for d in self.vix_history if d.date >= cutoff_date]
        
        return []


def mock_fii_dii_fetcher(date: date) -> FII_DIIData:
    """Mock FII/DII fetcher for testing"""
    return FII_DIIData(
        date=date,
        fii_net_flow_cr=np.random.uniform(-1000, 1000),
        dii_net_flow_cr=np.random.uniform(-500, 500),
        fii_buy_cr=np.random.uniform(0, 2000),
        fii_sell_cr=np.random.uniform(0, 2000),
        dii_buy_cr=np.random.uniform(0, 1000),
        dii_sell_cr=np.random.uniform(0, 1000),
    )


def mock_delivery_fetcher(date: date, symbol: str) -> DeliveryData:
    """Mock delivery fetcher for testing"""
    return DeliveryData(
        date=date,
        symbol=symbol,
        delivery_percentage=np.random.uniform(30, 70),
        total_traded_quantity=np.random.uniform(1e6, 1e7),
        delivery_quantity=np.random.uniform(3e5, 7e6),
    )


def mock_vix_fetcher(date: date) -> VIXData:
    """Mock VIX fetcher for testing"""
    vix_value = np.random.uniform(10, 25)
    return VIXData(
        date=date,
        timestamp=datetime.combine(date, datetime.min.time()),
        vix_value=vix_value,
        vix_change=np.random.uniform(-2, 2),
        vix_change_pct=np.random.uniform(-10, 10),
    )
