"""
Alternative Indian Data Pipeline (FII/DII, Delivery, OI, PCR)
Based on V3 Blueprint - India-Specific Alternative Data

Key findings from research:
- FII/DII net flows are high-value Indian market edges
- NSE delivery percentage indicates institutional vs retail activity
- Open interest changes for NIFTY & BANKNIFTY
- Put-Call ratio (OI and volume) as sentiment indicator
- India VIX term structure

V3 Upgrade - Expected Sharpe increase: +0.2–0.3
Priority: High
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
import json


@dataclass
class AlternativeDataPoint:
    """Alternative data point for a given date"""
    date: str
    fii_net_flow: float  # FII net flow (₹ crore)
    dii_net_flow: float  # DII net flow (₹ crore)
    delivery_percentage: float  # NSE delivery percentage
    oi_change_nifty: float  # OI change for NIFTY
    oi_change_banknifty: float  # OI change for BANKNIFTY
    pcr_oi: float  # Put-Call ratio (OI)
    pcr_volume: float  # Put-Call ratio (volume)
    india_vix: float  # India VIX
    vix_term_slope: float  # VIX term structure slope


@dataclass
class AlternativeFeatures:
    """Computed features from alternative data"""
    date: str
    fii_flow_change_1d: float
    fii_flow_change_5d: float
    dii_flow_change_1d: float
    delivery_ratio_trend: float
    oi_trend_nifty: float
    oi_trend_banknifty: float
    pcr_signal: str  # "bullish", "bearish", "neutral"
    vix_signal: str  # "low", "normal", "high", "extreme"


class AlternativeIndianDataPipeline:
    """
    Alternative Indian Data Pipeline.
    
    Data Sources:
    - FII/DII net flows: NSE website (daily)
    - NSE delivery percentage: NSE website (daily)
    - Open interest changes: NSE website (intraday)
    - Put-Call ratio: NSE website (hourly)
    - India VIX: NSE website (1-min)
    """
    
    def __init__(self):
        self.data_history: List[AlternativeDataPoint] = []
        self.features_history: List[AlternativeFeatures] = []
        
        # NSE API endpoints (placeholder URLs)
        self.nse_urls = {
            "fii_dii": "https://www.nseindia.com/products/content/equities/equities/fii_dii.htm",
            "delivery": "https://www.nseindia.com/products/content/equities/equities/delivery.htm",
            "oi": "https://www.nseindia.com/products/content/derivatives/equities/oi.htm",
            "pcr": "https://www.nseindia.com/products/content/derivatives/equities/pcr.htm",
            "vix": "https://www.nseindia.com/products/content/derivatives/equities/vix.htm"
        }
    
    def fetch_fii_dii_data(self, date: str) -> Tuple[float, float]:
        """
        Fetch FII/DII net flows from NSE.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            (fii_net_flow, dii_net_flow) in ₹ crore
        """
        # Placeholder implementation
        # In production, use NSE API or web scraping
        np.random.seed(abs(hash(date)) % (2**32))
        fii_flow = np.random.normal(100, 500)  # Mean ₹100 crore, std ₹500 crore
        dii_flow = np.random.normal(50, 300)  # Mean ₹50 crore, std ₹300 crore
        
        return fii_flow, dii_flow
    
    def fetch_delivery_percentage(self, date: str) -> float:
        """
        Fetch NSE delivery percentage.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Delivery percentage (0-100)
        """
        # Placeholder implementation
        np.random.seed((abs(hash(date)) + 1) % (2**32))
        delivery = np.random.normal(35, 15)  # Mean 35%, std 15%
        return np.clip(delivery, 0, 100)
    
    def fetch_oi_changes(self, date: str) -> Tuple[float, float]:
        """
        Fetch OI changes for NIFTY and BANKNIFTY.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            (oi_change_nifty, oi_change_banknifty)
        """
        # Placeholder implementation
        np.random.seed((abs(hash(date)) + 2) % (2**32))
        oi_nifty = np.random.normal(0.02, 0.05)  # Mean 2%, std 5%
        oi_banknifty = np.random.normal(0.015, 0.04)
        
        return oi_nifty, oi_banknifty
    
    def fetch_pcr(self, date: str) -> Tuple[float, float]:
        """
        Fetch Put-Call ratio (OI and volume).
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            (pcr_oi, pcr_volume)
        """
        # Placeholder implementation
        np.random.seed((abs(hash(date)) + 3) % (2**32))
        pcr_oi = np.random.normal(1.0, 0.3)  # Mean 1.0, std 0.3
        pcr_volume = np.random.normal(0.9, 0.25)
        
        return pcr_oi, pcr_volume
    
    def fetch_india_vix(self, date: str) -> float:
        """
        Fetch India VIX.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            India VIX level
        """
        # Placeholder implementation
        np.random.seed((abs(hash(date)) + 4) % (2**32))
        vix = np.random.normal(15, 5)  # Mean 15, std 5
        return max(10, vix)  # Minimum 10
    
    def fetch_vix_term_structure(self, date: str) -> float:
        """
        Fetch VIX term structure slope.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Term structure slope (near VIX - far VIX)
        """
        # Placeholder implementation
        np.random.seed((abs(hash(date)) + 5) % (2**32))
        slope = np.random.normal(-0.5, 1.0)  # Mean -0.5, std 1.0
        return slope
    
    def collect_daily_data(self, date: str) -> AlternativeDataPoint:
        """
        Collect all alternative data for a given date.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            AlternativeDataPoint
        """
        fii_flow, dii_flow = self.fetch_fii_dii_data(date)
        delivery = self.fetch_delivery_percentage(date)
        oi_nifty, oi_banknifty = self.fetch_oi_changes(date)
        pcr_oi, pcr_volume = self.fetch_pcr(date)
        vix = self.fetch_india_vix(date)
        vix_slope = self.fetch_vix_term_structure(date)
        
        data_point = AlternativeDataPoint(
            date=date,
            fii_net_flow=fii_flow,
            dii_net_flow=dii_flow,
            delivery_percentage=delivery,
            oi_change_nifty=oi_nifty,
            oi_change_banknifty=oi_banknifty,
            pcr_oi=pcr_oi,
            pcr_volume=pcr_volume,
            india_vix=vix,
            vix_term_slope=vix_slope
        )
        
        self.data_history.append(data_point)
        
        return data_point
    
    def compute_features(self, date: str) -> AlternativeFeatures:
        """
        Compute features from alternative data.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            AlternativeFeatures
        """
        # Get recent data
        recent_data = [d for d in self.data_history if d.date <= date]
        recent_data.sort(key=lambda x: x.date)
        
        if len(recent_data) < 5:
            return AlternativeFeatures(
                date=date,
                fii_flow_change_1d=0.0,
                fii_flow_change_5d=0.0,
                dii_flow_change_1d=0.0,
                delivery_ratio_trend=0.0,
                oi_trend_nifty=0.0,
                oi_trend_banknifty=0.0,
                pcr_signal="neutral",
                vix_signal="normal"
            )
        
        latest = recent_data[-1]
        
        # 1-day changes
        if len(recent_data) >= 2:
            fii_change_1d = latest.fii_net_flow - recent_data[-2].fii_net_flow
            dii_change_1d = latest.dii_net_flow - recent_data[-2].dii_net_flow
        else:
            fii_change_1d = 0.0
            dii_change_1d = 0.0
        
        # 5-day changes
        if len(recent_data) >= 5:
            fii_change_5d = latest.fii_net_flow - recent_data[-5].fii_net_flow
        else:
            fii_change_5d = 0.0
        
        # Delivery ratio trend (5-day moving average)
        if len(recent_data) >= 5:
            delivery_ma5 = np.mean([d.delivery_percentage for d in recent_data[-5:]])
            delivery_trend = latest.delivery_percentage - delivery_ma5
        else:
            delivery_trend = 0.0
        
        # OI trends
        if len(recent_data) >= 5:
            oi_nifty_trend = np.mean([d.oi_change_nifty for d in recent_data[-5:]])
            oi_banknifty_trend = np.mean([d.oi_change_banknifty for d in recent_data[-5:]])
        else:
            oi_nifty_trend = 0.0
            oi_banknifty_trend = 0.0
        
        # PCR signal
        if latest.pcr_oi < 0.8:
            pcr_signal = "bullish"  # Low PCR = bullish
        elif latest.pcr_oi > 1.2:
            pcr_signal = "bearish"  # High PCR = bearish
        else:
            pcr_signal = "neutral"
        
        # VIX signal
        if latest.india_vix < 12:
            vix_signal = "low"
        elif latest.india_vix < 18:
            vix_signal = "normal"
        elif latest.india_vix < 25:
            vix_signal = "high"
        else:
            vix_signal = "extreme"
        
        features = AlternativeFeatures(
            date=date,
            fii_flow_change_1d=fii_change_1d,
            fii_flow_change_5d=fii_change_5d,
            dii_flow_change_1d=dii_change_1d,
            delivery_ratio_trend=delivery_trend,
            oi_trend_nifty=oi_nifty_trend,
            oi_trend_banknifty=oi_banknifty_trend,
            pcr_signal=pcr_signal,
            vix_signal=vix_signal
        )
        
        self.features_history.append(features)
        
        return features
    
    def get_ml_features(self, window: int = 20) -> pd.DataFrame:
        """
        Get ML-ready features from alternative data.
        
        Args:
            window: Rolling window
            
        Returns:
            DataFrame with features
        """
        if len(self.features_history) < window:
            return pd.DataFrame()
        
        features_list = []
        for i in range(window, len(self.features_history)):
            recent_features = self.features_history[i-window:i]
            
            feature_dict = {
                "date": self.features_history[i].date,
                "fii_change_1d_mean": np.mean([f.fii_flow_change_1d for f in recent_features]),
                "fii_change_1d_std": np.std([f.fii_flow_change_1d for f in recent_features]),
                "fii_change_5d_mean": np.mean([f.fii_flow_change_5d for f in recent_features]),
                "dii_change_1d_mean": np.mean([f.dii_flow_change_1d for f in recent_features]),
                "delivery_trend_mean": np.mean([f.delivery_ratio_trend for f in recent_features]),
                "oi_nifty_trend_mean": np.mean([f.oi_trend_nifty for f in recent_features]),
                "oi_banknifty_trend_mean": np.mean([f.oi_trend_banknifty for f in recent_features]),
                "pcr_bullish_count": sum(1 for f in recent_features if f.pcr_signal == "bullish"),
                "vix_high_count": sum(1 for f in recent_features if f.vix_signal in ["high", "extreme"]),
            }
            
            features_list.append(feature_dict)
        
        return pd.DataFrame(features_list)
    
    def print_data_summary(self, date: str) -> None:
        """Print summary of alternative data."""
        data_point = self.collect_daily_data(date)
        features = self.compute_features(date)
        
        print("\n" + "="*60)
        print(f"ALTERNATIVE INDIAN DATA: {date}")
        print("="*60)
        print(f"FII Net Flow: ₹{data_point.fii_net_flow:.2f} crore")
        print(f"DII Net Flow: ₹{data_point.dii_net_flow:.2f} crore")
        print(f"Delivery Percentage: {data_point.delivery_percentage:.2f}%")
        print(f"OI Change NIFTY: {data_point.oi_change_nifty:.2%}")
        print(f"OI Change BANKNIFTY: {data_point.oi_change_banknifty:.2%}")
        print(f"PCR (OI): {data_point.pcr_oi:.2f}")
        print(f"PCR (Volume): {data_point.pcr_volume:.2f}")
        print(f"India VIX: {data_point.india_vix:.2f}")
        print(f"VIX Term Slope: {data_point.vix_term_slope:.2f}")
        
        print("\nComputed Features:")
        print(f"FII Flow Change 1d: ₹{features.fii_flow_change_1d:.2f} crore")
        print(f"FII Flow Change 5d: ₹{features.fii_flow_change_5d:.2f} crore")
        print(f"Delivery Ratio Trend: {features.delivery_ratio_trend:.2f}%")
        print(f"PCR Signal: {features.pcr_signal.upper()}")
        print(f"VIX Signal: {features.vix_signal.upper()}")
        print("="*60)


def run_sample_pipeline():
    """Run sample alternative data pipeline."""
    pipeline = AlternativeIndianDataPipeline()
    
    # Collect data for last 30 days
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    
    for date in dates:
        pipeline.collect_daily_data(date.strftime("%Y-%m-%d"))
    
    # Compute features for latest date
    latest_date = dates[-1].strftime("%Y-%m-%d")
    pipeline.print_data_summary(latest_date)
    
    # Get ML features
    ml_features = pipeline.get_ml_features(window=20)
    
    if not ml_features.empty:
        print("\nML Features (last 5 days):")
        print(ml_features.tail())
    
    return pipeline


if __name__ == "__main__":
    run_sample_pipeline()
