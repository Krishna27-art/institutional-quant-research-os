"""
FII/DII Flows Data Pipeline for Indian Market

This module implements a comprehensive data pipeline for Foreign Institutional
Investor (FII) and Domestic Institutional Investor (DII) flows in the Indian market.

Key Features:
- Daily FII/DII buy/sell flows
- Net flow computation and cumulative tracking
- Flow momentum signals
- Anomaly detection in flow patterns
- Alpha generation from institutional flows
- Integration with NSE data sources

Based on V4 Blueprint - Institutional Architecture
Priority: High (Phase 1)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from collections import deque
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of institutional flows."""
    FII = "FII"
    DII = "DII"


@dataclass
class FlowData:
    """Daily flow data point."""
    date: datetime
    flow_type: FlowType
    buy_cr: float  # Buy in crores
    sell_cr: float  # Sell in crores
    net_cr: float  # Net flow in crores
    derivatives_buy_cr: float = 0.0
    derivatives_sell_cr: float = 0.0
    derivatives_net_cr: float = 0.0


@dataclass
class FlowSignal:
    """Alpha signal generated from flows."""
    signal_id: str
    symbol: str
    timestamp: datetime
    signal_type: str  # "momentum", "divergence", "anomaly"
    signal_value: float  # Normalized -1 to 1
    confidence: float
    metadata: Dict = field(default_factory=dict)


class FIIDIIIngester:
    """Ingest FII/DII flow data from NSE."""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.flow_data: List[FlowData] = []
        self.cache_dir = "data/cache/fii_dii"
    
    def fetch_daily_flows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[FlowData]:
        """
        Fetch daily FII/DII flows from NSE.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of FlowData
        """
        # In production, this would scrape NSE website or use API
        # For now, generate synthetic data for demonstration
        return self._generate_synthetic_flows(start_date, end_date)
    
    def _generate_synthetic_flows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[FlowData]:
        """Generate synthetic FII/DII flow data for testing."""
        flows = []
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Skip weekends
        dates = [d for d in dates if d.weekday() < 5]
        
        # Generate realistic flow patterns
        fii_base = 500  # Base FII flow in crores
        dii_base = 300  # Base DII flow in crores
        
        for date in dates:
            # Add some randomness
            fii_buy = fii_base + np.random.normal(0, 100)
            fii_sell = fii_base * 0.8 + np.random.normal(0, 80)
            dii_buy = dii_base + np.random.normal(0, 50)
            dii_sell = dii_base * 0.9 + np.random.normal(0, 40)
            
            # Ensure non-negative
            fii_buy = max(0, fii_buy)
            fii_sell = max(0, fii_sell)
            dii_buy = max(0, dii_buy)
            dii_sell = max(0, dii_sell)
            
            # FII flow
            flows.append(FlowData(
                date=date,
                flow_type=FlowType.FII,
                buy_cr=fii_buy,
                sell_cr=fii_sell,
                net_cr=fii_buy - fii_sell
            ))
            
            # DII flow
            flows.append(FlowData(
                date=date,
                flow_type=FlowType.DII,
                buy_cr=dii_buy,
                sell_cr=dii_sell,
                net_cr=dii_buy - dii_sell
            ))
        
        return flows
    
    def store_flows(self, flows: List[FlowData]) -> None:
        """Store flow data in database."""
        self.flow_data.extend(flows)
        logger.info(f"Stored {len(flows)} flow data points")


class FlowProcessor:
    """Process and analyze FII/DII flows."""
    
    def __init__(self):
        self.flows: List[FlowData] = []
        self.cumulative_flows: Dict[FlowType, pd.Series] = {}
        self.momentum_signals: List[FlowSignal] = []
    
    def load_flows(self, flows: List[FlowData]) -> None:
        """Load flow data for processing."""
        self.flows = flows
        logger.info(f"Loaded {len(flows)} flow data points")
    
    def compute_net_flow(self, window_days: int = 5) -> pd.DataFrame:
        """
        Compute net flows over a rolling window.
        
        Args:
            window_days: Rolling window in days
            
        Returns:
            DataFrame with net flows
        """
        # Convert to DataFrame
        fii_flows = [f for f in self.flows if f.flow_type == FlowType.FII]
        dii_flows = [f for f in self.flows if f.flow_type == FlowType.DII]
        
        fii_df = pd.DataFrame([{
            'date': f.date,
            'net_cr': f.net_cr
        } for f in fii_flows])
        
        dii_df = pd.DataFrame([{
            'date': f.date,
            'net_cr': f.net_cr
        } for f in dii_flows])
        
        fii_df = fii_df.set_index('date').sort_index()
        dii_df = dii_df.set_index('date').sort_index()
        
        # Compute rolling net flows
        fii_df['net_rolling'] = fii_df['net_cr'].rolling(window=window_days).sum()
        dii_df['net_rolling'] = dii_df['net_cr'].rolling(window=window_days).sum()
        
        # Combine
        combined = pd.DataFrame({
            'fii_net': fii_df['net_cr'],
            'fii_net_rolling': fii_df['net_rolling'],
            'dii_net': dii_df['net_cr'],
            'dii_net_rolling': dii_df['net_rolling'],
            'combined_net': fii_df['net_cr'] + dii_df['net_cr'],
            'combined_net_rolling': fii_df['net_rolling'] + dii_df['net_rolling']
        })
        
        return combined
    
    def compute_cumulative_flows(self) -> Dict[FlowType, pd.Series]:
        """
        Compute cumulative flows from inception.
        
        Returns:
            Dict mapping flow type to cumulative series
        """
        fii_flows = [f for f in self.flows if f.flow_type == FlowType.FII]
        dii_flows = [f for f in self.flows if f.flow_type == FlowType.DII]
        
        fii_df = pd.DataFrame([{
            'date': f.date,
            'net_cr': f.net_cr
        } for f in fii_flows]).set_index('date').sort_index()
        
        dii_df = pd.DataFrame([{
            'date': f.date,
            'net_cr': f.net_cr
        } for f in dii_flows]).set_index('date').sort_index()
        
        self.cumulative_flows[FlowType.FII] = fii_df['net_cr'].cumsum()
        self.cumulative_flows[FlowType.DII] = dii_df['net_cr'].cumsum()
        
        return self.cumulative_flows
    
    def detect_anomalies(self, threshold: float = 2.0) -> List[FlowData]:
        """
        Detect anomalous flow patterns using z-score.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of anomalous flow data points
        """
        fii_flows = [f for f in self.flows if f.flow_type == FlowType.FII]
        dii_flows = [f for f in self.flows if f.flow_type == FlowType.DII]
        
        anomalies = []
        
        # FII anomalies
        fii_net = [f.net_cr for f in fii_flows]
        if fii_net:
            mean = np.mean(fii_net)
            std = np.std(fii_net)
            
            for flow in fii_flows:
                z_score = (flow.net_cr - mean) / std if std > 0 else 0
                if abs(z_score) > threshold:
                    anomalies.append(flow)
        
        # DII anomalies
        dii_net = [f.net_cr for f in dii_flows]
        if dii_net:
            mean = np.mean(dii_net)
            std = np.std(dii_net)
            
            for flow in dii_flows:
                z_score = (flow.net_cr - mean) / std if std > 0 else 0
                if abs(z_score) > threshold:
                    anomalies.append(flow)
        
        logger.info(f"Detected {len(anomalies)} anomalous flow patterns")
        return anomalies
    
    def compute_flow_momentum(self, window_days: int = 20) -> pd.DataFrame:
        """
        Compute flow momentum signals.
        
        Args:
            window_days: Lookback window for momentum
            
        Returns:
            DataFrame with momentum signals
        """
        net_flows = self.compute_net_flow(window_days)
        
        # Compute momentum as z-score of rolling net flow
        combined_rolling = net_flows['combined_net_rolling'].dropna()
        
        if len(combined_rolling) > 0:
            mean = combined_rolling.mean()
            std = combined_rolling.std()
            
            net_flows['momentum'] = (combined_rolling - mean) / std if std > 0 else 0
            net_flows['momentum'] = net_flows['momentum'].fillna(0)
        
        return net_flows


class FlowAlphaGenerator:
    """Generate alpha signals from FII/DII flows."""
    
    def __init__(self):
        self.signals: List[FlowSignal] = []
    
    def generate_momentum_signal(
        self,
        net_flows: pd.DataFrame,
        symbol: str = "NIFTY"
    ) -> List[FlowSignal]:
        """
        Generate momentum signals from flow data.
        
        Args:
            net_flows: DataFrame with net flows
            symbol: Symbol (default NIFTY)
            
        Returns:
            List of FlowSignal
        """
        signals = []
        
        for date, row in net_flows.iterrows():
            if pd.isna(row['momentum']):
                continue
            
            # Normalize signal to -1 to 1
            signal_value = np.tanh(row['momentum'] / 2)
            
            signal = FlowSignal(
                signal_id=f"flow_momentum_{date.strftime('%Y%m%d')}",
                symbol=symbol,
                timestamp=date,
                signal_type="momentum",
                signal_value=signal_value,
                confidence=0.6,
                metadata={
                    'fii_net': row['fii_net'],
                    'dii_net': row['dii_net'],
                    'combined_net': row['combined_net']
                }
            )
            
            signals.append(signal)
        
        self.signals.extend(signals)
        return signals
    
    def generate_divergence_signal(
        self,
        net_flows: pd.DataFrame,
        symbol: str = "NIFTY"
    ) -> List[FlowSignal]:
        """
        Generate divergence signals when FII and DII flows diverge.
        
        Args:
            net_flows: DataFrame with net flows
            symbol: Symbol (default NIFTY)
            
        Returns:
            List of FlowSignal
        """
        signals = []
        
        for date, row in net_flows.iterrows():
            # Divergence: FII and DII moving in opposite directions
            fii_sign = np.sign(row['fii_net'])
            dii_sign = np.sign(row['dii_net'])
            
            if fii_sign != dii_sign and abs(row['fii_net']) > 100 and abs(row['dii_net']) > 100:
                # Strong divergence
                signal_value = fii_sign  # Follow FII direction
                
                signal = FlowSignal(
                    signal_id=f"flow_divergence_{date.strftime('%Y%m%d')}",
                    symbol=symbol,
                    timestamp=date,
                    signal_type="divergence",
                    signal_value=float(signal_value),
                    confidence=0.5,
                    metadata={
                        'fii_net': row['fii_net'],
                        'dii_net': row['dii_net'],
                        'divergence_strength': abs(row['fii_net']) + abs(row['dii_net'])
                    }
                )
                
                signals.append(signal)
        
        self.signals.extend(signals)
        return signals
    
    def get_all_signals(self) -> List[FlowSignal]:
        """Get all generated signals."""
        return self.signals


class FlowStorage:
    """Store flow data and signals in database."""
    
    def __init__(self, clickhouse_host: str = "localhost", clickhouse_port: int = 8123):
        self.clickhouse_host = clickhouse_host
        self.clickhouse_port = clickhouse_port
    
    def store_flows(self, flows: List[FlowData]) -> None:
        """Store flow data in ClickHouse."""
        try:
            import clickhouse_connect
            
            client = clickhouse_connect.get_client(
                host=self.clickhouse_host,
                port=self.clickhouse_port,
                database='quant_research'
            )
            
            # Prepare data for insertion
            data = []
            for flow in flows:
                data.append({
                    'date': flow.date,
                    'fii_buy_cr': flow.buy_cr if flow.flow_type == FlowType.FII else 0,
                    'fii_sell_cr': flow.sell_cr if flow.flow_type == FlowType.FII else 0,
                    'fii_net_cr': flow.net_cr if flow.flow_type == FlowType.FII else 0,
                    'dii_buy_cr': flow.buy_cr if flow.flow_type == FlowType.DII else 0,
                    'dii_sell_cr': flow.sell_cr if flow.flow_type == FlowType.DII else 0,
                    'dii_net_cr': flow.net_cr if flow.flow_type == FlowType.DII else 0
                })
            
            df = pd.DataFrame(data)
            client.insert_df('fii_dii_flows', df)
            
            logger.info(f"Stored {len(flows)} flow records in ClickHouse")
            
        except ImportError:
            logger.warning("ClickHouse not available, skipping storage")
        except Exception as e:
            logger.error(f"Failed to store flows: {e}")


def sample_fii_dii_pipeline():
    """Demonstrate FII/DII flows pipeline."""
    print("=== FII/DII Flows Data Pipeline Demo ===\n")
    
    # Initialize components
    ingester = FIIDIIIngester()
    processor = FlowProcessor()
    alpha_generator = FlowAlphaGenerator()
    storage = FlowStorage()
    
    # Fetch data (synthetic for demo)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print(f"Fetching FII/DII flows from {start_date.date()} to {end_date.date()}...")
    flows = ingester.fetch_daily_flows(start_date, end_date)
    print(f"Fetched {len(flows)} flow data points\n")
    
    # Store flows
    storage.store_flows(flows)
    
    # Process flows
    processor.load_flows(flows)
    
    # Compute net flows
    print("Computing net flows (5-day window)...")
    net_flows = processor.compute_net_flow(window_days=5)
    print(f"Net flows computed for {len(net_flows)} days\n")
    
    # Compute cumulative flows
    print("Computing cumulative flows...")
    cumulative = processor.compute_cumulative_flows()
    print(f"FII cumulative: {cumulative[FlowType.FII].iloc[-1]:.2f} Cr")
    print(f"DII cumulative: {cumulative[FlowType.DII].iloc[-1]:.2f} Cr\n")
    
    # Detect anomalies
    print("Detecting flow anomalies...")
    anomalies = processor.detect_anomalies(threshold=2.0)
    print(f"Found {len(anomalies)} anomalies\n")
    
    # Compute momentum
    print("Computing flow momentum (20-day window)...")
    momentum = processor.compute_flow_momentum(window_days=20)
    print(f"Momentum computed\n")
    
    # Generate alpha signals
    print("Generating alpha signals...")
    momentum_signals = alpha_generator.generate_momentum_signal(momentum)
    divergence_signals = alpha_generator.generate_divergence_signal(momentum)
    
    print(f"Generated {len(momentum_signals)} momentum signals")
    print(f"Generated {len(divergence_signals)} divergence signals")
    print(f"Total signals: {len(alpha_generator.get_all_signals())}\n")
    
    # Display sample signals
    print("Sample Momentum Signals:")
    for signal in momentum_signals[:5]:
        print(f"  {signal.date}: {signal.signal_value:.4f}")
    
    print("\nSample Divergence Signals:")
    for signal in divergence_signals[:5]:
        print(f"  {signal.date}: {signal.signal_value:.4f}")
    
    print("\n=== Pipeline Demo Complete ===")
    print("Key capabilities:")
    print("- Daily FII/DII flow ingestion")
    print("- Net flow computation and cumulative tracking")
    print("- Flow momentum signals")
    print("- Anomaly detection in flow patterns")
    print("- Alpha generation from institutional flows")
    print("- Integration with ClickHouse for storage")


if __name__ == "__main__":
    sample_fii_dii_pipeline()
