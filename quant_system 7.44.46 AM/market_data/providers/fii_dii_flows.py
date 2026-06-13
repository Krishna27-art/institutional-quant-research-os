"""
FII/DII Flow Data Integration
Based on the critique: Build data moat with institutional flow data

Critical for institutional edge:
- Foreign Institutional Investor (FII) flows
- Domestic Institutional Investor (DII) flows
- Net flow analysis
- Sector-wise flows
- Historical flow patterns
- Flow-based signals

Data Sources:
- NSE/BSE daily disclosures
- SEBI FII/DII data
- Third-party providers (NSE India, Bloomberg)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class InvestorType(Enum):
    """Types of institutional investors."""
    FII = "fii"  # Foreign Institutional Investor
    DII = "dii"  # Domestic Institutional Investor
    PROPFUND = "propfund"  # Proprietary Funds
    MUTUAL_FUNDS = "mutual_funds"
    INSURANCE = "insurance"
    BANKS = "banks"


class FlowDirection(Enum):
    """Direction of flow."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass
class FlowData:
    """Daily flow data for a symbol."""
    date: datetime
    symbol: str
    investor_type: InvestorType
    buy_value: float  # In INR crores
    sell_value: float  # In INR crores
    net_flow: float  # buy - sell
    sector: Optional[str] = None
    
    def __post_init__(self):
        """Calculate net flow."""
        self.net_flow = self.buy_value - self.sell_value


@dataclass
class FlowSignal:
    """Signal derived from flow data."""
    date: datetime
    symbol: str
    signal_type: str
    direction: FlowDirection
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    reasoning: str


class FIIDIIManager:
    """
    Manager for FII/DII flow data.
    
    Features:
    - Daily FII/DII flow tracking
    - Net flow analysis
    - Sector-wise flow aggregation
    - Historical flow patterns
    - Flow-based trading signals
    - Contrarian indicators
    """
    
    def __init__(self):
        self.flow_data: Dict[str, List[FlowData]] = {}  # symbol -> flows
        self.sector_flows: Dict[str, List[FlowData]] = {}  # sector -> flows
        self.flow_signals: List[FlowSignal] = []
        
        # Configuration
        self.lookback_days = 20  # For calculating flow averages
        self.flow_threshold = 100  # INR crores threshold for significant flow
        self.contrarian_threshold = 0.7  # For contrarian signals
    
    def add_flow_data(self, flow: FlowData) -> None:
        """Add flow data."""
        symbol = flow.symbol
        
        if symbol not in self.flow_data:
            self.flow_data[symbol] = []
        
        self.flow_data[symbol].append(flow)
        
        # Also add to sector flows if sector specified
        if flow.sector:
            if flow.sector not in self.sector_flows:
                self.sector_flows[flow.sector] = []
            self.sector_flows[flow.sector].append(flow)
    
    def get_net_flow(
        self,
        symbol: str,
        investor_type: InvestorType,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """
        Get net flow for a symbol and investor type.
        
        Args:
            symbol: Trading symbol
            investor_type: Type of investor
            start_date: Start date
            end_date: End date
            
        Returns:
            Net flow in INR crores
        """
        if symbol not in self.flow_data:
            return 0.0
        
        flows = [
            flow for flow in self.flow_data[symbol]
            if flow.investor_type == investor_type
            and start_date <= flow.date <= end_date
        ]
        
        return sum(flow.net_flow for flow in flows)
    
    def get_flow_trend(
        self,
        symbol: str,
        investor_type: InvestorType,
        days: int = 20
    ) -> Dict[str, float]:
        """
        Get flow trend metrics.
        
        Args:
            symbol: Trading symbol
            investor_type: Type of investor
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if symbol not in self.flow_data:
            return {}
        
        flows = [
            flow for flow in self.flow_data[symbol]
            if flow.investor_type == investor_type
            and start_date <= flow.date <= end_date
        ]
        
        if not flows:
            return {}
        
        net_flows = [flow.net_flow for flow in flows]
        
        return {
            'total_net_flow': sum(net_flows),
            'avg_daily_flow': np.mean(net_flows),
            'flow_std': np.std(net_flows),
            'flow_trend': np.polyfit(range(len(net_flows)), net_flows, 1)[0],  # Linear trend
            'positive_days': len([f for f in net_flows if f > 0]),
            'negative_days': len([f for f in net_flows if f < 0]),
            'consistency': len([f for f in net_flows if f > 0]) / len(net_flows) if net_flows else 0
        }
    
    def get_sector_flow(
        self,
        sector: str,
        investor_type: InvestorType,
        days: int = 20
    ) -> Dict[str, float]:
        """
        Get sector-level flow metrics.
        
        Args:
            sector: Sector name
            investor_type: Type of investor
            days: Number of days to analyze
            
        Returns:
            Dictionary with sector flow metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if sector not in self.sector_flows:
            return {}
        
        flows = [
            flow for flow in self.sector_flows[sector]
            if flow.investor_type == investor_type
            and start_date <= flow.date <= end_date
        ]
        
        if not flows:
            return {}
        
        net_flows = [flow.net_flow for flow in flows]
        
        return {
            'total_net_flow': sum(net_flows),
            'avg_daily_flow': np.mean(net_flows),
            'flow_std': np.std(net_flows),
            'max_inflow': max(net_flows),
            'max_outflow': min(net_flows)
        }
    
    def generate_flow_signal(
        self,
        symbol: str,
        investor_type: InvestorType = InvestorType.FII
    ) -> Optional[FlowSignal]:
        """
        Generate trading signal based on flow data.
        
        Signal logic:
        - Strong positive flow: Bullish
        - Strong negative flow: Bearish
        - Contrarian: If flow is extremely one-sided, take opposite position
        
        Args:
            symbol: Trading symbol
            investor_type: Type of investor to analyze
            
        Returns:
            FlowSignal or None
        """
        trend = self.get_flow_trend(symbol, investor_type, days=20)
        
        if not trend:
            return None
        
        # Determine direction
        if trend['total_net_flow'] > self.flow_threshold:
            direction = FlowDirection.BUY
            strength = min(trend['total_net_flow'] / (self.flow_threshold * 3), 1.0)
            reasoning = f"Strong {investor_type.value} buying: {trend['total_net_flow']:.0f} crores"
        elif trend['total_net_flow'] < -self.flow_threshold:
            direction = FlowDirection.SELL
            strength = min(abs(trend['total_net_flow']) / (self.flow_threshold * 3), 1.0)
            reasoning = f"Strong {investor_type.value} selling: {trend['total_net_flow']:.0f} crores"
        else:
            direction = FlowDirection.NEUTRAL
            strength = 0.0
            reasoning = "Neutral flow"
        
        # Confidence based on consistency
        confidence = trend['consistency']
        
        # Contrarian signal check
        if confidence > self.contrarian_threshold and strength > 0.8:
            # Extremely one-sided flow - contrarian signal
            direction = FlowDirection.SELL if direction == FlowDirection.BUY else FlowDirection.BUY
            reasoning = f"Contrarian: {investor_type.value} flow extremely one-sided ({confidence:.0%} consistency)"
            confidence = 0.6  # Lower confidence for contrarian
        
        signal = FlowSignal(
            date=datetime.now(),
            symbol=symbol,
            signal_type=f"{investor_type.value}_flow",
            direction=direction,
            strength=strength,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.flow_signals.append(signal)
        return signal
    
    def get_flow_dataframe(
        self,
        symbol: str,
        investor_type: InvestorType,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get flow data as DataFrame."""
        if symbol not in self.flow_data:
            return pd.DataFrame()
        
        flows = [
            flow for flow in self.flow_data[symbol]
            if flow.investor_type == investor_type
            and start_date <= flow.date <= end_date
        ]
        
        data = []
        for flow in flows:
            data.append({
                'date': flow.date,
                'symbol': flow.symbol,
                'investor_type': flow.investor_type.value,
                'buy_value': flow.buy_value,
                'sell_value': flow.sell_value,
                'net_flow': flow.net_flow,
                'sector': flow.sector
            })
        
        return pd.DataFrame(data)
    
    def get_top_flows(
        self,
        investor_type: InvestorType,
        days: int = 5,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top symbols by net flow.
        
        Args:
            investor_type: Type of investor
            days: Number of days to analyze
            top_n: Number of top symbols to return
            
        Returns:
            List of (symbol, net_flow) tuples
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        symbol_flows = {}
        
        for symbol, flows in self.flow_data.items():
            net_flow = sum(
                flow.net_flow for flow in flows
                if flow.investor_type == investor_type
                and start_date <= flow.date <= end_date
            )
            if net_flow != 0:
                symbol_flows[symbol] = net_flow
        
        # Sort by net flow
        sorted_flows = sorted(symbol_flows.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_flows[:top_n]
    
    def get_flow_summary(self) -> pd.DataFrame:
        """Get summary of recent flow signals."""
        data = []
        
        for signal in self.flow_signals[-20:]:  # Last 20 signals
            data.append({
                'date': signal.date,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'direction': signal.direction.value,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the FII/DII Manager
    print("Testing FII/DII Flow Manager...")
    
    manager = FIIDIIManager()
    
    # Generate sample flow data
    print("\nGenerating sample flow data...")
    base_date = datetime.now()
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    for i in range(30):
        for symbol in symbols:
            # FII flows
            fii_flow = FlowData(
                date=base_date - timedelta(days=i),
                symbol=symbol,
                investor_type=InvestorType.FII,
                buy_value=np.random.uniform(0, 500),
                sell_value=np.random.uniform(0, 500),
                sector='IT' if symbol in ['TCS', 'INFY'] else 'FINANCE'
            )
            manager.add_flow_data(fii_flow)
            
            # DII flows
            dii_flow = FlowData(
                date=base_date - timedelta(days=i),
                symbol=symbol,
                investor_type=InvestorType.DII,
                buy_value=np.random.uniform(0, 300),
                sell_value=np.random.uniform(0, 300),
                sector='IT' if symbol in ['TCS', 'INFY'] else 'FINANCE'
            )
            manager.add_flow_data(dii_flow)
    
    print(f"Added flow data for {len(manager.flow_data)} symbols")
    
    # Get net flow
    print("\nGetting net flow for RELIANCE (FII)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20)
    net_flow = manager.get_net_flow('RELIANCE', InvestorType.FII, start_date, end_date)
    print(f"Net FII flow: {net_flow:.0f} crores")
    
    # Get flow trend
    print("\nGetting flow trend for RELIANCE (FII)...")
    trend = manager.get_flow_trend('RELIANCE', InvestorType.FII, days=20)
    for key, value in trend.items():
        print(f"  {key}: {value}")
    
    # Get sector flow
    print("\nGetting sector flow for IT (FII)...")
    sector_flow = manager.get_sector_flow('IT', InvestorType.FII, days=20)
    for key, value in sector_flow.items():
        print(f"  {key}: {value}")
    
    # Generate flow signal
    print("\nGenerating flow signal for RELIANCE...")
    signal = manager.generate_flow_signal('RELIANCE', InvestorType.FII)
    if signal:
        print(f"Direction: {signal.direction.value}")
        print(f"Strength: {signal.strength:.2f}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Reasoning: {signal.reasoning}")
    
    # Get top flows
    print("\nGetting top FII flows (last 5 days)...")
    top_flows = manager.get_top_flows(InvestorType.FII, days=5, top_n=5)
    for symbol, flow in top_flows:
        print(f"  {symbol}: {flow:.0f} crores")
    
    # Get flow DataFrame
    print("\nGetting flow DataFrame...")
    flow_df = manager.get_flow_dataframe('RELIANCE', InvestorType.FII, start_date, end_date)
    print(f"DataFrame shape: {flow_df.shape}")
    print(flow_df.head())
    
    # Get flow summary
    print("\nFlow signal summary:")
    summary = manager.get_flow_summary()
    print(summary.to_string(index=False))
