"""
Agency Theory Signals: Insider Transactions, Promoter Activity, FII/DII Tracker
Based on the critique: Build Agency Theory signals for institutional money behavior

Institutional money behaves differently. Signals:
- Promoter selling
- Buybacks
- Pledging
- Insider buying
- Mutual fund accumulation
- FII accumulation

These often predict moves before price.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class TransactionType(Enum):
    """Types of transactions."""
    BUY = "buy"
    SELL = "sell"
    PLEDGE = "pledge"
    RELEASE_PLEDGE = "release_pledge"


@dataclass
class InsiderTransaction:
    """Insider transaction data."""
    symbol: str
    timestamp: datetime
    insider_name: str
    transaction_type: TransactionType
    shares: int
    price: float
    value: float


@dataclass
class PromoterActivity:
    """Promoter activity data."""
    symbol: str
    timestamp: datetime
    activity_type: str  # "buy", "sell", "pledge", "buyback"
    shares: int
    percentage_change: float
    signal: float  # -1 to 1


@dataclass
class FIIActivity:
    """FII/DII activity data."""
    symbol: str
    timestamp: datetime
    fii_buy_value: float
    fii_sell_value: float
    dii_buy_value: float
    dii_sell_value: float
    net_fii_flow: float
    net_dii_flow: float
    signal: float  # -1 to 1


@dataclass
class MutualFundActivity:
    """Mutual fund activity data."""
    symbol: str
    timestamp: datetime
    fund_name: str
    shares_change: int
    percentage_change: float
    signal: float  # -1 to 1


class AgencyTheoryEngine:
    """
    Agency Theory Engine for institutional money behavior.
    
    Features:
    - Insider transaction tracking
    - Promoter activity monitoring
    - FII/DII flow tracking
    - Mutual fund accumulation
    - Signal generation based on institutional behavior
    """
    
    def __init__(self):
        self.insider_transactions: Dict[str, List[InsiderTransaction]] = {}
        self.promoter_activities: Dict[str, List[PromoterActivity]] = {}
        self.fii_activities: Dict[str, List[FIIActivity]] = {}
        self.mutual_fund_activities: Dict[str, List[MutualFundActivity]] = {}
        
        # Signal thresholds
        self.insider_buy_threshold = 0.01  # 1% of market cap
        self.promoter_sell_threshold = 0.05  # 5% stake change
        self.fii_flow_threshold = 10000000  # 10 crore INR
        self.mf_accumulation_threshold = 0.02  # 2% stake change
    
    def add_insider_transaction(
        self,
        symbol: str,
        timestamp: datetime,
        insider_name: str,
        transaction_type: TransactionType,
        shares: int,
        price: float
    ) -> InsiderTransaction:
        """
        Add insider transaction.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            insider_name: Name of insider
            transaction_type: Type of transaction
            shares: Number of shares
            price: Price per share
            
        Returns:
            InsiderTransaction
        """
        value = shares * price
        
        transaction = InsiderTransaction(
            symbol=symbol,
            timestamp=timestamp,
            insider_name=insider_name,
            transaction_type=transaction_type,
            shares=shares,
            price=price,
            value=value
        )
        
        # Store in history
        if symbol not in self.insider_transactions:
            self.insider_transactions[symbol] = []
        self.insider_transactions[symbol].append(transaction)
        
        return transaction
    
    def analyze_insider_activity(
        self,
        symbol: str,
        window_days: int = 30
    ) -> Dict:
        """
        Analyze insider activity for a symbol.
        
        Args:
            symbol: Trading symbol
            window_days: Lookback window
            
        Returns:
            Dictionary with analysis results
        """
        if symbol not in self.insider_transactions:
            return {'signal': 0.0, 'buy_volume': 0, 'sell_volume': 0}
        
        # Filter transactions in window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_transactions = [
            t for t in self.insider_transactions[symbol]
            if t.timestamp >= cutoff_date
        ]
        
        if not recent_transactions:
            return {'signal': 0.0, 'buy_volume': 0, 'sell_volume': 0}
        
        # Calculate buy and sell volumes
        buy_volume = sum(t.value for t in recent_transactions if t.transaction_type == TransactionType.BUY)
        sell_volume = sum(t.value for t in recent_transactions if t.transaction_type == TransactionType.SELL)
        
        # Calculate net flow
        net_flow = buy_volume - sell_volume
        
        # Generate signal
        # Strong insider buying = positive signal
        # Strong insider selling = negative signal
        signal = np.tanh(net_flow / self.insider_buy_threshold) if self.insider_buy_threshold > 0 else 0
        
        return {
            'signal': signal,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_flow': net_flow,
            'transaction_count': len(recent_transactions)
        }
    
    def add_promoter_activity(
        self,
        symbol: str,
        timestamp: datetime,
        activity_type: str,
        shares: int,
        total_shares: int
    ) -> PromoterActivity:
        """
        Add promoter activity.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            activity_type: Type of activity
            shares: Number of shares
            total_shares: Total shares outstanding
            
        Returns:
            PromoterActivity
        """
        percentage_change = shares / total_shares
        
        # Generate signal
        if activity_type == "buy":
            signal = min(percentage_change / self.promoter_sell_threshold, 1.0)
        elif activity_type == "sell":
            signal = -min(percentage_change / self.promoter_sell_threshold, 1.0)
        elif activity_type == "pledge":
            signal = -0.5  # Pledging is negative
        elif activity_type == "buyback":
            signal = 0.8  # Buyback is strongly positive
        else:
            signal = 0.0
        
        activity = PromoterActivity(
            symbol=symbol,
            timestamp=timestamp,
            activity_type=activity_type,
            shares=shares,
            percentage_change=percentage_change,
            signal=signal
        )
        
        # Store in history
        if symbol not in self.promoter_activities:
            self.promoter_activities[symbol] = []
        self.promoter_activities[symbol].append(activity)
        
        return activity
    
    def analyze_promoter_activity(
        self,
        symbol: str,
        window_days: int = 30
    ) -> Dict:
        """
        Analyze promoter activity for a symbol.
        
        Args:
            symbol: Trading symbol
            window_days: Lookback window
            
        Returns:
            Dictionary with analysis results
        """
        if symbol not in self.promoter_activities:
            return {'signal': 0.0, 'buy_activity': 0, 'sell_activity': 0}
        
        # Filter activities in window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_activities = [
            a for a in self.promoter_activities[symbol]
            if a.timestamp >= cutoff_date
        ]
        
        if not recent_activities:
            return {'signal': 0.0, 'buy_activity': 0, 'sell_activity': 0}
        
        # Calculate buy and sell activity
        buy_activity = sum(a.percentage_change for a in recent_activities if a.activity_type == "buy")
        sell_activity = sum(a.percentage_change for a in recent_activities if a.activity_type == "sell")
        
        # Calculate net signal
        net_signal = sum(a.signal for a in recent_activities)
        signal = np.tanh(net_signal)
        
        return {
            'signal': signal,
            'buy_activity': buy_activity,
            'sell_activity': sell_activity,
            'activity_count': len(recent_activities)
        }
    
    def add_fii_activity(
        self,
        symbol: str,
        timestamp: datetime,
        fii_buy_value: float,
        fii_sell_value: float,
        dii_buy_value: float,
        dii_sell_value: float
    ) -> FIIActivity:
        """
        Add FII/DII activity.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            fii_buy_value: FII buy value
            fii_sell_value: FII sell value
            dii_buy_value: DII buy value
            dii_sell_value: DII sell value
            
        Returns:
            FIIActivity
        """
        net_fii_flow = fii_buy_value - fii_sell_value
        net_dii_flow = dii_buy_value - dii_sell_value
        
        # Generate signal
        # FII buying is positive
        # DII buying is also positive but less weight
        signal = np.tanh(net_fii_flow / self.fii_flow_threshold) * 0.7 + \
                 np.tanh(net_dii_flow / self.fii_flow_threshold) * 0.3
        
        activity = FIIActivity(
            symbol=symbol,
            timestamp=timestamp,
            fii_buy_value=fii_buy_value,
            fii_sell_value=fii_sell_value,
            dii_buy_value=dii_buy_value,
            dii_sell_value=dii_sell_value,
            net_fii_flow=net_fii_flow,
            net_dii_flow=net_dii_flow,
            signal=signal
        )
        
        # Store in history
        if symbol not in self.fii_activities:
            self.fii_activities[symbol] = []
        self.fii_activities[symbol].append(activity)
        
        return activity
    
    def analyze_fii_activity(
        self,
        symbol: str,
        window_days: int = 30
    ) -> Dict:
        """
        Analyze FII/DII activity for a symbol.
        
        Args:
            symbol: Trading symbol
            window_days: Lookback window
            
        Returns:
            Dictionary with analysis results
        """
        if symbol not in self.fii_activities:
            return {'signal': 0.0, 'net_fii_flow': 0, 'net_dii_flow': 0}
        
        # Filter activities in window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_activities = [
            a for a in self.fii_activities[symbol]
            if a.timestamp >= cutoff_date
        ]
        
        if not recent_activities:
            return {'signal': 0.0, 'net_fii_flow': 0, 'net_dii_flow': 0}
        
        # Calculate net flows
        net_fii_flow = sum(a.net_fii_flow for a in recent_activities)
        net_dii_flow = sum(a.net_dii_flow for a in recent_activities)
        
        # Calculate net signal
        net_signal = sum(a.signal for a in recent_activities)
        signal = np.tanh(net_signal)
        
        return {
            'signal': signal,
            'net_fii_flow': net_fii_flow,
            'net_dii_flow': net_dii_flow,
            'activity_count': len(recent_activities)
        }
    
    def add_mutual_fund_activity(
        self,
        symbol: str,
        timestamp: datetime,
        fund_name: str,
        shares_change: int,
        total_shares: int
    ) -> MutualFundActivity:
        """
        Add mutual fund activity.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            fund_name: Name of mutual fund
            shares_change: Change in shares held
            total_shares: Total shares outstanding
            
        Returns:
            MutualFundActivity
        """
        percentage_change = shares_change / total_shares
        
        # Generate signal
        # Accumulation is positive
        # Selling is negative
        signal = np.tanh(percentage_change / self.mf_accumulation_threshold)
        
        activity = MutualFundActivity(
            symbol=symbol,
            timestamp=timestamp,
            fund_name=fund_name,
            shares_change=shares_change,
            percentage_change=percentage_change,
            signal=signal
        )
        
        # Store in history
        if symbol not in self.mutual_fund_activities:
            self.mutual_fund_activities[symbol] = []
        self.mutual_fund_activities[symbol].append(activity)
        
        return activity
    
    def analyze_mutual_fund_activity(
        self,
        symbol: str,
        window_days: int = 30
    ) -> Dict:
        """
        Analyze mutual fund activity for a symbol.
        
        Args:
            symbol: Trading symbol
            window_days: Lookback window
            
        Returns:
            Dictionary with analysis results
        """
        if symbol not in self.mutual_fund_activities:
            return {'signal': 0.0, 'accumulation': 0, 'selling': 0}
        
        # Filter activities in window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_activities = [
            a for a in self.mutual_fund_activities[symbol]
            if a.timestamp >= cutoff_date
        ]
        
        if not recent_activities:
            return {'signal': 0.0, 'accumulation': 0, 'selling': 0}
        
        # Calculate accumulation and selling
        accumulation = sum(a.percentage_change for a in recent_activities if a.percentage_change > 0)
        selling = sum(abs(a.percentage_change) for a in recent_activities if a.percentage_change < 0)
        
        # Calculate net signal
        net_signal = sum(a.signal for a in recent_activities)
        signal = np.tanh(net_signal)
        
        return {
            'signal': signal,
            'accumulation': accumulation,
            'selling': selling,
            'activity_count': len(recent_activities)
        }
    
    def get_aggregate_signal(
        self,
        symbol: str,
        window_days: int = 30
    ) -> Dict:
        """
        Get aggregate agency theory signal.
        
        Combines insider, promoter, FII/DII, and mutual fund signals.
        
        Args:
            symbol: Trading symbol
            window_days: Lookback window
            
        Returns:
            Dictionary with aggregate signal
        """
        insider_analysis = self.analyze_insider_activity(symbol, window_days)
        promoter_analysis = self.analyze_promoter_activity(symbol, window_days)
        fii_analysis = self.analyze_fii_activity(symbol, window_days)
        mf_analysis = self.analyze_mutual_fund_activity(symbol, window_days)
        
        # Weighted aggregate signal
        # Insider: 30%
        # Promoter: 25%
        # FII/DII: 30%
        # Mutual Fund: 15%
        
        aggregate_signal = (
            insider_analysis['signal'] * 0.30 +
            promoter_analysis['signal'] * 0.25 +
            fii_analysis['signal'] * 0.30 +
            mf_analysis['signal'] * 0.15
        )
        
        return {
            'aggregate_signal': aggregate_signal,
            'insider_signal': insider_analysis['signal'],
            'promoter_signal': promoter_analysis['signal'],
            'fii_signal': fii_analysis['signal'],
            'mf_signal': mf_analysis['signal'],
            'insider_net_flow': insider_analysis.get('net_flow', 0),
            'promoter_net_activity': promoter_analysis.get('buy_activity', 0) - promoter_analysis.get('sell_activity', 0),
            'fii_net_flow': fii_analysis.get('net_fii_flow', 0),
            'mf_accumulation': mf_analysis.get('accumulation', 0)
        }


if __name__ == "__main__":
    # Test the Agency Theory Engine
    print("Testing Agency Theory Signals: Insider Transactions, Promoter Activity, FII/DII Tracker...")
    
    engine = AgencyTheoryEngine()
    
    # Add insider transactions
    print("\nAdding Insider Transactions...")
    engine.add_insider_transaction(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(days=5),
        insider_name="John Doe",
        transaction_type=TransactionType.BUY,
        shares=10000,
        price=2500
    )
    
    engine.add_insider_transaction(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(days=3),
        insider_name="Jane Smith",
        transaction_type=TransactionType.BUY,
        shares=5000,
        price=2510
    )
    
    # Analyze insider activity
    print("\nAnalyzing Insider Activity...")
    insider_analysis = engine.analyze_insider_activity("RELIANCE", window_days=30)
    print(f"Signal: {insider_analysis['signal']:.2f}")
    print(f"Buy Volume: {insider_analysis['buy_volume']:.0f}")
    print(f"Sell Volume: {insider_analysis['sell_volume']:.0f}")
    print(f"Net Flow: {insider_analysis['net_flow']:.0f}")
    
    # Add promoter activity
    print("\nAdding Promoter Activity...")
    engine.add_promoter_activity(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(days=10),
        activity_type="buy",
        shares=1000000,
        total_shares=1000000000
    )
    
    # Analyze promoter activity
    print("\nAnalyzing Promoter Activity...")
    promoter_analysis = engine.analyze_promoter_activity("RELIANCE", window_days=30)
    print(f"Signal: {promoter_analysis['signal']:.2f}")
    print(f"Buy Activity: {promoter_analysis['buy_activity']:.2%}")
    print(f"Sell Activity: {promoter_analysis['sell_activity']:.2%}")
    
    # Add FII/DII activity
    print("\nAdding FII/DII Activity...")
    engine.add_fii_activity(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(days=2),
        fii_buy_value=50000000,
        fii_sell_value=10000000,
        dii_buy_value=30000000,
        dii_sell_value=5000000
    )
    
    # Analyze FII/DII activity
    print("\nAnalyzing FII/DII Activity...")
    fii_analysis = engine.analyze_fii_activity("RELIANCE", window_days=30)
    print(f"Signal: {fii_analysis['signal']:.2f}")
    print(f"Net FII Flow: {fii_analysis['net_fii_flow']:.0f}")
    print(f"Net DII Flow: {fii_analysis['net_dii_flow']:.0f}")
    
    # Add mutual fund activity
    print("\nAdding Mutual Fund Activity...")
    engine.add_mutual_fund_activity(
        symbol="RELIANCE",
        timestamp=datetime.now() - timedelta(days=7),
        fund_name="HDFC MF",
        shares_change=500000,
        total_shares=1000000000
    )
    
    # Analyze mutual fund activity
    print("\nAnalyzing Mutual Fund Activity...")
    mf_analysis = engine.analyze_mutual_fund_activity("RELIANCE", window_days=30)
    print(f"Signal: {mf_analysis['signal']:.2f}")
    print(f"Accumulation: {mf_analysis['accumulation']:.2%}")
    print(f"Selling: {mf_analysis['selling']:.2%}")
    
    # Get aggregate signal
    print("\nAggregate Agency Theory Signal:")
    aggregate = engine.get_aggregate_signal("RELIANCE", window_days=30)
    print(f"Aggregate Signal: {aggregate['aggregate_signal']:.2f}")
    print(f"Insider Signal: {aggregate['insider_signal']:.2f}")
    print(f"Promoter Signal: {aggregate['promoter_signal']:.2f}")
    print(f"FII Signal: {aggregate['fii_signal']:.2f}")
    print(f"MF Signal: {aggregate['mf_signal']:.2f}")
