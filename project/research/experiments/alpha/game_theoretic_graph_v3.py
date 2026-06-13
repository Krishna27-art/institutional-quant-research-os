"""
Game-Theoretic Heterogeneous Graph for Stock Selection
Based on V3 Blueprint - Investor Type Analysis

Key findings from research:
- Game-theoretic heterogeneous graph for stock price forecasting
- Investors (institutions, hot money, retail) play game; equilibrium yields signal
- Dragon & Tiger List events (A-share), game triples (buy/sell/hold actions)
- Heterogeneous GCN (RGCN) with DWT for multi-scale temporal features
- Transfer to India: Proxy institutions (FII buying), retail (delivery % low), hot money (high delivery + volume spike)

V3 Upgrade - Expected Sharpe increase: +0.1–0.2
Priority: Low (research)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class InvestorType:
    """Investor type classification"""
    type_name: str  # "institutional", "retail", "hot_money"
    score: float  # 0-1
    confidence: float


@dataclass
class GameTriple:
    """Game triple (buy/sell/hold actions)"""
    symbol: str
    institutional_action: str  # "buy", "sell", "hold"
    retail_action: str
    hot_money_action: str
    equilibrium_signal: float  # -1 to 1


@dataclass
class GraphNode:
    """Graph node for a stock"""
    symbol: str
    investor_types: Dict[str, InvestorType]
    edges: List[str]  # Connected symbols
    embedding: Optional[np.ndarray]


class GameTheoreticGraphEngine:
    """
    Game-Theoretic Heterogeneous Graph for stock selection.
    
    Investor Types:
    - Institutional: FII buying, high delivery percentage
    - Retail: Low delivery percentage, high retail volume
    - Hot Money: High delivery + volume spike (unusual activity)
    
    Game Theory:
    - Each investor type plays a game (buy/sell/hold)
    - Equilibrium yields signal
    - Heterogeneous GCN to learn from graph structure
    """
    
    def __init__(self):
        self.graph: Dict[str, GraphNode] = {}
        self.game_triples: List[GameTriple] = []
        self.correlation_threshold = 0.30
    
    def classify_investor_type(
        self,
        fii_flow: float,
        delivery_percentage: float,
        volume_spike: float,
        price_change: float
    ) -> Dict[str, InvestorType]:
        """
        Classify investor types for a stock.
        
        Args:
            fii_flow: FII net flow
            delivery_percentage: Delivery percentage
            volume_spike: Volume spike ratio
            price_change: Price change
            
        Returns:
            Dictionary of investor_type -> InvestorType
        """
        # Institutional: High FII buying, high delivery
        if fii_flow > 100 and delivery_percentage > 40:
            inst_score = 0.8
        elif fii_flow > 50 and delivery_percentage > 30:
            inst_score = 0.6
        else:
            inst_score = 0.2
        
        # Retail: Low delivery, high volume without price move
        if delivery_percentage < 20 and volume_spike > 1.5 and abs(price_change) < 0.02:
            retail_score = 0.8
        elif delivery_percentage < 30:
            retail_score = 0.5
        else:
            retail_score = 0.2
        
        # Hot Money: High delivery + volume spike + price move
        if delivery_percentage > 50 and volume_spike > 2.0 and abs(price_change) > 0.03:
            hot_score = 0.8
        elif delivery_percentage > 40 and volume_spike > 1.5:
            hot_score = 0.5
        else:
            hot_score = 0.2
        
        return {
            "institutional": InvestorType("institutional", inst_score, 0.7),
            "retail": InvestorType("retail", retail_score, 0.6),
            "hot_money": InvestorType("hot_money", hot_score, 0.5)
        }
    
    def determine_investor_action(
        self,
        investor_type: InvestorType,
        price_change: float
    ) -> str:
        """
        Determine investor action based on type and price change.
        
        Args:
            investor_type: Investor type
            price_change: Price change
            
        Returns:
            Action: "buy", "sell", "hold"
        """
        if investor_type.type_name == "institutional":
            # Institutions buy on dips, sell on rallies
            if price_change < -0.02 and investor_type.score > 0.6:
                return "buy"
            elif price_change > 0.03 and investor_type.score > 0.6:
                return "sell"
            else:
                return "hold"
        
        elif investor_type.type_name == "retail":
            # Retail tends to chase momentum
            if price_change > 0.02 and investor_type.score > 0.6:
                return "buy"
            elif price_change < -0.02 and investor_type.score > 0.6:
                return "sell"
            else:
                return "hold"
        
        else:  # hot_money
            # Hot money moves fast
            if investor_type.score > 0.7:
                if price_change > 0.01:
                    return "buy"
                elif price_change < -0.01:
                    return "sell"
            return "hold"
    
    def compute_equilibrium_signal(
        self,
        institutional_action: str,
        retail_action: str,
        hot_money_action: str
    ) -> float:
        """
        Compute equilibrium signal from game triple.
        
        Args:
            institutional_action: Institutional action
            retail_action: Retail action
            hot_money_action: Hot money action
            
        Returns:
            Equilibrium signal (-1 to 1)
        """
        # Assign values to actions
        action_values = {"buy": 1, "hold": 0, "sell": -1}
        
        inst_val = action_values[institutional_action]
        retail_val = action_values[retail_action]
        hot_val = action_values[hot_money_action]
        
        # Weighted average (institutions have highest weight)
        signal = 0.5 * inst_val + 0.3 * retail_val + 0.2 * hot_val
        
        return signal
    
    def build_graph(
        self,
        symbols: List[str],
        correlation_matrix: pd.DataFrame
    ) -> None:
        """
        Build heterogeneous graph from correlation matrix.
        
        Args:
            symbols: List of symbols
            correlation_matrix: Correlation matrix
        """
        for symbol in symbols:
            self.graph[symbol] = GraphNode(
                symbol=symbol,
                investor_types={},
                edges=[],
                embedding=None
            )
        
        # Add edges based on correlation
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:
                    continue
                
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > self.correlation_threshold:
                    self.graph[symbol1].edges.append(symbol2)
                    self.graph[symbol2].edges.append(symbol1)
    
    def update_node(
        self,
        symbol: str,
        fii_flow: float,
        delivery_percentage: float,
        volume_spike: float,
        price_change: float
    ) -> GameTriple:
        """
        Update node with investor types and compute game triple.
        
        Args:
            symbol: Stock symbol
            fii_flow: FII net flow
            delivery_percentage: Delivery percentage
            volume_spike: Volume spike ratio
            price_change: Price change
            
        Returns:
            GameTriple
        """
        # Classify investor types
        investor_types = self.classify_investor_type(
            fii_flow, delivery_percentage, volume_spike, price_change
        )
        
        # Determine actions
        inst_action = self.determine_investor_action(
            investor_types["institutional"], price_change
        )
        retail_action = self.determine_investor_action(
            investor_types["retail"], price_change
        )
        hot_action = self.determine_investor_action(
            investor_types["hot_money"], price_change
        )
        
        # Compute equilibrium signal
        equilibrium_signal = self.compute_equilibrium_signal(
            inst_action, retail_action, hot_action
        )
        
        # Update node
        if symbol in self.graph:
            self.graph[symbol].investor_types = investor_types
        
        # Create game triple
        triple = GameTriple(
            symbol=symbol,
            institutional_action=inst_action,
            retail_action=retail_action,
            hot_money_action=hot_action,
            equilibrium_signal=equilibrium_signal
        )
        
        self.game_triples.append(triple)
        
        return triple
    
    def get_top_signals(self, n: int = 10) -> List[GameTriple]:
        """
        Get top signals from game triples.
        
        Args:
            n: Number of top signals
            
        Returns:
            List of GameTriple sorted by equilibrium signal
        """
        sorted_triples = sorted(
            self.game_triples,
            key=lambda x: x.equilibrium_signal,
            reverse=True
        )
        return sorted_triples[:n]
    
    def print_graph_summary(self) -> None:
        """Print graph summary."""
        print("\n" + "="*60)
        print("GAME-THEORETIC GRAPH SUMMARY")
        print("="*60)
        print(f"Number of nodes: {len(self.graph)}")
        print(f"Number of edges: {sum(len(node.edges) for node in self.graph.values()) // 2}")
        print(f"Number of game triples: {len(self.game_triples)}")
        
        print("\nTop Signals:")
        for triple in self.get_top_signals(5):
            print(f"  {triple.symbol}: {triple.equilibrium_signal:.2f}")
            print(f"    Institutional: {triple.institutional_action}")
            print(f"    Retail: {triple.retail_action}")
            print(f"    Hot Money: {triple.hot_money_action}")
        
        print("="*60)


def run_sample_game_theoretic_graph():
    """Run sample game-theoretic graph."""
    engine = GameTheoreticGraphEngine()
    
    # Sample symbols
    symbols = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK"]
    
    # Generate sample correlation matrix
    np.random.seed(42)
    corr_matrix = pd.DataFrame(
        np.random.uniform(-0.5, 0.8, (len(symbols), len(symbols))),
        index=symbols,
        columns=symbols
    )
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Build graph
    engine.build_graph(symbols, corr_matrix)
    
    # Update nodes with sample data
    for symbol in symbols:
        fii_flow = np.random.uniform(-200, 300)
        delivery = np.random.uniform(15, 60)
        volume_spike = np.random.uniform(0.8, 2.5)
        price_change = np.random.uniform(-0.05, 0.05)
        
        engine.update_node(symbol, fii_flow, delivery, volume_spike, price_change)
    
    # Print summary
    engine.print_graph_summary()
    
    return engine


if __name__ == "__main__":
    run_sample_game_theoretic_graph()
