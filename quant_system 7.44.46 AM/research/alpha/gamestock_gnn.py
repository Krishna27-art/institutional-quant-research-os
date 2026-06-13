"""
GameStock Heterogeneous GNN for Investor Flow Analysis

Implements the GameStock model from Zhang et al. for analyzing investor flows
and generating alpha signals based on game-theoretic principles. This model
classifies investors into institutional, hot-money, and retail categories and
uses heterogeneous graph neural networks to capture flow dynamics.

Key Features:
- Three investor type classification (institutional, hot-money, retail)
- Flow aggregation and regime-aware alpha signals
- Correlation graph construction
- Game equilibrium modeling (Nash equilibrium)
- Heterogeneous GNN with relation types

Based on Blueprint Week 7-8: Advanced Alpha (Papers)
Reference: Zhang et al. - GameStock: Investor Flow Analysis with GNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InvestorType:
    """Investor type classification."""
    INSTITUTIONAL = 0
    HOT_MONEY = 1
    RETAIL = 2


class GameStockHeterogeneousGNN(nn.Module):
    """
    GameStock Heterogeneous Graph Neural Network.
    
    This model analyzes investor flows across different investor types
    and stocks using a heterogeneous graph neural network. It models
    the game-theoretic interactions between different investor types
    and stocks to generate alpha signals.
    """
    
    def __init__(
        self,
        n_stocks: int,
        n_investor_types: int = 3,
        hidden_dim: int = 64,
        n_relations: int = 2  # buy/sell relations
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_investor_types = n_investor_types
        self.hidden_dim = hidden_dim
        self.n_relations = n_relations
        
        # Embeddings for stocks and investor types
        self.stock_embed = nn.Embedding(n_stocks, hidden_dim)
        self.investor_embed = nn.Embedding(n_investor_types, hidden_dim)
        
        # Relation-specific transformations
        self.relation_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_relations)
        ])
        
        # Graph convolution layers
        self.conv1 = self._build_conv_layer(hidden_dim, hidden_dim)
        self.conv2 = self._build_conv_layer(hidden_dim, hidden_dim)
        
        # Game equilibrium layer
        self.game_layer = GameEquilibriumLayer(hidden_dim)
        
        # Payoff function
        self.payoff = nn.Linear(hidden_dim, 1)
        
        # Alpha signal output
        self.alpha_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1] for long/short signals
        )
    
    def _build_conv_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Build a graph convolution layer."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self,
        stock_ids: torch.Tensor,
        investor_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            stock_ids: Stock node indices [n_stocks]
            investor_ids: Investor type node indices [n_investor_types]
            edge_index: Edge indices [2, n_edges]
            edge_type: Edge type indices [n_edges] (0=buy, 1=sell)
            edge_attr: Edge attributes [n_edges, edge_dim]
            
        Returns:
            Tuple of (alpha_signals, equilibrium_payoffs)
        """
        # Get embeddings
        stock_features = self.stock_embed(stock_ids)
        investor_features = self.investor_embed(investor_ids)
        
        # Concatenate all node features
        x = torch.cat([stock_features, investor_features], dim=0)
        
        # Apply relation-specific transformations
        x_transformed = torch.zeros_like(x)
        for i in range(self.n_relations):
            mask = (edge_type == i)
            if mask.any():
                x_transformed += self.relation_weights[i](x)
        
        # Graph convolution layers
        h = self.conv1(x_transformed)
        h = self._message_passing(h, edge_index, edge_type)
        
        h = self.conv2(h)
        h = self._message_passing(h, edge_index, edge_type)
        
        # Game equilibrium
        equilibrium = self.game_layer(h)
        
        # Extract stock nodes only
        stock_h = equilibrium[:len(stock_ids)]
        
        # Calculate payoffs
        payoffs = self.payoff(stock_h)
        
        # Generate alpha signals
        alpha_signals = self.alpha_output(stock_h)
        
        return alpha_signals, payoffs
    
    def _message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Message passing for heterogeneous graph.
        
        Args:
            x: Node features [n_nodes, hidden_dim]
            edge_index: Edge indices [2, n_edges]
            edge_type: Edge type indices [n_edges]
            
        Returns:
            Updated node features
        """
        src, dst = edge_index
        
        # Aggregate messages
        out = torch.zeros_like(x)
        for i in range(self.n_relations):
            mask = (edge_type == i)
            if mask.any():
                # Get source and destination nodes for this relation type
                src_i = src[mask]
                dst_i = dst[mask]
                
                # Aggregate messages
                messages = x[src_i]
                # Simple mean aggregation
                out.index_add_(0, dst_i, messages)
        
        # Normalize by degree
        degree = torch.zeros(x.size(0), device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = torch.clamp(degree, min=1).unsqueeze(1)
        
        return out / degree


class GameEquilibriumLayer(nn.Module):
    """
    Game Equilibrium Layer.
    
    Models the Nash equilibrium of the game between different investor types
    and stocks using iterative best-response dynamics.
    """
    
    def __init__(self, hidden_dim: int, n_iterations: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_iterations = n_iterations
        
        # Best response function
        self.best_response = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute Nash equilibrium through iterative best-response dynamics.
        
        Args:
            h: Node features [n_nodes, hidden_dim]
            
        Returns:
            Equilibrium node features
        """
        equilibrium = h.clone()
        
        for _ in range(self.n_iterations):
            # Best response dynamics
            response = self.best_response(equilibrium)
            
            # Update towards equilibrium (gradient descent on potential)
            equilibrium = equilibrium + 0.1 * (response - equilibrium)
            
            # Apply non-negativity constraint
            equilibrium = F.relu(equilibrium)
        
        return equilibrium


class FlowAggregator:
    """
    Aggregates investor flows and generates regime-aware signals.
    """
    
    def __init__(self, window: int = 20):
        self.window = window
        self.flow_history: Dict[str, List[float]] = {}
    
    def aggregate_flow(
        self,
        symbol: str,
        investor_type: int,
        flow: float
    ) -> Dict[str, float]:
        """
        Aggregate flow for a symbol and investor type.
        
        Args:
            symbol: Stock symbol
            investor_type: Investor type (0=institutional, 1=hot-money, 2=retail)
            flow: Flow value (positive for buy, negative for sell)
            
        Returns:
            Dictionary with aggregated flow metrics
        """
        key = f"{symbol}_{investor_type}"
        
        if key not in self.flow_history:
            self.flow_history[key] = []
        
        self.flow_history[key].append(flow)
        
        # Keep only recent history
        if len(self.flow_history[key]) > self.window:
            self.flow_history[key] = self.flow_history[key][-self.window:]
        
        flows = self.flow_history[key]
        
        return {
            'mean_flow': np.mean(flows),
            'std_flow': np.std(flows),
            'cumulative_flow': np.sum(flows),
            'recent_flow': flows[-1] if flows else 0.0,
            'flow_momentum': np.mean(flows[-5:]) if len(flows) >= 5 else 0.0
        }
    
    def detect_regime(self, symbol: str) -> str:
        """
        Detect flow regime for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Regime classification
        """
        institutional_flow = self.aggregate_flow(symbol, InvestorType.INSTITUTIONAL, 0)
        hot_money_flow = self.aggregate_flow(symbol, InvestorType.HOT_MONEY, 0)
        retail_flow = self.aggregate_flow(symbol, InvestorType.RETAIL, 0)
        
        # Simple regime detection
        if institutional_flow['cumulative_flow'] > 0:
            if hot_money_flow['cumulative_flow'] > 0:
                return 'STRONG_BULLISH'
            else:
                return 'INSTITUTIONAL_BULLISH'
        elif institutional_flow['cumulative_flow'] < 0:
            if hot_money_flow['cumulative_flow'] < 0:
                return 'STRONG_BEARISH'
            else:
                return 'INSTITUTIONAL_BEARISH'
        else:
            if retail_flow['cumulative_flow'] > 0:
                return 'RETAIL_DRIVEN'
            else:
                return 'NEUTRAL'


class CorrelationGraphBuilder:
    """
    Builds correlation graphs for stock relationships.
    """
    
    @staticmethod
    def build_correlation_graph(
        returns: pd.DataFrame,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Build correlation graph from returns.
        
        Args:
            returns: DataFrame with returns for each stock
            threshold: Correlation threshold for edge creation
            
        Returns:
            Adjacency matrix
        """
        corr_matrix = returns.corr()
        
        # Create adjacency matrix
        adj = (corr_matrix.abs() > threshold).astype(float)
        
        # Remove self-loops
        np.fill_diagonal(adj.values, 0)
        
        return torch.from_numpy(adj.values).float()
    
    @staticmethod
    def build_heterogeneous_edges(
        stock_ids: List[int],
        investor_ids: List[int],
        flow_data: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build heterogeneous edges for investor-stock graph.
        
        Args:
            stock_ids: Stock node indices
            investor_ids: Investor type node indices
            flow_data: Dictionary with flow information
            
        Returns:
            Tuple of (edge_index, edge_type)
        """
        edges = []
        edge_types = []
        
        # Create edges between investor types and stocks
        for investor_id in investor_ids:
            for stock_id in stock_ids:
                # Buy edge (type 0)
                edges.append([investor_id, stock_id])
                edge_types.append(0)
                
                # Sell edge (type 1)
                edges.append([stock_id, investor_id])
                edge_types.append(1)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return edge_index, edge_type


if __name__ == "__main__":
    # Test GameStock GNN
    print("Testing GameStock Heterogeneous GNN...")
    
    # Create sample data
    n_stocks = 10
    n_investor_types = 3
    hidden_dim = 32
    
    stock_ids = torch.arange(n_stocks)
    investor_ids = torch.arange(n_investor_types)
    
    # Create sample edges
    edge_index = []
    edge_type = []
    
    for investor_id in range(n_investor_types):
        for stock_id in range(n_stocks):
            edge_index.append([investor_id, stock_id])
            edge_type.append(0)  # buy
            edge_index.append([stock_id, investor_id])
            edge_type.append(1)  # sell
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    # Create model
    model = GameStockHeterogeneousGNN(n_stocks, n_investor_types, hidden_dim)
    
    # Forward pass
    alpha_signals, payoffs = model(stock_ids, investor_ids, edge_index, edge_type)
    
    print(f"Alpha signals shape: {alpha_signals.shape}")
    print(f"Payoffs shape: {payoffs.shape}")
    print(f"Alpha signals range: [{alpha_signals.min():.3f}, {alpha_signals.max():.3f}]")
    
    # Test flow aggregator
    aggregator = FlowAggregator()
    flow_metrics = aggregator.aggregate_flow('RELIANCE', InvestorType.INSTITUTIONAL, 1000.0)
    print(f"\nFlow metrics: {flow_metrics}")
    
    regime = aggregator.detect_regime('RELIANCE')
    print(f"Regime: {regime}")
    
    print("\nGameStock Heterogeneous GNN test completed.")
