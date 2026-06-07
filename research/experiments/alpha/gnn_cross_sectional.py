"""
GNN for Cross-Sectional Alpha

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#13)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Graph Neural Networks for cross-sectional alpha generation
- Captures inter-stock dependencies
- Identifies contagion risk and improves diversification
- Used by emerging research tools
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torch-geometric")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class GNNConfig:
    """Configuration for GNN Cross-Sectional Alpha"""
    # Graph construction
    graph_type: str = "correlation"  # "correlation", "sector", "hybrid"
    correlation_threshold: float = 0.5  # Threshold for correlation edges
    n_neighbors: int = 10  # KNN neighbors
    
    # GNN architecture
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    gnn_type: str = "GCN"  # "GCN", "GAT"
    
    # Training
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    
    # Regularization
    l2_lambda: float = 0.01
    
    # Features
    use_price_features: bool = True
    use_volume_features: bool = True
    use_volatility_features: bool = True


class StockGraph:
    """Construct stock graph based on relationships"""
    
    def __init__(self, config: GNNConfig):
        self.config = config
        self.adjacency_matrix: Optional[np.ndarray] = None
        self.edge_index: Optional[np.ndarray] = None
        self.node_features: Optional[np.ndarray] = None
    
    def build_correlation_graph(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Build graph based on return correlations
        
        Args:
            returns: DataFrame of stock returns (stocks x time)
            
        Returns:
            Adjacency matrix
        """
        # Compute correlation matrix
        corr_matrix = returns.corr().values
        
        # Threshold to create edges
        adj_matrix = (np.abs(corr_matrix) > self.config.correlation_threshold).astype(float)
        np.fill_diagonal(adj_matrix, 0)  # No self-loops
        
        self.adjacency_matrix = adj_matrix
        return adj_matrix
    
    def build_knn_graph(self, features: pd.DataFrame) -> np.ndarray:
        """
        Build K-nearest neighbors graph
        
        Args:
            features: DataFrame of stock features (stocks x features)
            
        Returns:
            Adjacency matrix
        """
        from sklearn.neighbors import NearestNeighbors
        
        features_array = features.values
        nbrs = NearestNeighbors(n_neighbors=self.config.n_neighbors).fit(features_array)
        distances, indices = nbrs.kneighbors(features_array)
        
        n_stocks = features_array.shape[0]
        adj_matrix = np.zeros((n_stocks, n_stocks))
        
        for i in range(n_stocks):
            for j, idx in enumerate(indices[i]):
                adj_matrix[i, idx] = 1.0
        
        self.adjacency_matrix = adj_matrix
        return adj_matrix
    
    def get_edge_index(self) -> np.ndarray:
        """Convert adjacency matrix to edge index format for PyG"""
        if self.adjacency_matrix is None:
            return np.array([[], []])
        
        rows, cols = np.where(self.adjacency_matrix > 0)
        edge_index = np.vstack([rows, cols])
        
        self.edge_index = edge_index
        return edge_index


class GNNAlphaModel(nn.Module):
    """GNN model for cross-sectional alpha generation"""
    
    def __init__(self, config: GNNConfig, input_dim: int, output_dim: int = 1):
        super(GNNAlphaModel, self).__init__()
        self.config = config
        
        self.input_dim = input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = output_dim
        
        # GNN layers
        if config.gnn_type == "GCN":
            self.conv1 = GCNConv(input_dim, config.hidden_dim)
            self.conv2 = GCNConv(config.hidden_dim, config.hidden_dim)
        elif config.gnn_type == "GAT":
            self.conv1 = GATConv(input_dim, config.hidden_dim, heads=4, concat=False)
            self.conv2 = GATConv(config.hidden_dim, config.hidden_dim, heads=4, concat=False)
        else:
            self.conv1 = GCNConv(input_dim, config.hidden_dim)
            self.conv2 = GCNConv(config.hidden_dim, config.hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(config.hidden_dim, output_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features (n_nodes x input_dim)
            edge_index: Edge indices (2 x n_edges)
            
        Returns:
            Predictions (n_nodes x output_dim)
        """
        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc(x)
        
        return x


class CrossSectionalGNN:
    """
    Cross-Sectional Alpha Generation using GNN
    
    Captures inter-stock dependencies and generates alpha signals
    that account for stock relationships.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: GNNConfig):
        self.config = config
        
        self.graph_builder = StockGraph(config)
        self.model = None
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history: List[Dict] = []
    
    def prepare_features(self, returns: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare node features for each stock
        
        Args:
            returns: DataFrame of returns (stocks x time)
            volumes: Optional volume data
            
        Returns:
            Feature DataFrame (stocks x features)
        """
        features = []
        
        for stock in returns.columns:
            stock_features = {}
            stock_returns = returns[stock]
            
            # Price features
            if self.config.use_price_features:
                stock_features["mean_return"] = stock_returns.mean()
                stock_features["std_return"] = stock_returns.std()
                stock_features["skew_return"] = stock_returns.skew()
                stock_features["momentum_5d"] = stock_returns.rolling(5).mean().iloc[-1] if len(stock_returns) >= 5 else 0
                stock_features["momentum_20d"] = stock_returns.rolling(20).mean().iloc[-1] if len(stock_returns) >= 20 else 0
            
            # Volume features
            if self.config.use_volume_features and volumes is not None:
                stock_volume = volumes[stock]
                stock_features["mean_volume"] = stock_volume.mean()
                stock_features["volume_trend"] = stock_volume.pct_change().mean()
            
            # Volatility features
            if self.config.use_volatility_features:
                stock_features["volatility_5d"] = stock_returns.rolling(5).std().iloc[-1] if len(stock_returns) >= 5 else 0
                stock_features["volatility_20d"] = stock_returns.rolling(20).std().iloc[-1] if len(stock_returns) >= 20 else 0
            
            features.append(stock_features)
        
        feature_df = pd.DataFrame(features, index=returns.columns)
        feature_df = feature_df.fillna(0)
        
        return feature_df
    
    def build_graph(self, returns: pd.DataFrame, features: pd.DataFrame) -> None:
        """
        Build stock graph
        
        Args:
            returns: Returns DataFrame
            features: Feature DataFrame
        """
        if self.config.graph_type == "correlation":
            self.graph_builder.build_correlation_graph(returns)
        elif self.config.graph_type == "knn":
            self.graph_builder.build_knn_graph(features)
        else:
            # Hybrid: combine correlation and KNN
            corr_adj = self.graph_builder.build_correlation_graph(returns)
            knn_adj = self.graph_builder.build_knn_graph(features)
            self.graph_builder.adjacency_matrix = (corr_adj + knn_adj).clip(0, 1)
        
        self.graph_builder.get_edge_index()
    
    def train(self, returns: pd.DataFrame, target_returns: pd.Series, 
              volumes: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train GNN model
        
        Args:
            returns: Historical returns
            target_returns: Target returns for prediction
            volumes: Optional volume data
            
        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using fallback")
            return self._train_fallback(returns, target_returns)
        
        # Prepare features
        features = self.prepare_features(returns, volumes)
        feature_array = self.scaler.fit_transform(features.values)
        
        # Build graph
        self.build_graph(returns, features)
        
        # Prepare PyG data
        edge_index = torch.tensor(self.graph_builder.edge_index, dtype=torch.long)
        x = torch.tensor(feature_array, dtype=torch.float)
        y = torch.tensor(target_returns.values, dtype=torch.float).unsqueeze(1)
        
        # Initialize model
        self.model = GNNAlphaModel(self.config, input_dim=feature_array.shape[1])
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            out = self.model(x, edge_index)
            loss = criterion(out, y) + self.config.l2_lambda * sum(p.pow(2).sum() for p in self.model.parameters())
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x, edge_index)
            mse = mean_squared_error(target_returns.values, predictions.numpy())
        
        metrics = {
            "final_loss": losses[-1],
            "mse": mse,
            "n_epochs": self.config.epochs
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def _train_fallback(self, returns: pd.DataFrame, target_returns: pd.Series) -> Dict:
        """Fallback training without PyTorch"""
        # Simple linear model as fallback
        features = self.prepare_features(returns)
        feature_array = self.scaler.fit_transform(features.values)
        
        # Simple linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(feature_array, target_returns)
        
        predictions = model.predict(feature_array)
        mse = mean_squared_error(target_returns, predictions)
        
        return {"mse": mse, "n_epochs": 0}
    
    def predict(self, returns: pd.DataFrame, volumes: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            returns: Returns DataFrame
            volumes: Optional volume data
            
        Returns:
            Predictions for each stock
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback prediction
            features = self.prepare_features(returns, volumes)
            feature_array = self.scaler.transform(features.values)
            return np.zeros(len(returns.columns))
        
        # Prepare features
        features = self.prepare_features(returns, volumes)
        feature_array = self.scaler.transform(features.values)
        
        # Build graph
        self.build_graph(returns, features)
        
        # Prepare PyG data
        edge_index = torch.tensor(self.graph_builder.edge_index, dtype=torch.long)
        x = torch.tensor(feature_array, dtype=torch.float)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x, edge_index)
        
        return predictions.numpy().flatten()
    
    def get_contagion_risk(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate contagion risk using graph structure
        
        Args:
            returns: Returns DataFrame
            
        Returns:
            Dictionary of contagion metrics
        """
        # Build correlation graph
        adj_matrix = self.graph_builder.build_correlation_graph(returns)
        
        # Calculate network metrics
        degree_centrality = adj_matrix.sum(axis=1)
        clustering_coeff = []
        
        for i in range(len(adj_matrix)):
            neighbors = np.where(adj_matrix[i] > 0)[0]
            if len(neighbors) < 2:
                clustering_coeff.append(0)
                continue
            
            # Local clustering coefficient
            local_edges = 0
            for j in neighbors:
                for k in neighbors:
                    if j < k and adj_matrix[j, k] > 0:
                        local_edges += 1
            
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeff.append(local_edges / possible_edges if possible_edges > 0 else 0)
        
        return {
            "avg_degree_centrality": degree_centrality.mean(),
            "max_degree_centrality": degree_centrality.max(),
            "avg_clustering_coefficient": np.mean(clustering_coeff),
            "network_density": adj_matrix.sum() / (len(adj_matrix) * (len(adj_matrix) - 1))
        }


def simulate_stock_returns(n_stocks: int = 50, n_days: int = 252) -> pd.DataFrame:
    """Simulate correlated stock returns"""
    np.random.seed(42)
    
    # Generate correlation structure
    base_returns = np.random.randn(n_days, 5)
    stock_loadings = np.random.randn(5, n_stocks)
    
    returns = base_returns @ stock_loadings + np.random.randn(n_days, n_stocks) * 0.5
    
    stock_names = [f"STOCK_{i}" for i in range(n_stocks)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    return pd.DataFrame(returns, index=dates, columns=stock_names)


if __name__ == "__main__":
    # Example usage
    config = GNNConfig(
        graph_type="correlation",
        correlation_threshold=0.3,
        hidden_dim=32,
        epochs=50
    )
    
    gnn = CrossSectionalGNN(config)
    
    # Simulate data
    print("Simulating stock returns...")
    returns = simulate_stock_returns(30, 252)
    
    # Train
    print("\nTraining GNN...")
    target_returns = returns.iloc[-1]  # Use last day as target
    metrics = gnn.train(returns, target_returns)
    
    print(f"\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Predict
    print("\nGenerating predictions...")
    predictions = gnn.predict(returns)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Mean prediction: {predictions.mean():.6f}")
    
    # Contagion risk
    print("\nCalculating contagion risk...")
    contagion = gnn.get_contagion_risk(returns)
    for key, value in contagion.items():
        print(f"  {key}: {value:.4f}")
