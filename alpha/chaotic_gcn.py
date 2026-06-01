"""
Chaotic Graph Convolutional Network Alpha for Indian Markets.
Combines chaotic time series analysis with sector graph neural networks.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class GCNSignal:
    symbol: str
    direction: int  # 0=short, 1=neutral, 2=long
    confidence: float
    position_scale: float
    chaotic_score: float
    sector_momentum: float


class ChaoticGCNModel(nn.Module):
    """
    Graph Convolutional Network with chaotic feature fusion.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Chaotic fusion layer
        self.chaotic_fusion = nn.Linear(hidden_dim + 4, hidden_dim)  # +4 for chaotic features

        # Output layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        chaotic_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GCN with chaotic fusion.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            chaotic_features: Chaotic features [num_nodes, 4]

        Returns:
            Class logits [num_nodes, num_classes]
        """
        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Fuse with chaotic features
        x_fused = torch.cat([x, chaotic_features], dim=1)
        x_fused = self.chaotic_fusion(x_fused)
        x_fused = F.relu(x_fused)
        x_fused = self.dropout(x_fused)

        # Classification
        logits = self.classifier(x_fused)
        return logits


class ChaoticGCNAlpha:
    """
    Chaotic GCN-based alpha generation.

    Uses:
    1. Chaotic transformations of returns (logistic/tent maps)
    2. Sector correlation graph
    3. Graph Convolutional Network for cross-asset learning
    """

    def __init__(self, config: dict):
        self.config = config
        gcn_config = config.get("alpha", {}).get("chaotic_gcn", {})

        self.sectors = gcn_config.get("sectors", 12)
        self.chaotic_map = gcn_config.get("chaotic_map", "logistic")
        self.logistic_r = gcn_config.get("logistic_r", 3.99)
        self.tent_mu = gcn_config.get("tent_mu", 0.5)
        self.gcn_layers = gcn_config.get("gcn_layers", 2)
        self.hidden_dim = gcn_config.get("hidden_dim", 64)
        self.dropout = gcn_config.get("dropout", 0.3)
        self.retrain_frequency = gcn_config.get("retrain_frequency", "1W")

        self.model: Optional[ChaoticGCNModel] = None
        self._is_trained = False
        self._symbol_to_idx: Dict[str, int] = {}
        self._idx_to_symbol: Dict[int, str] = {}

    def _prepare_features(
        self,
        data_dict: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for GCN training.

        Returns:
            (features, chaotic_features, labels)
        """
        symbols = list(data_dict.keys())
        n = len(symbols)

        # Map symbols to indices
        self._symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        self._idx_to_symbol = {i: s for i, s in enumerate(symbols)}

        # Extract features for each symbol
        features_list = []
        chaotic_list = []
        labels_list = []

        for symbol in symbols:
            df = data_dict[symbol]

            if len(df) < 60:
                # Pad with zeros if insufficient data
                features_list.append(np.zeros(20))
                chaotic_list.append(np.zeros(4))
                labels_list.append(1)  # Neutral
                continue

            # Price features (last 20 returns)
            returns = df["Close"].pct_change(20).tail(20).values
            if len(returns) < 20:
                returns = np.pad(returns, (20 - len(returns), 0))

            # Volatility features
            vol = df["Close"].pct_change().tail(20).std()

            # Volume features
            rel_vol = df["Volume"].tail(20).mean() / df["Volume"].tail(60).mean()

            # Momentum features
            momentum_5 = df["Close"].pct_change(5).iloc[-1]
            momentum_20 = df["Close"].pct_change(20).iloc[-1]

            # Combine features
            features = np.concatenate([
                returns[:15],  # 15 return features
                [vol, rel_vol, momentum_5, momentum_20],  # 4 summary features
                [0, 0]  # Padding
            ])
            features_list.append(features)

            # Chaotic features
            log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
            if len(log_returns) > 50:
                chaotic_features = self._compute_chaotic_features(log_returns[-50:])
            else:
                chaotic_features = np.zeros(4)
            chaotic_list.append(chaotic_features)

            # Label (5-day forward return)
            if len(df) > 5:
                forward_return = df["Close"].pct_change(5).iloc[-1]
                if forward_return > 0.01:
                    label = 2  # Long
                elif forward_return < -0.01:
                    label = 0  # Short
                else:
                    label = 1  # Neutral
            else:
                label = 1
            labels_list.append(label)

        return np.array(features_list), np.array(chaotic_list), np.array(labels_list)

    def _compute_chaotic_features(self, returns: np.ndarray) -> np.ndarray:
        """Compute chaotic features from returns."""
        if len(returns) < 20:
            return np.zeros(4)

        # Normalize to [0, 1]
        ret_min, ret_max = returns.min(), returns.max()
        if ret_max == ret_min:
            return np.zeros(4)

        normalized = (returns - ret_min) / (ret_max - ret_min)

        # Apply chaotic map
        chaotic_series = np.zeros(len(normalized))
        chaotic_series[0] = normalized[0]

        for i in range(1, len(normalized)):
            x = chaotic_series[i - 1]
            if self.chaotic_map == "logistic":
                chaotic_series[i] = self.logistic_r * x * (1 - x)
            else:
                chaotic_series[i] = self.tent_mu * min(x, 1 - x)

        # Compute chaotic features
        window = 10
        if len(chaotic_series) < window:
            return np.zeros(4)

        # Chaotic entropy
        hist, _ = np.histogram(chaotic_series[-window:], bins=10, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist / hist.sum())) if len(hist) > 0 else 0

        # Chaotic Lyapunov exponent
        if self.chaotic_map == "logistic":
            lambdas = np.log(np.abs(self.logistic_r - 2 * self.logistic_r * chaotic_series[-window:]))
        else:
            lambdas = np.full(len(chaotic_series[-window:]), np.log(self.tent_mu))
        lyapunov = np.mean(lambdas) if len(lambdas) > 0 else 0

        # Chaotic deviation
        deviation = np.abs(normalized[-window:] - chaotic_series[-window:]).mean()

        # Chaotic variance
        chaotic_var = np.var(chaotic_series[-window:])

        return np.array([entropy, lyapunov, deviation, chaotic_var])

    def _build_graph(
        self,
        data_dict: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build sector correlation graph.

        Returns:
            (edge_index, edge_weights)
        """
        symbols = list(data_dict.keys())
        n = len(symbols)

        # Compute correlation matrix
        returns_dict = {}
        for symbol in symbols:
            df = data_dict[symbol]
            if len(df) >= 60:
                returns_dict[symbol] = df["Close"].pct_change().tail(60).values
            else:
                returns_dict[symbol] = np.zeros(60)

        # Build correlation matrix
        corr_matrix = np.zeros((n, n))
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    try:
                        corr = np.corrcoef(
                            returns_dict[sym_i],
                            returns_dict[sym_j]
                        )[0, 1]
                        corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
                    except:
                        corr_matrix[i, j] = 0.0

        # Create edges based on correlation threshold
        threshold = 0.3
        edges = []
        edge_weights = []

        for i in range(n):
            for j in range(n):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    edges.append([i, j])
                    edge_weights.append(abs(corr_matrix[i, j]))

        if len(edges) == 0:
            # Fallback: create a fully connected graph
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edges.append([i, j])
                        edge_weights.append(0.1)

        edge_index = np.array(edges).T if edges else np.array([[0], [0]])
        edge_weights = np.array(edge_weights)

        return edge_index, edge_weights

    def train_model(
        self,
        data_dict: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str],
        labels: Dict[str, int],
        epochs: int = 50,
        learning_rate: float = 0.001
    ) -> None:
        """
        Train the Chaotic GCN model.

        Args:
            data_dict: {symbol: DataFrame}
            sector_map: {symbol: sector}
            labels: {symbol: label} (0=short, 1=neutral, 2=long)
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        logger.info("Training Chaotic GCN model...")

        # Prepare features
        features, chaotic_features, _ = self._prepare_features(data_dict, sector_map)

        # Build graph
        edge_index, edge_weights = self._build_graph(data_dict, sector_map)

        # Prepare labels
        label_array = np.array([labels.get(s, 1) for s in self._symbol_to_idx.keys()])

        # Convert to tensors
        x = torch.FloatTensor(features)
        chaotic_feat = torch.FloatTensor(chaotic_features)
        edge_idx = torch.LongTensor(edge_index)
        y = torch.LongTensor(label_array)

        # Initialize model
        input_dim = features.shape[1]
        self.model = ChaoticGCNModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.gcn_layers,
            num_classes=3,
            dropout=self.dropout
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(x, edge_idx, chaotic_feat)
            loss = criterion(logits, y)

            # Backward pass
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self._is_trained = True
        logger.info("Chaotic GCN model trained successfully")

    def generate_signal(
        self,
        data_dict: Dict[str, pd.DataFrame],
        sector_map: Dict[str, str],
        symbol: str
    ) -> Dict:
        """
        Generate signal for a specific symbol.

        Args:
            data_dict: {symbol: DataFrame}
            sector_map: {symbol: sector}
            symbol: Symbol to generate signal for

        Returns:
            Signal dictionary with direction, confidence, etc.
        """
        if not self._is_trained:
            logger.warning("GCN model not trained, returning neutral signal")
            return {
                "direction": 1,
                "confidence": 0.0,
                "position_scale": 0.0,
                "chaotic_score": 0.0,
                "sector_momentum": 0.0,
            }

        if symbol not in self._symbol_to_idx:
            logger.warning(f"Symbol {symbol} not in training data")
            return {
                "direction": 1,
                "confidence": 0.0,
                "position_scale": 0.0,
                "chaotic_score": 0.0,
                "sector_momentum": 0.0,
            }

        # Prepare features
        features, chaotic_features, _ = self._prepare_features(data_dict, sector_map)

        # Build graph
        edge_index, edge_weights = self._build_graph(data_dict, sector_map)

        # Convert to tensors
        x = torch.FloatTensor(features)
        chaotic_feat = torch.FloatTensor(chaotic_features)
        edge_idx = torch.LongTensor(edge_index)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, edge_idx, chaotic_feat)
            probs = F.softmax(logits, dim=1)

        # Get signal for specific symbol
        idx = self._symbol_to_idx[symbol]
        symbol_probs = probs[idx].cpu().numpy()

        direction = int(np.argmax(symbol_probs))
        confidence = float(symbol_probs[direction])

        # Calculate position scale based on confidence
        position_scale = confidence if direction != 1 else 0.0

        # Get chaotic score
        chaotic_score = float(chaotic_features[idx, 0])  # Entropy

        # Get sector momentum
        sector = sector_map.get(symbol, "Unknown")
        sector_symbols = [s for s, sec in sector_map.items() if sec == sector and s in self._symbol_to_idx]
        if sector_symbols:
            sector_probs = probs[[self._symbol_to_idx[s] for s in sector_symbols]].mean(dim=0)
            sector_momentum = float(sector_probs[2] - sector_probs[0])  # Long - Short
        else:
            sector_momentum = 0.0

        return {
            "direction": direction,
            "confidence": confidence,
            "position_scale": position_scale,
            "chaotic_score": chaotic_score,
            "sector_momentum": sector_momentum,
        }

    def reset(self) -> None:
        """Reset model state."""
        self._is_trained = False
        self._symbol_to_idx.clear()
        self._idx_to_symbol.clear()
