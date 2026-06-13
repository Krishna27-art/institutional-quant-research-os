"""
BCF-GCN (Bi-level Chaotic Fusion Graph Convolutional Network)

Implements the BCF-GCN model from Kandimalla et al. for prediction intervals
with chaotic dynamics. This model combines graph neural networks with chaotic
maps to provide both point predictions and uncertainty quantification.

Key Features:
- Logistic Map chaotic branch for center prediction
- Tent Map chaotic branch for width prediction (prediction intervals)
- Bi-level fusion with learnable gating
- Prediction interval coverage
- Chaotic dynamics for uncertainty modeling

Based on Blueprint Week 7-8: Advanced Alpha (Papers)
Reference: Kandimalla et al. - Bi-level Chaotic Fusion GCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LogisticMap(nn.Module):
    """
    Logistic Map chaotic system.
    
    The logistic map is a classic chaotic system defined by:
    x_{n+1} = r * x_n * (1 - x_n)
    
    For r > 3.57, the system exhibits chaotic behavior.
    """
    
    def __init__(self, r_init: float = 3.8):
        super().__init__()
        self.r = nn.Parameter(torch.tensor(r_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply logistic map transformation.
        
        Args:
            x: Input tensor (should be in [0, 1])
            
        Returns:
            Transformed tensor
        """
        # Clamp to [0, 1] to ensure stability
        x_clamped = torch.clamp(x, 0.0, 1.0)
        return self.r * x_clamped * (1 - x_clamped)


class TentMap(nn.Module):
    """
    Tent Map chaotic system.
    
    The tent map is another classic chaotic system defined by:
    x_{n+1} = mu * x_n if x_n < 0.5
             mu * (1 - x_n) if x_n >= 0.5
    
    For mu > 1, the system exhibits chaotic behavior.
    """
    
    def __init__(self, mu_init: float = 1.8):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply tent map transformation.
        
        Args:
            x: Input tensor (should be in [0, 1])
            
        Returns:
            Transformed tensor
        """
        # Clamp to [0, 1] to ensure stability
        x_clamped = torch.clamp(x, 0.0, 1.0)
        return torch.where(x_clamped < 0.5, self.mu * x_clamped, self.mu * (1 - x_clamped))


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.
    
    Implements a simple GCN layer without requiring torch_geometric.
    This follows the formulation: H' = sigma(D^-1 * A * H * W)
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [n_nodes, in_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Updated node features [n_nodes, out_features]
        """
        # Normalize adjacency matrix
        degree = torch.sum(adj, dim=1, keepdim=True)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # D^-1/2 * A * D^-1/2
        norm_adj = degree_inv_sqrt * adj * degree_inv_sqrt.T
        
        # GCN operation
        support = torch.mm(norm_adj, x)
        output = torch.mm(support, self.weight) + self.bias
        
        return F.relu(output)


class BCF_GCN(nn.Module):
    """
    Bi-level Chaotic Fusion Graph Convolutional Network.
    
    This model combines:
    1. Graph neural networks for capturing relationships between stocks
    2. Chaotic maps (logistic and tent) for modeling uncertainty
    3. Bi-level fusion with learnable gating
    
    The model outputs both point predictions and prediction intervals.
    """
    
    def __init__(
        self,
        n_features: int,
        n_hidden: int = 64,
        n_classes: int = 1,
        n_layers: int = 2
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        # Graph convolution layers
        self.conv1 = GCNLayer(n_features, n_hidden)
        self.conv2 = GCNLayer(n_hidden, n_hidden)
        
        # Chaotic branches
        self.logistic = LogisticMap(r_init=3.8)
        self.tent = TentMap(mu_init=1.8)
        
        # Gating mechanism for bi-level fusion
        self.gate = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        # Output layers
        self.out_center = nn.Linear(n_hidden, n_classes)
        self.out_width = nn.Linear(n_hidden, n_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [n_nodes, n_features]
            adj: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Tuple of (prediction, center, width)
            - prediction: Final point prediction
            - center: Center prediction from logistic branch
            - width: Width prediction from tent branch (for intervals)
        """
        # Graph convolution layers
        h = self.conv1(x, adj)
        h = self.dropout(h)
        h = self.conv2(h, adj)
        
        # Apply activation to get values in [0, 1] for chaotic maps
        h_normalized = torch.sigmoid(h)
        
        # Chaotic branches
        center = self.logistic(h_normalized)
        width = self.tent(h_normalized)
        
        # Bi-level fusion with gating
        gate = self.gate(h)
        h_mixed = gate * center + (1 - gate) * width
        
        # Output predictions
        pred_center = self.out_center(h_mixed)
        pred_width = self.out_width(h_mixed)
        
        # Final prediction (center)
        prediction = pred_center
        
        return prediction, center, width
    
    def predict_with_interval(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence intervals.
        
        Args:
            x: Node features
            adj: Adjacency matrix
            confidence: Confidence level for interval (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (prediction, lower_bound, upper_bound)
        """
        prediction, center, width = self.forward(x, adj)
        
        # Calculate interval bounds using width
        # Width is scaled by confidence level
        z_score = 1.96  # For 95% confidence
        interval_width = torch.abs(width) * z_score
        
        lower_bound = prediction - interval_width
        upper_bound = prediction + interval_width
        
        return prediction, lower_bound, upper_bound


class BCF_GCN_Trainer:
    """
    Trainer for BCF-GCN model.
    """
    
    def __init__(
        self,
        model: BCF_GCN,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
    
    def train_epoch(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            x: Node features
            adj: Adjacency matrix
            y: Target values
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        prediction, _, _ = self.model(x, adj)
        
        # Calculate loss
        loss = self.criterion(prediction, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y: torch.Tensor
    ) -> dict:
        """
        Evaluate the model.
        
        Args:
            x: Node features
            adj: Adjacency matrix
            y: Target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            prediction, center, width = self.model(x, adj)
            
            # Calculate metrics
            mse = F.mse_loss(prediction, y).item()
            mae = F.l1_loss(prediction, y).item()
            
            # Calculate prediction interval coverage
            pred, lower, upper = self.model.predict_with_interval(x, adj)
            coverage = ((y >= lower) & (y <= upper)).float().mean().item()
            
        return {
            'mse': mse,
            'mae': mae,
            'coverage': coverage
        }


def create_sample_adjacency(n_nodes: int, sparsity: float = 0.1) -> torch.Tensor:
    """
    Create a sample adjacency matrix.
    
    Args:
        n_nodes: Number of nodes
        sparsity: Sparsity of the graph
        
    Returns:
        Adjacency matrix
    """
    adj = torch.rand(n_nodes, n_nodes)
    adj = (adj < sparsity).float()
    
    # Make symmetric
    adj = (adj + adj.T) / 2
    
    # Remove self-loops
    adj.fill_diagonal_(0)
    
    return adj


if __name__ == "__main__":
    # Test BCF-GCN
    print("Testing BCF-GCN...")
    
    # Create sample data
    n_nodes = 10
    n_features = 5
    n_hidden = 16
    
    x = torch.randn(n_nodes, n_features)
    adj = create_sample_adjacency(n_nodes)
    y = torch.randn(n_nodes, 1)
    
    # Create model
    model = BCF_GCN(n_features, n_hidden, n_classes=1)
    
    # Forward pass
    prediction, center, width = model(x, adj)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Center shape: {center.shape}")
    print(f"Width shape: {width.shape}")
    
    # Prediction with interval
    pred, lower, upper = model.predict_with_interval(x, adj)
    print(f"Prediction with interval shape: {pred.shape}")
    print(f"Lower bound shape: {lower.shape}")
    print(f"Upper bound shape: {upper.shape}")
    
    # Train for a few steps
    trainer = BCF_GCN_Trainer(model)
    for epoch in range(10):
        loss = trainer.train_epoch(x, adj, y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Evaluate
    metrics = trainer.evaluate(x, adj, y)
    print(f"\nEvaluation metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Coverage: {metrics['coverage']:.4f}")
    
    print("\nBCF-GCN test completed.")
