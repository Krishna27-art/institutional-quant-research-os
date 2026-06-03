"""
LSTM for Time-Series Prediction

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#21)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- LSTM networks for time-series prediction
- Captures temporal dependencies in financial data
- Multi-step ahead forecasting
- Used by Renaissance Technologies
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
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


@dataclass
class LSTMConfig:
    """Configuration for LSTM Time-Series Model"""
    # Model architecture
    input_size: int = 10  # Number of input features
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1  # Number of output features
    
    # Sequence parameters
    sequence_length: int = 20  # Lookback window
    prediction_horizon: int = 1  # Steps ahead to predict
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout: float = 0.2
    
    # Regularization
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


class TimeSeriesDataset(Dataset):
    """Dataset for time-series data"""
    
    def __init__(self, data: np.ndarray, sequence_length: int, prediction_horizon: int = 1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMModel(nn.Module):
    """LSTM model for time-series prediction"""
    
    def __init__(self, config: LSTMConfig):
        super(LSTMModel, self).__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Output layer
        self.fc = nn.Linear(config.hidden_size, config.output_size * config.prediction_horizon)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions (batch_size, prediction_horizon, output_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_output = lstm_out[:, -1, :]
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # Output layer
        output = self.fc(last_output)
        
        # Reshape for multi-step prediction
        output = output.view(-1, self.config.prediction_horizon, self.config.output_size)
        
        return output


class LSTMTimeSeriesPredictor:
    """
    LSTM Time-Series Predictor
    
    Uses LSTM networks to predict time-series data with
    temporal dependencies.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        
        # Model
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Training history
        self.training_losses: List[float] = []
        self.validation_losses: List[float] = []
        
        # Scaler
        self.scaler = None
    
    def prepare_data(self, data: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        # Separate features and target
        features = data.drop(columns=[target_col]).values
        target = data[target_col].values
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        return features, target
    
    def create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM
        
        Args:
            features: Feature array
            target: Target array
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(features) - self.config.sequence_length - self.config.prediction_horizon + 1):
            X.append(features[i:i + self.config.sequence_length])
            y.append(target[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, target_col: str, 
              validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model
        
        Args:
            data: Training data
            target_col: Target column name
            validation_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping training")
            return {}
        
        # Prepare data
        features, target = self.prepare_data(data, target_col)
        
        # Create sequences
        X, y = self.create_sequences(features, target)
        
        # Split train/validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, self.config.sequence_length, self.config.prediction_horizon)
        val_dataset = TimeSeriesDataset(X_val, self.config.sequence_length, self.config.prediction_horizon)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        self.model = LSTMModel(self.config)
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        self.criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.training_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = self.model(batch_X)
                    loss = self.criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.validation_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        return {
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1
        }
    
    def predict(self, data: pd.DataFrame, target_col: str) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            data: Input data
            target_col: Target column name
            
        Returns:
            Predictions
        """
        if not TORCH_AVAILABLE or self.model is None:
            print("Model not trained or PyTorch not available")
            return np.zeros(len(data))
        
        # Prepare data
        features, _ = self.prepare_data(data, target_col)
        
        # Create sequences
        X, _ = self.create_sequences(features, np.zeros(len(features)))
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                batch_X = torch.FloatTensor(X[i:i + self.config.batch_size])
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred.numpy())
        
        predictions = np.vstack(predictions)
        
        # Reshape for single-step prediction
        if self.config.prediction_horizon == 1:
            predictions = predictions.squeeze(-1)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get model summary"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon
        }


def simulate_time_series(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Simulate time-series data for testing"""
    np.random.seed(42)
    
    # Generate correlated features
    base_signal = np.cumsum(np.random.randn(n_samples) * 0.01)
    
    features = np.random.randn(n_samples, n_features)
    for i in range(n_features):
        features[:, i] += base_signal * 0.1
    
    # Generate target with temporal dependencies
    target = np.zeros(n_samples)
    for i in range(20, n_samples):
        target[i] = 0.3 * features[i, 0] + 0.2 * features[i, 1] + 0.1 * target[i-1] + np.random.randn() * 0.01
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    data = pd.DataFrame(features, columns=feature_names)
    data["target"] = target
    
    return data


if __name__ == "__main__":
    # Example usage
    config = LSTMConfig(
        input_size=10,
        hidden_size=32,
        num_layers=2,
        sequence_length=20,
        prediction_horizon=1,
        epochs=50,  # Reduced for testing
        batch_size=32
    )
    
    predictor = LSTMTimeSeriesPredictor(config)
    
    # Simulate data
    print("Simulating time-series data...")
    data = simulate_time_series(500, 10)
    
    # Train
    print("\nTraining LSTM model...")
    if TORCH_AVAILABLE:
        metrics = predictor.train(data, "target")
        print(f"\nTraining Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("Skipping training (PyTorch not available)")
    
    # Predict
    print("\nGenerating predictions...")
    predictions = predictor.predict(data, "target")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Mean prediction: {predictions.mean():.6f}")
    
    # Model summary
    print("\nModel Summary:")
    summary = predictor.get_model_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
