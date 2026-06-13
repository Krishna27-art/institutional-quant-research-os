"""
Centralized configuration management for the institutional quant research platform.

Provides:
- Single source of truth for all configuration parameters
- Validation of configuration values
- Environment-specific settings (dev, staging, prod)
- Type-safe configuration access
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path


class Environment(Enum):
    """Environment types."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


@dataclass
class RiskLimits:
    """Risk management limits."""
    # VaR limits
    var_confidence: float = 0.99
    var_cap_pct: float = 0.02  # 2% of AUM
    cvar_confidence: float = 0.95
    
    # Position limits
    max_position_pct: float = 0.05  # 5% per position
    max_sector_pct: float = 0.30  # 30% per sector
    max_strategy_weight: float = 0.50  # 50% max single strategy
    
    # Risk per trade
    risk_per_trade: float = 0.005  # 0.5% of AUM
    risk_per_strategy: float = 0.05  # 5% of AUM
    total_portfolio_risk: float = 0.15  # 15% of AUM
    
    # Circuit breakers
    max_daily_loss_pct: float = 0.03  # 3% daily
    max_weekly_loss_pct: float = 0.08  # 8% weekly
    max_drawdown_from_peak_pct: float = 0.10  # 10% from peak
    
    # Leverage
    max_leverage: float = 1.0  # Reduced from 4x for safety
    warn_leverage: float = 3.0
    
    # Volatility targeting
    risk_target: float = 0.15  # 15% annual vol
    
    def validate(self) -> None:
        """Validate risk limits are within reasonable bounds."""
        if not 0 < self.var_confidence < 1:
            raise ValueError(f"var_confidence must be between 0 and 1, got {self.var_confidence}")
        if not 0 < self.var_cap_pct < 0.10:
            raise ValueError(f"var_cap_pct must be < 10%, got {self.var_cap_pct}")
        if not 0 < self.max_position_pct < 0.20:
            raise ValueError(f"max_position_pct must be < 20%, got {self.max_position_pct}")
        if not 0 < self.max_leverage <= 5.0:
            raise ValueError(f"max_leverage must be <= 5x, got {self.max_leverage}")


@dataclass
class TradingParameters:
    """Trading execution parameters."""
    # Order sizing
    min_order_size: float = 100000  # ₹1 lakh minimum
    max_order_size: float = 50000000  # ₹5 crore maximum
    
    # Slippage
    default_slippage_bps: float = 5.0  # 5 bps default
    
    # Execution
    order_timeout_seconds: int = 30
    max_retries: int = 3
    
    # Market hours (NSE)
    market_open: str = "09:15"
    market_close: str = "15:30"
    
    # Lot sizes (NSE F&O)
    lot_sizes: Dict[str, int] = field(default_factory=lambda: {
        'NIFTY': 50,
        'BANKNIFTY': 15,
        'FINNIFTY': 40,
    })
    
    def validate(self) -> None:
        """Validate trading parameters."""
        if self.min_order_size <= 0:
            raise ValueError(f"min_order_size must be > 0, got {self.min_order_size}")
        if self.max_order_size <= self.min_order_size:
            raise ValueError(f"max_order_size must be > min_order_size")


@dataclass
class DataParameters:
    """Data management parameters."""
    # Market data
    data_feed: str = "mock"  # mock, websocket, api
    tick_aggregation: str = "1min"  # 1min, 5min, 1h
    
    # Historical data
    min_history_days: int = 252  # 1 year
    max_history_days: int = 1260  # 5 years
    
    # Data quality
    price_min: float = 0.1
    price_max: float = 100000
    volume_min: int = 100
    
    # Corporate actions
    corporate_actions_source: str = "nse"  # nse, vendor
    
    def validate(self) -> None:
        """Validate data parameters."""
        if self.min_history_days < 30:
            raise ValueError(f"min_history_days must be >= 30, got {self.min_history_days}")


@dataclass
class BacktestParameters:
    """Backtesting parameters."""
    # Simulation
    initial_capital: float = 250_000_000  # ₹25 Crore
    commission_per_trade: float = 20.0  # ₹20 per trade
    
    # Slippage model
    slippage_model: str = "volume_aware"  # fixed, volume_aware, none
    
    # Fills
    fill_model: str = "close"  # close, open, vwap, realistic
    
    # Warmup period
    warmup_bars: int = 50
    
    # Walk-forward
    train_test_split: float = 0.7  # 70% train, 30% test
    purge_window: int = 10  # bars to purge between train/test
    
    def validate(self) -> None:
        """Validate backtest parameters."""
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {self.initial_capital}")
        if not 0 < self.train_test_split < 1:
            raise ValueError(f"train_test_split must be between 0 and 1, got {self.train_test_split}")


@dataclass
class MonitoringParameters:
    """Monitoring and alerting parameters."""
    # Metrics
    enable_prometheus: bool = False
    prometheus_port: int = 8000
    
    # Alerts
    alert_on_var_exceeded: bool = True
    alert_on_drawdown: bool = True
    alert_on_order_failure: bool = True
    
    # Dashboard
    dashboard_port: int = 8080
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/platform.log"
    
    def validate(self) -> None:
        """Validate monitoring parameters."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level}")


@dataclass
class Settings:
    """Main settings container."""
    environment: Environment = Environment.DEV
    
    # Sub-settings
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    trading: TradingParameters = field(default_factory=TradingParameters)
    data: DataParameters = field(default_factory=DataParameters)
    backtest: BacktestParameters = field(default_factory=BacktestParameters)
    monitoring: MonitoringParameters = field(default_factory=MonitoringParameters)
    
    # Paths
    base_path: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    def __post_init__(self):
        """Initialize settings from environment variables."""
        # Override with environment variables if set
        env = os.environ.get("PLATFORM_ENV", "dev")
        self.environment = Environment(env)
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
    
    def validate_all(self) -> None:
        """Validate all settings."""
        self.risk_limits.validate()
        self.trading.validate()
        self.data.validate()
        self.backtest.validate()
        self.monitoring.validate()
    
    def to_dict(self) -> Dict:
        """Convert settings to dictionary."""
        return {
            'environment': self.environment.value,
            'risk_limits': self.risk_limits.__dict__,
            'trading': self.trading.__dict__,
            'data': self.data.__dict__,
            'backtest': self.backtest.__dict__,
            'monitoring': self.monitoring.__dict__,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_all()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
