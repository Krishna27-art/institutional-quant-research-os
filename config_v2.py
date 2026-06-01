"""
Architecture V2 Configuration File
Quantitative Trading System for Indian Markets (NIFTY/BANKNIFTY)
Based on 8-Agent Debate Final Resolution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Regime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"


@dataclass
class AlphaConfig:
    """Configuration for individual alpha strategies"""
    name: str
    expected_sharpe: float
    capacity_cr: float  # Capacity in Crores
    decay_months: int
    confidence: float
    status: str  # "Must Build", "Worth Testing", "Low Priority", "Reject"


@dataclass
class GlobalConfig:
    """Global system configuration"""
    # Target AUM
    aum_target: float = 250_000_000  # ₹25 Crore initial live
    aum_final: float = 1_000_000_000  # ₹100 Crore target
    
    # Instrument Universe
    instrument_universe: List[str] = field(default_factory=lambda: [
        "NIFTY 50",
        "BANKNIFTY",
        "FINNIFTY",
        "MIDCPNIFTY"
    ])
    top_liquid_stocks: int = 100
    
    # Data Parameters
    data_frequency: str = "1-minute"
    target_latency_ms: int = 1000  # 1 second end-to-end
    
    # Team
    team_size: int = 3  # engineers + 1 quant researcher


@dataclass
class AlphaRankingConfig:
    """Alpha ranking table from debate resolution"""
    alphas: List[AlphaConfig] = field(default_factory=lambda: [
        AlphaConfig(
            name="5-min ORB (Stocks in Play)",
            expected_sharpe=1.1,
            capacity_cr=100,
            decay_months=6,
            confidence=0.70,
            status="Must Build"
        ),
        AlphaConfig(
            name="VWAP Trend (NIFTY futures)",
            expected_sharpe=0.9,
            capacity_cr=500,
            decay_months=12,
            confidence=0.60,
            status="Must Build"
        ),
        AlphaConfig(
            name="Put-Call Carry (Weekly options)",
            expected_sharpe=0.7,
            capacity_cr=200,
            decay_months=24,
            confidence=0.75,
            status="Must Build"
        ),
        AlphaConfig(
            name="Volatility Carry (Short straddle)",
            expected_sharpe=0.6,
            capacity_cr=150,
            decay_months=18,
            confidence=0.65,
            status="Worth Testing"
        ),
        AlphaConfig(
            name="GCN (Game-theoretic stock)",
            expected_sharpe=0.5,
            capacity_cr=50,
            decay_months=0,
            confidence=0.40,
            status="Low Priority"
        ),
        AlphaConfig(
            name="LSTM / Transformer",
            expected_sharpe=0.3,
            capacity_cr=0,
            decay_months=1,
            confidence=0.20,
            status="Reject"
        ),
    ])


@dataclass
class AlphaCombinationConfig:
    """Alpha combination engine configuration"""
    method: str = "risk_parity_kelly"
    kelly_fraction: float = 0.15  # 15% of optimal Kelly
    
    # Regime-based weights
    regime_weights: Dict[Regime, Dict[str, float]] = field(default_factory=lambda: {
        Regime.BULL_TREND: {
            "ORB": 0.40,
            "VWAP": 0.30,
            "PCP": 0.15,
            "VolCarry": 0.10,
            "Others": 0.05
        },
        Regime.BEAR_TREND: {
            "ORB": 0.20,
            "VWAP": 0.40,
            "PCP": 0.20,
            "VolCarry": 0.15,
            "Others": 0.05
        },
        Regime.SIDEWAYS: {
            "ORB": 0.10,
            "VWAP": 0.10,
            "PCP": 0.30,
            "VolCarry": 0.40,
            "Others": 0.10
        },
        Regime.HIGH_VOL: {
            "ORB": 0.15,
            "VWAP": 0.15,
            "PCP": 0.20,
            "VolCarry": 0.40,
            "Others": 0.10
        }
    })
    
    rebalance_frequency: str = "daily"
    correlation_penalty: bool = True
    correlation_threshold: float = 0.5


@dataclass
class RegimeEngineConfig:
    """HMM Regime Detection Engine configuration"""
    algorithm: str = "HMM"
    n_states: int = 4
    states: List[str] = field(default_factory=lambda: ["bull_trend", "bear_trend", "sideways", "high_vol"])
    
    # Features for regime detection
    features: List[str] = field(default_factory=lambda: [
        "realized_vol_5d",
        "implied_vol",
        "nifty_return_5d",
        "turnover_ratio_5d"
    ])
    
    training_window_days: int = 252  # 1 year
    retraining_frequency: str = "daily"
    change_point_detection: str = "CUSUM"
    change_point_window_minutes: int = 10


@dataclass
class PortfolioConfig:
    """Portfolio construction engine configuration"""
    method: str = "risk_parity"
    optimizer: str = "SLSQP"
    
    # Constraints
    max_single_strategy_weight: float = 0.50
    max_sector_weight: float = 0.30
    max_leverage: float = 4.0
    max_position_size_pct: float = 0.05  # 5% of AUM
    
    # Objective
    target_volatility: float = 0.15  # 15% annual
    
    # Rebalancing
    rebalance_time: str = "market_open"


@dataclass
class RiskEngineConfig:
    """Risk engine configuration"""
    # Pre-trade checks
    max_position_size_pct: float = 0.05  # 5% of AUM
    max_sector_exposure_pct: float = 0.30  # 30% of AUM
    var_99_1day_cap_pct: float = 0.02  # 2% of AUM
    correlation_heat_threshold: float = 0.7
    
    # Intraday controls
    trailing_stop_atr_pct: float = 0.10  # 10% ATR
    daily_circuit_breaker_pct: float = -0.03  # -3% daily PnL
    leverage_warning_threshold: float = 3.0
    leverage_hard_stop: float = 4.0
    
    # Post-trade
    kelly_adjustment_frequency: str = "monthly"
    
    # Tail risk
    vix_threshold: float = 12.0
    tail_hedge_cost_pct_aum: float = 0.01  # 1% of AUM/year


@dataclass
class ExecutionConfig:
    """Execution engine configuration"""
    # VWAP execution
    vwap_slicing: bool = True
    
    # Order types
    limit_order_patience_bps_min: float = 0.5
    limit_order_patience_bps_max: float = 2.0
    
    # Stop loss
    stop_loss_atr_pct: float = 0.10  # 10% ATR
    
    # Slippage model (conservative)
    slippage_large_cap_bps: float = 2.0
    slippage_mid_cap_bps: float = 5.0
    reject_small_caps: bool = True


@dataclass
class FeatureConfig:
    """Feature pipeline configuration"""
    n_features: int = 50  # Target after selection
    feature_selection_method: str = "Boruta"
    
    # Core features
    core_features: List[str] = field(default_factory=lambda: [
        "relative_volume",
        "vwap_distance",
        "realized_volatility",
        "implied_volatility",
        "put_call_ratio",
        "fii_dii_flow",
        "day_of_week",
        "time_of_day",
        "atr",
        "rsi",
        "macd",
        "bollinger_band_width",
        "volume_profile",
        "tick_volume_ratio",
        "bid_ask_spread",
        "order_flow_imbalance",
        "momentum_5d",
        "momentum_20d",
        "volatility_5d",
        "volatility_20d"
    ])
    
    # Rolling retraining
    rolling_retrain_frequency_days: int = 5
    online_learning: bool = False  # Disabled per debate


@dataclass
class DatabaseConfig:
    """Database architecture configuration"""
    # Redis (hot cache)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_cache_duration_hours: int = 24
    
    # ClickHouse (analytics)
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "quant_trading"
    
    # PostgreSQL (metadata)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "quant_metadata"
    
    # Parquet (archive)
    parquet_archive_path: str = "./data/archive"


@dataclass
class TechStackConfig:
    """Technology stack configuration"""
    # Languages
    python_version: str = "3.11"
    go_version: str = "1.21"
    
    # Core libraries
    use_polars: bool = True
    use_numpy: bool = True
    use_numba: bool = True
    
    # ML
    ml_framework: str = "LightGBM"
    use_shap: bool = True
    
    # API
    api_framework: str = "FastAPI"
    use_websockets: bool = True
    
    # Orchestration
    use_docker_compose: bool = True
    use_kubernetes: bool = False  # Phase 2 only


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    # Metrics
    use_prometheus: bool = True
    prometheus_port: int = 9090
    
    # Visualization
    use_grafana: bool = True
    grafana_port: int = 3000
    
    # Alerting
    alert_on_latency_spike: bool = True
    alert_on_circuit_breaker: bool = True
    alert_slack_channel: Optional[str] = None


@dataclass
class ResearchRoadmapConfig:
    """Research roadmap configuration"""
    # Phase 1: Research (Months 1-2)
    phase1_data_ingestion: bool = True
    phase1_backtest_reproduction: bool = True
    phase1_feature_pipeline: bool = True
    
    # Phase 2: Model Training (Months 3-4)
    phase2_lightgbm_training: bool = True
    phase2_hmm_regime: bool = True
    phase2_portfolio_construction: bool = True
    
    # Phase 3: Paper Trading (Months 5-6)
    phase3_paper_trading: bool = True
    phase3_target_sharpe: float = 1.0
    phase3_max_drawdown: float = 0.12
    
    # Phase 4: Live Trading (Months 7-9)
    phase4_live_initial_aum: float = 5_000_000  # ₹5 Cr
    phase4_monitoring: bool = True
    
    # Phase 5: Scale (Months 10-12)
    phase5_target_aum: float = 25_000_000  # ₹25 Cr
    phase5_weekly_options: bool = True
    phase5_cpp_migration: bool = False  # Future


@dataclass
class ArchitectureV2Config:
    """Main configuration class for Architecture V2"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    alpha_ranking: AlphaRankingConfig = field(default_factory=AlphaRankingConfig)
    alpha_combination: AlphaCombinationConfig = field(default_factory=AlphaCombinationConfig)
    regime_engine: RegimeEngineConfig = field(default_factory=RegimeEngineConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk_engine: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tech_stack: TechStackConfig = field(default_factory=TechStackConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    roadmap: ResearchRoadmapConfig = field(default_factory=ResearchRoadmapConfig)


# Singleton instance
config = ArchitectureV2Config()
