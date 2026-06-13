"""
Architecture V3 Configuration File
Quantitative Trading System for Indian Markets (NIFTY/BANKNIFTY)
Based on 8-Agent Debate Final Resolution - Production-Ready Version
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Regime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"


class Phase(Enum):
    PHASE1_RESEARCH = "phase1_research"
    PHASE2_PAPER_TRADING = "phase2_paper_trading"
    PHASE3_LIVE_SMALL = "phase3_live_small"
    PHASE4_SCALE = "phase4_scale"


@dataclass
class AlphaConfig:
    """Configuration for individual alpha strategies"""
    name: str
    expected_sharpe: float
    capacity_cr: float  # Capacity in Crores
    decay_months: int
    confidence: float
    status: str  # "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Rejected"
    priority: int  # 1 = highest


@dataclass
class GlobalConfig:
    """Global system configuration"""
    # Target AUM by phase
    phase1_aum: float = 0  # Research only
    phase2_aum: float = 0  # Paper trading
    phase3_aum: float = 5_000_000  # ₹5 Cr live
    phase4_aum: float = 100_000_000  # ₹100 Cr target
    
    # Instrument Universe
    instrument_universe: List[str] = field(default_factory=lambda: [
        "NIFTY FUTURES",
        "BANKNIFTY FUTURES",
        "NIFTY 50 STOCKS"
    ])
    top_liquid_stocks: int = 500  # CRITICAL FIX: Aligned with NIFTY500 constituents
    
    # Data Parameters
    data_frequency: str = "1-minute"
    target_latency_ms: int = 100  # 100ms end-to-end (reduced from 1000ms)
    
    # Team
    team_size: int = 6  # Increased from 3
    team_composition: Dict[str, int] = field(default_factory=lambda: {
        "senior_engineers": 2,
        "junior_engineers": 1,
        "quant_researcher": 1,
        "devops_engineer": 1,
        "risk_manager": 1
    })


@dataclass
class AlphaRankingConfig:
    """Alpha ranking table from debate resolution - Phased deployment"""
    alphas: List[AlphaConfig] = field(default_factory=lambda: [
        # Phase 1: VWAP Futures Only
        AlphaConfig(
            name="VWAP Trend (NIFTY futures)",
            expected_sharpe=0.9,
            capacity_cr=500,
            decay_months=12,
            confidence=0.70,
            status="Phase 1",
            priority=1
        ),
        # Phase 2: Add ORB
        AlphaConfig(
            name="5-min ORB (Stocks in Play)",
            expected_sharpe=0.8,
            capacity_cr=100,
            decay_months=6,
            confidence=0.60,
            status="Phase 2",
            priority=2
        ),
        # Phase 3: Tail Hedging
        AlphaConfig(
            name="Tail Hedging (Long OTM puts)",
            expected_sharpe=-0.2,  # Cost of insurance
            capacity_cr=1000,
            decay_months=0,
            confidence=0.90,
            status="Phase 3",
            priority=3
        ),
        # Phase 4: Re-evaluate
        AlphaConfig(
            name="ML Ensemble (Simplified)",
            expected_sharpe=0.3,
            capacity_cr=200,
            decay_months=12,
            confidence=0.30,
            status="Phase 4",
            priority=4
        ),
        # Rejected per debate
        AlphaConfig(
            name="Put-Call Carry",
            expected_sharpe=0.0,
            capacity_cr=0,
            decay_months=0,
            confidence=0.0,
            status="Rejected",
            priority=99
        ),
        AlphaConfig(
            name="Volatility Carry (Short straddle)",
            expected_sharpe=0.0,
            capacity_cr=0,
            decay_months=0,
            confidence=0.0,
            status="Rejected",
            priority=99
        ),
    ])


@dataclass
class AlphaCombinationConfig:
    """Alpha combination engine configuration - Simplified"""
    method: str = "risk_parity_simple"  # Simplified from risk_parity_kelly
    kelly_fraction: float = 0.10  # Reduced from 0.15 for safety
    
    # Regime-based weights - Simplified
    regime_weights: Dict[Regime, Dict[str, float]] = field(default_factory=lambda: {
        Regime.BULL_TREND: {
            "VWAP": 0.70,
            "ORB": 0.30,
            "TailHedge": 0.00
        },
        Regime.BEAR_TREND: {
            "VWAP": 0.50,
            "ORB": 0.20,
            "TailHedge": 0.30
        },
        Regime.SIDEWAYS: {
            "VWAP": 0.40,
            "ORB": 0.40,
            "TailHedge": 0.20
        },
        Regime.HIGH_VOL: {
            "VWAP": 0.30,
            "ORB": 0.20,
            "TailHedge": 0.50
        }
    })
    
    rebalance_frequency: str = "daily"
    correlation_penalty: bool = True
    correlation_threshold: float = 0.5


@dataclass
class RegimeEngineConfig:
    """Regime Detection Engine configuration - Simplified"""
    algorithm: str = "RULE_BASED"  # Changed from HMM (overfit)
    n_states: int = 4
    states: List[str] = field(default_factory=lambda: ["bull_trend", "bear_trend", "sideways", "high_vol"])
    
    # Features for regime detection
    features: List[str] = field(default_factory=lambda: [
        "realized_vol_5d",
        "nifty_return_5d",
        "vix"
    ])
    
    training_window_days: int = 0  # Not needed for rule-based
    retraining_frequency: str = "never"  # Rule-based doesn't need retraining
    change_point_detection: str = "CUSUM"
    change_point_window_minutes: int = 10


@dataclass
class PortfolioConfig:
    """Portfolio construction engine configuration - Conservative"""
    method: str = "risk_parity"
    optimizer: str = "SLSQP"
    
    # Constraints - Stricter
    max_single_strategy_weight: float = 0.70  # Increased from 0.50 for Phase 1
    max_sector_weight: float = 0.30
    max_leverage: float = 1.0  # Reduced from 4.0 for safety
    max_position_size_pct: float = 0.02  # Reduced from 0.05 (2% of AUM)
    
    # Liquidity constraints
    max_participation_rate: float = 0.05  # Reduced from 0.10 (5% of ADV)
    min_adv_cr: float = 10.0  # Minimum ₹10 Cr ADV
    
    # Objective
    target_volatility: float = 0.12  # Reduced from 0.15 (12% annual)
    
    # Rebalancing
    rebalance_time: str = "market_open"


@dataclass
class RiskEngineConfig:
    """Risk engine configuration - Enhanced"""
    # Pre-trade checks - Stricter
    max_position_size_pct: float = 0.02  # Reduced from 0.05
    max_sector_exposure_pct: float = 0.25  # Reduced from 0.30
    var_99_1day_cap_pct: float = 0.015  # Reduced from 0.02 (1.5% of AUM)
    correlation_heat_threshold: float = 0.6  # Reduced from 0.7
    
    # Intraday controls - Enhanced
    trailing_stop_atr_pct: float = 0.15  # Increased from 0.10 (wider stops)
    daily_circuit_breaker_pct: float = -0.02  # Tightened from -0.03 (-2% daily PnL)
    leverage_warning_threshold: float = 0.8  # Reduced from 3.0
    leverage_hard_stop: float = 1.0  # Reduced from 4.0
    
    # Post-trade
    kelly_adjustment_frequency: str = "quarterly"  # Changed from monthly
    
    # Tail risk - Enhanced
    vix_threshold: float = 15.0  # Increased from 12.0 (earlier hedging)
    tail_hedge_cost_pct_aum: float = 0.01  # 1% of AUM/year
    tail_hedge_notional_pct: float = 0.01  # 1% of AUM in OTM puts
    
    # Stress testing - New
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "2008_financial_crisis",
        "covid_2020",
        "demonetization_2016",
        "flash_crash_2010",
        "custom_correlation_breakdown"
    ])
    stress_test_frequency: str = "weekly"


@dataclass
class ExecutionConfig:
    """Execution engine configuration - Realistic"""
    # VWAP execution
    vwap_slicing: bool = True
    vwap_participation_rate: float = 0.05  # Reduced from 0.08-0.10
    
    # Order types
    limit_order_patience_bps_min: float = 1.0  # Increased from 0.5
    limit_order_patience_bps_max: float = 5.0  # Increased from 2.0
    
    # Stop loss - Wider
    stop_loss_atr_pct: float = 0.15  # Increased from 0.10
    
    # Slippage model - Realistic
    slippage_large_cap_bps: float = 10.0  # Increased from 2.0
    slippage_mid_cap_bps: float = 20.0  # Increased from 5.0
    slippage_futures_bps: float = 5.0  # New
    market_impact_bps: float = 5.0  # New
    
    # Bid-ask spread - New
    bid_ask_spread_cost_bps: float = 2.0  # New
    
    # Order book - New
    use_order_book_imbalance: bool = True
    order_book_depth_threshold: float = 0.5  # 50% of average depth


@dataclass
class FeatureConfig:
    """Feature pipeline configuration - Simplified"""
    n_features: int = 10  # Reduced from 50
    feature_selection_method: str = "manual"  # Changed from Boruta
    
    # Core features - Reduced
    core_features: List[str] = field(default_factory=lambda: [
        "relative_volume",
        "vwap_distance",
        "realized_volatility_5d",
        "atr",
        "rsi",
        "day_of_week",
        "time_of_day",
        "gap_pct",
        "bid_ask_spread",
        "order_flow_imbalance"
    ])
    
    # Rolling retraining - Less frequent
    rolling_retrain_frequency_days: int = 30  # Changed from 7 (monthly)
    online_learning: bool = False
    
    # Feature validation - New
    future_info_detection: bool = True
    leakage_detection: bool = True
    decay_detection: bool = True
    psi_threshold: float = 0.1


@dataclass
class MLConfig:
    """ML configuration - Simplified"""
    # Model
    ml_framework: str = "LightGBM"
    model_depth: int = 3  # Shallow trees for speed
    n_estimators: int = 100  # Reduced from default
    learning_rate: float = 0.1
    
    # Training
    training_window_days: int = 252  # 1 year
    time_series_cv: str = "purged_kfold"
    n_folds: int = 5
    purge_days: int = 5
    
    # Inference
    inference_language: str = "C++"  # Via ONNX
    target_inference_ms: float = 10.0  # 10ms
    
    # Monitoring
    shap_monitoring: bool = True
    feature_importance_drift_threshold: float = 0.3
    
    # Status
    enabled: bool = False  # Disabled until Phase 4


@dataclass
class DatabaseConfig:
    """Database architecture configuration - Simplified"""
    # Redis (hot cache)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_cache_duration_hours: int = 24
    
    # PostgreSQL (metadata + positions + trades)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "quant_trading"
    
    # TimescaleDB (time-series data)
    timescale_host: str = "localhost"
    timescale_port: int = 5432
    timescale_database: str = "quant_timeseries"
    
    # S3 (archive)
    s3_bucket: str = "quant-archive"
    s3_region: str = "ap-south-1"
    
    # Removed: Arctic, ClickHouse, Parquet local


@dataclass
class TechStackConfig:
    """Technology stack configuration - Production-focused"""
    # Languages
    python_version: str = "3.11"
    cpp_version: str = "17"
    
    # Core libraries
    use_polars: bool = True
    use_numpy: bool = True
    use_numba: bool = True
    
    # ML
    ml_framework: str = "LightGBM"
    use_onnx: bool = True  # For C++ inference
    use_shap: bool = True
    
    # API
    api_framework: str = "FastAPI"
    use_websockets: bool = True
    
    # Infrastructure
    cloud_provider: str = "AWS"  # Managed services
    use_docker: bool = True
    use_kubernetes: bool = False  # Phase 4 only
    
    # Monitoring
    use_prometheus: bool = True
    use_grafana: bool = True
    use_alertmanager: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration - Enhanced"""
    # Metrics
    use_prometheus: bool = True
    prometheus_port: int = 9090
    
    # Visualization
    use_grafana: bool = True
    grafana_port: int = 3000
    
    # Alerting
    alert_on_latency_spike: bool = True
    alert_on_circuit_breaker: bool = True
    alert_on_drawdown: bool = True
    alert_on_var_breach: bool = True
    alert_slack_channel: str = "quant-alerts"
    alert_email: str = "team@quantfund.com"
    
    # Real-time monitoring - New
    real_time_pnl_monitoring: bool = True
    pnl_monitoring_interval_seconds: int = 60
    real_time_risk_monitoring: bool = True
    risk_monitoring_interval_seconds: int = 60
    
    # Latency monitoring - New
    latency_monitoring: bool = True
    latency_threshold_ms: float = 100.0
    
    # Health checks - New
    health_check_interval_seconds: int = 30
    component_health_checks: List[str] = field(default_factory=lambda: [
        "data_feed",
        "database",
        "execution_engine",
        "risk_engine"
    ])


@dataclass
class OperationalConfig:
    """Operational configuration - New"""
    # Disaster recovery
    disaster_recovery_enabled: bool = True
    dr_region: str = "ap-south-2"
    rto_hours: int = 1  # Recovery Time Objective
    rpo_minutes: int = 5  # Recovery Point Objective
    
    # Backup
    backup_frequency: str = "daily"
    backup_retention_days: int = 90
    
    # Incident response
    incident_response_plan: bool = True
    on_call_rotation: bool = True
    incident_slack_channel: str = "quant-incidents"
    
    # Compliance
    sebi_registration: bool = True
    algo_trading_registration: bool = True
    audit_trail_enabled: bool = True
    
    # Investor reporting
    investor_reporting_frequency: str = "monthly"
    performance_attribution: bool = True
    risk_reporting: bool = True


@dataclass
class ResearchRoadmapConfig:
    """Research roadmap configuration - Realistic timeline"""
    # Phase 1: Research (Months 1-6)
    phase1_data_ingestion: bool = True
    phase1_backtest_reproduction: bool = True
    phase1_feature_pipeline: bool = True
    phase1_vwap_strategy: bool = True
    phase1_target_sharpe: float = 0.8
    phase1_duration_months: int = 6
    
    # Phase 2: Paper Trading (Months 7-12)
    phase2_paper_trading: bool = True
    phase2_vwap_live: bool = True
    phase2_target_sharpe: float = 0.8
    phase2_max_drawdown: float = 0.10
    phase2_duration_months: int = 6
    
    # Phase 3: Live Trading - Small (Months 13-18)
    phase3_live_initial_aum: float = 5_000_000  # ₹5 Cr
    phase3_orb_strategy: bool = True
    phase3_tail_hedging: bool = True
    phase3_target_sharpe: float = 1.0
    phase3_max_drawdown: float = 0.12
    phase3_duration_months: int = 6
    
    # Phase 4: Scale (Months 19-24)
    phase4_target_aum: float = 100_000_000  # ₹100 Cr
    phase4_ml_ensemble: bool = False  # Re-evaluate
    phase4_options_strategies: bool = False  # Re-evaluate
    phase4_target_sharpe: float = 1.2
    phase4_max_drawdown: float = 0.15
    phase4_duration_months: int = 6
    
    # Total timeline
    total_duration_months: int = 24


@dataclass
class BudgetConfig:
    """Budget configuration - New"""
    # Annual budget in USD
    technology_budget: float = 100_000  # $100K/year
    data_budget: float = 150_000  # $150K/year
    personnel_budget: float = 800_000  # $800K/year
    total_budget: float = 1_050_000  # $1.05M/year
    
    # As percentage of AUM
    budget_pct_of_aum_100m: float = 0.0105  # 1.05% of $100M


@dataclass
class ArchitectureV3Config:
    """Main configuration class for Architecture V3"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    alpha_ranking: AlphaRankingConfig = field(default_factory=AlphaRankingConfig)
    alpha_combination: AlphaCombinationConfig = field(default_factory=AlphaCombinationConfig)
    regime_engine: RegimeEngineConfig = field(default_factory=RegimeEngineConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk_engine: RiskEngineConfig = field(default_factory=RiskEngineConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tech_stack: TechStackConfig = field(default_factory=TechStackConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    operational: OperationalConfig = field(default_factory=OperationalConfig)
    roadmap: ResearchRoadmapConfig = field(default_factory=ResearchRoadmapConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)


# Singleton instance
config = ArchitectureV3Config()
