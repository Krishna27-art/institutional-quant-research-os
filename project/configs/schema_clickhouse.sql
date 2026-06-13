-- ClickHouse Schema for Analytics and Backtest Results
-- Phase 10 Implementation

-- Backtest results (billions of rows support)
CREATE TABLE IF NOT EXISTS backtest_results (
    run_id UUID,
    strategy_id UUID,
    symbol String,
    timestamp DateTime,
    position Float64,
    pnl Float64,
    return Float64,
    regime String,
    INDEX idx_strategy (strategy_id) TYPE minmax GRANULARITY 4,
    INDEX idx_symbol (symbol) TYPE set(100) GRANULARITY 4,
    INDEX idx_regime (regime) TYPE set(10) GRANULARITY 4
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (run_id, strategy_id, symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Feature importance over time
CREATE TABLE IF NOT EXISTS feature_importance (
    model_id UUID,
    feature_name String,
    importance Float64,
    timestamp DateTime,
    regime String,
    INDEX idx_model (model_id) TYPE minmax GRANULARITY 4,
    INDEX idx_feature (feature_name) TYPE set(1000) GRANULARITY 4
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (model_id, feature_name, timestamp)
SETTINGS index_granularity = 8192;

-- Aggregated performance metrics
CREATE TABLE IF NOT EXISTS aggregated_performance (
    strategy_id UUID,
    date Date,
    sharpe_ratio Float64,
    hit_rate Float64,
    max_drawdown Float64,
    total_return Float64,
    volatility Float64,
    turnover Float64,
    regime String,
    INDEX idx_strategy (strategy_id) TYPE minmax GRANULARITY 4
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (strategy_id, date)
SETTINGS index_granularity = 8192;

-- Alpha decay tracking
CREATE TABLE IF NOT EXISTS alpha_decay (
    alpha_id UUID,
    timestamp DateTime,
    sharpe_21d Float64,
    sharpe_63d Float64,
    percentile_21d Float64,
    decay_status String,
    INDEX idx_alpha (alpha_id) TYPE minmax GRANULARITY 4
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (alpha_id, timestamp)
SETTINGS index_granularity = 8192;

-- Risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    timestamp DateTime,
    portfolio_id UUID,
    var_1d Float64,
    var_5d Float64,
    cvar_1d Float64,
    beta Float64,
    correlation_heat Float64,
    leverage Float64,
    exposure_sector String,
    exposure_value Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, portfolio_id)
SETTINGS index_granularity = 8192;

-- Execution analytics
CREATE TABLE IF NOT EXISTS execution_analytics (
    order_id UUID,
    symbol String,
    timestamp DateTime,
    side String,
    quantity Float64,
    price Float64,
    fill_price Float64,
    slippage_bps Float64,
    market_impact_bps Float64,
    fill_time_ms Float64,
    broker String,
    INDEX idx_symbol (symbol) TYPE set(1000) GRANULARITY 4,
    INDEX idx_broker (broker) TYPE set(10) GRANULARITY 4
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, symbol)
SETTINGS index_granularity = 8192;
