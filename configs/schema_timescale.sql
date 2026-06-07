-- TimescaleDB Schema for Time-Series Data
-- Phase 10 Implementation

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1-minute OHLCV bars
CREATE TABLE IF NOT EXISTS bars_1min (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC DEFAULT 0,
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('bars_1min', 'time', partitioning_column => 'symbol', number_partitions => 64)
IF NOT EXISTS;

-- Add retention policy (3 years)
SELECT add_retention_policy('bars_1min', INTERVAL '3 years')
IF NOT EXISTS;

-- 5-minute OHLCV bars (continuous aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM bars_1min
GROUP BY time_bucket('5 minutes', time), symbol;

-- 15-minute OHLCV bars (continuous aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_15min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 minutes', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM bars_1min
GROUP BY time_bucket('15 minutes', time), symbol;

-- 1-hour OHLCV bars (continuous aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1hour
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM bars_1min
GROUP BY time_bucket('1 hour', time), symbol;

-- 1-day OHLCV bars (continuous aggregate)
CREATE MATERIALIZED VIEW IF NOT EXISTS bars_1day
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS time,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume
FROM bars_1min
GROUP BY time_bucket('1 day', time), symbol;

-- Features time series
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value NUMERIC NOT NULL,
    version TEXT DEFAULT 'v1',
    PRIMARY KEY (time, symbol, feature_name)
);

SELECT create_hypertable('features', 'time', partitioning_column => 'symbol', number_partitions => 32)
IF NOT EXISTS;

-- Trade records
CREATE TABLE IF NOT EXISTS trades (
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT CHECK (side IN ('buy', 'sell')),
    quantity NUMERIC NOT NULL,
    price NUMERIC NOT NULL,
    commission NUMERIC DEFAULT 0,
    order_id TEXT,
    strategy_id UUID,
    PRIMARY KEY (time, symbol, trade_id)
);

SELECT create_hypertable('trades', 'time', partitioning_column => 'symbol', number_partitions => 16)
IF NOT EXISTS;

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    quantity NUMERIC NOT NULL,
    avg_cost NUMERIC NOT NULL,
    market_value NUMERIC,
    unrealized_pnl NUMERIC,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('positions', 'time', partitioning_column => 'symbol', number_partitions => 16)
IF NOT EXISTS;

-- Regime history
CREATE TABLE IF NOT EXISTS regime_history (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    regime TEXT NOT NULL,
    probability JSONB DEFAULT '{}',
    hmm_state INTEGER,
    change_point_detected BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('regime_history', 'time')
IF NOT EXISTS;

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    portfolio_value NUMERIC NOT NULL,
    daily_return NUMERIC,
    cumulative_return NUMERIC,
    sharpe_ratio NUMERIC,
    max_drawdown NUMERIC,
    volatility NUMERIC,
    PRIMARY KEY (time)
);

SELECT create_hypertable('performance_metrics', 'time')
IF NOT EXISTS;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_bars_symbol ON bars_1min(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_symbol_feature ON features(symbol, feature_name, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_regime_regime ON regime_history(regime, time DESC);
