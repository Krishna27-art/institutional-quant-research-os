-- Dashboard Database Schema
-- This schema defines the tables needed for the Quant Research OS dashboard

-- Market Data Table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    value DECIMAL(15, 2),
    change DECIMAL(15, 2),
    change_pct DECIMAL(10, 4),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

-- Regime Data Table
CREATE TABLE IF NOT EXISTS regime_data (
    id SERIAL PRIMARY KEY,
    regime_name VARCHAR(50) NOT NULL,
    state INTEGER NOT NULL,
    confidence DECIMAL(5, 2),
    duration_days INTEGER,
    transition_prob DECIMAL(5, 2),
    regime_sharpe DECIMAL(10, 4),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alpha Performance Table
CREATE TABLE IF NOT EXISTS alpha_performance (
    id SERIAL PRIMARY KEY,
    alpha_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL, -- live, paper, research
    sharpe DECIMAL(10, 4),
    decay_days INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk Metrics Table
CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    var DECIMAL(15, 2), -- Value at Risk
    cvar DECIMAL(15, 2), -- Conditional Value at Risk
    gross_exposure DECIMAL(10, 2),
    max_drawdown DECIMAL(10, 4),
    portfolio_heat DECIMAL(10, 4),
    leverage DECIMAL(10, 2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio Data Table
CREATE TABLE IF NOT EXISTS portfolio_data (
    id SERIAL PRIMARY KEY,
    aum DECIMAL(20, 2),
    daily_pnl DECIMAL(20, 2),
    mtd_pnl DECIMAL(20, 2),
    net_exposure DECIMAL(10, 2),
    long_exposure DECIMAL(10, 2),
    short_exposure DECIMAL(10, 2),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Options Data Table
CREATE TABLE IF NOT EXISTS options_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    atm_iv DECIMAL(10, 4),
    iv_rank INTEGER,
    pcr_oi DECIMAL(10, 4),
    max_pain DECIMAL(15, 2),
    expiry_date DATE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signals Table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- LONG, SHORT
    alpha_name VARCHAR(100) NOT NULL,
    strength DECIMAL(5, 4),
    time TIME,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL, -- critical, warning, info
    time VARCHAR(50),
    source VARCHAR(50),
    status VARCHAR(20), -- UNACKNOWLEDGED, ACKNOWLEDGED, RESOLVED
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Holdings Table
CREATE TABLE IF NOT EXISTS holdings (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(15, 2),
    ltp DECIMAL(15, 2),
    pnl DECIMAL(20, 2),
    pct_aum DECIMAL(10, 4),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sector Exposure Table
CREATE TABLE IF NOT EXISTS sector_exposure (
    id SERIAL PRIMARY KEY,
    sector_name VARCHAR(50) NOT NULL,
    exposure DECIMAL(10, 4),
    color VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Status Table
CREATE TABLE IF NOT EXISTS system_status (
    id SERIAL PRIMARY KEY,
    component_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- OPERATIONAL, DEGRADED, DOWN
    latency VARCHAR(20),
    uptime_pct DECIMAL(10, 4),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol);
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_regime_data_timestamp ON regime_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_alpha_performance_name ON alpha_performance(alpha_name);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_portfolio_data_timestamp ON portfolio_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_system_status_timestamp ON system_status(timestamp);
