-- PostgreSQL Schema for Registry and Metadata
-- Phase 10 Implementation

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Feature Registry
CREATE TABLE IF NOT EXISTS feature_registry (
    feature_id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    formula TEXT,
    version INTEGER DEFAULT 1,
    category TEXT CHECK (category IN ('price', 'volume', 'volatility', 'breadth', 'sector', 'options', 'cross_asset', 'sentiment')),
    parameters JSONB DEFAULT '{}',
    dependencies TEXT[] DEFAULT '{}',
    update_frequency TEXT DEFAULT '1min',
    complexity TEXT DEFAULT 'O(1)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deprecated BOOLEAN DEFAULT FALSE,
    UNIQUE(name, version)
);

-- Alpha Registry
CREATE TABLE IF NOT EXISTS alpha_registry (
    alpha_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    alpha_type TEXT CHECK (alpha_type IN ('momentum', 'mean_reversion', 'volatility', 'options', 'factor')),
    logic TEXT,
    parameters JSONB DEFAULT '{}',
    expected_sharpe FLOAT,
    capacity_cr FLOAT,
    decay_months INTEGER,
    confidence FLOAT,
    status TEXT CHECK (status IN ('phase1_research', 'phase2_paper_trading', 'phase3_live_small', 'phase4_scale', 'rejected')),
    priority INTEGER DEFAULT 99,
    training_start DATE,
    training_end DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    regime_dependencies TEXT[] DEFAULT '{}',
    UNIQUE(name, version)
);

-- Alpha Performance Tracking
CREATE TABLE IF NOT EXISTS alpha_performance (
    id SERIAL PRIMARY KEY,
    alpha_id UUID REFERENCES alpha_registry(alpha_id) ON DELETE CASCADE,
    regime_id INTEGER,
    is_online BOOLEAN DEFAULT FALSE,
    date DATE NOT NULL,
    daily_return NUMERIC,
    sharpe_rolling_21d NUMERIC,
    hit_rate_21d NUMERIC,
    turnover_21d NUMERIC,
    drawdown_21d NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(alpha_id, regime_id, is_online, date)
);

-- Model Registry (MLflow compatible)
CREATE TABLE IF NOT EXISTS model_registry (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    model_type TEXT CHECK (model_type IN ('alpha', 'regime', 'volatility')),
    parameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    trained_on DATE,
    deployed_at TIMESTAMP,
    is_production BOOLEAN DEFAULT FALSE,
    file_path TEXT,
    UNIQUE(name, version)
);

-- Prediction Registry
CREATE TABLE IF NOT EXISTS prediction_registry (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES model_registry(model_id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    prediction_time TIMESTAMP NOT NULL,
    prediction_value NUMERIC,
    actual_value NUMERIC,
    regime_at_prediction INTEGER,
    confidence NUMERIC
);

-- User Accounts
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    api_key TEXT UNIQUE,
    hashed_password TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Strategy Definitions
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    parameters JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(user_id)
);

-- Audit Logs
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    action TEXT NOT NULL,
    table_name TEXT,
    record_id TEXT,
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_alpha_performance_alpha_id ON alpha_performance(alpha_id);
CREATE INDEX IF NOT EXISTS idx_alpha_performance_date ON alpha_performance(date);
CREATE INDEX IF NOT EXISTS idx_prediction_registry_symbol ON prediction_registry(symbol);
CREATE INDEX IF NOT EXISTS idx_prediction_registry_time ON prediction_registry(prediction_time);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
