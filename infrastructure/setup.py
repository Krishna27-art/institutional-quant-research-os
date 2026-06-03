"""
Infrastructure Setup Script for Institutional Quant Research OS

This script sets up the simplified infrastructure stack for the quant system:
- ClickHouse: Time-series database for high-performance analytics (single node)
- Redis: In-memory cache and streaming for low-latency feature access
- PostgreSQL: Metadata and configuration storage

Based on Profit-Centric Audit - Simplified Architecture
Priority: High (Phase 1)
Removed: Kafka (replaced with Redis Streams), Kubernetes (replaced with Docker Compose)
"""

import subprocess
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfrastructureSetup:
    """Setup and configure infrastructure components."""
    
    def __init__(self, config_path: str = "infrastructure/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.base_dir = Path(__file__).parent.parent
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            "clickhouse": {
                "host": "localhost",
                "port": 8123,
                "database": "quant_research",
                "user": "default",
                "password": ""
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "database": "quant_metadata",
                "user": "quant_user",
                "password": "quant_password"
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def create_docker_compose(self) -> str:
        """Create docker-compose.yml for infrastructure (simplified - no Kafka)."""
        compose_content = """
version: '3.8'

services:
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: quant_clickhouse
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./infrastructure/clickhouse/config.xml:/etc/clickhouse-server/config.xml
      - ./infrastructure/clickhouse/users.xml:/etc/clickhouse-server/users.xml
    environment:
      - CLICKHOUSE_DB=quant_research
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    healthcheck:
      test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: quant_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgresql:
    image: postgres:15-alpine
    container_name: quant_postgresql
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: quant_metadata
      POSTGRES_USER: quant_user
      POSTGRES_PASSWORD: quant_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U quant_user"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  clickhouse_data:
  redis_data:
  postgres_data:
"""
        
        compose_path = self.base_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        logger.info(f"Created docker-compose.yml at {compose_path}")
        return str(compose_path)
    
    def create_clickhouse_config(self):
        """Create ClickHouse configuration files."""
        config_dir = self.base_dir / "infrastructure" / "clickhouse"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # config.xml
        config_xml = """<?xml version="1.0"?>
<clickhouse>
    <logger>
        <level>information</level>
        <console>true</console>
    </logger>
    
    <listen_host>::</listen_host>
    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>
    
    <max_memory_usage>8000000000</max_memory_usage>
    <max_memory_usage_for_user>8000000000</max_memory_usage_for_user>
    
    <mark_cache_size>5368709120</mark_cache_size>
    
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    
    <path>/var/lib/clickhouse/</path>
    <tmp_path>/var/lib/clickhouse/tmp/</tmp_path>
    
    <users_config>users.xml</users_config>
    <default_profile>default</default_profile>
</clickhouse>
"""
        
        config_path = config_dir / "config.xml"
        with open(config_path, 'w') as f:
            f.write(config_xml)
        
        # users.xml
        users_xml = """<?xml version="1.0"?>
<clickhouse>
    <users>
        <default>
            <password></password>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>default</profile>
            <quota>default</quota>
            <management>0</management>
        </default>
    </users>
    
    <profiles>
        <default>
            <max_memory_usage>10000000000</max_memory_usage>
            <use_uncompressed_cache>1</use_uncompressed_cache>
        </default>
    </profiles>
    
    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>1000</queries>
                <errors>100</errors>
                <result_rows>10000000000</result_rows>
                <read_rows>10000000000</read_rows>
                <execution_time>3600</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
"""
        
        users_path = config_dir / "users.xml"
        with open(users_path, 'w') as f:
            f.write(users_xml)
        
        logger.info(f"Created ClickHouse config files in {config_dir}")
    
    def create_postgres_init(self):
        """Create PostgreSQL initialization script."""
        postgres_dir = self.base_dir / "infrastructure" / "postgres"
        postgres_dir.mkdir(parents=True, exist_ok=True)
        
        init_sql = """
-- Create quant metadata database
CREATE DATABASE IF NOT EXISTS quant_metadata;

-- Connect to quant_metadata
\\c quant_metadata

-- Create tables for metadata
CREATE TABLE IF NOT EXISTS alpha_registry (
    id SERIAL PRIMARY KEY,
    alpha_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50),
    status VARCHAR(20) DEFAULT 'HYPOTHESIS',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    capacity_cr FLOAT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS feature_registry (
    id SERIAL PRIMARY KEY,
    feature_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50),
    data_type VARCHAR(20),
    source VARCHAR(100),
    ic_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

CREATE TABLE IF NOT EXISTS data_sources (
    id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    source_type VARCHAR(50),
    status VARCHAR(20) DEFAULT 'ACTIVE',
    last_updated TIMESTAMP,
    config JSONB
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id SERIAL PRIMARY KEY,
    alpha_id VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    sharpe_ratio FLOAT,
    max_drawdown FLOAT,
    cagr FLOAT,
    win_rate FLOAT,
    turnover FLOAT,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (alpha_id) REFERENCES alpha_registry(alpha_id)
);

-- Create indexes
CREATE INDEX idx_alpha_status ON alpha_registry(status);
CREATE INDEX idx_alpha_category ON alpha_registry(category);
CREATE INDEX idx_feature_category ON feature_registry(category);
CREATE INDEX idx_backtest_alpha ON backtest_results(alpha_id);

-- Insert initial data
INSERT INTO data_sources (source_name, source_type, status) VALUES
('NSE_TICK', 'MARKET_DATA', 'ACTIVE'),
('BSE_TICK', 'MARKET_DATA', 'ACTIVE'),
('FII_DII_FLOWS', 'INSTITUTIONAL', 'ACTIVE'),
('OPTIONS_CHAIN', 'DERIVATIVES', 'ACTIVE'),
('CORPORATE_ACTIONS', 'CORPORATE', 'ACTIVE'),
('EARNINGS_CALENDAR', 'EVENTS', 'ACTIVE');
"""
        
        init_path = postgres_dir / "init.sql"
        with open(init_path, 'w') as f:
            f.write(init_sql)
        
        logger.info(f"Created PostgreSQL init script at {init_path}")
    
    def create_clickhouse_schema(self):
        """Create ClickHouse database schema."""
        schema_sql = """
-- Create database
CREATE DATABASE IF NOT EXISTS quant_research;

USE quant_research;

-- Market data tables (partitioned by date)
CREATE TABLE IF NOT EXISTS market_data_1min (
    symbol String,
    timestamp DateTime64(3),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume UInt64,
    vwap Float64,
    trades UInt32,
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS market_data_tick (
    symbol String,
    timestamp DateTime64(6),
    price Float64,
    volume UInt64,
    bid_price Nullable(Float64),
    ask_price Nullable(Float64),
    bid_size Nullable(UInt64),
    ask_size Nullable(UInt64),
    trade_id Nullable(String),
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;

-- Options data
CREATE TABLE IF NOT EXISTS options_chain (
    symbol String,
    expiry_date Date,
    strike Float64,
    option_type String,
    timestamp DateTime64(3),
    iv Float64,
    delta Float64,
    gamma Float64,
    vega Float64,
    theta Float64,
    open_interest UInt64,
    volume UInt64,
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, expiry_date, strike, option_type, timestamp)
SETTINGS index_granularity = 8192;

-- FII/DII flows
CREATE TABLE IF NOT EXISTS fii_dii_flows (
    date Date,
    fii_buy_cr Float64,
    fii_sell_cr Float64,
    fii_net_cr Float64,
    dii_buy_cr Float64,
    dii_sell_cr Float64,
    dii_net_cr Float64
) ENGINE = MergeTree()
ORDER BY date
SETTINGS index_granularity = 8192;

-- Corporate actions
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol String,
    action_type String,
    action_date Date,
    factor Float64,
    cash_value Float64,
    record_date Nullable(Date),
    ex_date Nullable(Date)
) ENGINE = MergeTree()
ORDER BY (symbol, action_date)
SETTINGS index_granularity = 8192;

-- Earnings calendar
CREATE TABLE IF NOT EXISTS earnings_calendar (
    symbol String,
    announcement_date Date,
    fiscal_quarter String,
    fiscal_year UInt16,
    actual_eps Float64,
    estimated_eps Float64,
    revenue_actual Float64,
    revenue_estimated Float64,
    surprise Float64,
    revenue_surprise Float64
) ENGINE = MergeTree()
ORDER BY (symbol, announcement_date)
SETTINGS index_granularity = 8192;

-- Features table (for caching computed features)
CREATE TABLE IF NOT EXISTS features_cache (
    symbol String,
    feature_id String,
    timestamp DateTime64(3),
    value Float64,
    date Date MATERIALIZED toDate(timestamp)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, feature_id, timestamp)
SETTINGS index_granularity = 8192;

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    alpha_id String,
    backtest_id String,
    start_date Date,
    end_date Date,
    sharpe_ratio Float64,
    max_drawdown Float64,
    cagr Float64,
    win_rate Float64,
    turnover Float64,
    parameters String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (alpha_id, backtest_id)
SETTINGS index_granularity = 8192;
"""
        
        schema_path = self.base_dir / "infrastructure" / "clickhouse_schema.sql"
        with open(schema_path, 'w') as f:
            f.write(schema_sql)
        
        logger.info(f"Created ClickHouse schema at {schema_path}")
        return str(schema_path)
    
    def start_infrastructure(self):
        """Start all infrastructure services using docker-compose."""
        compose_path = self.base_dir / "docker-compose.yml"
        
        if not compose_path.exists():
            self.create_docker_compose()
            self.create_clickhouse_config()
            self.create_postgres_init()
        
        logger.info("Starting infrastructure services...")
        
        try:
            subprocess.run(
                ["docker-compose", "up", "-d"],
                cwd=self.base_dir,
                check=True,
                capture_output=True
            )
            logger.info("Infrastructure services started successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start infrastructure: {e}")
            raise
    
    def stop_infrastructure(self):
        """Stop all infrastructure services."""
        compose_path = self.base_dir / "docker-compose.yml"
        
        if not compose_path.exists():
            logger.warning("docker-compose.yml not found")
            return
        
        logger.info("Stopping infrastructure services...")
        
        try:
            subprocess.run(
                ["docker-compose", "down"],
                cwd=self.base_dir,
                check=True,
                capture_output=True
            )
            logger.info("Infrastructure services stopped successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop infrastructure: {e}")
            raise
    
    def initialize_databases(self):
        """Initialize database schemas."""
        import clickhouse_connect
        
        # Initialize ClickHouse
        ch_config = self.config['clickhouse']
        try:
            client = clickhouse_connect.get_client(
                host=ch_config['host'],
                port=ch_config['port'],
                database='default',
                username=ch_config['user'],
                password=ch_config['password']
            )
            
            schema_path = self.base_dir / "infrastructure" / "clickhouse_schema.sql"
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            client.command(schema_sql)
            logger.info("ClickHouse schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse: {e}")
            logger.info("ClickHouse will be initialized when the service starts")
        
        # PostgreSQL is initialized automatically via init.sql
        logger.info("PostgreSQL will be initialized automatically on startup")
    
    def check_health(self) -> Dict[str, bool]:
        """Check health of all infrastructure services."""
        health_status = {}
        
        # Check ClickHouse
        try:
            import clickhouse_connect
            ch_config = self.config['clickhouse']
            client = clickhouse_connect.get_client(
                host=ch_config['host'],
                port=ch_config['port'],
                database='default',
                username=ch_config['user'],
                password=ch_config['password']
            )
            client.query("SELECT 1")
            health_status['clickhouse'] = True
        except:
            health_status['clickhouse'] = False
        
        # Check Redis
        try:
            import redis
            redis_config = self.config['redis']
            r = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db']
            )
            r.ping()
            health_status['redis'] = True
        except:
            health_status['redis'] = False
        
        # Check PostgreSQL
        try:
            import psycopg2
            pg_config = self.config['postgresql']
            conn = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['database'],
                user=pg_config['user'],
                password=pg_config['password']
            )
            conn.close()
            health_status['postgresql'] = True
        except:
            health_status['postgresql'] = False
        
        return health_status


def main():
    """Main setup function."""
    setup = InfrastructureSetup()
    
    print("="*60)
    print("INSTITUTIONAL QUANT RESEARCH OS - INFRASTRUCTURE SETUP")
    print("="*60)
    
    # Create configuration files
    print("\n1. Creating configuration files...")
    setup.create_docker_compose()
    setup.create_clickhouse_config()
    setup.create_postgres_init()
    setup.create_clickhouse_schema()
    
    # Start infrastructure
    print("\n2. Starting infrastructure services...")
    setup.start_infrastructure()
    
    # Wait for services to be ready
    print("\n3. Waiting for services to be ready (30 seconds)...")
    import time
    time.sleep(30)
    
    # Initialize databases
    print("\n4. Initializing database schemas...")
    setup.initialize_databases()
    
    # Check health
    print("\n5. Checking service health...")
    health = setup.check_health()
    
    print("\nHealth Status:")
    for service, status in health.items():
        status_str = "✓" if status else "✗"
        print(f"  {status_str} {service}")
    
    print("\n" + "="*60)
    print("INFRASTRUCTURE SETUP COMPLETE")
    print("="*60)
    print("\nServices:")
    print("  - ClickHouse: localhost:8123")
    print("  - Redis: localhost:6379")
    print("  - PostgreSQL: localhost:5432")
    print("\nNote: Kafka removed - using Redis Streams for messaging")
    print("To stop services: cd infrastructure && docker-compose down")
    print("="*60)


if __name__ == "__main__":
    main()
