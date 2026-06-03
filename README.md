# Institutional Quant Research OS - Architecture V2

A production-grade quantitative trading system for Indian markets (NSE/BSE) based on agent debate consensus. Features simplified Phase 1 stack (Python + Polars + PostgreSQL + Redis + LightGBM), risk-parity portfolio construction, and comprehensive risk management.

## Architecture V2 Overview

### Phase 1 Tech Stack (₹25Cr AUM)
- **Language**: Python 3.11 (core), Go (data ingestion)
- **Data Processing**: Polars, NumPy, Numba
- **ML**: LightGBM, Scikit-learn, SHAP
- **Database**: PostgreSQL (metadata), Redis (state), ClickHouse (analytics)
- **Streaming**: Redis Streams
- **API**: FastAPI (order entry), WebSocket (market data)
- **Orchestration**: Docker Compose (single VM)
- **Monitoring**: Prometheus + Grafana
- **Cloud**: AWS Mumbai (t3.xlarge for research, c5.4xlarge for live)

### Global Parameters
- **Target AUM**: ₹25,00,00,000 (initial live)
- **Instrument Universe**: NIFTY 50 + BANKNIFTY + top 100 liquid stocks
- **Data Frequency**: 1-minute bars (from tick aggregation)
- **Target Latency**: 1 second (end-to-end)
- **Team**: 3 engineers + 1 quant researcher

## Alpha Ranking Table

| Strategy | Expected Sharpe | Capacity (₹Cr) | Decay (months) | Confidence | Status |
|----------|----------------|----------------|---------------|------------|--------|
| 5-min ORB (Stocks in Play) | 1.1 | 100 | 6 | 70% | Must Build |
| VWAP Trend (NIFTY futures) | 0.9 | 500 | 12 | 60% | Must Build |
| Put-Call Carry (Weekly options) | 0.7 | 200 | 24 | 75% | Must Build |
| Volatility Carry (Short straddle) | 0.6 | 150 | 18 | 65% | Worth Testing |
| GCN (Game-theoretic stock) | 0.5 | 50 | unknown | 40% | Low Priority |
| LSTM / Transformer | 0.3 | 0 | 1 | 20% | Reject |

## Core Components

### 1. Alpha Strategies (`alpha/`)
- **ORB (Zarattini)**: 5-minute Opening Range Breakout with Relative Volume filtering
- **VWAP Trend (Zarattini/Aziz)**: Trend-following on NIFTY futures with VWAP crossover
- **Put-Call Carry (Shin)**: Weekly options carry gap strategy
- **Volatility Carry**: Short straddle with delta hedging on NIFTY options
- **Game-Theoretic Selection (Zhang)**: Heterogeneous investor participation modeling

### 2. Regime Engine (`regime/`)
- **HMM + Change Point Detection**: 4-state Gaussian HMM (bull_trend, bear_trend, sideways, high_vol)
- Features: realized_vol_5d, implied_vol, nifty_return_5d, turnover_ratio_5d
- Training window: 252 days (1 year)
- Retraining: Daily after close
- Change point detection: CUSUM with 10-minute windows

### 3. Alpha Combination Engine (`alpha/lightgbm_ensemble.py`)
- **Method**: Risk-parity + Kelly (15% of optimal)
- Regime-based weights:
  - bull_trend: ORB 40%, VWAP 30%, PCP 15%, VolCarry 10%
  - bear_trend: ORB 20%, VWAP 40%, PCP 20%, VolCarry 15%
  - sideways: ORB 10%, VWAP 10%, PCP 30%, VolCarry 40%
  - high_vol: ORB 15%, VWAP 15%, PCP 20%, VolCarry 40%
- Correlation penalty: Shrink weights if inter-alpha correlation > 0.5
- Rebalance: Daily

### 4. Portfolio Construction (`portfolio/`)
- **Method**: Risk-parity (equal volatility contribution)
- Optimizer: Sequential Least Squares (SLSQP) - daily
- Constraints:
  - Max single strategy weight: 50%
  - Max sector weight: 30%
  - Max leverage: 4x
  - Max position size: 5% of AUM
- Objective: Minimize sqrt(w' Σ w) subject to target vol = 15%
- Rebalance: At market open, using previous close data

### 5. Risk Engine (`risk/institutional_risk_engine.py`)
- **Pre-trade**:
  - Position limit check (5% AUM)
  - Sector limit check (30% AUM)
  - VaR (99% 1-day) using historical simulation, cap at 2% of AUM
  - Correlation check (stop if portfolio heat > 0.7)
- **Intraday**:
  - Trailing stop on each position (10% ATR)
  - Global circuit breaker: if daily PnL < -3%, flatten all and halt
  - Leverage monitor (warn if >3x, hard stop at 4x)
- **Post-trade**:
  - Compute realized Sharpe, VaR, drawdown
  - Adjust Kelly fraction monthly (15% of observed Sharpe²)
- **Tail risk**: Buy OTM puts on NIFTY if VIX < 12 (cost ≤ 1% of AUM/year)

### 6. Execution Engine (`execution/vwap_pov_execution.py`)
- **VWAP-based slicing** for large orders
- **Limit orders** with 0.5-2 bps patience
- **Stop-loss** at 10% ATR (market order)
- **Venue selection**: NSE vs BSE based on liquidity
- **Market impact model**: Square-root model

### 7. Feature Engineering (`features/feature_pipeline.py`)
- **50 core features** reduced via Boruta
- Categories:
  - Volume: RV, volume ratio, tick volume, volume profile slope
  - Price: VWAP distance, ATR, momentum, high/low ratio
  - Volatility: Realized vol, IV, IV percentile, IV-RV spread
  - Options: PCR, IV skew, term structure, gamma exposure
  - Flow: FII/DII flow, order flow imbalance
  - Time: Day-of-week, time-of-day, expiry week flag
  - Technical: RSI, MACD, Bollinger Bands, Stochastic, Williams %R
  - Microstructure: Bid-ask spread, depth imbalance
  - Market structure: Gap, gap fill, inside/outside bar, engulfing

### 8. Backtesting (`backtesting/`)
- **Hybrid backtester**: Vectorized + event-driven
- Indian market costs: Brokerage, stamp duty, SEBI turnover fee, GST
- Slippage model: Fixed + variable based on order size
- Performance analytics: Sharpe, Sortino, Calmar, win rate, profit factor

### 9. Data Storage (`data/`)
- **Hot cache** (last 24 hours): Redis (in-memory, 5ms)
- **Real-time ingest**: Redis Streams (one stream per symbol, consumer group for features)
- **Historical 1-min bars**: ClickHouse (partition by symbol, time)
- **Raw ticks**: Parquet on S3 (partitioned by symbol/year/month)
- **Features & signals**: ClickHouse + Redis (latest only)
- **Research**: DuckDB on local Parquet files

### 10. Live Trading (`live/`)
- **Data feed**: WebSocket → Go routine → Redis Streams
- **Feature calculation**: Python + Polars (vectorized) → Redis Hash
- **Signal generation**: LightGBM (C API via Python) → Redis Pub/Sub
- **Risk & order generation**: Python (single thread) → Redis Queue
- **Execution**: FastAPI (order submission to broker) → HTTP/2
- **Monitoring**: Prometheus + Grafana (per-symbol, per-strategy metrics)
- **Alerting**: PagerDuty + Slack (latency spikes, circuit breaker hits)

## Project Structure

```
institutional-quant-research-os/
├── CMakeLists.txt              # C++ build system (Phase 2)
├── Dockerfile                  # FastAPI application container
├── Dockerfile.jupyter          # Jupyter notebook container
├── docker-compose.yml          # Local development environment
├── .env.example                # Environment variables template
├── .dockerignore               # Docker build exclusions
├── config/
│   └── config.yaml            # Production configuration
├── alpha/
│   ├── __init__.py
│   ├── orb_zarattini.py        # 5-min ORB (Zarattini methodology)
│   ├── vwap_trend_zarattini.py # VWAP Trend (Zarattini/Aziz)
│   ├── put_call_carry_shin.py  # Put-Call Carry (Shin)
│   ├── vol_carry.py            # Volatility Carry
│   ├── game_theoretic.py       # Game-Theoretic stock selection
│   ├── chaotic_gcn.py          # Chaotic Graph Convolutional Network
│   └── lightgbm_ensemble.py    # Alpha combination engine
├── alpha_engines/
│   ├── __init__.py
│   ├── base.py                 # Base alpha engine
│   ├── orb_engine.py           # ORB engine implementation
│   ├── pcp_engine.py           # Put-Call Carry engine
│   ├── vol_engine.py           # Volatility Carry engine
│   └── ensemble.py             # Ensemble engine
├── backtesting/
│   ├── __init__.py
│   ├── backtest_orb.py         # ORB backtesting
│   ├── backtest_vwap.py        # VWAP backtesting
│   ├── backtest_pcp.py         # PCP backtesting
│   ├── backtest_vol_carry.py   # Volatility Carry backtesting
│   ├── backtest_vwap.py        # VWAP backtesting
│   ├── hybrid_backtester.py    # Hybrid vectorized + event-driven
│   └── performance_analytics.py # Performance metrics
├── brokers/
│   ├── __init__.py
│   └── kite_connect.py         # Zerodha Kite Connect adapter
├── core/
│   ├── __init__.py
│   ├── data_layer.py           # Data management
│   ├── dsa_structures.py       # DSA implementations
│   ├── event_engine.py         # Event-driven architecture
│   └── events.py               # Event definitions
├── data/
│   ├── __init__.py
│   ├── nse_adapter.py          # NSE data adapter
│   ├── nse_ingestion.py        # Historical tick data ingestion
│   ├── catalog/                # Data catalog
│   ├── processed/              # Processed data storage
│   ├── experiments/            # Experiment databases
│   ├── audit.py                # Data auditing
│   └── corporate_actions.py    # Corporate actions handling
├── database/
│   └── db_architecture.py      # Database schema definitions
├── engine/
│   ├── bindings.cpp            # pybind11 bindings (Phase 2)
│   ├── execution_engine.cpp    # C++ execution engine (Phase 2)
│   ├── execution_engine.h      # C++ execution header (Phase 2)
│   └── order_book.cpp          # C++ order book (Phase 2)
├── execution/
│   ├── __init__.py
│   ├── cost_model.py           # Transaction cost model
│   ├── execution_aware_alpha.py # Execution-aware alpha
│   ├── fill_model.py           # Fill simulation model
│   └── vwap_pov_execution.py   # VWAP + POV hybrid execution
├── features/
│   ├── __init__.py
│   └── feature_pipeline.py     # 50-feature engineering pipeline
├── hypothesis/
│   ├── __init__.py
│   ├── falsification.py        # Hypothesis falsification
│   ├── mechanism.py            # Behavioral mechanism modeling
│   └── registry.py             # Hypothesis registry
├── legacy/
│   ├── __init__.py
│   ├── behavioral_hypothesis.py # Behavioral regimes framework
│   ├── signal_validity_tracker.py # Signal veto engine
│   ├── garch_volatility.py     # GARCH volatility modeling
│   ├── long_memory_volatility.py # Long memory volatility (Deep et al.)
│   └── research_database.py   # Research database
├── live/
│   ├── __init__.py
│   ├── broker_api.py           # Live broker API
│   └── server.py               # Live trading server
├── live_trading/
│   └── api_server.py           # API server for live trading
├── market/
│   ├── __init__.py
│   ├── intraday_structure.py    # Intraday market structure
│   ├── liquidity.py            # Liquidity metrics
│   └── opening_auction_participants.py # Opening auction analysis
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py              # Monitoring metrics
│   ├── alerts.py               # Alerting system
│   ├── decay.py                # Signal decay tracking
│   └── prometheus.yml          # Prometheus configuration
├── paper_trading/
│   └── simulator.py            # Paper trading simulator
├── portfolio/
│   ├── __init__.py
│   ├── allocator.py            # Portfolio allocator
│   ├── alpha_combination.py     # Alpha combination
│   ├── capacity_intelligence.py # Capacity analysis
│   └── risk.py                 # Portfolio risk
├── regime/
│   ├── __init__.py
│   ├── hmm_engine.py           # HMM regime engine
│   └── hybrid_hmm_cpd.py       # Hybrid HMM + Change Point Detection
├── risk/
│   ├── __init__.py
│   └── institutional_risk_engine.py # Institutional risk engine
├── signals/
│   ├── __init__.py
│   ├── base.py                 # Base signal class
│   ├── gap_fade.py             # Gap fade signal
│   └── validator.py           # Signal validator
├── stats/
│   ├── __init__.py
│   ├── decision_replay.py      # Decision replay analysis
│   ├── distributions.py        # Statistical distributions
│   └── evidence.py             # Evidence tracking
├── trading_research/
│   ├── __init__.py
│   ├── app.py                  # Research application
│   ├── config.py               # Research configuration
│   ├── backtesting/            # Research backtesting
│   ├── data/                   # Research data
│   └── hypothesis/            # Research hypotheses
├── web/
│   ├── dashboard.html          # Web dashboard
│   ├── css/
│   │   └── dashboard.css       # Dashboard styling
│   └── js/
│       └── dashboard.js        # Dashboard JavaScript
├── reports/                    # Generated reports
├── research_os/                # Research OS components
│   ├── __init__.py
│   ├── challenge.py            # Pre-deployment challenge
│   ├── experiment.py          # Experiment framework
│   └── replay.py              # Decision replay
├── app.py                      # Main application entry
├── config.py                   # Configuration management
├── config_v2.py                # Architecture V2 configuration
├── main.py                     # Main orchestrator
└── requirements.txt            # Python dependencies
```

## Detailed File Descriptions

### Core System Files

#### `main.py`
The main orchestrator of the NiftyQuant system. It initializes and coordinates all components including data manager, feature engine, regime engine, alpha strategies, risk engine, and the C++/Python execution engine. It supports backtesting and live trading modes, manages market hours, executes routines (pre-market, opening range, trading), handles order submission with risk checks, and performs EOD cleanup.

#### `app.py`
The main entry point for the old "Market OS" system. It handles various demo commands related to market state, event-driven replay, smart money concepts, data auditing, leakage detection, NSE data normalization, and experiment logging. This file is likely to be replaced by the new `main.py` orchestrator in the upgraded system.

#### `config.py`
Defines a `SystemConfig` dataclass for the old system's basic configuration, including project paths, initial capital, max daily loss, max position percentage, and gap fade parameters. This configuration is likely superseded by the new `config/config.yaml` and `config_v2.py` for the upgraded system.

#### `config_v2.py`
Defines a comprehensive set of dataclasses for the upgraded system's configuration. It includes `GlobalConfig`, `AlphaRankingConfig`, `AlphaCombinationConfig`, `RegimeEngineConfig`, `PortfolioConfig`, `RiskEngineConfig`, `ExecutionConfig`, `FeatureConfig`, `DatabaseConfig`, `TechStackConfig`, `MonitoringConfig`, and `ResearchRoadmapConfig`. This file provides detailed, structured configuration for various aspects of the trading system.

#### `CMakeLists.txt`
Defines the CMake build system for the C++ execution engine. It sets the C++ standard to 17, configures optimization flags for release builds, finds Python and pybind11 (fetching them if not found), finds spdlog (fetching if not found), defines a static library for the execution engine, creates the pybind11 module `niftyquant_cpp`, and sets up installation for the module.

#### `requirements.txt`
Lists all Python dependencies for the project, categorized by function (Core Data Processing, Data Sources, Statistical & ML, Deep Learning & Graph Neural Networks, Optimization, Database & Caching, API & Streaming, HMM & Regime Detection, Performance, Visualization, Utilities, Monitoring, C++ Build Dependencies). It includes libraries like pandas, numpy, polars, yfinance, nselib, kiteconnect, torch, torch-geometric, arctic, fastapi, uvicorn, pybind11, and cmake.

### Core Module (`core/`)

#### `core/__init__.py`
Exports `EventDrivenEngine`, `Event`, `EventBus`, and `EventType` as part of the core module, indicating an event-driven architecture for the system.

#### `core/data_layer.py`
Implements the `DataManager` which is responsible for handling all market data. It defines `Instrument`, `OHLCV`, and `Tick` data structures. It includes abstract `DataFeed` and concrete implementations for `ZerodhaFeed` (for live Indian market data via Kite Connect) and `YahooFeed` (as a fallback). It also integrates `ArcticStore` for historical data caching in MongoDB and uses Redis for instrument caching and tick publishing. The `DataManager` also contains a hardcoded `SECTOR_MAP` for NIFTY 50 stocks.

#### `core/feature_layer.py`
Defines the `FeatureEngine` class, which is responsible for computing various features from OHLCV data. It includes methods for calculating candlestick microstructure features (body ratio, shadows, log returns, realized volatility, ATR), volume-based features (relative volume, volume imbalance, VWAP distance), momentum features (multi-scale returns, RSI, MACD, Bollinger Band position), and chaotic features (chaotic entropy, Lyapunov exponent, deviation) based on logistic or tent maps. It also provides methods for building sector correlation matrices and adjacency/feature matrices for Graph Convolutional Networks (GCN).

#### `core/regime_engine.py`
Implements the `HybridRegimeEngine` which combines a `HMMRegimeDetector` for long-term market regimes (Bull, Bear, Sideways, Crisis) and a `VVGClassifier` for intraday regimes (trending, choppy). It defines `Regime` and `StrategyType` enums and a `RegimeState` dataclass. The HMM uses daily log-returns and volatility, while the VVG uses opening gap, volume, and volatility ranks. The engine provides strategy recommendations and position sizing multipliers based on the detected hybrid regime.

#### `core/event_engine.py`
Defines the `EventDrivenEngine`, a minimal event loop designed for decoupling strategy, risk, and execution components. It uses an `EventBus` to emit and handle events and can optionally integrate with a `ReplayJournal` for recording and replaying events. It provides methods to emit events of various types (e.g., `MARKET_OPEN`, `BAR`, `MARKET_CLOSE`) and to replay a sequence of events from a journal.

#### `core/events.py`
Defines the core event structures for the event-driven system. It includes an `EventType` Enum (e.g., `MARKET_OPEN`, `BAR`, `SIGNAL`, `ORDER`, `FILL`, `RISK`, `PORTFOLIO`, `MARKET_CLOSE`) and an `Event` dataclass to encapsulate event type, payload, timestamp, and sequence. It also defines an `EventBus` for synchronous event dispatching, allowing handlers to subscribe to specific event types and events to be emitted and stored in a history.

#### `core/dsa_structures.py`
Implements high-performance data structures and algorithms for trading: Fenwick Tree for cumulative volume (O(log N) operations), Max-Heap for top-N stocks by relative volume, Ring Buffer for tick stream processing, Segment Tree for range min/max queries on OHLC data, Bloom Filter for duplicate tick detection, Sparse Table for pre-computed volatility ranges, Symbol Cache with LRU eviction, and Order Book using priority queues for bid/ask management.

### Alpha Module (`alpha/`)

#### `alpha/__init__.py`
Exports several strategy classes and related data structures: `ORBStrategy`, `ORBSignal`, `ORBPosition` for Opening Range Breakout; `VWAPStrategy`, `VWAPSignal`, `VWAPPosition` for VWAP-based strategies; `ChaoticGCNAlpha` for the Graph Convolutional Network alpha; and `GameTheoreticAlpha` for the game-theoretic alpha.

#### `alpha/orb_strategy.py`
Implements the `ORBStrategy` (Opening Range Breakout) for Indian markets. It defines `ORBSignal` and `ORBPosition` dataclasses. The strategy identifies breakout candidates based on relative volume, defines an opening range (e.g., 9:15-9:20 AM), and generates long/short signals when price breaks out of this range with volume confirmation. It includes logic for setting stop-loss and target prices based on ATR and range size, and manages active positions, including handling failed breakouts and forced closures.

#### `alpha/vwap_strategy.py`
Implements the `VWAPStrategy`, which supports both trend-following and mean-reversion modes based on Volume Weighted Average Price. It defines `VWAPSignal` and `VWAPPosition` dataclasses. The strategy calculates VWAP and generates signals based on price deviation from VWAP, considering momentum and volume for trend mode, and extreme deviation with reversal patterns for reversion mode. It includes logic for setting stop-loss and target prices and managing active positions.

#### `alpha/chaotic_gcn.py`
Implements the `ChaoticGCNAlpha` strategy, which combines chaotic time series analysis with Graph Convolutional Networks (GCNs) for cross-asset learning. It defines a `ChaoticGCNModel` (a PyTorch GCN) and a `GCNSignal` dataclass. The strategy prepares features including price, volume, momentum, and chaotic features (entropy, Lyapunov exponent, deviation) derived from logistic or tent maps. It builds a sector correlation graph to represent asset relationships and trains the GCN to predict future price movements (long, neutral, short) for individual symbols, incorporating both individual asset features and graph-based relationships.

#### `alpha/game_theoretic.py`
Implements the `GameTheoreticAlpha` strategy, which models market participants as strategic agents (hot money and institutional money) to generate alpha. It defines a `GameTheoreticSignal` dataclass. The strategy computes "hot money" signals based on short-term momentum and volume acceleration, and "institutional" signals based on longer-term trends and volume consistency. It then analyzes the equilibrium and deviation between these two types of signals to determine trading direction and confidence, aiming to either follow strong trends near equilibrium or fade "hot money" when there's significant deviation.

#### `alpha/lightgbm_ensemble.py`
Implements a LightGBM Ensemble for Alpha Combination. The architecture uses LightGBM as the primary model (fastest, lowest latency) with XGBoost as secondary (slightly better accuracy). It uses stacked ensemble with rolling Sharpe optimization for weight rebalancing (daily) and applies correlation penalty for risk management. The ensemble prepares features including individual alpha signals, lagged signals, market features, time features, and regime features. It optimizes weights using rolling Sharpe ratio and applies correlation penalty to reduce redundancy.

#### `alpha/put_call_carry_shin.py`
Implements a Put-Call Carry Gap System based on Shin (2026a,b) methodology. The strategy calculates the carry gap between option-implied and OIS discount factors, and when the carry gap exceeds 20bp, enters a short strangle. It holds for 7-30 days to capture theta decay, applies GBM path-risk adjustment (rσ√τ), and exits at 1 DTE or if the carry gap closes. The backtester validates Shin's findings of 37bp annualized carry with 98.4% positive observations.

#### `alpha/orb_zarattini.py`
Implements a 5-Minute Opening Range Breakout Strategy based on Zarattini et al. (2024) methodology. The strategy calculates a 5-minute opening range (9:15-9:20), computes Relative Volume (RV = OR volume / avg volume), and only trades if RV > 100%. It enters on breakout above/below OR, uses 10% ATR for optimal stop loss (Zarattini finding), and targets at 2x risk. The backtester validates Zarattini's findings of 1,637% return (2016-2023) with Sharpe 2.81.

#### `alpha/vwap_trend_zarattini.py`
Implements a VWAP Trend Trading System based on Zarattini/Aziz (2023) methodology. The strategy calculates VWAP for the day, and when price > VWAP by > 2σ, there's a 56% continuation probability. It enters long when price crosses above VWAP with volume confirmation, enters short when price crosses below VWAP with volume confirmation, uses stop loss at 1.5σ from VWAP, and targets at 3σ from VWAP. It focuses on first and last hour where 80% of profits occur (Zarattini finding).

### Alpha Engines Module (`alpha_engines/`)

#### `alpha_engines/__init__.py`
Exports base alpha engine classes and specific alpha engines like ORBEngine, VWAPEngine, PCPEngine, and VolCarryEngine. It also provides a factory function to create alpha engines by name.

#### `alpha_engines/base.py`
Defines abstract base classes for alpha engines, including signal generation, validation, and metrics. It provides the foundation for implementing specific alpha strategies with a consistent interface.

#### `alpha_engines/orb_engine.py`
Implements a 5-minute Opening Range Breakout alpha engine with relative volume filtering, day-of-week weighting, and signal creation. It extends the base alpha engine to provide ORB-specific functionality.

### Backtesting Module (`backtesting/`)

#### `backtesting/__init__.py`
Exposes `VectorizedBacktester`, `BacktestConfig`, `BacktestResult`, and helper functions like `compute_transaction_cost` and `compute_market_impact`.

#### `backtesting/backtester.py`
Implements the `VectorizedBacktester` for strategy evaluation. It defines `BacktestConfig` for Indian market transaction costs (brokerage, STT, charges, GST, SEBI, stamp duty, slippage, market impact) and `BacktestResult` for comprehensive metrics (returns, equity, drawdown, Sharpe, Sortino, CAGR, win rate, profit factor, costs). It includes methods to `run_signal_backtest` (for generic signals) and `run_orb_backtest` (specific to ORB strategy with intraday simulation), and helper functions to compute costs and metrics.

#### `backtesting/backtest_orb.py`
Backtesting framework for ORB strategy with realistic slippage, performance metrics, and synthetic data example. It provides specific backtesting capabilities for the ORB strategy.

#### `backtesting/hybrid_backtester.py`
Hybrid vectorized + event-driven backtester with realistic execution simulation, slippage, partial fills, and performance metrics. It combines the speed of vectorized backtesting with the realism of event-driven simulation.

#### `backtesting/performance_analytics.py`
Comprehensive performance metrics and analytics for backtests including risk, drawdown, trade, holding period, and tail risk metrics. It provides detailed analysis capabilities for evaluating strategy performance.

### Brokers Module (`brokers/`)

#### `brokers/__init__.py`
Exports `BrokerType`, `BrokerConfig`, `Tick`, `Order`, `KiteConnectClient`, and `BrokerManager` from the `kite_connect` module, indicating the system's broker API abstraction and Zerodha integration.

#### `brokers/kite_connect.py`
Implements the Zerodha Kite Connect API client and broker manager. It supports authentication, market data quotes, historical data, order placement, WebSocket live ticks, and rate limiting. It defines data structures for ticks and orders, and a unified broker manager that currently supports Zerodha. It includes async methods for API calls, WebSocket connection management, and order lifecycle handling.

### Data Module (`data/`)

#### `data/__init__.py`
Exports several classes related to data auditing (`AuditResult`, `CorporateActionAudit`, `SurvivorshipAudit`), NSE data adaptation (`NSELibAdapter`, `NSEMarketDataset`, `NSERequest`), data sourcing (`DataSource`, `OHLCVColumns`), and data validation (`DataQualityReport`, `DataValidator`).

#### `data/audit.py`
Defines data auditing classes: `CorporateActionAudit` to flag suspicious price jumps around corporate actions, and `SurvivorshipAudit` to ensure trades only involve symbols active in the universe at the time of the trade. It uses `AuditResult` to report findings, including issues and metadata.

#### `data/nse_adapter.py`
Provides `NSELibAdapter` for fetching and normalizing data from NSE (National Stock Exchange) using the `nselib` library. It includes methods to normalize bhavcopy, FII/DII data, VIX, and delivery data into a consistent pandas DataFrame format. `NSERequest` defines parameters for data requests, and `NSEMarketDataset` acts as a container for various market context datasets.

#### `data/source.py`
Defines the `DataSource` class for loading and saving OHLCV data from local CSV files or fetching it from `yfinance`. It includes helper functions for normalizing DataFrame columns and converting to naive datetimes. The `DataSource` manages raw and processed data directories and ensures data consistency with standard OHLCV columns.

#### `data/store.py`
Implements `DataStore`, a SQLite-backed persistence layer for cleaned research datasets. It uses `StoredFrameMeta` to store metadata about each DataFrame. The `DataStore` provides methods to initialize the database, save and load pandas DataFrames (e.g., OHLCV data) into specific tables, set validation scores, and retrieve metadata.

#### `data/universe.py`
Defines `UniverseRegistry` for managing point-in-time universe membership of symbols. It uses `UniverseMembership` to store the active period for each symbol. The registry can load membership data from a CSV file and provides methods to query which symbols were active on a given date or check if a specific symbol was active.

#### `data/corporate_actions.py`
Provides classes for corporate action adjustments. It defines a `CorporateAction` dataclass and a `CorporateActionAdjuster` class that applies multiplicative price adjustments (splits, bonuses, reverse splits) to historical OHLCV data before the corporate action date, adjusting prices and volumes accordingly to maintain data consistency.

#### `data/validator.py`
Implements `DataValidator` which validates OHLCV data integrity before research use. It checks for missing columns, duplicate dates, non-positive prices/volumes, stale price runs, extreme price moves, and missing sessions. It produces a `DataQualityReport` with a score and flagged issues. It also provides a `clean` method to enforce minimum quality.

#### `data/nse_ingestion.py`
Implements NSE/BSE data ingestion using `nselib` and `yfinance`. It supports historical tick and minute bar data download, corporate actions handling, survivorship bias correction, and Parquet storage with partitioning. It includes `NSEDataIngestion` class with async methods for downloading, saving, and ingesting data for symbols, and a `CorporateActionsHandler` for fetching and adjusting corporate actions like splits and dividends.

### Database Module (`database/`)

#### `database/__init__.py`
Exports `DatabaseConfig`, `RedisCache`, `ClickHouseManager`, `PostgreSQLManager`, and `DatabaseManager` from the `db_architecture` module, indicating a layered database design involving Redis, ClickHouse, and PostgreSQL.

#### `database/db_architecture.py`
Implements the database architecture layer for the system. It defines `DatabaseConfig` for Redis, ClickHouse, PostgreSQL, and Parquet storage configuration. It implements `RedisCache` for hot data caching and streaming, `ClickHouseManager` for historical analytics with table schemas and queries, `PostgreSQLManager` for metadata management with tables for symbols, strategies, experiments, and users, and a unified `DatabaseManager` to coordinate these components and provide caching, streaming, and persistence of market data, features, and signals.

### Execution Module (`execution/`)

#### `execution/__init__.py`
Exports `ExecutionCostBreakdown`, `IndianCostModel` from `cost_model`, `FillEstimate`, `FillModel` from `fill_model`, and `ImpactEstimate`, `MarketImpactModel` from `impact`.

#### `execution/cost_model.py`
Defines an Indian equity transaction cost model. It includes `ExecutionCostBreakdown` dataclass for detailed cost components and `IndianCostModel` class to estimate brokerage, STT, exchange charges, SEBI charges, GST, stamp duty, and total cost for trades, differentiating intraday and delivery trades.

#### `execution/fill_model.py`
Defines a fill probability and slippage estimation model. It includes `FillEstimate` dataclass and `FillModel` class which estimates fill ratio and slippage in basis points based on liquidity tiers and market conditions such as at open or price gaps.

#### `execution/impact.py`
Implements a market impact approximation using a square-root impact model. It defines `ImpactEstimate` dataclass and `MarketImpactModel` class which estimates participation rate and impact in basis points given order notional, average daily volume notional, and daily volatility.

### Engine Module (`engine/`)

#### `engine/order_book.h`
Defines the core data structures and the `OrderBook` class for the execution engine. It includes type aliases for `Price`, `Quantity`, `OrderId`, `Timestamp`, and enums for `Side`, `OrderType`, and `OrderStatus`. Key structs are `Order` (representing a single order) and `MarketDepth` (for storing bid/ask levels). The `OrderBook` class manages orders, provides market depth, and includes methods for adding/canceling orders and matching them.

#### `engine/order_book.cpp`
Implements the `OrderBook` class. It uses `std::map` for bids (descending price) and asks (ascending price) to maintain price-time priority. The `add_order` method assigns an ID and timestamp, then places the order in the appropriate map and calls `match_orders`. `cancel_order` removes an order and adjusts quantities. `get_depth` provides the top N levels. The `match_orders` method performs a simple price-time matching algorithm, updating `last_trade_price_` and `total_volume_` upon a fill.

#### `engine/execution_engine.h`
Defines the `ExecutionEngine` class, which handles order submission and execution, including advanced order types like VWAP. It defines `VWAPParams` and `ExecutionMetrics` structs. The `ExecutionEngine` manages multiple `OrderBook` instances, uses worker threads for execution, and provides methods for submitting market, limit, and VWAP orders. It also supports callbacks for market data and fill events, and calculates execution metrics.

#### `engine/execution_engine.cpp`
Implements the `ExecutionEngine`. It manages worker threads to process orders, particularly complex ones like VWAP. `submit_market_order` and `submit_limit_order` interact directly with the `OrderBook`. `submit_vwap_order` creates a `ParentOrder` and detaches a thread to execute it, breaking it down into child orders based on `compute_slice_size` and `compute_limit_price`, which consider factors like participation rate, market depth, and urgency. It also tracks execution metrics and handles callbacks for fills.

#### `engine/bindings.cpp`
Uses pybind11 to create Python bindings for the C++ execution engine. It exposes C++ enums (`Side`, `OrderType`, `OrderStatus`), data structures (`MarketDepth`, `Fill`, `VWAPParams`, `ExecutionMetrics`), and the `OrderBook` and `ExecutionEngine` classes to Python. This allows Python code to interact with the high-performance C++ components for order management and execution.

### Live Trading Module (`live/`)

#### `live/__init__.py`
Exports `LiveServer` and `BrokerAPI`, indicating that these are the primary components for the live trading module.

#### `live/server.py`
Implements a FastAPI-based `LiveServer` for real-time trading. It defines `OrderRequest` and `OrderResponse` models for API interaction. It includes endpoints for submitting orders, checking order status, getting positions, and portfolio summary. A `ConnectionManager` handles WebSocket connections for broadcasting real-time updates. The server uses `uvicorn` for deployment and includes CORS middleware.

#### `live/broker_api.py`
Defines an abstract `BrokerAPI` class and a concrete `ZerodhaAPI` implementation for interacting with trading brokers. It includes enums for `BrokerType`, `OrderSide`, `OrderType`, `OrderStatus`, and dataclasses for `Order` and `Position`. The `ZerodhaAPI` connects to Kite Connect, places/cancels orders, and retrieves order status, positions, holdings, and account balance. A `BrokerAPIFactory` is provided to create instances of specific broker APIs.

### Risk Module (`risk/`)

#### `risk/__init__.py`
Exports `RiskEngine`, `RiskAction`, and `RiskCheckResult`, indicating that these are the primary components of the risk management module.

#### `risk/risk_engine.py`
Defines the `RiskEngine` class, which provides comprehensive risk management functionalities. It includes methods for pre-trade checks (position size, sector concentration, daily loss, drawdown, correlation, portfolio VaR), volatility-targeted position sizing, and updating positions/PnL. It uses `RiskAction` and `RiskCheckResult` enums/dataclasses to communicate risk decisions. The engine monitors portfolio value, drawdown, and sector exposure, and can force liquidate all positions in emergency scenarios.

### Regime Module (`regime/`)

#### `regime/__init__.py`
Exports `HMMRegimeEngine`, `HMMConfig`, `Regime`, and `RegimeState` from the `hmm_engine` module.

#### `regime/hmm_engine.py`
Implements a Hidden Markov Model Regime Detection Engine that detects 4 market regimes: Bull Trend (positive returns, moderate volatility), Bear Trend (negative returns, moderate volatility), Sideways (low returns, low volatility), and High Vol (high volatility, uncertain direction). It uses a Gaussian HMM with 4 states trained on realized volatility, implied volatility, NIFTY return, and turnover ratio. The engine provides regime-based weights for alpha combination and includes change point detection using CUSUM.

#### `regime/hybrid_hmm_cpd.py`
Implements a Hybrid HMM + Change Point Detection Regime Engine based on research recommendations for Indian markets. The architecture uses HMM as primary (4 states: trend_up, trend_down, sideways, high_vol) with online CPD for rapid regime shifts. Features include realized_vol, implied_vol, return, turnover, spread, and skew. It uses daily re-estimation (252-day window) and online CPD with 10-minute window. The engine provides regime-specific alpha weights and transition matrix analysis.

### Features Module (`features/`)

#### `features/feature_pipeline.py`
Implements a Feature Pipeline for Architecture V2 that computes 50 core features for alpha generation. Features include volume features (RV, volume ratio, tick volume), price features (VWAP distance, ATR, momentum), volatility features (realized vol, IV, IV percentile), options features (PCR, skew, term structure), flow features (FII/DII flow, order flow imbalance), time features (day-of-week, time-of-day), technical features (RSI, MACD, Bollinger Bands), microstructure features (bid-ask spread, depth imbalance), and market structure features (gap, gap fill, inside/outside bar, engulfing).

### Monitoring Module (`monitoring/`)

#### `monitoring/__init__.py`
Exports `TradingMetrics`, `AlertManager`, `start_metrics_server`, `metrics`, and `alert_manager`.

#### `monitoring/metrics.py`
Implements monitoring with Prometheus/Grafana structure. The `TradingMetrics` class defines Prometheus metrics for the trading system including counters (total orders, signals, trades, risk checks), gauges (current PnL, positions, latency, leverage, VaR, regime, alpha confidence), histograms (order execution time, signal generation time, feature computation time, risk check time), and summaries (latency, market data latency). The `AlertManager` provides alerting for latency spikes, circuit breaker hits, VaR exceeded, leverage exceeded, and position limit exceeded.

### Legacy Module (`legacy/`)

#### `legacy/__init__.py`
Exports components integrated from institutional_quant, quant_probability_engine, and quant_research_platform folders: `BehavioralRegime`, `BehavioralHypothesis`, `BehavioralTaxonomy`, `VetoReason`, `VetoEvent`, `SignalValidityTracker`, `GARCHParams`, `GARCHModel`, `RegimeGARCHManager`, `StrategyEvidence`, `RetiredStrategy`, and `ResearchDatabase`.

#### `legacy/behavioral_hypothesis.py`
Implements a Behavioral Hypothesis Framework. It defines `BehavioralRegime` enum for market behavioral regimes (volatility_expansion, liquidity_vacuum, mean_reversion, inventory_rebalancing, panic_squeeze) and `BehavioralHypothesis` dataclass for formal behavioral hypotheses. The `BehavioralTaxonomy` models market as probability mixture of behaviors instead of static enums, estimating behavioral mixture from market features (volatility z-score, volume z-score, spread z-score, institutional ratio, intraday return).

#### `legacy/signal_validity_tracker.py`
Implements a Signal Validity Tracker (SVT), a pre-trade safety layer that answers whether the behavioral mechanism has reason to be active. It provides vetoes for macro freeze (block trending strategies near major events), expiry pinning (block momentum strategies near expiry with high OI clustering), participant flow (block buying if FIIs net-short and cash fleeing), volatility spike (block strategies during extreme volatility), and liquidity crunch (block strategies during low liquidity).

#### `legacy/garch_volatility.py`
Implements GARCH(1,1) Volatility Modeling. It defines `GARCHParams` dataclass for GARCH(1,1) parameters (omega, alpha, beta, mu) and `GARCHModel` class with expiry segmentation, fitting two separate GARCH models for regular sessions and expiry week sessions. The model uses the equation σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2. The `RegimeGARCHManager` manages GARCH models for different regimes (regular, expiry, high vol, low vol).

### Research OS Module (`research_os/`)

#### `research_os/__init__.py`
Exports components for the research operating system.

#### `research_os/experiment.py`
Implements immutable experiment tracking. It defines `ExperimentRecord` dataclass with experiment_id, hypothesis_id, data_fingerprint, created_at, params, metrics, and parent_experiment_id. The `ExperimentStore` provides SQLite-backed persistence for experiment records with fingerprinting for reproducibility.

#### `research_os/replay.py`
Implements a deterministic replay journal inspired by event-sourced trading engines. It defines `ReplayEvent` dataclass with sequence, event_type, timestamp, and payload. The `ReplayJournal` provides an append-only replay log for research decisions with fingerprinting for verification and methods to append events, load events, and verify integrity.

### Trading Research Module (`trading_research/`)

#### `trading_research/__init__.py`
Exports `Config`, `GapFadeV2Config`, and `WalkForwardConfig` from the config module.

#### `trading_research/app.py`
Command-line entrypoint for the trading research OS with commands for extract-gaps, analyze-participants, paper-trade, and validate.

#### `trading_research/config.py`
Defines core configuration for the trading research OS including project paths, market hours, gap thresholds, lookback days, position sizing, and data source settings. It defines `GapFadeV2Config` for simple gap fade strategy, `WalkForwardConfig` for walk-forward validation, and top-level `Config` dataclass.

#### `trading_research/backtesting/walk_forward.py`
Provides walk-forward splitting and evaluation utilities. It defines `WalkForwardResult` dataclass and functions for expanding walk-forward splits, date-aware walk-forward splits, Sharpe calculation, and running walk-forward evaluation.

#### `trading_research/data/timestamp_validator.py`
Implements timestamp validation for anti-leakage checks. The `TimestampValidator` detects future timestamps in time series data with optional IST timezone enforcement.

#### `trading_research/data/validation.py`
Provides high-level data validation using the timestamp validator.

#### `trading_research/hypothesis/gap_participant_models.py`
Implements participant models for gap events. It defines `ParticipantRegime` dataclass, `RetailPanicDetector` for detecting retail panic behavior based on gap percentage, volume shock, and delivery percentage, `InstitutionalAbsorptionDetector` for detecting institutional absorption based on delivery percentage and auction imbalance, and `ParticipantRegimeClassifier` that combines both detectors to classify participant regimes.

#### `trading_research/research_os/layer_boundaries.py`
Defines layer boundaries to eliminate conceptual redundancy. It defines `SystemLayer` enum (RAW_STATE, MARKET_REGIME, PARTICIPANT_STATE, BEHAVIORAL_ACTIVATION, STRATEGY_CONTEXT) and `BoundaryEnforcer` with module-layer mapping and boundary violation tracking to prevent duplicate logic across layers.

#### `trading_research/strategies/gap_fade.py`
Implements a simple gap fade strategy used for walk-forward validation tests. It defines `StrategyResult` dataclass and `GapFadeV2` class with signal generation and backtest methods.

### Web Module (`web/`)

#### `web/dashboard.html`
Provides a web dashboard for the quant trading system with portfolio overview, active positions, strategy performance, market regime display, risk metrics, and recent signals table.

### Stats Module (`stats/`)

#### `stats/__init__.py`
Exports statistical evaluation helpers: `OutcomeDistribution`, `EvidenceBreakdown`, `EvidenceScorer`, `FeatureValidationReport`, `FeatureValidator`, `LeakageGuard`, `LeakageReport`, `ADFResult`, `TestResults`, autocorrelation, deflated_sharpe_ratio, one_sample_t_test, `WalkForwardAnalyzer`, and `WalkForwardResult`.

### Market Module (`market/`)

#### `market/__init__.py`
Exports market understanding primitives: `IntradayStructureAnalyzer`, `LiquidityAnalyzer`, `ParticipationAnalyzer`, `RegimeEngine`, `RegimeSnapshot`, `SmartMoneyStructure`, `StructureSignal`, `MarketState`, `MarketStateEngine`, and `VolatilityForecaster`.

### Hypothesis Module (`hypothesis/`)

#### `hypothesis/__init__.py`
Exports hypothesis definitions and validation: `FalsificationResult`, `FalsificationSuite`, `MechanismEvaluator`, `MechanismResult`, `HypothesisRegistry`, and `Hypothesis`.

### Portfolio Module (`portfolio/`)

#### `portfolio/__init__.py`
Exports portfolio allocation and risk sizing components: `PortfolioAllocator`, `PortfolioAllocation`, `RiskManager`, `PositionSizer`, and `PositionSizingDecision`.

### Signals Module (`signals/`)

#### `signals/__init__.py`
Exports signal primitives: `Signal` base class.

## Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Redis (for hot cache and streaming)
- PostgreSQL (for metadata)
- ClickHouse (for analytics, optional for Phase 1)
- Go 1.20+ (for data ingestion, optional for Phase 1)

### Quick Start with Docker Compose
```bash
# Clone repository
git clone <repository-url>
cd institutional-quant-research-os

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
vim .env

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Phase 1: Python-Only Setup (No Docker)
```bash
# Install Redis
docker run -d -p 6379:6379 redis:alpine

# Install PostgreSQL
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15

# Install ClickHouse (optional, for analytics)
docker run -d -p 8123:8123 clickhouse/clickhouse-server

# Install Python dependencies
pip install -r requirements.txt
```

### Phase 2: Build C++ Execution Engine (Optional)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp niftyquant_cpp*.so ../
```

## Configuration

Edit `.env` or `config/config.yaml` to configure:

### System Configuration
- System mode (paper/live/backtest)
- Data feed selection (Zerodha/Upstox)
- Target AUM and risk limits
- Trading hours and market calendar

### Alpha Strategy Settings
- ORB: Relative volume threshold, ATR multiplier
- VWAP: Trend/reversion mode, VWAP distance threshold
- PCP: Carry gap threshold, DTE range
- Vol Carry: IV percentile, delta hedge frequency

### Risk Limits
- Max position size: 5% of AUM
- Max sector exposure: 30%
- Daily circuit breaker: -3% of NAV
- VaR limit: 2% of AUM (99% 1-day)

### Execution Parameters
- Slippage model: Fixed + variable
- Venue selection: NSE vs BSE
- Order types: Market, Limit, VWAP
- Participation rates

## Usage

### Run Backtest
```bash
# Run ORB backtest
python alpha/orb_zarattini.py

# Run VWAP backtest
python alpha/vwap_trend_zarattini.py

# Run PCP backtest
python alpha/put_call_carry_shin.py

# Run hybrid backtester
python backtesting/hybrid_backtester.py
```

### Run Regime Detection
```bash
python regime/hybrid_hmm_cpd.py
```

### Run Feature Engineering
```bash
python features/feature_pipeline.py
```

### Run Paper Trading
```bash
python paper_trading/simulator.py
```

### Start Live Server
```bash
# Start FastAPI server
python live/server.py

# Or using Docker
docker-compose up quant-api
```

### Start Jupyter Notebook
```bash
# Using Docker
docker-compose up jupyter

# Access at http://localhost:8888
```

### Access Web Dashboard
```bash
# Open in browser
open web/dashboard.html

# Or serve with Python
python -m http.server 8000 --directory web
# Access at http://localhost:8000/dashboard.html
```

## Research Roadmap (0-12 months)

### Month 1-2: Data & Backtests
- Ingest NSE/BSE tick data (2020-2024)
- Reproduce 5-min ORB and VWAP backtests
- Build feature pipeline (50 features)

### Month 3-4: ML & Regime
- Train LightGBM on 2020-2022, test 2023-2024
- Build HMM regime detector
- Construct risk-parity portfolio

### Month 5-6: Paper Trading
- Paper trade with simulated slippage (2 bps)
- Achieve Sharpe > 1.0 (net) and max DD < 12%
- Go/No-Go assessment

### Month 7-9: Live Deployment (Phase 1)
- Go live with ₹5Cr (10% of target)
- Monitor daily, fix operational issues
- Scale to ₹25Cr

### Month 10-12: Scale & Enhance
- Add weekly options strategy
- Start development of C++ signal path (Phase 2)
- Add ClickHouse for analytics

## Trading Strategies

### 5-min ORB (Zarattini)
- **Time**: 9:15-9:20 AM IST (opening range), entries after 9:20
- **Logic**: Stocks with RV > 200% breaking 5-minute range
- **Universe**: Top 20 stocks by relative volume
- **Stop Loss**: 1.5x ATR
- **Target**: 2x range size
- **Best in**: Trending regimes with high volume
- **Expected Sharpe**: 1.1

### VWAP Trend (Zarattini/Aziz)
- **Instrument**: NIFTY futures
- **Logic**: Price > VWAP by 2σ with volume confirmation
- **Time-of-day**: First hour (9:15-10:15) and last hour (14:30-15:30)
- **Stop Loss**: 1.5σ from VWAP
- **Target**: 3σ from VWAP
- **Expected Sharpe**: 0.9

### Put-Call Carry (Shin)
- **Instrument**: Weekly NIFTY options
- **Logic**: Short OTM strangle on Wednesday, close Thursday
- **Carry gap**: >20bp (option-implied vs OIS)
- **GBM path-risk**: rσ√τ adjustment
- **Expected Sharpe**: 0.7

### Volatility Carry
- **Instrument**: NIFTY options
- **Logic**: Short straddle, delta-hedged
- **Entry**: IV percentile > 80
- **Hedge**: Daily delta rebalancing
- **Expected Sharpe**: 0.6

### Game-Theoretic Selection (Zhang)
- **Logic**: Model heterogeneous investor participation
- **Participants**: Retail, institutional, HFT, foreign
- **Equilibrium states**: Balanced, retail_dominant, institutional_dominant
- **Signal**: Fade dominant participant when divergent
- **Expected RankIC improvement**: 3-5%

## Risk Management

### Position Limits
- Max position size: 5% of AUM
- Max sector exposure: 30% of AUM
- Max single strategy weight: 50%
- Max leverage: 4x (warn at 3x, hard stop at 4x)

### Portfolio Controls
- Risk per trade: 0.5% of AUM
- Risk per strategy: 5% of AUM
- Total portfolio at risk: 15% (unlevered)
- Daily circuit breaker: -3% of NAV
- Weekly circuit breaker: -8% of NAV

### Correlation Risk
- Portfolio heat limit: 0.7
- Correlation penalty: Shrink weights if inter-alpha correlation > 0.5
- Portfolio-level VaR: Aggregates exposures, scales down leverage when correlation high

### Kelly Criterion
- Use 15% of optimal Kelly fraction
- Monthly adjustment based on observed Sharpe²
- Conservative sizing to account for estimation error

## Execution

### VWAP + POV Hybrid
- **VWAP schedule**: Based on historical volume profile
- **POV (Percentage of Volume)**: 10% of market volume
- **Venue selection**: NSE (5-5 lakhs/min) vs BSE (2 lakhs/min)
- **Order slicing**: 30-minute typical duration
- **Market impact**: Square-root model (k * sqrt(Q/V))

### Slippage Model
- Fixed component: 0.2 bps
- Variable component: 0.05 bps per ₹1 Cr
- Indian market: 2-5 bps for large caps, 5-10 bps for mid-caps
- Reject small-caps entirely

### Order Types
- **Market orders**: For stop-loss (10% ATR)
- **Limit orders**: With 0.5-2 bps patience
- **VWAP orders**: For large position entries/exits

## Performance Metrics

### Expected Performance (Net of Fees)
- **CAGR**: 18% - 25%
- **Sharpe Ratio**: 1.2 - 1.5
- **Maximum Drawdown**: 18% - 22%
- **Win Rate**: 22% - 26%
- **Profit Factor**: 1.3 - 1.6

### Backtest Metrics
- Total Return, CAGR
- Sharpe Ratio, Sortino Ratio
- Max Drawdown, Calmar Ratio
- Win Rate, Profit Factor
- Average Trade Return

### Execution Metrics
- Implementation Shortfall (bps)
- Average Slippage (bps)
- Participation Rate Achieved
- VWAP Performance (bps)
- Total Execution Time

### Go/No-Go Criteria
- Paper trade for 6 months
- Must achieve Sharpe > 1.0 (net of costs)
- Max drawdown < 12%
- Only then proceed to live with ₹25Cr

## DSA Choices

### Data Structures Used
- **Fenwick Tree**: Cumulative volume for VWAP (O(log N) update, O(log N) query)
- **Heap (min-heap)**: Top-20 stocks by RV (O(log N) insert, O(1) top)
- **Ring Buffer (deque)**: Tick stream (O(1) append, O(1) pop)
- **Hash Map**: Symbol → latest bar (O(1) average)
- **Segment Tree**: Range min/max for OHLC aggregates (O(log N) query)
- **Bloom Filter**: Duplicate tick detection (O(k) with 1% false positive)
- **Sparse Table**: Pre-computed volatility ranges (O(1) query after O(N log N) build)
- **Priority Queue (max-heap)**: Order book bids (O(log N) insert/delete)

### Complexity Analysis
- Tick ingestion: O(log N) per tick (Fenwick + heap)
- Feature computation: O(F * N) vectorized → ~50 * 2000 = 100k ops, <200ms
- ML inference: O(T * depth) with LightGBM → ~10ms per symbol
- Portfolio optimization: O(N * K²) with factor model (K=10) → <500ms

## Deployment

### Phase 1: Production Checklist
- [ ] Configure broker API credentials (Zerodha/Upstox)
- [ ] Set up PostgreSQL for metadata
- [ ] Configure Redis for hot cache and streaming
- [ ] Configure risk limits (0.5% per trade, 3% daily breaker)
- [ ] Test backtest with historical data (2020-2024)
- [ ] Run paper trading for 6 months
- [ ] Achieve Sharpe > 1.0 and max DD < 12%
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure alerting (PagerDuty + Slack)
- [ ] Go live with ₹5Cr initial allocation

### Phase 2: Scale Checklist (if >₹200Cr AUM)
- [ ] Migrate to C++ for signal generation
- [ ] Add ClickHouse for analytics
- [ ] Add Kafka for streaming
- [ ] Implement co-location for lower latency
- [ ] Add more sophisticated execution algorithms

### Monitoring
- Portfolio value and PnL
- Position exposure and sector breakdown
- Risk metrics (VaR, drawdown, portfolio heat)
- Execution quality metrics (slippage, implementation shortfall)
- Regime state and strategy allocation
- Latency metrics (end-to-end, per-component)
- Circuit breaker hits and recovery status

### Fallback Procedures
- If Redis down: Use local SQLite for critical state
- If Python process dies: Systemd restarts within 5 seconds
- If broker API down: Hold orders for 15 minutes, then cancel
- If data feed interruption: Halt trading, alert operations team

## Top 5 Risks

1. **Execution slippage**: Especially on small-caps (5-10× higher than backtest assumptions)
2. **Regime shift**: Prolonged sideways market where all alphas underperform
3. **Alpha decay**: Crowding of ORB strategy as it becomes well-known
4. **Technology failures**: Data feed interruption, Redis corruption
5. **Correlation spike**: All alphas lose simultaneously during crisis

## What NOT to Build

- GCN / graph neural networks (too complex, unproven)
- LSTM / Transformers (data insufficient, latency too high)
- Dispersion trading (not liquid in India)
- High-frequency market making (requires co-location, C++ expertise)
- Multi-layer ensemble of 5 models (overkill, no added value)

## License

Proprietary - Institutional Use Only

## Support

For issues and questions, refer to the internal documentation or contact the quant research team.

---

**Final Approval Status**:
- Investment Committee: ✅ CONDITIONAL (paper trade first)
- Risk Committee: ✅ APPROVED (with circuit breakers and position limits)
- CTO: ✅ APPROVED (simplified stack)
- Quant Research: ⚠️ ABSTAIN (wants longer backtest)

**Go / No-Go Decision**: **PROCEED WITH PAPER TRADING**
Live deployment only after 6 consecutive months of positive Sharpe > 1.0.

---

# QUANT RESEARCH OS v3 – INSTITUTIONAL UPGRADE PLAN

**Based on independent review by Quant PM & CTO (simulated)**  
**Current scores:** Research 85, Engineering 90, Production 70, Institutional 60  
**Target scores:** Research 95, Engineering 95, Production 90, Institutional 85  

## Overview

The v3 upgrade addresses critical gaps in institutional readiness through 12 major components covering governance, monitoring, positioning, data, market understanding, and automated alpha generation. This upgrade transforms the system from a research prototype to a production-grade institutional platform capable of managing ₹500Cr AUM.

## 1. Research Governance Layer

**Problem:** No formal process to decide when a strategy is dead, degrading, or ready to scale.

**Solution: Strategy Lifecycle Management**

The system implements a formal 6-phase lifecycle for all strategies:

- **RESEARCH**: Jupyter exploration, backtesting, peer review
- **PAPER_TRADE**: Live data without real capital, 60 days minimum
- **LIVE_10PCT**: 10% of target allocation, 30 days
- **LIVE_FULL**: Full allocation
- **DECAY**: Monitoring phase with potential position reduction
- **RETIRED**: Deactivated and archived

**Gate Criteria:**
- PAPER_TRADE → LIVE_10PCT: Sharpe > 1.0 (net), maxDD < 15%, win rate > 40%
- LIVE_10PCT → LIVE_FULL: Rolling 20d Sharpe > 0.8, no consecutive 5 losing days
- LIVE_FULL maintenance: Rolling 20d Sharpe > 0.6 for 60 days

**Health Metrics Tracked:**
- Rolling Sharpe (20d)
- Rolling Sortino (20d)
- Alpha half-life (exponential decay fit)
- Turnover stability (coefficient of variation)
- Feature drift (PSI per feature)
- Regime stability

**Decision Rules:**
- Rolling Sharpe < 0.3 for 10 days → enter DECAY
- Rolling Sharpe < 0 for 15 days → RETIRE
- Alpha half-life < 30 days → reduce allocation by 50%
- Feature drift > 0.3 for any top-5 feature → flag for review

## 2. Alpha Decay Detection

**Problem:** No automated monitoring of whether alpha is still alive.

**Solution: Signal Monitor**

Daily monitoring of strategy performance with automated alerts:

**Metrics Tracked Per Strategy:**
- Current rolling Sharpe (20d) vs expected Sharpe
- Current rolling win rate vs expected win rate
- Current information coefficient vs expected IC
- Current max drawdown (20d) vs expected max drawdown

**Alert Levels:**
- **P1 (Slack urgent)**: Sharpe drop > 30% relative to expected, IC drop > 40%
- **P2**: Drawdown exceeds expected 1.5x, win rate drop > 20%

**Automated Actions:**
- P1 alert → automatically reduce strategy allocation by 50%
- P2 alert → notify quant lead for manual review

**Implementation:** Store daily returns, expected returns, and IC in ClickHouse. Run SQL query each morning. Dashboard in Grafana with historical overlay.

## 3. Feature Drift Monitoring

**Problem:** LightGBM assumes training distribution equals production distribution – often false.

**Solution: Feature Drift Monitor**

Daily monitoring of feature distribution changes:

**Metrics:**
- Population Stability Index (PSI) for each feature
- KL Divergence for categorical features
- Feature importance change (relative rank)

**Thresholds:**
- PSI < 0.1 → stable
- 0.1 ≤ PSI < 0.2 → moderate drift (alert P2)
- PSI ≥ 0.2 → high drift (alert P1, trigger model retraining)

**Automatic Retraining:**
- Any top-10 feature has PSI > 0.2 → schedule retraining within 24 hours
- Average PSI > 0.15 → increase retraining frequency to every 3 days

**Output:** Drift report (JSON) stored in ClickHouse, Grafana panel showing PSI over time.

**Implementation:** Use `scipy.stats` for KL divergence, custom PSI function in Python.

## 4. Meta Alpha Layer

**Problem:** Currently signal → portfolio directly. No layer that learns which alpha works in which regime.

**Solution: Meta Model**

A meta-learning layer that predicts alpha-specific weights based on current market state:

**Inputs:**
- Daily returns of each base alpha (lagged)
- Regime (from MarketStateEngine)
- Feature drift scores
- Alpha health metrics (rolling Sharpe, IC)

**Model:**
- LightGBM classifier per alpha: should this alpha be active? (binary)
- Second model: weight multiplier (regression)

**Training:**
- Rolling 3 years, predict next 1 year
- Features: regime (one-hot), alpha lag returns (5d, 20d), VIX change, turnover change

**Live Operation:**
- Inference daily after close
- Output: alpha_weights = base_weights * meta_multiplier (if active)
- Fallback: equal weights if meta model confidence < 0.6

**Expected Impact:** Sharpe improvement of +0.2–0.3

## 5. Capacity Analysis

**Problem:** No scalability curve; unknown at which AUM Sharpe degrades.

**Solution: Capacity Simulator**

Systematic analysis of strategy scalability:

**Process:**
- Run backtest with increasing position sizes: ₹1Cr, ₹5Cr, ₹10Cr, ₹25Cr, ₹50Cr, ₹100Cr, ₹200Cr, ₹500Cr
- Use realistic market impact model (square root law, calibrated to Indian data)
- Compute Sharpe at each size

**Output:**
- Capacity curve (CSV) stored in ClickHouse
- Sharpe(AUM) = Sharpe0 * exp(-AUM / capacity_limit)

**Capacity Limit Definition:**
- AUM at which Sharpe drops by 20% from its peak

**Live Monitoring:**
- If current AUM > 0.7 × capacity_limit → alert to reduce allocation or cap inflows

**Example Outputs:**
- 5-min ORB: capacity_limit = ₹120Cr
- VWAP futures: capacity_limit = ₹800Cr

**Implementation:** Modify backtester to scale positions proportionally to AUM. Add impact model that increases slippage with trade size.

## 6. Walk-Forward Research OS

**Problem:** Single backtest not trusted; need continuous validation.

**Solution: Automated Walk-Forward Framework**

Monthly automated validation for all active strategies:

**Data Splits:**
- Train: earliest 3 years
- Validate: next 1 year
- Test: next 1 year
- Forward (out-of-sample): most recent 1 year (not used in training)

**Process:**
1. For each strategy, optimize hyperparameters on train+validate
2. Evaluate on test
3. If test Sharpe > 1.0, proceed to forward validation
4. Compare forward Sharpe to test Sharpe; if drop > 30%, flag for review

**Storage:**
- All splits, parameters, results stored in ClickHouse + Git LFS
- Versioned by strategy and date

**Dashboard:**
- Shows Sharpe progression: train → test → forward
- Highlights strategies with consistent degradation

**Alert:**
- If forward Sharpe < 0.5 × test Sharpe → P1 alert (possible overfitting)

## 7. Probabilistic Forecasting

**Problem:** Binary BUY/SELL signals ignore uncertainty.

**Solution: Prediction with Confidence**

Enhanced signal output with uncertainty quantification:

**Output Format:**
```json
{
  "expected_return": 1.3,
  "prob_positive": 0.64,
  "confidence": 0.78,
  "uncertainty_band": [0.7, 2.1]
}
```

**Calibration:**
- Use validation set to fit logistic regression: predicted_prob → actual_prob
- Store calibration curve

**Use in Position Sizing:**
- position_size ∝ expected_return * prob_positive * confidence

**Implementation:** Modify LightGBM to output raw logits; apply sigmoid; calibrate.

## 8. Bayesian Position Sizing

**Problem:** Kelly alone is too aggressive, ignores confidence and regime.

**Solution: Bayesian Position Sizing**

Multi-factor position sizing with risk adjustments:

**Formula:**
```
base_position = Kelly_fraction * (expected_return / variance)
adjusted_position = base_position
    * confidence_score
    * regime_score (0.5 for high_vol, 1.0 for normal)
    * liquidity_score (1.0 if position < capacity_limit * 0.7, else decays)
    * feature_drift_penalty (1 - PSI/0.5 capped at 0)
```

**Risk Caps:**
- Single trade risk ≤ 0.5% of AUM (unlevered)
- Total portfolio risk ≤ 10% of AUM (unlevered)

**Clipping:**
- If adjusted_position > 2× Kelly → clip to 2× Kelly
- If adjusted_position < 0.1× Kelly → set to 0 (skip trade)

**Implementation:** Compute all factors in real-time using Redis; pass to order generator.

## 9. Crisis Simulator

**Problem:** Strategies not tested against historical crashes.

**Solution: Stress Testing Suite**

Comprehensive stress testing against historical market crises:

**Scenarios:**
- COVID_2020: NIFTY dropped 40% in March 2020
- Adani_2023: Short seller report, Adani stocks crashed 70%
- Russia_Ukraine_2022: NIFTY gap down 5%, volatility spike
- Flash_Crash_2015: 5% drop in 10 minutes (synthetic)
- Rate_hike_2022: 200bps increase in 3 months
- Liquidity_crisis: Zero volume for 1 hour

**Simulation:**
- Replay market data during each crisis window
- Run all strategies with current parameters
- Record max drawdown, daily loss, VaR violation count

**Pass Criteria:**
- Max drawdown < 25% (unlevered)
- No day with > 5% loss (unlevered)
- VaR violation rate < 5%

**Action if Fail:**
- Strategy must have a crisis-override rule (e.g., stop all trading when VIX > 35)
- Or reduce allocation permanently

## 10. Alternative Data (India-Specific)

**Problem:** Missing high-value Indian market edges.

**Solution: Alternative Data Pipeline**

Priority data sources for Indian markets:

**Priority High:**
- FII/DII net flows (daily, source: NSE website)
- NSE delivery percentage (daily)
- Open interest changes for NIFTY & BANKNIFTY (intraday)
- Put-Call ratio (OI and volume, hourly)
- India VIX (1-min)

**Priority Medium:**
- RBI policy calendar (dates, expectations)
- Earnings calendar (company-wise)
- Sector breadth (advance/decline per sector)
- NIFTY 50 constituent changes (quarterly)

**Implementation:**
- Scrapers for NSE/BSE websites (daily)
- Store in ClickHouse with timestamps
- Features: flow_change_1d, flow_change_5d, delivery_ratio_trend, OI_trend

**Expected Alpha Increase:** Using FII/DII as a feature alone can add 0.1-0.2 Sharpe (India-specific edge).

## 11. Market State Engine (Enhanced)

**Problem:** Regimes are too coarse; need more granular "states".

**Solution: Market State Engine**

Multi-dimensional market state classification:

**Dimensions:**
- Trend: strong_up, weak_up, sideways, weak_down, strong_down
- Volatility: very_low, low, medium, high, very_high
- Breadth: expanding, contracting, neutral
- Sentiment: extreme_fear, fear, neutral, greed, extreme_greed

**States (12 total):**
- bull_overextended: trend=strong_up + breadth=expanding + sentiment=extreme_greed
- bull_accumulation: trend=weak_up + breadth=neutral + sentiment=neutral
- bull_distribution: trend=weak_up + breadth=contracting + sentiment=greed
- panic_pullback: trend=strong_down + volatility=very_high + sentiment=extreme_fear
- (8 additional states)

**Detection:**
- Daily clustering (k-means) on the four dimensions
- Supervised classifier to assign state name

**Usage:**
- Alpha weights change per state
- Position sizing multiplier per state
- Risk limits tighten in panic states

## 12. Alpha Factory

**Problem:** Alphas are manually researched, not systematically generated.

**Solution: Automated Alpha Factory**

Genetic programming for systematic alpha generation:

**Input:** 50 core features (1-min, daily)

**Process:**
- Generate candidate alphas via genetic programming (GP)
  - Population: 500
  - Generations: 20
  - Operators: +, -, *, /, log, rank, lag, rolling_mean, rolling_std, crossover
- Each candidate evaluated on 3-year train, 1-year test
- Keep candidates with OOS Sharpe > 1.0
- Rank by Sharpe / turnover (efficiency)

**Output:** Top 10 new alphas per month

**Human-in-the-Loop:**
- Quant reviews top 3, rejects overfit or non-interpretable ones
- Accepted alphas enter paper trading pipeline

**Constraints:**
- Max feature depth = 3 (no over-complex expressions)
- Max turnover < 200% per day

**Infrastructure:**
- Runs overnight on EC2 spot instances
- Results stored in ClickHouse

**Expected Outcome:** 1–2 new production alphas per quarter

## Final Architecture V3 Summary

```yaml
Quant_Research_OS_v3:
  governance: StrategyLifecycle + MetaAlphaLayer + WalkForwardOS
  monitoring: SignalMonitor + FeatureDriftMonitor + CrisisSimulator
  positioning: ProbabilisticForecasting + BayesianPositionSizing
  data: AlternativeData pipeline (FII/DII, delivery, OI, PCR)
  market: MarketStateEngine (12 states)
  alpha_gen: AlphaFactory (genetic programming)
  capacity: CapacityAnalysis (curves per strategy)
  compliance: Audit trail + blacklist + limit monitoring
  infrastructure: CI/CD, staging, multi-AZ, backup broker
```

## Revised Implementation Roadmap

| Phase | Duration | Focus | Key Deliverables |
|-------|----------|-------|------------------|
| Phase 1 | 30 days | Governance & Monitoring | StrategyLifecycle, SignalMonitor, FeatureDriftMonitor |
| Phase 2 | 30 days | Positioning & Data | Probabilistic forecasting, Bayesian sizing, alternative data pipeline |
| Phase 3 | 30 days | Meta & Factory | MetaAlphaLayer, AlphaFactory (GP) |
| Phase 4 | 30 days | Testing & Capacity | WalkForwardOS, CapacityAnalysis, CrisisSimulator |
| Phase 5 | 30 days | Production Hardening | DR, CI/CD, staging, audit trail |

**Total effort:** 5 months, 3 engineers + 2 quants.

## Final Scores (Post-Upgrade)

| Dimension | Score |
|-----------|-------|
| Research completeness | 95/100 |
| Engineering completeness | 95/100 |
| Production readiness | 90/100 |
| Institutional readiness | 85/100 |

## Expected Live Performance (Post-Upgrade, ₹500Cr AUM)

- **Sharpe:** 1.2 – 1.6
- **CAGR:** 15 – 22%
- **Max DD:** 18 – 22%

## Recommended Annual Research Budget

₹3–4 crore (including data, compute, and 5 FTE).

## Final Sign-Off

- **Chief Quant Officer:** ✅ Approved
- **CTO:** ✅ Approved
- **Risk Committee:** ✅ Approved with condition that crisis simulator passes all scenarios before live scaling

**We are ready to build.**

---

# QUANT RESEARCH OS V3.5: THE CONSTRAINED EVOLUTIONARY ORG

**A realistic, compliant, and cost-constrained evolutionary framework for automated alpha discovery.**

## PREAMBLE: REVISING THE V4 VISION

The original V4 blueprint proposed "The Civilization," an autonomous network of 12+ AI agents executing massive genetic programming cycles across thousands of AWS EC2 instances. While visionary, a rigorous institutional audit revealed severe structural flaws: prohibitive computational costs, coordination chaos, look-ahead and regime biases, lack of regulatory compliance (SEBI guidelines on explainability), and unrealistic return/funding expectations.

This revised **V3.5 specification** addresses all 23 audit issues directly. It pivots the architecture toward a highly disciplined, resource-constrained, and fully compliant agentic framework—**The Constrained Evolutionary Org**.

| Dimension | V4 Assumption | V3.5 Reality (Solution) |
|---|---|---|
| **Compute Cost** | Parallel 10,000 EC2 instances | Adaptive population (starts at 100) + surrogate fitness proxies + early stopping |
| **Agent Coordination** | 12+ independent agents (chaos) | Reduced to 5 core agents + centralized priority-queue message bus |
| **GP Feature Discovery** | General Python code (look-ahead leak) | Constrained GP DSL + timestamp-aware operators + correlation leakage checks |
| **Validation Overfitting** | Nested walk-forward (leaky time series) | Purged walk-forward validation + 3-month embargo periods |
| **Synthetic Data** | Unconstrained VAE/Diffusion | Regime-conditional GANs constrained by stylized facts + discriminator check |
| **Online Learning Drift** | Continuous streaming updates (forgetting) | Hybrid model (weekly base retraining + Kalman residual corrections) + experience replay |
| **Position Sizing** | Infinite liquidity Kelly fraction | Liquidity decay factor + 10% ADV participation cap + correlation-adjusted capacity |
| **Risk Counterfactuals** | O(N²) all-portfolio simulations | Local counterfactuals (positions > 5% AUM) + predefined scenario library |
| **SEBI Compliance** | Explainability hand-waved | Post-hoc SHAP/LIME explanation + Human-in-the-Loop strategy approval |
| **Project Roadmap** | 10-year fantasy roadmap | Rolling 2-year modular plan |
| **Funding & Returns** | ₹500 crore upfront budget, 500% returns | Phased, milestone-based budget (₹10cr Year 1), 20-30% net CAGR targets |
| **Knowledge Graph** | Infinite O(N) query growth | Temporal sharding (yearly shards) + importance pruning of inactive nodes |

---

## PHASE 1: SYSTEM ARCHITECTURE & 5 CORE AGENTS

To prevent coordination chaos and communication explosion, we consolidate the system into 5 core agents managed by a centralized Orchestrator over a priority message bus.

```
                  ┌─────────────────────────────────────┐
                  │        Centralized Message Bus      │
                  │         (with Priority Queues)      │
                  └──────────────────┬──────────────────┘
                                     │
         ┌───────────────┬───────────┼───────────┬───────────────┐
         ▼               ▼           ▼           ▼               ▼
   ┌───────────┐   ┌───────────┐ ┌───────────┐ ┌───────────┐   ┌───────────┐
   │ Research  │   │   Alpha   │ │ Validator │ │   Risk    │   │ Execution │
   │   Agent   │   │ Generator │ │   Agent   │ │   Agent   │   │   Agent   │
   │           │   │  (GP/DSL) │ │ (Purged)  │ │(Capacity) │   │ (ADV Cap) │
   └───────────┘   └───────────┘ └───────────┘ └───────────┘   └───────────┘
```

### 1.1 Agent Roles

1. **Research Agent (Orchestrator)**: Formulates high-level research hypotheses based on academic papers and knowledge graph. Proposes feature sets and directions.
2. **Alpha Generator Agent**: Receives research directions and conducts genetic programming (GP) based search using a highly constrained Domain-Specific Language (DSL). Applies surrogate fitness checks.
3. **Validator Agent**: Evaluates candidate strategies submitted by the Generator using purged walk-forward validation and embargoes. Applies complexity penalties.
4. **Risk Agent**: Computes portfolio risks (VaR, CVaR), runs stress tests using the scenario library, performs local counterfactual simulations, and maintains the correlation-adjusted aggregate capacity model.
5. **Execution Agent**: Translates validated strategy weights into execution orders. Applies liquidity decay factors and handles the 10% ADV participation caps.

### 1.2 Communication Protocol
Agents communicate via a centralized message bus using priority queues to prevent race conditions and message flooding. All agents share a unified reward metric: **global portfolio Net Sharpe ratio improvement**.

---

## PHASE 2: COST-CONSTRAINED ALPHA SEARCH (GP)

Rather than assuming infinite compute, the Alpha Generator Agent implements strict budget-preserving algorithms:

1. **Adaptive Population Sizing**: GP search starts with a population of 100 candidates. The search space is expanded only for lineages that show promising out-of-sample improvements.
2. **Surrogate Fitness Proxy**: Before running expensive multi-year backtests, candidate strategies are evaluated using a cheap proxy model (evaluating basic Information Coefficient and turnover over a 6-month subset). 99% of unpromising candidates are filtered here.
3. **Fitness Inheritance**: Child strategies inherit the fitness score of their parent unless significant structural mutations are applied, saving compute on minor parameter tweaks.
4. **Early Stopping**: Backtest runs terminate early if a candidate exhibits high drawdowns or zero information coefficient in the first year of the evaluation window.
5. **Complexity Cap**: We enforce minimum description length (MDL) and AIC/BIC penalties ($-\text{complexity\_penalty} \times k$). Any GP expression exceeding 10 total operations is rejected.

---

## PHASE 3: ROBUST VALIDATION & LEAKAGE CONTROL

To prevent strategies from memorizing noise or using future information, the Validator Agent applies three structural checks:

1. **Purged Walk-Forward**: Time-series cross-validation includes a 1-year **purge gap** between training and testing folds to eliminate overlapping returns.
2. **Embargo Period**: A 3-month **embargo period** is added immediately after each test fold to prevent look-ahead spillover from auto-correlated features.
3. **Leakage Detection**: All GP operators are timestamp-aware. A correlation check is performed; any candidate showing correlation with future returns that is not fully explained by lagged features (leakage score > 0.01) is instantly rejected.

---

## PHASE 4: RISK & POSITION SIZING

V3.5 models market liquidity constraints and correlation-adjusted risk at scale:

1. **Liquidity Decay Factor**: Position sizing incorporates market impact. The allocated size is scaled as:
   $$\text{position} = \text{base\_position} \times \exp\left(-\frac{\text{position}}{\text{capacity\_limit}}\right)$$
2. **Participation Cap**: Strategy executions are restricted to a maximum participation rate of 10% of the average daily volume (ADV) per minute.
3. **Correlation-Adjusted Capacity**: Portfolio capacity is modeled dynamically. If two strategies are highly correlated, their combined capacity allocation is reduced.
4. **Local Counterfactual Risk**: Instead of running full portfolio simulations across the entire timeline, the Risk Agent runs counterfactuals only for the largest positions (>5% of AUM).
5. **Predefined Scenario Library**: Incorporates 100 historical and synthetic market stress scenarios (e.g., VIX +10%, 2008 Lehman collapse, 2020 COVID crash) to run fast parametric evaluations.

---

## PHASE 5: REGULATORY COMPLIANCE & MLOPS

To comply with SEBI algorithmic trading guidelines and ensure operational stability:

1. **Strategy Approval Workflow (Human-in-the-Loop)**: AI agents cannot deploy strategies live. Any strategy passing the Validator Agent enters a staging environment for 60 days. Final deployment requires sign-off from a human quant.
2. **Human-Readable Decision Logs**: The system logs a human-readable text explanation detailing the mathematical logic and input signals behind every trade placement.
3. **Post-Hoc Explainability**: MLOps pipelines output SHAP or LIME value breakdowns for all model decisions.
4. **Online Learning Drift Control**: We implement a hybrid learning loop. A base model is retrained weekly, and online corrections are applied via a Kalman filter on residuals. To prevent catastrophic forgetting, retraining utilizes experience replay (sampling historical regimes evenly).

---

## PHASE 6: PRUNED KNOWLEDGE GRAPH

To maintain low latency on graph queries, the Knowledge Graph is optimized with:

1. **Temporal Sharding**: Nodes (Papers, Features, Alphas) and edges are partitioned into yearly shards. Queries default to the current year's active shard.
2. **Importance Pruning**: The database automatically archives nodes and edges that have not been cited or reused in any experiment for over 6 months and have zero citations.

---

## PHASE 7: BUDGET, ROADMAP, & EXPECTED PERFORMANCE

### 7.1 Phased Funding
- **Year 1 (Proof of Concept)**: ₹10 crore (budget for 10 base features, 5 core agents, tick data, and core infrastructure).
- **Year 2 (Scaling)**: ₹50 crore (scale GP search, add news/alternative data, deploy live trading).
- **Year 3 (Full Deployment)**: ₹200 crore (expand AUM, optimize execution hot paths).

### 7.2 Expected Performance
- **Target Sharpe**: 1.2 – 1.6 (out-of-sample)
- **Net CAGR**: 20% – 30% (realistic target for Indian equity markets)
- **Expected Alpha Velocity**: 10 new high-quality alphas per year (net of decay).

### 7.3 Rolling 2-Year Roadmap
- **Q1-Q2 Year 1**: Consolidate agents into 5 core roles, implement priority message bus.
- **Q3-Q4 Year 1**: Deploy purged walk-forward validation and surrogate fitness GP.
- **Q1-Q2 Year 2**: Build temporal sharding in Knowledge Graph and SEBI compliance logs.
- **Q3-Q4 Year 2**: Integrate local counterfactual risk engine and execute live trading with AUM scaling.
