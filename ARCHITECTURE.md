# Institutional Quant Research OS - Layered Architecture

## Architecture Overview

This system follows a strict layered architecture with clear separation of concerns. Each layer has exactly one responsibility and well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                             │
│  (Dashboard, API, CLI)                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Execution Layer                                │
│  (Order Management, Trade Execution, Position Management)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Portfolio Layer                                │
│  (Portfolio Construction, Risk Management, Allocation)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Research Layer                                │
│  (Alpha Generation, Signal Generation, Strategy Research)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Layer                                  │
│  (Feature Engineering, Feature Store, Feature Computation)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Layer                                     │
│  (Data Ingestion, Validation, Truth Database)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                            │
│  (Event Bus, Caching, Storage, Monitoring)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### 1. Infrastructure Layer (Bottom)
**Responsibility:** Provide foundational services for all layers.

**Components:**
- Event Bus (architecture/event_bus.py)
- Caching (Redis)
- Storage (SQLite, PostgreSQL, Arctic)
- Monitoring (Prometheus, Grafana)
- Logging (Structured logging)

**Interfaces:**
- Event publishing/subscribing
- Cache get/set operations
- Storage read/write operations
- Metric emission

**Constraints:**
- No business logic
- No knowledge of financial concepts
- Pure infrastructure services

### 2. Data Layer
**Responsibility:** Ingest, validate, and provide clean market data.

**Components:**
- Data Ingestion (core/data_layer.py)
- Data Validation Pipeline (core/data_validation_pipeline.py)
- Truth Database (core/truth_database.py)
- Data Quality Engine (core/data_quality_engine.py)

**Interfaces:**
```python
class DataProvider:
    def get_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame
    def get_latest_tick(symbol: str) -> Tick
    def subscribe(symbols: List[str], callback: Callable)
```

**Constraints:**
- Only returns validated data
- No feature computation
- No signal generation
- Single source of truth (Truth Database)

### 3. Feature Layer
**Responsibility:** Compute and store features from raw market data.

**Components:**
- Feature Engineering (features/advanced_feature_engineering.py)
- Feature Store (features/feature_store.py)
- Feature Pipeline (features/feature_pipeline.py)

**Interfaces:**
```python
class FeatureProvider:
    def compute_features(data: pd.DataFrame) -> pd.DataFrame
    def get_features(symbol: str, features: List[str]) -> pd.DataFrame
    def store_features(features: pd.DataFrame, metadata: Dict)
```

**Constraints:**
- Only reads from Data Layer
- No signal generation
- No trading logic
- Feature versioning and lineage

### 4. Research Layer
**Responsibility:** Generate alpha signals and conduct research.

**Components:**
- Alpha Generation (alpha/momentum_strategies.py, alpha/mean_reversion_strategies.py)
- Signal Generation (src/alpha/manager.py)
- Strategy Research (research/experiments/)
- Model Training (ml/, models/)

**Interfaces:**
```python
class SignalGenerator:
    def generate_signals(data: pd.DataFrame) -> List[Signal]
    def get_signal_confidence(signal: Signal) -> float
    def validate_signal(signal: Signal) -> bool
```

**Constraints:**
- Only reads from Feature Layer
- No execution logic
- No position management
- Pure research and signal generation

### 5. Portfolio Layer
**Responsibility:** Construct portfolios and manage risk.

**Components:**
- Portfolio Construction (portfolio/construction/)
- Risk Management (risk/risk_engine.py)
- Allocation (portfolio/hrp_optimizer.py)
- Position Sizing

**Interfaces:**
```python
class PortfolioManager:
    def construct_portfolio(signals: List[Signal]) -> Portfolio
    def manage_risk(portfolio: Portfolio) -> Portfolio
    def allocate_capital(signals: List[Signal]) -> Dict[str, float]
```

**Constraints:**
- Only reads from Research Layer
- No order execution
- No direct market interaction
- Pure portfolio construction and risk management

### 6. Execution Layer
**Responsibility:** Execute trades and manage positions.

**Components:**
- Order Management (execution/order_types.py)
- Trade Execution (execution/live/, execution/paper/)
- Position Management (portfolio/trade_logger.py)
- Cost Models (execution/cost_models/)

**Interfaces:**
```python
class ExecutionEngine:
    def execute_order(order: Order) -> ExecutionResult
    def get_positions() -> Dict[str, Position]
    def cancel_order(order_id: str) -> bool
```

**Constraints:**
- Only reads from Portfolio Layer
- No research logic
- No portfolio construction
- Pure execution and position management

### 7. Presentation Layer (Top)
**Responsibility:** Present information to users and accept commands.

**Components:**
- Dashboard (web/dashboard.html, dashboard/api/)
- API Server (dashboard/api/api_server.py)
- CLI (src/monitoring/dashboard_cli.py)

**Interfaces:**
```python
class PresentationLayer:
    def display_dashboard()
    def handle_api_request(request) -> Response
    def execute_command(command) -> Result
```

**Constraints:**
- Only reads from Execution and Research layers
- No business logic
- No trading decisions
- Pure presentation and user interaction

## Data Flow

### Read Path (Bottom to Top)
```
Infrastructure → Data → Features → Research → Portfolio → Execution → Presentation
```

### Write Path (Top to Bottom)
```
Presentation → Execution → Portfolio → Research → Features → Data → Infrastructure
```

## Event-Driven Communication

Layers communicate via events through the Event Bus:

```python
# Event Types
DATA_UPDATED = "data.updated"
FEATURES_COMPUTED = "features.computed"
SIGNAL_GENERATED = "signal.generated"
PORTfolio_CONSTRUCTED = "portfolio.constructed"
ORDER_EXECUTED = "order.executed"
```

## Layer Interface Rules

1. **Upward calls only:** A layer can only call the layer directly below it
2. **No skipping:** Never skip layers (e.g., Research never calls Data directly)
3. **Event-driven:** Cross-layer communication via events only
4. **Interface-based:** All communication via defined interfaces
5. **Versioned interfaces:** Interfaces are versioned for backward compatibility

## Module Organization

```
institutional-quant-research-os/
├── core/                    # Infrastructure + Data Layer
│   ├── data_layer.py       # Data ingestion
│   ├── data_validation_pipeline.py  # Data validation
│   ├── truth_database.py    # Single source of truth
│   ├── data_quality_engine.py  # Data quality monitoring
│   └── event_bus.py        # Event bus (infrastructure)
├── features/                # Feature Layer
│   ├── feature_store.py    # Feature storage
│   ├── feature_pipeline.py # Feature computation
│   └── advanced_feature_engineering.py
├── alpha/                   # Research Layer
│   ├── momentum_strategies.py
│   ├── mean_reversion_strategies.py
│   └── manager.py          # Signal generation
├── portfolio/               # Portfolio Layer
│   ├── construction/
│   ├── trade_logger.py
│   └── hrp_optimizer.py
├── execution/               # Execution Layer
│   ├── live/
│   ├── paper/
│   └── cost_models/
├── models/                  # Research Layer (ML)
│   ├── model_registry.py
│   └── prediction_registry.py
├── research/                # Research Layer
│   ├── experiments/
│   └── validation/
├── risk/                    # Portfolio Layer
│   └── risk_engine.py
├── dashboard/               # Presentation Layer
│   └── api/
└── web/                     # Presentation Layer
    ├── dashboard.html
    └── dashboard.js
```

## Migration Strategy

### Phase 1: Data Layer (Completed)
- ✅ Data validation pipeline
- ✅ Truth database
- ✅ Data quality engine
- ✅ Integration with data_layer.py

### Phase 2: Feature Layer
- Create feature layer interfaces
- Migrate feature computation
- Implement feature store integration

### Phase 3: Research Layer
- Create research layer interfaces
- Separate signal generation from execution
- Implement proper event communication

### Phase 4: Portfolio Layer
- Create portfolio layer interfaces
- Separate portfolio construction from execution
- Implement risk management integration

### Phase 5: Execution Layer
- Create execution layer interfaces
- Separate execution from research
- Implement proper order management

### Phase 6: Presentation Layer
- Create presentation layer interfaces
- Separate presentation from business logic
- Implement proper API structure

## Testing Strategy

Each layer has its own test suite:

```
tests/
├── test_data_layer.py
├── test_feature_layer.py
├── test_research_layer.py
├── test_portfolio_layer.py
├── test_execution_layer.py
└── test_presentation_layer.py
```

Integration tests verify layer interactions:

```
tests/integration/
├── test_data_to_feature_flow.py
├── test_feature_to_research_flow.py
├── test_research_to_portfolio_flow.py
└── test_portfolio_to_execution_flow.py
```

## Performance Considerations

1. **Data Layer:** Optimized for read throughput (caching, indexing)
2. **Feature Layer:** Batch computation, vectorized operations
3. **Research Layer:** Parallel signal generation
4. **Portfolio Layer:** Fast optimization algorithms
5. **Execution Layer:** Low-latency order execution
6. **Presentation Layer:** Responsive UI, efficient data transfer

## Security Considerations

1. **Data Layer:** Encrypted storage, secure connections
2. **Feature Layer:** Feature access control
3. **Research Layer:** Model versioning, audit trail
4. **Portfolio Layer:** Position limits, risk checks
5. **Execution Layer:** Order authentication, trade confirmation
6. **Presentation Layer:** API authentication, role-based access

## Monitoring

Each layer emits metrics:

1. **Data Layer:** Data freshness, validation pass rate, source health
2. **Feature Layer:** Feature computation time, cache hit rate
3. **Research Layer:** Signal generation rate, IC values
4. **Portfolio Layer:** Portfolio turnover, risk metrics
5. **Execution Layer:** Order latency, fill rates
6. **Presentation Layer:** API response time, error rates

## Ownership Documentation

Each layer has a designated owner responsible for:
- Layer architecture and design
- Interface definitions
- Performance monitoring
- Bug fixes and improvements
- Documentation updates

See OWNERSHIP.md for detailed ownership information.
