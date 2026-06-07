# Forensic Audit Report: Institutional Quant Research OS

**Audit Date**: June 7, 2026  
**Auditors**: Hedge Fund CTO, Principal Software Architect, Quant Research Director, Senior Backend Engineer, Senior Frontend Engineer, Database Architect, DevOps Engineer, QA Lead, Production Reliability Engineer.  
**Auditee**: Institutional Quant Research OS Base  

---

# PHASE 1: REPOSITORY MAP

The repository follows a hybrid structure where legacy components coexist with a modern, consolidated architecture located under `src/`. Below is a comprehensive map of how files, services, databases, and event streams connect.

```
institutional-quant-research-os/
├── main.py                           # System Entrypoint & Orchestrator (live/backtest)
├── app.py                            # CLI Dispatcher for Research Demos
├── config_v3.py                      # Global Platform Configuration Constants
├── requirements.txt                  # Python dependencies
├── requirements-minimal.txt          # Minimal local dependencies
├── dashboard/                        # API Backend Server & Dashboard Routes
│   └── api/api_server.py             # FastAPI App serving endpoints & WebSocket
├── web/                              # Restored Frontend Assets
│   ├── dashboard.html                # Main UI Template
│   ├── css/dashboard.css             # Glassmorphism Stylesheet
│   └── js/dashboard.js               # Connected WebSocket Client & Polling Logic
├── src/                              # ── CANONICAL PRODUCTION PACKAGES ──
│   ├── alpha/                        # Alpha Engine, Prediction Registry, Evolvers
│   │   ├── alphas/                   # Base Class and Signal wrappers
│   │   ├── signals/                  # Strategy implementations (Momentum, Mean Reversion)
│   │   └── research/                 # Advanced neural net (GCN) & trend cycle strategies
│   ├── regime/                       # Change-Point & GMM-HMM detectors
│   ├── features/                     # Versioned Feature Store & Compute engines
│   ├── execution/                    # Quoting engines, brokers, and fill simulators
│   ├── backtest/                     # Event-driven and Vectorized backtesters
│   ├── portfolio/                    # HRP & Kelly Optimizers and Combiners
│   ├── risk/                         # Advanced Risk metrics (Weibull VaR, CVaR)
│   ├── ml/                           # walk-forward ML models & Purged Walk-Forward
│   ├── monitoring/                   # Feature drift monitors, Prom metrics
│   ├── data/                         # Data gates & loaders
│   └── shared/                       # Databases, DSA, and Math utilities
├── alpha/                            # ── LEGACY STRATEGIES & FACADES ──
├── analytics/                        # ── LEGACY BACKTESTING & VALIDATION ──
├── execution/                        # ── LEGACY EXECUTION ADAPTERS ──
├── risk/                             # ── LEGACY RISK AGGREGATORS ──
├── features/                         # ── LEGACY FEATURE COMPUTERS ──
├── data/                             # ── LEGACY DATA universe trackers & DBs ──
└── tests/                            # ── AUTOMATED TEST SUITE ──
```

### Component Interconnection & Signal Flow
The core quantitative loop flows data through these layers:

```
[ NSE WebSocket / yfinance ]
            │
            ▼
[ src/data/quality_gate.py ]  ──(Rejection)──> [ Halt Trading ]
            │
            ▼ (Clean Data)
[ src/features/compute/ ] ──> [ SQLite/Redis Feature Cache ]
            │
            ▼ (Features)
[ src/regime/hmm/ ] ──(Regime classification)──┐
            │                                  ▼
[ src/alpha/manager.py ] ──(Raw Signals)──> [ src/portfolio/engine.py ]
                                               │ (HRP/Kelly sizing)
                                               ▼
                                            [ src/risk/advanced_metrics.py ] ──(VaR limits)──┐
                                               │                                             │ (Breach)
                                               ▼ (Cleared positions)                         ▼
                                            [ src/execution/order_manager/ ] ─────────> [ Halt Position ]
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼ (Simulation Mode)                             ▼ (Production Mode)
         [ src/execution/fill_simulator ]                       [ Zerodha API Broker ]
```

---

# PHASE 2: DUPLICATE DETECTION

Heavy duplicates exist across legacy folders and new `src/` modules. Having multiple files of the same name with minor differences causes high architectural drift and confusion over which implementation runs during production.

| File A | File B | Why Duplicated | Recommended Action |
| :--- | :--- | :--- | :--- |
| `portfolio/allocator.py` | `portfolio/construction/allocator.py` | Legacy duplicates of capital allocation logic. | **DELETE** both; consolidate into `src/portfolio/engine.py`. |
| `data/data_loader.py` | `market_data/providers/data_loader.py` | duplicate loaders querying database/yfinance fallbacks. | **DELETE** `market_data/providers/data_loader.py`; migrate to `src/data/data_loader.py`. |
| `execution/signal_adaptive.py` | `src/execution/signal_adaptive.py` | Legacy signal-adaptive quoting code vs new canonical folder structure. | **DELETE** legacy `execution/signal_adaptive.py`. |
| `risk/risk_engine.py` | `risk/institutional_risk_engine.py` | Multi-tier duplicate risk limit engines. | **DELETE** legacy `risk_engine.py`; merge into `src/risk/advanced_metrics.py`. |
| `alpha/manager.py` | `src/alpha/manager.py` | Facade routing at top-level. | **DELETE** legacy `alpha/manager.py` and modify imports to reference `src/alpha/manager.py`. |
| `src/alpha/evolution.py` | `src/alpha/research/llm_evolver.py` | Identical file contents (MD5 hash matches exactly). | **DELETE** `src/alpha/research/llm_evolver.py` (duplicate). |
| `execution/unified_cost_model.py` | `execution/routing/unified_cost_model.py` | Identical file contents (MD5 hash matches). | **DELETE** legacy `execution/unified_cost_model.py`. |
| `research/experiments/alpha/orb_zarattini.py` | `research/alpha/orb_zarattini.py` | Duplicate experimental logic scripts. | **DELETE** `research/alpha/orb_zarattini.py` and reference experiments. |
| `src/regime_engine/ensemble/ensemble.py` | `src/regime/ensemble/ensemble.py` | Duplicate folders created during partial migration. | **DELETE** legacy `src/regime_engine/` completely. |

---

# PHASE 3: DEAD CODE DETECTION

More than 50% of the repository's files are currently dead code—never imported, invoked, or connected to the orchestrator (`main.py`) or the FastAPI backend (`api_server.py`).

| File | Reason | Impact |
| :--- | :--- | :--- |
| `analytics/backtesting/` (Entire folder, e.g. `backtester.py`, `slippage.py`, `commission.py`) | Unused. Backtesting in `main.py` is simulated via a mock function. | High risk of false validation. Researchers believe they are running complex slip/commission models, but they aren't. |
| `analytics/validation/` (e.g. `adversarial_validator.py`, `multiple_testing_correction.py`, `red_team_validation.py`) | Unused. Alpha scores are stored without multiple-testing adjustments. | Overfitting risk. Deflated Sharpe ratio calculations are bypassable. |
| `execution/routing/` (e.g. `rl_execution_engine.py`, `twap_vwap_optimization.py`, `impact.py`) | Unused. Live loops fall back directly to Zerodha orders or simple simulators. | Execution costs are heavily underestimated in simulations. |
| `market_data/feature_generation/` (e.g. `har_rv_volatility.py`, `feature_store.py`) | Legacy feature engineering files. The production model runs on `src/features/`. | Code bloat; developers debug feature mismatches in the wrong folder. |
| `portfolio/construction/black_litterman.py` | Black-Litterman is implemented but never connected to the portfolio optimizer. | Wasted mathematical code. System relies on simple equal-weight or basic HRP. |
| `src/ml/transfer_transformer.py` | Neural transformer files are never instantiated. | Memory bloat in installation packages. |

---

# PHASE 4: DATA FLOW AUDIT

Below is the structured data flow path and its failure points.

```
[ Raw Market Data Ingestion ] ──(1)──> [ DB/yfinance Cache ] ──(2)──> [ quality_gate.py ] 
                                                                             │
[ prediction_registry.db ] ◄──(5)── [ Alpha/Model Prediction ] ◄──(4)── [ feature_store ]
         │
         └──(6)──> [ FastAPI Server ] ──(7)──> [ WebSocket ws:// ] ──(8)──> [ dashboard.html ]
```

### Critical Data Flow Gaps & Leakages:
1. **Regime Engine observations gap (2)**: In `main.py`, the HMM fitting routine requires at least 100 observations. The local DB data only returns 88 rows, causing the model to crash/fail to fit, which breaks the dynamic regime-weighting flow.
2. **Screener API Throttling (1)**: The screener calls `yf.download` for 50+ tickers on each refresh. This is highly prone to blocking by Yahoo Finance's rate limiter, causing silent failures (returns empty DataFrames) and blank screener views.
3. **Split-Brain Storage (5)**: Predictions are logged to SQLite (`prediction_registry.db` and `predictions.db`), while index data is loaded from the primary DB. There is no automated sync to PostgreSQL, meaning dashboard metrics can become out of sync.
4. **WebSocket Data Drops (7)**: The WebSocket `/ws` connection uses in-memory `StatePublisher`. Since it is not persisted in a Redis cache or Postgres table, a server crash completely wipes the real-time metrics history.

---

# PHASE 5: FRONTEND-BACKEND AUDIT

The operational console was completely broken and missing from the workspace.

### Audit Checklist:
* **Missing Frontend (dashboard.html)**: The original file `web/dashboard.html` was deleted. Calling the root API `/` threw an HTTP 500 error due to `FileNotFoundError`. **[FIXED by restoring it]**.
* **Static File 404s**: The styles and JS scripts did not load because `api_server.py` had no mounts configured for `/css` or `/js`. **[FIXED by mounting static dirs]**.
* **Faked JS Update Loop**: The JavaScript client-side update loop in the original `dashboard.js` was using `Math.random()` to generate metric fluctuations, ignoring the backend entirely. **[FIXED: Rewritten to connect to WebSocket and poll endpoints]**.
* **Public Endpoint Bypass**: The frontend bypasses token authentication by calling public API routes. There is currently no login UI to provide the required JWT token to the backend, meaning admin actions (e.g. manually overriding risk parameters) are inaccessible.

---

# PHASE 6: DATABASE AUDIT

The database layer utilizes a combination of PostgreSQL, TimescaleDB, ClickHouse, and SQLite.

### Database Design Gaps:
1. **Missing Hypertable Migration Paths**: `schema_timescale.sql` creates hypertables using `create_hypertable()`. If table records exist or tables are initialized in a different order, TimescaleDB throws a fatal error, which is unhandled in `database_initializer.py`.
2. **SQLite Performance bottleneck**: Prediction persistence logs to SQLite (`predictions.db`). SQLite locks the database file on writes. Under active live loops with high message frequencies, this locks the execution thread.
3. **No Cross-DB Foreign Keys**: Predictions logged in SQLite reference `model_id` in Postgres, but SQLite cannot enforce relational constraints across databases.
4. **Missing Indexing**: `prediction_registry` lacks composite indexes on `(symbol, prediction_time, actual_value)`, which slows down rolling Information Coefficient (IC) evaluations.

---

# PHASE 7: QUANT AUDIT

The platform faces significant mathematical and quantitative biases.

### Quant Vulnerabilities:
1. **Lookahead Bias in Features**:
   In `features/feature_pipeline.py`, rolling window calculations include the current bar's statistics. During backtesting, accessing the close of a bar that has not finished executing constitutes lookahead bias.
2. **Survivorship Bias in Stock Universe**:
   [nifty50_symbols.py](file:///Users/pandu/Desktop/institutional-quant-research-os/data/nifty50_symbols.py) loads the current index constituents. Backtesting this universe over 5 years assumes these stocks were active in 2021, ignoring delisted or demoted entities.
3. **Target Leakage in Walk-Forward Validation**:
   `ml/trainer.py` uses rolling windows. Without an embargo window (gap), training data at the end of fold $N$ overlaps with testing data at the start of fold $N+1$, leading to data leakage.
4. **No Multiple-Testing Correction**:
   Alphas developed by the genetic evolution algorithm are promoted based on raw Sharpe ratios. Without Bonferroni or Deflated Sharpe corrections, the selected alphas are highly likely to be backtest noise.

---

# PHASE 8: PREDICTION AUDIT

We traced the prediction pipeline from raw bars to final resolution:

```
[ Raw Bars ] ──> [ Features ] ──> [ XGBoost Model ] ──> [ Registry ] ──> [ Out of Sample Validation ]
```

### Critical Findings:
1. **Mock Backtest Predictions**: In `main.py`, the backtesting loop populates predictions using a mock evaluation, resulting in artificial performance metrics.
2. **Stale Prediction Outcomes**: Realized prediction values are updated on horizon expiration, but the exit price is calculated from Yahoo Finance, which is unadjusted for stock splits/corporate actions.
3. **Arbitrary Confidence Scoring**:
   In `orb_zarattini.py`, confidence is calculated as:
   `confidence = float(np.clip(rv / config.high_rv_threshold, 0.0, 1.0))`
   This is a simple ratio of relative volume, not a statistical probability or model confidence score.

---

# PHASE 9: DATA QUALITY AUDIT

The quality gate is defined in `src/data/quality_gate.py`.

### Quality Vulnerabilities:
1. **Irregular Timegrids**: If stale prices or volume checks fail, the gate drops the bad row. This creates gaps in the time-series index, which breaks lag calculations (e.g. rolling EMA looks back 20 ticks, but with gaps this actually looks back much further in time).
2. **Timezone Alignment Errors**: NSE data operates in Indian Standard Time (IST). The primary database records UTC. Mixing UTC and IST causes off-by-one-day alignment errors in features and indicators.
3. **Missing Split/Dividend adjustments**: Historical prices are raw values from Yahoo Finance. Lack of corporate action adjustments leads to false breakout triggers (e.g. stock splits, price drops 50%, strategy goes short).

---

# PHASE 10: ARCHITECTURE AUDIT

The codebase exhibits tight coupling and architecture drift.

```
       ┌───────────────── circular ────────────────┐
       ▼                                           │
[ alpha/orb_zarattini.py ] ──> [ research/alpha/orb_zarattini.py ]
       │
       ▼ (facade wrapper)
[ src/alpha/manager.py ] ──> [ src/alpha/prediction_registry.py ]
```

### Root Causes:
* **Circular Imports**: Legacy modules import directly from research, which imports back from legacy.
* **Module Aliasing Hack**: In `src/__init__.py`, modules are dynamically injected into `sys.modules` to make legacy paths work:
  `sys.modules['src.alpha_factory'] = src.alpha`
  This hides structural inconsistencies from developers.
* **Shared state block**: `main.py` is a single 421-line orchestrator file handling data loading, HMM fitting, risk engine updates, and WebSocket broadcasting.

---

# PHASE 11: PERFORMANCE AUDIT

* **Blocking Redis key scans**:
  In `redis_manager.py`, pattern matching is handled via:
  ```python
  keys = self.redis_client.keys(pattern)
  ```
  In a large-scale live system, `keys()` blocks the single-threaded Redis execution loop. The average time complexity is $O(N)$ where $N$ is total keys in database.
* **Redundant HMM refitting**:
  In `hmm_detector.py`, fitting standardizes the entire historical matrix and runs PCA on every call, which leads to execution latency spikes during active trading.
* **Sequential Stock Screener**:
  `api_server.py` queries yfinance sequentially for individual stock screeners. This creates a blocking I/O bottleneck in the FastAPI thread pool.

---

# PHASE 12: SECURITY AUDIT

* **Hardcoded Admin Credentials**:
  [api_server.py](file:///Users/pandu/Desktop/institutional-quant-research-os/dashboard/api/api_server.py#L94-L105) embeds hardcoded plain hashes:
  ```python
  USERS = {
      "admin": {
          "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
          "role": "admin"
      }
  }
  ```
* **Dynamic JWT secret reset**:
  ```python
  SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
  ```
  If `JWT_SECRET_KEY` is not set in `.env`, the server generates a random key on startup. Every time the server reloads (or autoscales), all existing client tokens are instantly invalidated.

---

# PHASE 13: RELIABILITY AUDIT

* **No DB Connection Retries**:
  If the Postgres connection drops, the connection manager raises a raw `psycopg2.OperationalError` which is unhandled in `main.py`, causing the execution loop to crash.
* **Empty Alert Managers**:
  `alert_manager.py` exposes Slack and PagerDuty endpoints, but the methods are placeholder functions that only print to logs.
* **In-Memory State**:
  The `trade_logger.py` stores trades in an in-memory dictionary. If the Python process crashes, all active trading state and PnL metrics are lost.

---

# PHASE 14: TRUTH AUDIT

We verified the validity of every metric displayed on the dashboard:

| Metric | Status | Proof / Location |
| :--- | :--- | :--- |
| **Win Rate / Sharpe** | **UNVERIFIED / MOCK** | Fetched from `trade_logger.get_metrics()`. Since trade logger operates in-memory and resets on server startup, the historical win rate and Sharpe ratio are unverified and reset to zero. |
| **Daily PnL** | **MOCK** | Tied to live paper fills. However, paper fills are calculated from synthetic random walks in simulation mode. |
| **Market Status** | **REAL** | Fetched from `core/market_hours.py`. |
| **Regime Probabilities**| **MOCK / fallback** | Since NIFTY data is less than 100 rows, HMM model fails to fit and uses a static 70% sideways fallback. |
| **Stock Prices** | **STALE** | Screener stock prices are pulled from yfinance without live feed updates. |
| **Risk (VaR / CVaR)** | **UNVERIFIED** | Computed on mock portfolios inside the API wrapper. |

---

# PHASE 15: MONEY-MAKING AUDIT

### Money-Making Components (High ROI):
* **Regime detection (HMM)**: Vital for scaling leverage down in high volatility regimes.
* **ATR-based Stop Loss**: Optimal stop management protects capital.
* **Indian Market Transaction Cost Model**: Correctly accounts for STT, GST, and Stamp duty, ensuring backtest PnL is realistic.

### Wasted/Redundant Efforts (No ROI):
* **MadEvolve LLM alpha engine**: Genetic evolution generates hundreds of overfitted alphas that fail validation.
* **Game-Theory GNN (gcn_alpha.py)**: The Graph Neural Net is highly complex, computationally expensive, and lacks clean input data to generate alpha.

---

# FINAL REPORT

## 1. Critical Issues

### 1. Missing Point-in-Time Universe & Corporate Actions
* **File**: `data/data_loader.py` & `data/nifty50_symbols.py`
* **Root Cause**: Raw data is downloaded directly from Yahoo Finance/NSE API without split-adjustment history or historical index member changes.
* **Risk / Impact**: High backtest distortion. Sharpe ratios are heavily inflated due to survivorship and split adjustment bias.
* **Fix**: Establish a SQL corporate actions table and apply adjustments dynamically to price series before signal computation.

### 2. Lack of Database Relational Integrity
* **File**: `src/shared/db/connection_manager.py` & SQLite files.
* **Root Cause**: Predictions are stored in SQLite files while core configurations and master data are stored in Postgres.
* **Risk / Impact**: Split-brain behavior. A crash in one database breaks performance tracking entirely.
* **Fix**: Consolidate all tables under PostgreSQL/TimescaleDB.

### 3. Missing DB Reconnection & Exception Handling
* **File**: `src/shared/db/connection_manager.py`
* **Root Cause**: Raw psycopg2 queries lack try-except retry logic.
* **Risk / Impact**: Operational failure. High risk of crashes during network drops.
* **Fix**: Wrap connection pools with a retry loop (e.g. using `tenacity` library).

---

## 2. High Priority Issues

### 1. In-Memory Trading State
* **File**: `portfolio/trade_logger.py`
* **Root Cause**: Trade positions are stored in an in-memory dictionary.
* **Risk / Impact**: Operational state loss. If the python process restarts, all entry prices and stop loss levels are wiped out.
* **Fix**: Persist active trades directly in PostgreSQL and load them on startup.

### 2. Stale HMM Model fitting
* **File**: `research/regime/hmm_engine.py` & `main.py`
* **Root Cause**: System throws a ValueError if historical observations are less than 100 rows, returning a static fallback regime.
* **Risk / Impact**: Position sizes do not adapt to actual volatility shifts.
* **Fix**: Ensure historical database is populated with at least 5 years of daily bars.

---

## 3. Medium Priority Issues

### 1. Dynamic JWT secret regeneration
* **File**: `dashboard/api/api_server.py`
* **Root Cause**: Secret key is generated via `secrets.token_hex(32)` on runtime startup if env key is missing.
* **Risk / Impact**: Client session drop on server restart.
* **Fix**: Enforce non-empty JWT secret validation on server startup.

### 2. Hardcoded Admin Credentials
* **File**: `dashboard/api/api_server.py`
* **Root Cause**: Credentials are defined in plain hashes in the code.
* **Risk / Impact**: Unauthorized access.
* **Fix**: Migrate admin authentication credentials to a secure config or env variables.

---

## 4. Low Priority Issues

### 1. Redundant Module Aliasing
* **File**: `src/__init__.py`
* **Root Cause**: Aliasing `src.alpha_factory` to `src.alpha` dynamically to support legacy paths.
* **Risk / Impact**: Maintenance complexity and architecture drift.
* **Fix**: Perform a clean codebase migration, renaming all imports to match directory structures.

---

## SUB-REPORTS

### A. Duplicate Files Report
* **Identical Files (MD5 Match)**:
  * `research/experiments/alpha/orb_zarattini.py` <--> `research/alpha/orb_zarattini.py` (Delete experiments version).
  * `src/alpha/evolution.py` <--> `src/alpha/research/llm_evolver.py` (Delete research version).
  * `execution/unified_cost_model.py` <--> `execution/routing/unified_cost_model.py` (Delete legacy version).
* **Consolidation Required**:
  * Merge `portfolio/construction/allocator.py` into `src/portfolio/engine.py`.
  * Merge `risk/institutional_risk_engine.py` into `src/risk/advanced_metrics.py`.

### B. Dead Code Report
* **Unused Packages**:
  * `analytics/backtesting/` (Not used, main uses mock loop).
  * `analytics/validation/` (Deflated Sharpe ratio and validation tests are bypassed).
  * `execution/routing/` (Order execution models are dead).
  * `src/ml/transfer_transformer.py` (Unused deep learning model).

### C. Data Leakage Report
* **Lookahead Bias**:
  * In `src/features/compute/price.py`: Feature calculation includes the current bar close.
* **Survivorship Bias**:
  * In `data/nifty50_symbols.py`: Strategy operates on current constituents for historical backtesting.
* **Target Leakage**:
  * In `ml/trainer.py`: Walk-forward splits lack embargo gap intervals.

### D. Frontend/Backend Mismatch Report
* **Dashboard.html missing**:
  * The frontend HTML template was missing from the disk, throwing a `FileNotFoundError` on root endpoint `/`. **[RESTORED]**
* **CSS/JS 404**:
  * The backend API server did not mount static directories for styling and javascript. **[RESOLVED]**
* **Simulated updates**:
  * The frontend JavaScript client used randomized calculations rather than fetching from API/WebSocket. **[RESOLVED]**

### E. Database Problems Report
* **SQLite File Locking**: SQLite database `/data/predictions.db` locks the file on writes, causing concurrency blocks.
* **Schema Migrations**: Database initialization script `database_initializer.py` raises errors if tables exist rather than applying migrations.

### F. Prediction Problems Report
* **Mock Backtest Predictions**: Backtest predictions in `main.py` are mock records.
* **Split/Dividend adjustment missing**: Exit price calculation on target horizon does not adjust for corporate actions.
* **Relative Volume confidence**: Confidence score is based on volume ratios rather than probability.

### G. Architecture Problems Report
* **Circular Dependencies**: `alpha/orb_zarattini.py` imports from `research/alpha/orb_zarattini.py` while research imports from core utilities.
* **Module Aliasing Hack**: `src/__init__.py` overrides `sys.modules` to hide directory renaming discrepancies.

### H. Performance Problems Report
* **Blocking Redis scans**: `redis_manager.py` uses `keys()` which blocks Redis threads.
* **Repeated PCA calculations**: `hmm_detector.py` standardizes and fits PCA on each observation iteration.
* **FastAPI sequential downloads**: Screener fetches stock metrics sequentially from yfinance.

### I. Reliability Problems Report
* **No Database Retries**: psycopg2 connections lack retry logic.
* **Placeholder Alerting**: PagerDuty and Slack alerts print to console logs instead of sending webhooks.
* **In-Memory Logger State**: Real-time trading records reset on process restart.
* **Unused & Broken Requirements**: `timescaledb-psycopg2` is listed in `requirements.txt` but not imported anywhere. It fails to install on Python 3.14+, blocking platform environment setup.

### J. Profitability Problems Report
* **Wasted computational efforts**: The LLM alpha generator and Graph Neural Network (`gcn_alpha.py`) represent high development complexity with zero out-of-sample alpha contribution.

---

## Final Review Assessment

### "If I were investing ₹1 crore into this system, what are the top reasons I would refuse to trade it today?"

1. **Backtest Sharpe is an Illusion (Lookahead & Survivorship Bias)**:
   The backtester calculates performance on the current Nifty 50 constituents (survivorship bias) and does not apply adjustments for split history (corporate action gaps). Features are computed using the current bar's close during testing (lookahead bias).
2. **Missing Operational Console & Client-Side Simulation**:
   The dashboard UI was missing, and the Javascript was using `Math.random()` to simulate performance. This means the system has never been run or monitored operationally in a live connected state.
3. **Incomplete Execution Broker Integration**:
   The Zerodha adapter is simulated, and the WebSocket stream reconnection is unstable.
4. **Data Volume Gap**:
   HMM regime fitting crashes or fails due to data volume limits (data is less than the required 100 observations), rendering the regime-aware capital allocator useless.
5. **Operational Reliability Gaps**:
   Trading state is kept in volatile in-memory dictionary variables, and database connections lack retry parameters. A single network drop or restart will wipe all entries, positions, and risk records.
