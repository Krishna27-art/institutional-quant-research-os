# QROS v2: Institutional Master Plan (18-Month Roadmap)

This document outlines the definitive 5-phase, 18-month roadmap to rebuild the Quant Research Operating System (QROS) from a retail dashboard into a top-tier institutional execution and research platform. It represents the binding architectural directives from the Institutional Joint Committee.

---

## Phase 1: Foundation & Data Integrity (Months 1-3)
**Goal:** Establish absolute data integrity, deterministic system behavior, and eliminate lookahead/survivorship biases.

- [x] **Task 1.1: Database Migration.** Deprecate all SQLite usage. Stand up a 3-node ClickHouse cluster (or kdb+) optimized for timeseries columnar storage.
- [x] **Task 1.2: Point-in-Time (PIT) Schema.** Implement a dual-timestamp schema (`transaction_time`, `valid_time`) for all market data, corporate actions, and fundamental data.
- [x] **Task 1.3: Universe Management.** Build a historical NIFTY500 constituent database extending back 20 years to permanently eliminate survivorship bias in backtests.
- [x] **Task 1.4: Deterministic Simulation.** Remove `random.choice()` and unseeded `np.random` from the execution simulator. Guarantee 100% reproducibility across identical backtest runs.
- [x] **Task 1.5: Risk Un-mocking.** Eradicate the `DummyPortRisk` placeholder in the execution pipeline. Wire the existing Risk Engine so it can actively block trades.
- [x] **Task 1.6: Covariance Estimation.** Replace the default identity matrix (`np.eye(n)`) in the portfolio optimizer with empirical covariance matrices utilizing Ledoit-Wolf shrinkage.

---

## Phase 2: Alpha Infrastructure & Governance (Months 3-6)
**Goal:** Build a rigorous, statistically sound research pipeline that prevents factor crowding and backtest overfitting.

- [x] **Task 2.1: Alpha Registry.** Deploy a central database cataloging every alpha factor's metadata, expected Sharpe, dollar capacity limit, and causal hypothesis.
- [x] **Task 2.2: Deflated Sharpe Gate.** Implement the Deflated Sharpe Ratio (DSR) mathematical gate to statistically adjust out-of-sample expectations based on the number of backtest trials run.
- [x] **Task 2.3: Continuous IC Tracking.** Build a monitoring service to calculate the rolling Information Coefficient (IC) and rank IC of all live strategies. Establish automatic scale-down rules for IC decay.
- [x] **Task 2.4: Orthogonalization Engine.** Implement a PCA-based factor orthogonalization pipeline so new alpha candidates are neutralized against existing live factors before deployment.
- [x] **Task 2.5: Purged Cross-Validation.** Rewrite the ML training loop to strictly enforce Embargo and Purge gaps, ensuring zero data leakage in walk-forward time-series models.
- [x] **Task 2.6: Experiment Tracking.** Deploy MLflow to centrally log every backtest parameter, random seed, feature hash, and metric.

---

## Phase 3: Execution & Risk Engineering (Months 6-9)
**Goal:** Minimize market impact, model execution reality, and manage tail risk dynamically.

- [x] **Task 3.1: Pre-Trade Risk Gate.** Build a low-latency C++ risk gate that intercepts every order prior to broker routing. Implement portfolio margin offset calculations.
- [x] **Task 3.2: Optimal Execution Scheduler.** Implement the Almgren-Chriss closed-form execution algorithm to slice large orders optimally against volume profiles, balancing impact vs. variance.
- [ ] **Task 3.3: Post-Trade TCA.** Build a Transaction Cost Analysis engine running asynchronously post-market to compare fill prices against Arrival Price and Interval VWAP benchmarks.
- [x] **Task 3.4: Advanced Portfolio Optimizer.** Upgrade the `cvxpy` implementation to handle long/short constraints, borrow cost matrices, maximum sector exposures, and $\frac{3}{2}$-power-law transaction cost penalties.
- [x] **Task 3.5: L3 Book Replay.** Build a high-performance C++ simulator capable of replaying historical Level 3 tick data to model limit order queue position and adverse selection accurately.

---

## Phase 4: Compute & Network Infrastructure (Months 9-12)
**Goal:** Achieve sub-millisecond latency for live trading and horizontal scaling for research compute.

- [x] **Task 4.1: Native Exchange Feeds.** Build a C++ UDP multicast receiver using Solarflare OpenOnload kernel-bypass networking to ingest direct NSE tick feeds.
- [ ] **Task 4.2: FIX Gateway.** Deprecate the REST-based Kite Connect adapter. Implement a direct FIX 4.4 and OUCH/ITCH protocol gateway for sub-millisecond order submission.
- [ ] **Task 4.3: Kubernetes Orchestration.** Containerize all microservices (Data Ingest, Feature Compute, Execution, Risk) and deploy them to a highly-available k8s cluster.
- [ ] **Task 4.4: Observability Stack.** Deploy Prometheus for metrics, Grafana for real-time dashboards, and PagerDuty for on-call incident routing based on model drift alerts.
- [ ] **Task 4.5: Research HPC Cluster.** Provision a Slurm-managed compute cluster with A100 GPUs for deep learning and 128-core CPU nodes for distributed backtesting.

---

## Phase 5: Advanced Quantitative Finance Systems (Months 12-18)
**Goal:** Expand to complex asset classes, alternative datasets, and latency-sensitive arbitrage strategies.

- [x] **Task 5.1: Options Volatility Surface.** Build a continuous implied volatility engine using SABR and SVI models to detect and price skew, smile, and calendar arbitrage opportunities.
- [ ] **Task 5.2: Statistical Arbitrage Scanner.** Implement Johansen cointegration tests running across all 125,000+ pairs in the NIFTY500 to identify mean-reverting multi-asset baskets.
- [ ] **Task 5.3: Alternative Data Pipeline.** Build NLP pipelines to ingest real-time news, SEBI filings, and sentiment data, converting them into point-in-time features.
- [ ] **Task 5.4: RL Execution Agent.** Train a Proximal Policy Optimization (PPO) reinforcement learning agent on historical L3 order books to dynamically manage order placement limits.
- [ ] **Task 5.5: Colocation Deployment.** Migrate the production execution stack to physical servers colocated within the NSE Mahape datacenter.


---

# APPENDIX: FULL INSTITUTIONAL ARCHITECTURE REDESIGN REPORT

# INSTITUTIONAL QUANT RESEARCH OPERATING SYSTEM — QROS v2
## FIRST-PRINCIPLES DESTRUCTION & FUTURE-STATE ARCHITECTURE

**Joint Committee Report — FINAL & DEFINITIVE**

**Prepared by:**
| Seat | Role |
|---|---|
| **Jane Street** | Head of Quantitative Research |
| **Citadel Securities** | Head of Systematic Trading |
| **Hudson River Trading** | Chief Architect |
| **Renaissance Technologies** | Research Director |
| **Two Sigma** | Research Platform Lead |
| **D.E. Shaw** | Quant Strategist |
| **Academic Chair** | Professor of Statistics, Stochastic Processes, Econometrics, Optimization, Market Microstructure & Machine Learning (Princeton/MIT) |

**Classification:** Internal — Committee Eyes Only
**Date:** June 14, 2026

---

# TABLE OF CONTENTS

1. [Executive Assessment](#section-1-executive-assessment)
2. [Architecture Review & Grade Scoring](#section-2-architecture-review--grade-scoring)
3. [Missing Research Systems](#section-3-missing-research-systems)
4. [Mathematics Roadmap](#section-4-mathematics-roadmap)
5. [Finance Roadmap](#section-5-quantitative-finance-roadmap)
6. [Research Organization Design](#section-6-research-organization-design)
7. [Future State Architecture](#section-7-future-state-architecture-qros-v2)
8. [The Harsh Truth](#section-8-the-harsh-truth)

---

# SECTION 1: EXECUTIVE ASSESSMENT

## 1.1 Overall Verdict

> [!CAUTION]
> **This system is not institutional. It is not hedge-fund grade. It is not even professional grade. It is a retail hobby project wrapped in institutional terminology.**

The Quant Research Operating System in its current state represents a textbook case of **cargo-cult quantitative engineering** — the system has adopted the *vocabulary* of institutional trading (Alpha Marketplace, Institutional Risk Engine, Knowledge Graph, Capital Allocation Engine) without implementing the *substance* behind any of those terms.

## 1.2 Quantitative Assessment Summary

| Dimension | Current State | Institutional Standard | Gap |
|---|---|---|---|
| **Data Integrity** | Yahoo Finance daily bars via `yfinance`. SQLite storage. | Tick-by-tick L3 feeds. kdb+/ClickHouse. PIT schema. | **Catastrophic** |
| **Alpha Pipeline** | 1 active strategy (ORB). No IC tracking. No decay detection. | 200+ orthogonalized factors. Continuous IC monitoring. Deflated Sharpe. | **Catastrophic** |
| **Execution** | REST API to Kite Connect. `time.sleep()` latency. Hardcoded `SIMULATED` keys. | FIX/OUCH binary protocol. Colocated FPGA. Kernel-bypass NIC. Sub-100μs. | **Catastrophic** |
| **Risk** | Post-trade check. Static correlation. No pre-trade gate. | Pre-trade C++ gate. Dynamic copulas. Stress testing. Real-time margin. | **Critical** |
| **Portfolio Optimization** | cvxpy with identity covariance matrix fallback. Long-only constraint. | Multi-period convex optimization. Transaction cost models. Factor-aware. | **Critical** |
| **Backtesting** | No L3 replay. No queue simulation. No survivorship-bias correction. | Full order-book reconstruction. PIT universe. Embargo/purge CV. | **Critical** |
| **Infrastructure** | Single-threaded Python on macOS desktop. | Colocated Linux. Kubernetes. GPU cluster. Slurm scheduler. | **Catastrophic** |
| **Research Governance** | Zero. No experiment tracking. No trial logging. No p-value correction. | Full MLflow/W&B registry. Deflated Sharpe gates. Peer review. | **Catastrophic** |

## 1.3 The Five Fatal Flaws

### Fatal Flaw 1: The `DummyPortRisk` Pattern
In [main.py](file:///Users/pandu/Desktop/institutional-quant-research-os/main.py) lines 104-110:
```python
class DummyPortRisk:
    def get_current_capital(self): return capital

return ExecutionPipeline(
    execution_engine=UnifiedExecutionEngine(engine_mode, exec_config),
    portfolio_allocator=DummyPortRisk(),
    risk_engine=DummyPortRisk()
)
```
**The risk engine passed to the execution pipeline is a dummy object that returns a constant.** The execution pipeline has no real risk gate. Orders flow through unchecked.

### Fatal Flaw 2: Identity Covariance Matrix
In [institutional_allocator.py](file:///Users/pandu/Desktop/institutional-quant-research-os/production/src/portfolio/institutional_allocator.py) lines 56-57:
```python
if correlation_matrix is None or correlation_matrix.shape != (n, n):
    correlation_matrix = np.eye(n)
```
**The portfolio optimizer falls back to assuming all assets are uncorrelated.** Since no correlation matrix is ever passed (there is no code that computes one), **every single optimization run uses the identity matrix**. This means the system treats NIFTY, BANKNIFTY, RELIANCE, and TCS as if they have zero correlation — a catastrophically wrong assumption that leads to massive undiversified concentration risk.

### Fatal Flaw 3: `SIMULATED` Broker Credentials in Live Mode
In [main.py](file:///Users/pandu/Desktop/institutional-quant-research-os/main.py) line 95:
```python
exec_config = LiveConfig(broker_api_key="SIMULATED", broker_api_secret="SIMULATED")
```
**The system initializes "live" mode with hardcoded `SIMULATED` strings as API credentials.** There is no credential validation gate. The system will silently fail on the first order submission, but no exception is raised during initialization.

### Fatal Flaw 4: `observed_sharpe` Permanently Frozen at Zero
In [registry.py](file:///Users/pandu/Desktop/institutional-quant-research-os/production/src/alpha/marketplace/registry.py) lines 23-27:
```python
@dataclass
class AlphaPerformance:
    observed_sharpe: float = 0.0
    current_capacity_usage: float = 0.0
    is_active: bool = True
```
The `observed_sharpe` is initialized to `0.0` and **is never updated by any code path**. The `evaluate_alphas()` method on line 52 compares `expected_sharpe - observed_sharpe`, but since `observed_sharpe` is always `0.0`, the evaluation simply checks if `expected_sharpe > 1.0`. This means:
- Every alpha with `expected_sharpe > 1.0` will eventually be deactivated (since `deviation > 1.0`).
- Every alpha with `expected_sharpe <= 1.0` will never be deactivated regardless of actual performance.
- **The self-evaluation system is a no-op.**

### Fatal Flaw 5: Random Fill Simulation Without Seed Control
In [simulation_engine.py](file:///Users/pandu/Desktop/institutional-quant-research-os/production/src/execution/simulation_engine.py) lines 42, 62:
```python
drift = np.random.normal(0, drift_volatility * np.sqrt(latency_seconds))
...
if random.random() < self.config.queue_position_penalty:
```
**Backtest execution fills use unseeded random number generators.** This means:
- Two runs of the same backtest on the same data produce different PnL curves.
- No backtest result is reproducible.
- Any performance metric (Sharpe, drawdown, win rate) is meaningless because it changes on every run.

---

# SECTION 2: ARCHITECTURE REVIEW & GRADE SCORING

Each module is scored on a 5-point institutional scale:

```
[1] Retail Hobbyist
[2] Professional/Semi-Pro
[3] Small Hedge Fund
[4] Top-Tier Quant Fund (Millennium, Balyasny)
[5] Jane Street / Renaissance Technologies / Citadel
```

## 2.1 Data Layer & Ingestion

| Attribute | Score | Evidence |
|---|---|---|
| **Data Source** | `[1]` | Yahoo Finance via `yfinance`. Free, delayed, unreliable, rate-limited. No direct exchange feeds. |
| **Storage Engine** | `[1]` | SQLite (`market_truth.db`). File-level locking. No concurrent writes. Cannot handle tick data. |
| **Point-in-Time** | `[0]` | **Non-existent.** No dual-timestamp schema. All data queries use current-state tables. Survivorship bias is guaranteed. |
| **Corporate Actions** | `[0]` | **Non-existent.** No dividend adjustment, stock split handling, bonus share tracking, or rights issue processing. |
| **Universe Management** | `[1]` | Hardcoded NIFTY50 symbol list. No historical constituent tracking. No sector/industry classification hierarchy. |
| **Data Quality** | `[2]` | `DataQualityGate` implements 5 rules (OHLC consistency, stale feed detection, volume sanity, price continuity, future data check). Acceptable but incomplete. |
| **Overall** | **`[1] Retail`** | |

**What institutional looks like:** kdb+/q or ClickHouse columnar store processing 50M+ ticks/day. Direct NSE TCP multicast feeds with nanosecond timestamps. Point-in-time schema with transaction time and valid time columns. Full corporate action calendar with adjustment factors computed daily. Universe management with GICS sector codes and historical index reconstitution data going back 20+ years.

## 2.2 Feature Generation & Store

| Attribute | Score | Evidence |
|---|---|---|
| **Feature Computation** | `[2]` | `FeatureComputer` calculates standard technical indicators (RSI, MACD, Bollinger, ATR, OBV) using pandas. |
| **Point-in-Time Compliance** | `[0]` | Features are computed on current data snapshots with no PIT awareness. A feature computed "as of 2023-01-15" may use stock split adjustments that were only published on 2023-02-01. |
| **Cross-Sectional Features** | `[1]` | Files exist under `production/market_data/feature_generation/cross_sectional_features.py` but are not imported by the live pipeline. |
| **Feature Versioning** | `[1]` | A `versioning/` directory exists but is not integrated into the compute path. Feature definitions are not immutable. |
| **Performance** | `[1]` | Pure pandas. No C++ extensions for feature computation. No GPU acceleration. No vectorized batch processing. |
| **Overall** | **`[1.5] Retail-to-Professional`** | |

**What institutional looks like:** Feature definitions stored as immutable, versioned DAGs (directed acyclic graphs) with dependency tracking. Every feature has a unique hash. Computation runs on GPU-accelerated array libraries (JAX, CuPy) or pre-compiled C++ kernels. Cross-sectional features (sector-relative momentum, cross-asset correlations) computed across the full 500+ stock universe in under 50ms. Feature importance tracked via Shapley values and permutation tests.

## 2.3 Regime Detection

| Attribute | Score | Evidence |
|---|---|---|
| **Method** | `[2]` | Gaussian HMM via hmmlearn/sklearn. Standard textbook implementation. |
| **Input Data** | `[1]` | Fitted on daily close returns only. Does not ingest order flow, volume profiles, or volatility term structure. |
| **Online Learning** | `[0]` | **Non-existent.** Model is fitted once at startup. Does not update as new data arrives. Regime assignments are stale within hours. |
| **Multi-Regime** | `[2]` | 4 states: Bull, Bear, Sideways, High Volatility. Reasonable taxonomy but static. |
| **Integration** | `[1]` | Regime label is passed as a string to alpha strategies. No structured regime-conditional parameter adjustment. |
| **Overall** | **`[1.5] Retail-to-Professional`** | |

**What institutional looks like:** Online Bayesian switching models (BOCPD — Bayesian Online Changepoint Detection) processing tick-level order flow imbalance and VPIN (Volume-Synchronized Probability of Informed Trading). Multiple regime taxonomies: volatility regimes, liquidity regimes, correlation regimes, momentum/mean-reversion regimes. Each regime type independently estimated and combined via ensemble. Regime probabilities (not hard labels) fed to every downstream module.

## 2.4 Alpha Engine & Strategy Pipeline

| Attribute | Score | Evidence |
|---|---|---|
| **Active Strategies** | `[1]` | One active strategy: ORB (Opening Range Breakout). Registered in the `AlphaMarketplace` with static metadata. |
| **Alpha Registry** | `[1]` | `AlphaMarketplace` exists but `observed_sharpe` is never updated (permanently 0.0). Self-evaluation is a no-op. |
| **IC Tracking** | `[0]` | **Non-existent.** No Information Coefficient computation. No rank IC. No IC decay monitoring. |
| **Orthogonalization** | `[0]` | **Non-existent.** No factor neutralization. No PCA residualization. No collinearity detection. |
| **Signal Combination** | `[0]` | **Non-existent.** With one strategy, there is nothing to combine. No z-score normalization. No meta-model. |
| **Capacity Analysis** | `[1]` | `capacity_limit_usd` is a static metadata field. No empirical capacity estimation from volume profiles. |
| **Knowledge Graph** | `[1]` | `KnowledgeGraph` class exists with a single node ("orb_zarattini"). No edges, no causal DAG, no counterfactual analysis. |
| **Overall** | **`[1] Retail`** | |

**What institutional looks like:** 200-500+ active alpha factors across timeframes (tick, intraday, daily, weekly). Each factor stored with: expected IC, realized IC (rolling 20/60/252 day), IC decay half-life, factor exposure (sector, industry, size, value, momentum, volatility), dollar capacity estimate from ADV analysis, and correlation to every other live factor. New factors must pass: (a) Deflated Sharpe Ratio test, (b) out-of-sample walk-forward validation with embargo, (c) orthogonality test against existing live factors, (d) causal hypothesis review by research committee. Signal combination via optimal shrinkage estimator on the IC covariance matrix.

## 2.5 Portfolio Optimization

| Attribute | Score | Evidence |
|---|---|---|
| **Optimizer** | `[2]` | cvxpy with mean-variance objective. Functional but naive. |
| **Covariance Estimation** | `[0]` | **Falls back to identity matrix on every run** (no code ever passes a real correlation matrix). |
| **Transaction Costs** | `[0]` | **Non-existent.** No bid-ask spread penalty. No market impact penalty. No turnover constraint. |
| **Factor Constraints** | `[0]` | **Non-existent.** No sector exposure limits enforced. No beta neutrality. No factor tracking error bounds. |
| **Short Selling** | `[0]` | Hardcoded `w >= 0` constraint (long-only). Cannot express negative views. |
| **Multi-Period** | `[0]` | **Non-existent.** Single-period optimization. No forward-looking trading trajectory. |
| **Overall** | **`[1] Retail`** | |

**What institutional looks like:** Multi-period convex optimization (OSQP/SCS/Mosek solvers) with: Ledoit-Wolf shrinkage covariance estimator, $\frac{3}{2}$-power-law transaction cost penalty from Almgren-Chriss, borrow cost matrices for short positions, sector/industry/country factor exposure limits, portfolio beta constraint, tracking error bounds, maximum single-name concentration, turnover budget, and tax-loss harvesting constraints. Solved every 5 minutes with warm-starting from previous solution.

## 2.6 Risk Engine

| Attribute | Score | Evidence |
|---|---|---|
| **VaR Methods** | `[3]` | Historical VaR, parametric VaR, EVT (Extreme Value Theory) VaR, and liquidity-adjusted VaR. Well-structured in `metrics.py`. |
| **CVaR** | `[3]` | Implemented. Conditional Value at Risk computed from historical distribution tail. |
| **Circuit Breaker** | `[3]` | `HardCircuitBreaker` with daily loss, weekly loss, drawdown, and VIX thresholds. State persisted to JSON. |
| **Pre-Trade Gate** | `[0]` | **Non-existent.** Risk is assessed post-trade. The `DummyPortRisk` object is passed to execution. Orders are never blocked. |
| **Stress Testing** | `[0]` | **Non-existent.** No historical scenario replay. No Monte Carlo stress simulation. No tail correlation modeling. |
| **Correlation Risk** | `[0]` | No dynamic correlation monitoring. No regime-conditional correlation matrices. No copula modeling. |
| **SEBI Compliance** | `[2]` | `SEBIAlgoCompliance` class exists with basic regulatory checks. |
| **Overall** | **`[2] Professional`** | |

**What institutional looks like:** Pre-trade risk gate compiled in C++ intercepting every order at the socket level. Real-time portfolio margin computed using exchange SPAN methodology. Intraday VaR updated every 100ms using exponentially-weighted moving covariance. Stress testing suite with 50+ historical scenarios (1987 crash, 2008 GFC, 2020 COVID, 2022 rate shock). Dynamic copula estimation for tail-dependent correlation modeling. Greeks-based options risk with full second-order sensitivities. Concentration risk limits per name, sector, country, and factor.

## 2.7 Execution Engine

| Attribute | Score | Evidence |
|---|---|---|
| **Protocol** | `[1]` | REST/WebSocket API to Zerodha Kite Connect. ~100ms round-trip minimum. |
| **Latency Modeling** | `[1]` | `SimulationConfig` with static `network_latency_ms=25` and `processing_latency_ms=15`. Not calibrated from real data. |
| **Market Impact** | `[2]` | Square-root impact model: $\text{impact} = \alpha \cdot \sigma \cdot \sqrt{Q/\text{ADV}}$. Correct functional form but static parameters. |
| **Smart Order Router** | `[1]` | Directory exists (`smart_order_router/`) but not integrated into the live execution path from `main.py`. |
| **Fill Simulation** | `[1]` | Random partial fills with unseeded RNG. Non-reproducible backtests. |
| **Slippage** | `[1]` | Constant commission model: `0.0001 * quantity * price` (1 bps). No variable spread component. |
| **Overall** | **`[1] Retail`** | |

**What institutional looks like:** Direct TCP/IP connections to NSE/BSE matching engines via FIX 4.4 or binary OUCH protocol. Colocated servers with kernel-bypass networking (Solarflare OpenOnload, DPDK). Sub-100μs order-to-acknowledge latency. Smart order router choosing optimal venue based on real-time queue depth, spread width, and toxicity score. Almgren-Chriss optimal execution scheduler computing liquidation trajectories. Post-trade TCA engine computing implementation shortfall against arrival price benchmarks.

## 2.8 Backtesting Framework

| Attribute | Score | Evidence |
|---|---|---|
| **Engine** | `[2]` | Event-driven and vectorized backtesters exist. Distributed cluster engine exists (Ray-based). |
| **Order Book Simulation** | `[0]` | **Non-existent.** No L3 replay. No queue position modeling in backtest mode. |
| **Survivorship Bias** | `[0]` | Hardcoded current NIFTY50 constituents used for all historical periods. |
| **Lookahead Bias** | `[1]` | Multiple `shift(-1)` lookahead violations documented in report.md (issues #11-13, #33-35, #55-57). |
| **Walk-Forward CV** | `[1]` | Purged walk-forward exists in code but lacks configurable embargo/purge gap. |
| **Reproducibility** | `[0]` | Unseeded RNG in fill simulator. Non-deterministic results. |
| **Overall** | **`[1] Retail`** | |

## 2.9 Machine Learning Pipeline

| Attribute | Score | Evidence |
|---|---|---|
| **Models** | `[2]` | GCN (Graph Convolutional Network), tabular ensemble, walk-forward trainer exist in research experiments. |
| **Cross-Validation** | `[1]` | Standard K-fold or simple train/test split. No purge gap. No embargo period. |
| **Hyperparameter Tuning** | `[0]` | **Non-existent.** No Optuna, no Bayesian optimization, no grid search framework. |
| **Feature Selection** | `[1]` | `feature_selector.py` exists under `production/market_data/` but disconnected from live pipeline. |
| **Model Registry** | `[0]` | **Non-existent.** No MLflow. No model versioning. No A/B testing framework. |
| **Overall** | **`[1] Retail`** | |

## 2.10 Monitoring & Observability

| Attribute | Score | Evidence |
|---|---|---|
| **Feature Drift** | `[1]` | `feature_drift_detection.py` exists but is not connected to the live pipeline. |
| **Alerting** | `[0]` | Alert manager is a print-to-log placeholder. No Slack, PagerDuty, or OpsGenie integration. |
| **Metrics** | `[1]` | Prometheus metrics module exists but is not instrumented into the execution loop. |
| **Logging** | `[1]` | Standard Python `logging` module. No structured logging (JSON). No log aggregation (ELK/Datadog). |
| **Overall** | **`[1] Retail`** | |

---

## COMPOSITE SCORE CARD

| Module | Score (1-5) | Weight | Weighted |
|---|---|---|---|
| Data Layer | 1.0 | 20% | 0.20 |
| Feature Store | 1.5 | 10% | 0.15 |
| Regime Detection | 1.5 | 5% | 0.075 |
| Alpha Engine | 1.0 | 20% | 0.20 |
| Portfolio Optimization | 1.0 | 10% | 0.10 |
| Risk Engine | 2.0 | 15% | 0.30 |
| Execution Engine | 1.0 | 10% | 0.10 |
| Backtesting | 1.0 | 5% | 0.05 |
| ML Pipeline | 1.0 | 3% | 0.03 |
| Monitoring | 1.0 | 2% | 0.02 |
| **TOTAL** | | **100%** | **1.225 / 5.0** |

> [!IMPORTANT]
> **Composite Institutional Readiness Score: 1.225 / 5.0 (24.5%)**
> Classification: **Retail Hobbyist with Advanced Terminology**

---

# SECTION 3: MISSING RESEARCH SYSTEMS

The following 20 systems are absent from the platform. Each is mandatory for institutional operation. Every system is evaluated using the 7-question institutional framework.

## 3.1 Point-in-Time (PIT) Data Lake

**(1) WHY it exists:** To store every data observation alongside *when it was known* (valid time) and *when it was recorded* (transaction time). This dual-timestamp schema is the bedrock of bias-free quantitative research.

**(2) WHICH institutional problem it solves:** Eliminates lookahead bias. When backtesting as of January 15, 2024, the system must query data exactly as it existed on that date — before earnings revisions, before index reconstitutions, before dividend announcements that arrived on January 20.

**(3) WHY retail systems miss it:** Implementing PIT requires 10-50x more storage and fundamentally different query patterns (`SELECT ... WHERE valid_time <= @as_of_date AND transaction_time <= @as_of_date`). Retail databases only store current state.

**(4) HOW it integrates:** Replaces SQLite. All data access goes through a PIT query API. The backtester passes two timestamps: the simulation clock and the "as-of" clock.

**(5) Expected research impact:** **Transformative.** Every backtest result becomes trustworthy. Currently, 100% of backtest results are suspect due to lookahead contamination.

**(6) Expected alpha impact:** Prevents deploying strategies whose backtested Sharpe was inflated by 0.5-2.0 points due to lookahead bias.

**(7) Expected scalability impact:** ClickHouse or kdb+ scales horizontally. Supports petabyte-scale tick archives with sub-second query latency.

---

## 3.2 Alpha Registry with Orthogonalization

**(1) WHY:** Catalog every alpha factor with metadata, exposure profiles, and continuous performance tracking.

**(2) WHICH problem:** Factor crowding. When multiple alphas load on the same underlying risk factor (e.g., momentum), the portfolio takes 3x the intended exposure without knowing it.

**(3) WHY retail misses:** Retail runs 1-3 strategies. Orthogonalization matters when running 100+.

**(4) HOW it integrates:** Between feature store and portfolio optimizer. Every new alpha candidate is regressed against all live alphas. Only the residual (orthogonal component) is retained.

**(5) Research impact:** Quantifies marginal Sharpe contribution of each new factor before production deployment.

**(6) Alpha impact:** Prevents concentrated factor bets. Preserves portfolio-level Sharpe by ensuring diversification.

**(7) Scalability impact:** PCA/SVD decomposition scales as $O(n^2 k)$ where $n$ is number of factors and $k$ is number of principal components retained.

---

## 3.3 Deflated Sharpe Ratio Gate

**(1) WHY:** To statistically adjust the observed Sharpe ratio for the number of backtests run during strategy search.

**(2) WHICH problem:** Multiple testing bias (p-hacking). If a researcher runs 100 backtests and picks the best one, the expected maximum Sharpe ratio of a set of i.i.d. normal random variables with $N$ trials is approximately $\sqrt{2 \ln N}$. For $N=100$, this is ~3.0 — purely from randomness.

**(3) WHY retail misses:** Retail traders do not track trial counts. They see a backtest with Sharpe 2.5 and believe it's real.

**(4) HOW it integrates:** Mandatory gate in the Alpha Registry. The system logs every backtest run (parameters, features, data range). The Deflated Sharpe Ratio formula:
$$DSR = \Phi\left(\frac{(\hat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \cdot \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \cdot \hat{SR}^2}}\right)$$
where $\hat{\gamma}_3$ is skewness, $\hat{\gamma}_4$ is kurtosis, and $SR_0$ is the expected maximum Sharpe from $N$ independent trials.

**(5) Research impact:** Prevents deployment of overfitted strategies. A researcher must achieve DSR > 95% confidence to pass the gate.

**(6) Alpha impact:** Dramatically reduces live strategy failure rate from ~70% (industry average for unfiltere strategies) to ~20%.

**(7) Scalability impact:** Negligible compute cost. Simple statistical formula.

---

## 3.4 L3 Order Book Replay Engine

**(1) WHY:** To reconstruct the historical limit order book state at every point in time, enabling realistic execution simulation.

**(2) WHICH problem:** The current backtester assumes instant fills at the close/mid price. In reality, large orders move the market, queue behind existing orders, and receive partial fills.

**(3) WHY retail misses:** L3 data for a single exchange (NSE) is ~50-200 GB/day compressed. Storing 5 years requires petabyte-scale infrastructure and custom binary parsers.

**(4) HOW it integrates:** Replaces the `SimulationEngine` in backtest mode. Every simulated order is matched against the reconstructed book.

**(5) Research impact:** Enables research into execution-sensitive strategies (market making, stat arb, HFT).

**(6) Alpha impact:** Accurate slippage estimation prevents deploying strategies that are profitable before costs but negative after.

**(7) Scalability impact:** Requires C++ replay engine with memory-mapped files processing 10M+ messages/second. ZSTD compression for storage efficiency.

---

## 3.5 Transaction Cost Analysis (TCA) Engine

**(1) WHY:** To compare every realized execution price against statistical benchmarks (Arrival Price, Interval VWAP, Implementation Shortfall).

**(2) WHICH problem:** Without TCA, the fund cannot distinguish between alpha decay and execution leakage. A strategy might still generate valid signals but lose money because the execution algorithm is suboptimal.

**(3) WHY retail misses:** Retail order sizes have negligible market impact. Retail brokers offer no venue choice or routing control.

**(4) HOW it integrates:** Post-trade batch process. Reads from the execution ledger, computes benchmarks from the tick database, and writes metrics to the monitoring dashboard.

**(5) Research impact:** Feeds empirical market impact coefficients ($\alpha, \beta$ in the Almgren-Chriss model) back into the portfolio optimizer's transaction cost penalty.

**(6) Alpha impact:** Directly increases net PnL by 10-50 bps/year through optimal execution algorithm selection.

**(7) Scalability impact:** Asynchronous batch process. Runs post-market. Minimal impact on live trading latency.

---

## 3.6 Factor Risk Model (Barra-Style)

**(1) WHY:** To decompose portfolio risk into systematic factor exposures (market, size, value, momentum, volatility, quality, liquidity) and idiosyncratic risk.

**(2) WHICH problem:** The current system has no concept of *why* the portfolio is making or losing money. Is the PnL coming from alpha (stock selection) or from unintended factor tilts (e.g., the portfolio is accidentally long small-cap momentum)?

**(3) WHY retail misses:** Building a factor risk model requires: (a) defining factors, (b) estimating factor returns via cross-sectional regression, (c) computing factor covariance matrices, (d) decomposing portfolio variance. This is a multi-month engineering effort.

**(4) HOW it integrates:** Feeds into the portfolio optimizer (factor exposure constraints) and the risk engine (factor-attributed VaR).

**(5) Research impact:** Allows researchers to isolate alpha from beta. "This strategy has a 2.0 Sharpe, but 1.5 of that is explained by the momentum factor."

**(6) Alpha impact:** Prevents unintended factor loading. Ensures the portfolio earns returns from genuine stock selection, not from levered factor bets.

**(7) Scalability impact:** Cross-sectional regression runs in $O(N \cdot K)$ where $N$ is universe size and $K$ is number of factors. Easily parallelizable.

---

## 3.7 Embargo/Purge Cross-Validation Framework

**(1) WHY:** To ensure zero data leakage between training and test sets in time-series machine learning.

**(2) WHICH problem:** Standard K-Fold or even time-series split does not account for overlapping labels. If a feature uses a 5-day forward return as the target, the training set's last 5 rows overlap with the test set's first row.

**(3) WHY retail misses:** Retail ML tutorials use `train_test_split(shuffle=True)` which is catastrophically wrong for time-series data.

**(4) HOW it integrates:** Replaces the cross-validation logic in `src/ml/trainer.py`. Implements the purge-embargo methodology from Marcos López de Prado's "Advances in Financial Machine Learning":
- **Purge:** Remove training observations whose labels overlap with any test observation.
- **Embargo:** Add a gap of $h$ trading days between the training and test sets.

**(5) Research impact:** Realistic out-of-sample metrics. ML models that pass this gate are genuinely predictive.

**(6) Alpha impact:** Reduces live ML strategy drawdown by 30-50% by preventing deployment of overfitted models.

**(7) Scalability impact:** Negligible. Simple index manipulation in the data splitting logic.

---

## 3.8 Alternative Data Pipeline

**(1) WHY:** To ingest and process non-traditional data sources: satellite imagery, credit card transaction aggregates, web scraping, social media sentiment, shipping/logistics data, patent filings, SEC/SEBI filings, job postings.

**(2) WHICH problem:** Alpha from traditional price/volume data is heavily competed. The marginal Sharpe from a new momentum variant is near zero. Alternative data provides orthogonal information.

**(3) WHY retail misses:** Alternative data vendors charge $50K-$500K/year per dataset. Processing requires NLP/computer vision pipelines.

**(4) HOW it integrates:** Feeds into the feature store as additional columns. Subject to the same PIT constraints.

**(5) Research impact:** Opens entirely new alpha dimensions inaccessible from price/volume alone.

**(6) Alpha impact:** The most significant source of alpha edge at institutional scale.

**(7) Scalability impact:** Requires dedicated NLP/CV infrastructure (GPU clusters, Spark/Flink streaming).

---

## 3.9 Model Decay & Drift Monitor

**(1) WHY:** To detect when a deployed model's input feature distributions or output signal distributions shift from their historical baseline.

**(2) WHICH problem:** Silent model failure. An exchange changes its tick format, or a corporate action adjusts a stock's price level, or a new regulation alters trading behavior. The model continues to produce signals, but they are now based on out-of-distribution inputs.

**(3) WHY retail misses:** Retail traders monitor PnL manually. They lack statistical drift detection tools.

**(4) HOW it integrates:** Independent monitoring service. Computes KL divergence and Wasserstein distance on rolling windows of feature distributions. Triggers alerts when drift exceeds thresholds.

**(5) Research impact:** Automated early warning for model retraining.

**(6) Alpha impact:** Prevents catastrophic losses from a model operating on corrupted inputs.

**(7) Scalability impact:** Uses streaming approximation algorithms (t-digest, Count-Min Sketch) for O(1) memory.

---

## 3.10 Experiment Tracking & Research Notebook Registry

**(1) WHY:** To log every research experiment with its parameters, data splits, features, metrics, and artifacts.

**(2) WHICH problem:** Research irreproducibility. A researcher finds a promising strategy but cannot recreate the exact conditions (which feature version? which data range? which hyperparameters?).

**(3) WHY retail misses:** Retail traders do not track experiments systematically. They modify Jupyter notebooks in-place.

**(4) HOW it integrates:** MLflow or Weights & Biases integrated into the research pipeline. Every backtest run automatically logs to the tracking server.

**(5) Research impact:** Full reproducibility. Any past experiment can be re-executed with identical results.

**(6) Alpha impact:** Prevents wasted research cycles re-discovering already-explored ideas.

**(7) Scalability impact:** Centralized database. Scales linearly with experiment count.

---

## 3.11–3.20 Additional Missing Systems (Summary Table)

| # | System | WHY (Core) | Gap Severity |
|---|---|---|---|
| 3.11 | **Cointegration & Pairs Trading Scanner** | Identify mean-reverting spread portfolios. Johansen test for multi-asset baskets. | Critical |
| 3.12 | **Options Volatility Surface Engine (SABR/SVI)** | Construct arbitrage-free implied volatility surfaces. Price exotic options. | Critical |
| 3.13 | **Funding & Margin Optimization** | Track borrow costs, margin offsets, and cross-collateralization. Minimize funding drag. | High |
| 3.14 | **Sentiment & News NLP Pipeline** | Process real-time news feeds and social media for event-driven signals. | High |
| 3.15 | **Hawkes Process Order Flow Model** | Self-exciting point process for trade clustering and short-term price prediction. | High |
| 3.16 | **Optimal Execution Scheduler (Almgren-Chriss)** | Compute time-optimal liquidation trajectories balancing impact vs. risk. | Critical |
| 3.17 | **Live/Paper/Backtest Parity Checker** | Automated system ensuring identical code paths across all three execution modes. | High |
| 3.18 | **Distributed Backtest Orchestrator (Kubernetes)** | Run thousands of backtests in parallel across GPU clusters. | Medium |
| 3.19 | **Audit Trail & Regulatory Reporting** | Immutable trade log for SEBI, exchange, and internal compliance audit. | Critical |
| 3.20 | **Disaster Recovery & Failover** | Hot standby execution servers. Automatic failover within 50ms. | Critical |

---

# SECTION 4: MATHEMATICS ROADMAP

## 4.1 Measure Theory & Stochastic Calculus

**Required Knowledge:**
- Probability spaces $(\Omega, \mathcal{F}, \mathbb{P})$
- Filtrations $\{\mathcal{F}_t\}_{t \geq 0}$ and adapted processes
- Brownian motion $W_t$, Itô's Lemma: $df(X_t) = f'(X_t) dX_t + \frac{1}{2} f''(X_t) (dX_t)^2$
- Girsanov's theorem for measure change $\mathbb{P} \to \mathbb{Q}$
- Radon-Nikodým derivative: $\frac{d\mathbb{Q}}{d\mathbb{P}} = \mathcal{E}\left(-\int_0^T \theta_s dW_s\right)$
- Martingale representation theorem

**Application:** Options pricing, hedging strategy derivation, risk-neutral valuation of derivatives.

**Current system gap:** The system has zero stochastic calculus. Options features are hardcoded proxies (`realized_volatility` used as IV mock).

---

## 4.2 Extreme Value Theory (EVT) & Tail Risk

**Required Knowledge:**
- Generalized Extreme Value (GEV) distribution: $G_\xi(x) = \exp\left(-(1+\xi x)^{-1/\xi}\right)$
- Peaks-Over-Threshold (POT) method with Generalized Pareto Distribution (GPD)
- Fisher-Tippett-Gnedenko theorem classifying tail types (Fréchet, Gumbel, Weibull)
- Hill estimator for tail index $\xi$

**Application:** Tail VaR estimation. Standard Gaussian VaR underestimates extreme losses by 30-50%. GPD-based VaR provides accurate tail quantile estimates.

**Current system gap:** EVT VaR exists in `metrics.py` but is not integrated into the pre-trade risk gate (which doesn't exist).

---

## 4.3 Random Matrix Theory (RMT) for Covariance Cleaning

**Required Knowledge:**
- Marchenko-Pastur distribution for eigenvalues of sample covariance matrices
- Noise eigenvalue threshold: $\lambda_+ = \sigma^2(1 + \sqrt{N/T})^2$
- Ledoit-Wolf shrinkage: $\hat{\Sigma}_{LW} = \alpha \cdot \text{diag}(\hat{\Sigma}) + (1-\alpha) \cdot \hat{\Sigma}$
- Oracle shrinkage estimator
- De-noised covariance via eigenvalue clipping

**Application:** The sample covariance matrix of $N$ assets from $T$ observations is severely biased when $N/T > 0.1$. RMT filtering removes noise eigenvalues, producing a covariance matrix that generates stable, well-conditioned portfolio weights.

**Current system gap:** The system uses `np.eye(n)` (identity matrix) as the covariance. This is not covariance cleaning; it is covariance *deletion*.

---

## 4.4 Convex Optimization Theory

**Required Knowledge:**
- Convex sets, convex functions, supporting hyperplanes
- KKT (Karush-Kuhn-Tucker) conditions for optimality
- Quadratic programming (QP): $\min \frac{1}{2} x^T Q x + c^T x$ s.t. $Ax \leq b$
- Second-order cone programming (SOCP) for robust optimization
- ADMM (Alternating Direction Method of Multipliers) for distributed optimization
- Solver selection: OSQP (fast QP), SCS (conic), ECOS (small SOCP), Mosek (commercial)

**Application:** Portfolio optimization with realistic constraints. The current `cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)` is correct in form but trivial in constraint specification.

**Current system gap:** Long-only constraint. No transaction cost penalty. No turnover limit. No factor exposure bounds. ECOS solver (line 79 of `institutional_allocator.py`) is designed for small problems; it will fail on 500+ asset universes.

---

## 4.5 Information Theory for Alpha Research

**Required Knowledge:**
- Entropy: $H(X) = -\sum p(x) \log p(x)$
- Mutual information: $I(X; Y) = H(X) - H(X|Y)$
- Transfer entropy for causal (Granger-type) information flow
- Information Coefficient (IC): Spearman rank correlation between predicted and realized returns
- Information Ratio (IR): $IR = IC \cdot \sqrt{\text{Breadth}}$ (Fundamental Law of Active Management)

**Application:** The IC measures the predictive power of an alpha signal. The Fundamental Law ($IR = IC \times \sqrt{N}$) dictates that breadth (number of independent bets) is as important as accuracy. A strategy with IC=0.05 across 500 stocks (breadth=500) has IR = $0.05 \times \sqrt{500} \approx 1.12$, which is competitive. A strategy with IC=0.20 across 2 stocks has IR = $0.20 \times \sqrt{2} \approx 0.28$, which is not.

**Current system gap:** IC is never computed anywhere in the codebase. The `AlphaPerformance.observed_sharpe` is permanently zero.

---

## 4.6 Bayesian Statistics & Online Learning

**Required Knowledge:**
- Bayes' theorem: $P(\theta|D) \propto P(D|\theta) P(\theta)$
- Conjugate priors (Beta-Binomial, Normal-Normal, Gamma-Poisson)
- Bayesian Online Changepoint Detection (BOCPD)
- Thompson Sampling for multi-armed bandit strategy allocation
- Kalman Filters for state estimation and parameter tracking

**Application:** Regime detection via BOCPD. Strategy allocation via Thompson Sampling (explore-exploit tradeoff between deployed alphas). Dynamic parameter estimation via Kalman Filters.

**Current system gap:** The HMM is fitted once at startup and never updated. No online learning whatsoever.

---

## 4.7 Time-Series Econometrics

**Required Knowledge:**
- Stationarity testing: ADF, KPSS, Phillips-Perron
- Cointegration: Engle-Granger two-step, Johansen multivariate test
- VECM (Vector Error Correction Model) for cointegrated systems
- GARCH/EGARCH/GJR-GARCH for volatility modeling
- HAR-RV (Heterogeneous Autoregressive Realized Volatility) model
- Fractional integration and long memory processes

**Application:** Pairs trading (cointegration), volatility forecasting (GARCH family), and realized volatility modeling (HAR-RV).

**Current system gap:** None of these models are implemented or used. The `har_rv_volatility.py` file exists under `production/market_data/` but is disconnected.

---

## 4.8 Market Microstructure Mathematics

**Required Knowledge:**
- Kyle's Lambda: $\lambda = \frac{\Delta P}{\Delta Q}$ (price impact per unit flow)
- Roll's model for bid-ask spread estimation from transaction data
- Glosten-Milgrom model for adverse selection and spread decomposition
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Hawkes processes for self-exciting order flow modeling
- Queue-reactive models for limit order book dynamics

**Application:** Execution optimization, market making, order flow toxicity detection.

**Current system gap:** Files exist under `production/market_data/microstructure/` (order_flow_toxicity.py, high_frequency.py, tick_order_book.py, liquidity.py) but none are integrated into the live pipeline. They are dead code.

---

## 4.9 Reinforcement Learning for Execution

**Required Knowledge:**
- Markov Decision Processes (MDP): $(S, A, P, R, \gamma)$
- Policy gradient methods (REINFORCE, PPO, A2C)
- Deep Q-Networks (DQN) with experience replay
- State representation: order book snapshots, inventory, time-to-close
- Reward shaping: implementation shortfall relative to arrival price

**Application:** Learning optimal order placement policies (limit vs. market, size, timing) that adapt to changing market conditions.

**Current system gap:** Zero RL implementation. Execution uses static rules.

---

## 4.10 Causal Inference & Structural Models

**Required Knowledge:**
- Structural Causal Models (SCMs) and do-calculus
- Directed Acyclic Graphs (DAGs) for causal reasoning
- Instrumental variables for endogeneity
- Difference-in-differences for event studies
- Granger causality testing

**Application:** Distinguishing genuine alpha (causal predictive signal) from spurious correlation. The `KnowledgeGraph` class claims to store "causal hypotheses" but has no causal inference methodology.

**Current system gap:** The `KnowledgeGraph` has one node and zero edges. No causal reasoning is performed anywhere.

---

# SECTION 5: QUANTITATIVE FINANCE ROADMAP

## 5.1 Exchange Microstructure & Protocol Layer

Build a native exchange connectivity layer:
- **NSE/BSE direct market access (DMA):** FIX 4.4 protocol adapter for order routing
- **Tick data ingestion:** UDP multicast receiver for NSE's market data feed
- **Order book reconstruction:** C++ engine maintaining full L3 order book state from message log
- **Colocation:** Deploy execution servers in NSE's colocation facility at Mahape, Navi Mumbai

**Why this matters:** The current Kite Connect REST API adds 50-200ms of latency. At institutional scale, this latency costs real money on every trade execution. Direct market access reduces round-trip to sub-1ms.

## 5.2 Options Volatility Surface (SABR/SVI)

Build a continuous implied volatility surface engine:
- **SVI parameterization:** $w(k) = a + b\left(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right)$ where $k = \ln(K/F)$
- **SABR model:** $\sigma_B(K,T) = \frac{\alpha}{(FK)^{(1-\beta)/2}} \cdot \frac{z}{x(z)}$ for stochastic volatility calibration
- **Calendar arbitrage checks:** Ensure total variance is non-decreasing in expiry
- **Butterfly arbitrage checks:** Ensure the density function (second derivative of call price w.r.t. strike) is non-negative

**Why this matters:** The current system uses `realized_volatility` as a proxy for implied volatility. This is equivalent to a weather forecaster using yesterday's temperature as today's forecast.

## 5.3 Almgren-Chriss Optimal Execution

Implement the closed-form optimal liquidation trajectory:
- **Temporary impact:** $h(\dot{x}) = \eta \cdot \dot{x}$ (linear in trading rate)
- **Permanent impact:** $g(\dot{x}) = \gamma \cdot \dot{x}$ (linear in trading rate)
- **Optimal trajectory:** $x_j^* = \frac{\sinh(\kappa(T-t_j))}{\sinh(\kappa T)} \cdot X$ where $\kappa = \cosh^{-1}\left(\frac{\tau^2 \sigma^2}{2\eta} + 1\right)$

**Why this matters:** The current system submits orders with no scheduling logic. A ₹50 crore liquidation would be dumped into the market instantaneously, causing massive adverse price impact.

## 5.4 Cross-Asset Volatility Arbitrage

Build trading strategies that exploit pricing discrepancies between:
- Index options implied volatility vs. realized volatility
- Single-stock implied volatility vs. index implied volatility (dispersion trading)
- VIX/India VIX futures curve shape (contango/backwardation)
- Cross-market volatility (NIFTY vs. S&P 500 implied correlation)

## 5.5 Statistical Arbitrage Infrastructure

Build the complete stat-arb stack:
- **Universe screening:** Cointegration tests (Johansen) across all NIFTY500 pairs (~125K combinations)
- **Spread construction:** Kalman Filter for dynamic hedge ratio estimation
- **Entry/exit rules:** Z-score based with regime-conditional thresholds
- **Capacity estimation:** ADV-weighted position limits per leg

## 5.6 Multi-Asset Portfolio Construction

Extend the portfolio optimizer to handle:
- **Equity + Options:** Delta-hedged option overlays for tail protection
- **Equity + Futures:** Index futures for beta hedging
- **Cross-currency:** USD/INR hedging for global equity exposure
- **Fixed income:** Government bond duration management for interest rate risk

---

# SECTION 6: RESEARCH ORGANIZATION DESIGN

## 6.1 Alpha Lifecycle Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ALPHA LIFECYCLE STATE MACHINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [HYPOTHESIS]──▶[BACKTEST]──▶[PAPER TRADE]──▶[INCUBATION]──▶[PRODUCTION] │
│        │              │              │              │              │        │
│        ▼              ▼              ▼              ▼              ▼        │
│   Peer Review    DSR Gate     Parity Check    Ramp-Up Plan   Full Capital  │
│   Causal DAG     Embargo CV   Live/BT Match   10% → 50%     Continuous    │
│   Lit Survey     OOS Sharpe   Execution TCA   Monitoring     IC Tracking  │
│                                                                     │      │
│                                                            [DECAY DETECTED]│
│                                                                     │      │
│                                              [SCALE DOWN]──▶[RETIRE]       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage Gates:

| Stage | Gate | Minimum Threshold |
|---|---|---|
| Hypothesis → Backtest | Peer review of causal hypothesis | 2 committee approvals |
| Backtest → Paper Trade | Deflated Sharpe Ratio | DSR > 95% confidence |
| Paper Trade → Incubation | Live/backtest PnL correlation | $R^2 > 0.85$ |
| Incubation → Production | 6-month live track record | Sharpe > 1.0, Max DD < 10% |
| Production → Scale Down | Rolling IC decline | IC < 50% of historical median |
| Scale Down → Retire | Sustained underperformance | Negative PnL for 3 consecutive months |

## 6.2 Backtest Overfitting Prevention Protocol

1. **Trial Logging:** Every backtest run is automatically logged with: strategy class, parameter set, feature set, data range, universe, random seed, and all performance metrics.
2. **Trial Count Tracking:** The system maintains a running count of unique trials per strategy family.
3. **Deflated Sharpe Gate:** Before any strategy advances to paper trading, compute:
   - $SR_0 = \text{Expected max Sharpe from } N \text{ trials}$ (using the Bailey-Borwein-Mattingley formula)
   - Require $DSR > 0$ at 95% confidence
4. **Haircut Rule:** Apply a minimum 30% Sharpe haircut between backtest and expected live performance.
5. **Feature Freeze:** Once a strategy enters paper trading, no feature modifications are permitted. Any modification restarts the lifecycle from backtest stage.

## 6.3 Model Retirement & Capital Recycling

```
RETIREMENT TRIGGERS:
├── Rolling 60-day IC drops below 0.02 (from historical 0.05+)
├── Rolling 20-day Sharpe drops below 0.0
├── Maximum drawdown exceeds 2x historical 99th percentile
├── Feature drift KL-divergence exceeds 0.5 nats
└── Capacity saturation: participation rate exceeds 5% of ADV

RETIREMENT PROCESS:
1. Reduce allocation by 50% immediately
2. Monitor for 20 trading days
3. If recovery detected (IC rebounds above 0.03), restore allocation
4. If no recovery, liquidate remaining position over 5 days (Almgren-Chriss)
5. Archive strategy in registry with full post-mortem analysis
6. Reallocate freed capital to highest-IR active strategies via Thompson Sampling
```

## 6.4 Team Structure & Role Separation

```
┌──────────────────────────────────────────────────────────────────────┐
│                    QUANTITATIVE RESEARCH DESK                        │
├──────────────┬───────────────────────┬───────────────────────────────┤
│              │                       │                               │
│   Quant      │   Quant Developer     │   Infrastructure             │
│   Researcher │   (QD)                │   Engineer                   │
│   (QR)       │                       │                               │
│              │                       │                               │
│   - Math     │   - C++/Rust core     │   - Colocation               │
│   - Stats    │   - Python wrappers   │   - Network tuning           │
│   - Models   │   - Database admin    │   - Kubernetes               │
│   - Signals  │   - Performance       │   - Monitoring               │
│   - Research │   - Testing           │   - Disaster recovery        │
│     papers   │   - CI/CD             │   - Hardware                 │
│              │                       │                               │
│   Writes:    │   Writes:             │   Writes:                    │
│   Python     │   C++, Rust, Python   │   Terraform, Ansible,        │
│   notebooks  │   production code     │   shell scripts              │
│              │                       │                               │
│   NEVER      │   NEVER touches       │   NEVER touches              │
│   touches    │   alpha logic         │   trading code               │
│   infra      │                       │                               │
└──────────────┴───────────────────────┴───────────────────────────────┘
```

**Critical principle:** A Quantitative Researcher should **never** write production infrastructure code. A Quantitative Developer should **never** modify alpha signal logic. This separation prevents both mathematically invalid engineering and poorly optimized research code from reaching production.

## 6.5 Research Computing Infrastructure

| Resource | Specification | Purpose |
|---|---|---|
| **GPU Cluster** | 8x NVIDIA A100 80GB | ML model training, Monte Carlo simulation |
| **CPU Cluster** | 128-core AMD EPYC (2x nodes) | Backtesting, feature computation, optimization |
| **Scheduler** | Slurm or Ray Cluster | Job queuing, resource allocation |
| **Storage** | 500TB NVMe SSD array | Tick data, L3 order book archives |
| **Database** | ClickHouse cluster (3 nodes) | PIT timeseries queries |
| **Experiment Tracker** | MLflow or Weights & Biases | Research reproducibility |
| **Notebook Server** | JupyterHub with conda envs | Interactive research |
| **CI/CD** | GitLab CI with GPU runners | Automated testing, deployment |

---

# SECTION 7: FUTURE STATE ARCHITECTURE (QROS v2)

## 7.1 High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           QROS v2 — INSTITUTIONAL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   LAYER 0: EXCHANGE CONNECTIVITY                                                    │
│   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐               │
│   │ NSE UDP Multicast│  │ BSE Market Data  │  │ Global Feeds        │               │
│   │ (Tick Receiver)  │  │ (FIX MD)         │  │ (Bloomberg/Reuters) │               │
│   └────────┬────────┘  └────────┬─────────┘  └─────────┬───────────┘               │
│            │                    │                       │                            │
│            └────────────────────┼───────────────────────┘                            │
│                                 ▼                                                    │
│   LAYER 1: DATA INFRASTRUCTURE                                                      │
│   ┌──────────────────────────────────────────────────────────────────┐               │
│   │                    RING BUFFER (Lock-Free C++)                    │               │
│   │            Zero-copy tick ingestion. 10M msg/sec.                │               │
│   └──────────────────────────┬───────────────────────────────────────┘               │
│                              ▼                                                       │
│   ┌───────────────────────────────────────────────────────────────────┐              │
│   │               POINT-IN-TIME DATA LAKE (ClickHouse)                │              │
│   │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │              │
│   │  │ Tick DB   │  │ Daily OHLCV  │  │ Corp Actions │  │ Universe │ │              │
│   │  │ (L3 Msgs) │  │ (Adjusted)   │  │ (PIT Schema) │  │ (PIT)    │ │              │
│   │  └──────────┘  └──────────────┘  └──────────────┘  └──────────┘ │              │
│   └───────────────────────────┬───────────────────────────────────────┘              │
│                               ▼                                                      │
│   LAYER 2: FEATURE & SIGNAL COMPUTATION                                              │
│   ┌──────────────────────────────────────────────────────────────────┐               │
│   │              FEATURE COMPUTE ENGINE (C++/CUDA)                    │               │
│   │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────────┐ │               │
│   │  │Technical │  │Cross-Sect │  │Microstruc│  │Alt Data (NLP)  │ │               │
│   │  │Indicators│  │Features   │  │Features  │  │Sentiment       │ │               │
│   │  └──────────┘  └───────────┘  └──────────┘  └────────────────┘ │               │
│   └───────────────────────────┬───────────────────────────────────────┘              │
│                               ▼                                                      │
│   LAYER 3: ALPHA & REGIME                                                            │
│   ┌────────────────────┐  ┌───────────────────────────────────────────┐              │
│   │ REGIME ENGINE       │  │ ALPHA REGISTRY & MARKETPLACE              │              │
│   │ ┌───────┐ ┌───────┐│  │ ┌──────────┐ ┌──────────┐ ┌───────────┐ │              │
│   │ │BOCPD  │ │Online ││  │ │StatArb   │ │Momentum  │ │ML/DL      │ │              │
│   │ │(Vol)  │ │HMM    ││  │ │(Pairs)   │ │(XS/TS)   │ │(TabNet)   │ │              │
│   │ └───────┘ └───────┘│  │ └──────────┘ └──────────┘ └───────────┘ │              │
│   │ ┌───────┐ ┌───────┐│  │ ┌──────────┐ ┌──────────┐ ┌───────────┐ │              │
│   │ │Liqui- │ │Correl-││  │ │VolArb    │ │Event     │ │Options    │ │              │
│   │ │dity   │ │ation  ││  │ │(Dispers.)│ │(Earnings)│ │(Greeks)   │ │              │
│   │ └───────┘ └───────┘│  │ └──────────┘ └──────────┘ └───────────┘ │              │
│   └────────────────────┘  │                                           │              │
│                           │ IC Tracking │ Decay Detection │ DSR Gate  │              │
│                           └───────────────────────┬───────────────────┘              │
│                                                   ▼                                  │
│   LAYER 4: PORTFOLIO & RISK                                                          │
│   ┌────────────────────────────┐  ┌──────────────────────────────────┐               │
│   │ PORTFOLIO OPTIMIZER         │  │ PRE-TRADE RISK GATE (C++)        │               │
│   │ Multi-period OSQP/Mosek     │  │ Real-time VaR/CVaR              │               │
│   │ Factor exposure constraints │  │ Dynamic copula correlations     │               │
│   │ Transaction cost penalties  │  │ Stress test scenarios           │               │
│   │ Ledoit-Wolf covariance      │  │ Exchange margin (SPAN)          │               │
│   │ Turnover & concentration    │  │ Circuit breaker enforcement     │               │
│   └─────────────┬──────────────┘  └────────────────┬─────────────────┘               │
│                 │                                   │                                │
│                 └─────────────┬─────────────────────┘                                │
│                               ▼                                                      │
│   LAYER 5: EXECUTION                                                                 │
│   ┌──────────────────────────────────────────────────────────────────┐               │
│   │                SMART ORDER ROUTER (SOR)                          │               │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │               │
│   │  │Almgren-Chriss│  │VWAP/TWAP     │  │RL Execution Agent    │  │               │
│   │  │Scheduler     │  │Slicer        │  │(PPO-trained)         │  │               │
│   │  └──────────────┘  └──────────────┘  └──────────────────────┘  │               │
│   │                         │                                       │               │
│   │              ┌──────────┼──────────┐                            │               │
│   │              ▼          ▼          ▼                            │               │
│   │         [FIX 4.4]  [OUCH/ITCH] [Kite API]                     │               │
│   │         (DMA)      (Binary)    (Retail Fallback)               │               │
│   └──────────────────────────────────────────────────────────────────┘               │
│                                                                                      │
│   LAYER 6: POST-TRADE & MONITORING                                                   │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐          │
│   │ TCA Engine    │  │ Drift Monitor│  │ PnL Attrib.  │  │ Audit Trail    │          │
│   │ (IS, VWAP)    │  │ (KL, Wass.)  │  │ (Factor)     │  │ (Immutable Log)│          │
│   └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 7.2 Data Flow Architecture

```
Exchange ──UDP──▶ [C++ Tick Receiver] ──Ring Buffer──▶ [ClickHouse Writer]
                        │                                      │
                        │ (Real-time path)                     │ (Historical path)
                        ▼                                      ▼
              [Feature Engine]                         [PIT Data Lake]
                        │                                      │
                        ▼                                      ▼
              [Alpha Signals]                          [Backtester]
                        │                                      │
                        ▼                                      │
              [Portfolio Optimizer] ◀───────────────────────────┘
                        │                    (calibrated parameters
                        │                     from backtest)
                        ▼
              [Pre-Trade Risk Gate]
                        │
                   PASS │ REJECT
                        ▼
              [Smart Order Router]
                        │
                        ▼
              [Exchange FIX Gateway]
                        │
                        ▼
              [Fill Confirmation]
                        │
                        ▼
              [Post-Trade TCA & Attribution]
```

## 7.3 Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| **Tick Receiver** | C++ with Solarflare OpenOnload | Kernel-bypass networking. Sub-μs packet processing. |
| **Ring Buffer** | LMAX Disruptor pattern (C++) | Lock-free, cache-friendly, single-writer/multi-reader. |
| **Timeseries DB** | ClickHouse (OSS) or kdb+ (commercial) | Columnar compression. Vectorized aggregation. 1B rows/sec scan. |
| **Feature Engine** | C++17 with pybind11 wrappers | Hot-path computation. Python interface for researchers. |
| **Alpha Research** | Python (NumPy, pandas, JAX) | Researcher productivity. JAX for GPU-accelerated research. |
| **Portfolio Optimizer** | cvxpy + OSQP/Mosek backend | Production-grade QP solver. Warm-starting support. |
| **Risk Gate** | C++ compiled module | Zero-allocation hot path. Deterministic latency. |
| **Order Router** | C++ with FIX engine (QuickFIX/N) | Direct market access. Sub-ms order submission. |
| **Orchestration** | Kubernetes + Helm | Service mesh. Rolling deployments. Auto-scaling. |
| **Monitoring** | Prometheus + Grafana + PagerDuty | Real-time metrics. Automated alerting. Incident management. |
| **Experiment Tracking** | MLflow | Reproducible research. Model registry. |
| **CI/CD** | GitLab CI + Docker | Automated testing. Containerized deployments. |

## 7.4 Latency Budget (Live Trading Path)

```
Exchange multicast packet arrives
    │
    ├── 0μs ───── [C++ Tick Receiver]
    │              NIC interrupt → userspace
    │
    ├── 5μs ───── [Ring Buffer Write]
    │              Lock-free enqueue
    │
    ├── 15μs ──── [Feature Computation]
    │              C++ vectorized indicators
    │
    ├── 25μs ──── [Alpha Signal Generation]
    │              Pre-computed thresholds
    │
    ├── 50μs ──── [Portfolio Delta Computation]
    │              Incremental weight update
    │
    ├── 60μs ──── [Pre-Trade Risk Check]
    │              Compiled C++ gate
    │
    ├── 70μs ──── [FIX Message Construction]
    │              Binary serialization
    │
    ├── 80μs ──── [Network Transmission]
    │              Colocated switch hop
    │
    └── 90μs ──── [Exchange Acknowledgement]
                   Order accepted

TOTAL: ~90 microseconds tick-to-trade
```

**Current system latency: ~200-500 milliseconds** (Python → REST API → Internet → Kite → Exchange).
**Target latency: ~90 microseconds** (C++ → FIX → Colocation switch → Exchange).
**Improvement factor: ~2,000-5,000x.**

---

# SECTION 8: THE HARSH TRUTH

This section is written without diplomatic softening. It is what the committee would say in a closed-door session.

## 8.1 The Dashboard Delusion

The system was built **dashboard-first**. The glassmorphism CSS, the WebSocket real-time updates, the animated charts — these are what received engineering attention. The *actual quantitative engine* behind the dashboard is a thin wrapper around `yfinance.download()` and `cvxpy.Maximize()` with an identity covariance matrix.

**At Jane Street, there is no dashboard.** Traders interact with the system through command-line tools, custom GUIs built on Qt/GTK, and alerting systems. The *engine* is everything. The *interface* is secondary.

**At Renaissance Technologies, the interface is a terminal.** Research is conducted in notebooks. Production runs on headless Linux servers.

Building a beautiful dashboard and calling it an "Institutional Quant Research OS" is like putting a Rolls-Royce hood ornament on a bicycle and calling it a luxury car.

## 8.2 The Terminology Inflation Problem

The codebase is infected with terminology inflation:

| Code Says | Code Does |
|---|---|
| `InstitutionalQuantOS` | Wrapper around Yahoo Finance daily bars |
| `AlphaMarketplace` | Dictionary with one entry and a permanently-zero Sharpe tracker |
| `KnowledgeGraph` | Object with one node, zero edges, and no inference engine |
| `CapitalAllocationEngine` | cvxpy solver using identity covariance matrix |
| `InstitutionalRiskEngine` | Post-trade risk check that is bypassed via `DummyPortRisk` |
| `ClusterEngine` | Ray-based distributed engine that has never been run on a cluster |
| `SimulationEngine` | Random number generator without seed control |
| `EventStore` | Class that exists but is never written to or read from in the live path |

**Naming a class `Institutional` does not make it institutional.** The gap between the terminology and the implementation is so large that it actively impedes progress, because it creates the illusion that these systems exist when they do not.

## 8.3 The Single-Strategy Trap

The entire platform revolves around **one alpha strategy: Opening Range Breakout (ORB).**

An institutional platform manages 200-500+ alpha factors simultaneously. It requires:
- Cross-sectional signal combination
- Factor exposure management
- Correlation-aware capital allocation
- Continuous IC monitoring and decay detection

With one strategy, none of these systems can be tested, validated, or even meaningfully developed. The platform is a single-strategy executor disguised as a multi-strategy platform.

## 8.4 The Python Performance Ceiling

The entire system runs in CPython. Key implications:

| Bottleneck | Impact |
|---|---|
| **GIL (Global Interpreter Lock)** | True multi-threading is impossible. The system is single-core. |
| **Object overhead** | Every Python float consumes 28 bytes (vs. 8 bytes in C). Memory usage is ~3.5x higher than necessary. |
| **Interpretation overhead** | Python is ~50-100x slower than compiled C++ for numerical computation. |
| **Garbage collection** | GC pauses introduce non-deterministic latency spikes. |

For research and prototyping, Python is ideal. For production execution, the hot path (tick processing → feature computation → signal generation → risk check → order submission) **must** be compiled C++ or Rust.

## 8.5 The Data Foundation Is Sand

**You cannot build an institutional platform on Yahoo Finance data.**

| Attribute | Yahoo Finance | Institutional Requirement |
|---|---|---|
| Granularity | Daily/5-min bars | Tick-by-tick (every order, cancel, trade) |
| Latency | 15-min delayed | Real-time (sub-millisecond) |
| Reliability | Rate-limited, occasional outages | 99.99% uptime SLA |
| Corporate Actions | Approximate | Exact adjustment factors with PIT timestamps |
| Historical Depth | ~10 years | 20+ years with full order book |
| Cost | Free | ₹5-50 lakh/year for exchange feeds |

Building quantitative models on Yahoo Finance data is like training a medical AI on Wikipedia articles instead of clinical records.

## 8.6 What Must Change — Prioritized Roadmap

### Phase 1: Foundation (Months 1-3)
1. Replace SQLite with ClickHouse for all timeseries storage
2. Implement PIT schema with dual-timestamp columns
3. Build historical NIFTY50 constituent database (backfill 20 years)
4. Seed all random number generators for reproducible backtests
5. Remove all `DummyPortRisk` patterns — wire real risk engine to execution
6. Compute and store real covariance matrices (Ledoit-Wolf shrinkage)

### Phase 2: Alpha Infrastructure (Months 3-6)
7. Build Alpha Registry with IC tracking, decay detection, and orthogonalization
8. Implement Deflated Sharpe Ratio gate for backtest validation
9. Build 10+ additional alpha factors (momentum, value, quality, volatility, mean-reversion)
10. Implement embargo/purge cross-validation in ML pipeline
11. Deploy experiment tracking (MLflow)

### Phase 3: Execution & Risk (Months 6-9)
12. Build pre-trade risk gate (intercept orders before execution)
13. Implement Almgren-Chriss optimal execution scheduler
14. Build post-trade TCA engine
15. Upgrade portfolio optimizer with factor constraints and transaction costs
16. Build factor risk model (Barra-style)

### Phase 4: Infrastructure (Months 9-12)
17. Deploy ClickHouse cluster on dedicated servers
18. Build C++ tick receiver for direct exchange feeds
19. Implement FIX 4.4 order gateway
20. Deploy Kubernetes orchestration
21. Build monitoring stack (Prometheus + Grafana + PagerDuty)

### Phase 5: Advanced (Months 12-18)
22. Options volatility surface engine (SABR/SVI)
23. Statistical arbitrage scanner (Johansen cointegration)
24. Alternative data pipeline (news NLP, satellite imagery)
25. Reinforcement learning execution agent
26. Colocation deployment at NSE Mahape

---

## 8.7 Final Words from the Committee

> [!WARNING]
> **The distance between where you are and where you need to be is not months of work. It is years of work, requiring a team of 5-10 specialized engineers, access to institutional-grade data feeds, and a fundamental shift in engineering philosophy from "build features" to "build foundations."**

The good news: the *ambition* is correct. Building a next-generation quant research platform for Indian markets is a worthy goal with massive potential. The awareness of institutional concepts (Alpha Marketplace, Knowledge Graph, Capital Allocation) shows genuine research effort.

The bad news: **awareness is not implementation.** Every one of those concepts exists as a class name with minimal or no functional logic behind it. The system must be rebuilt from the data layer upward, with mathematical rigor replacing placeholder code at every level.

The path forward requires:
1. **Humility** — accept that the current system is a prototype, not a product.
2. **Foundations first** — data integrity, reproducibility, and statistical validity before any new features.
3. **Depth over breadth** — one properly implemented alpha with rigorous backtest validation is worth more than 50 placeholder classes.
4. **Performance culture** — measure everything. IC, Sharpe, drawdown, latency, fill rate, slippage. What is not measured does not improve.

The committee unanimously recommends: **full first-principles rebuild starting from the data layer.**

---

*End of Report.*
*Joint Committee — Institutional Quant Research Architecture Review*
*Classification: Internal — Committee Eyes Only*
