"""Main orchestrator for the consolidated Quant Research OS."""

import argparse
import asyncio
import logging
from dataclasses import asdict, dataclass
import dataclasses
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

# Register PyYAML multi-representers for numpy types to prevent RepresenterError
yaml.SafeDumper.add_multi_representer(np.floating, lambda dumper, data: dumper.represent_float(float(data)))
yaml.SafeDumper.add_multi_representer(np.integer, lambda dumper, data: dumper.represent_int(int(data)))

import os
from src.alpha.manager import AlphaManager
from src.data.data_loader import NSEDataLoader
from market_data.feature_generation.feature_pipeline import FeaturePipeline, FeatureConfig
from src.portfolio.engine import PortfolioAllocator
from src.regime.detectors.hmm import RobustHMMRegime
from src.risk.institutional_risk_engine import InstitutionalRiskEngine
from execution.live.market_stream import NSEWebSocketStream
from alpha.orb_zarattini import scan_symbols
from src.data.quality_gate import get_quality_gate
from src.alpha.prediction_registry import get_prediction_registry, PredictionRecord
from data.nse_market_calendar import get_nse_calendar
from src.risk.sebi_algo_compliance import SEBIAlgoCompliance, Order as SEBIOrder

@dataclass
class Position:
    symbol: str
    sector: str
    quantity: float
    entry_price: float
    current_price: float
    side: str

logger = logging.getLogger("quant_os")

class QuantResearchOS:
    def __init__(self, config_path: str = "config_v3.py") -> None:
        # CRITICAL FIX: Add input validation
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("config_path must be a non-empty string")
        
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        data_config = self.config.get("data", {})
        
        # Validate data_dir
        data_dir = data_config.get("data_dir", "/data/nse_bars")
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string")
        
        self.data_loader = NSEDataLoader(data_dir=data_dir)
        self.feature_pipeline = FeaturePipeline(FeatureConfig())
        self.alpha_manager = AlphaManager(self.config)
        self.regime_manager = RobustHMMRegime()
        self.portfolio_allocator = PortfolioAllocator(total_capital=250_000_000)
        self.risk_engine = InstitutionalRiskEngine(capital=250_000_000)
        self.data_quality_gate = get_quality_gate()
        self.prediction_registry = get_prediction_registry()
        self.nse_calendar = get_nse_calendar()
        self.market_data = {}
        self.symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "INFY", "HDFCBANK"]
        self.current_time = datetime.now()
        self.daily_pnl = 0.0

        # SEBI compliance — pre-trade regulatory checks
        self.sebi_compliance = SEBIAlgoCompliance(broker_id="QUANT_OS")
        self.sebi_compliance.register_client("SYSTEM", {
            "max_position_value": 50_000_000,
            "max_daily_orders": 10_000,
            "max_exposure": 250_000_000,
        })

        # Startup environment validation
        self._validate_environment()

    def _validate_environment(self):
        """Validate environment and warn about missing configurations."""
        if not os.getenv("NSE_WEBSOCKET_URL") and not os.getenv("KITE_WEBSOCKET_URL"):
            logger.warning("ENV CHECK: No WebSocket URL set — will run in SIMULATION mode (historical replay)")
        if not os.getenv("KITE_API_KEY"):
            logger.warning("ENV CHECK: No Kite API key — broker adapter will NOT place real orders")

    def _load_config(self, path: Path) -> dict[str, Any]:
        if not path.exists() and path.as_posix() == "config/config.yaml":
            path = Path("core/config/config.yaml")
        if not path.exists():
            logger.warning("Config file %s not found; using runtime defaults", path)
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    async def initialize(self) -> None:
        logger.info("Initializing Quant Research OS")
        # Load historical bars
        for sym in self.symbols:
            try:
                end = date.today()
                # CRITICAL FIX: Use 5 years of historical data for meaningful backtesting
                # Previously only 370 days (~1 year), now 1825 days (5 years)
                start = end - timedelta(days=1825)
                bars = self.data_loader.get_historical_bars(sym, start.isoformat(), end.isoformat())
                if not bars.empty:
                    clean_bars, result = self.data_quality_gate.validate(sym, bars)
                    if result.passed:
                        self.market_data[sym] = clean_bars
                        logger.info(f"Loaded and validated {len(clean_bars)} bars for {sym}")
                    else:
                        logger.error(f"Data for {sym} rejected by quality gate: {result}")
                        self.market_data[sym] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                else:
                    self.market_data[sym] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            except Exception as e:
                logger.error(f"Failed to load data for {sym}: {e}")
                self.market_data[sym] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Fit regime model
        # Using a mock df for fitting to prevent crashes if no data
        try:
            self.regime_manager.fit(self.market_data['NIFTY'])
            logger.info("Regime model fitted successfully")
        except Exception as e:
            logger.warning(f"Could not fit regime: {e}")
            logger.warning("Using default regime: sideways")

    def _has_required_backtest_data(self) -> bool:
        nifty = self.market_data.get("NIFTY")
        if nifty is None or nifty.empty:
            logger.warning("NIFTY data is None or empty")
            return False
        # Check for required columns (handle multi-index columns from Yahoo Finance)
        required = {"open", "high", "low", "close", "volume"}
        available_columns = set()
        for col in nifty.columns:
            if isinstance(col, tuple):
                # MultiIndex column - take the first element
                available_columns.add(str(col[0]).lower())
            else:
                available_columns.add(str(col).lower())
        if not required.issubset(available_columns):
            logger.warning(f"Missing required columns. Available: {available_columns}")
            return False
        # Find the close column (handle multi-index)
        close_col = None
        for col in nifty.columns:
            col_str = str(col[0]).lower() if isinstance(col, tuple) else str(col).lower()
            if col_str == "close":
                close_col = col
                break
        if close_col is None:
            logger.warning("Close column not found")
            return False
        non_null_count = len(nifty.dropna(subset=[close_col]))
        logger.info(f"NIFTY data check: {len(nifty)} rows, {non_null_count} non-null close prices")
        return non_null_count >= 100

    def on_bar(self, bar):
        """Callback for incoming WebSocket bars."""
        sym = bar['symbol']
        if sym not in self.market_data:
            self.market_data[sym] = pd.DataFrame()
            
        bar_time = bar.get('timestamp', datetime.now())
        if isinstance(bar_time, str):
            bar_time = pd.to_datetime(bar_time).to_pydatetime()
        self.current_time = bar_time

        # Append bar
        new_row = pd.DataFrame([bar], index=[bar_time])
        combined = pd.concat([self.market_data[sym], new_row])
        
        # Validate through quality gate
        clean_df, result = self.data_quality_gate.validate(sym, combined)
        if result.passed:
            self.market_data[sym] = clean_df
            self._evaluate_state()
        else:
            logger.warning(f"Live bar for {sym} rejected by quality gate: {result}")

    def _evaluate_state(self):
        """Re-evaluates regime, risk, and allocation."""
        try:
            # 0. Market calendar gate — skip non-trading days
            if hasattr(self.current_time, 'date'):
                if not self.nse_calendar.is_trading_day(self.current_time.date()):
                    logger.info(f"Not a trading day ({self.current_time.date()}). Skipping.")
                    return {
                        "nav": self.risk_engine.capital, "daily_pnl": 0.0,
                        "regime": "holiday", "regime_confidence": 0.0,
                        "risk": {"var": 0.0, "cvar": 0.0, "tail_risk": 0.0},
                        "signals": [], "allocations": [],
                        "updated_at": datetime.now().isoformat(),
                    }

            # 1. Regime
            try:
                regime = self.regime_manager.predict_regime(self.market_data['NIFTY']).iloc[-1]
                conf = self.regime_manager.confidence(self.market_data['NIFTY'])
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}, using default")
                regime = 'sideways'
                conf = 0.5
                
            # 2. Risk (VaR on returns)
            try:
                nifty = self.market_data.get('NIFTY', pd.DataFrame())
                if 'close' in nifty and not nifty['close'].empty:
                    close_data = nifty['close'].values
                    if len(close_data.shape) == 2:
                        close_data = close_data.flatten()
                    returns = self.risk_engine.compute_returns(close_data)
                else:
                    returns = pd.Series(dtype=float)
                var = self.risk_engine.compute_weibull_var(self.risk_engine.capital, returns)
                cvar = self.risk_engine.compute_cvar(self.risk_engine.capital, returns)
                tail = self.risk_engine.tail_risk(self.risk_engine.capital, returns)
            except Exception as e:
                logger.error(f"Risk calculation failed: {e}")
                var, cvar, tail = 0.0, 0.0, 0.0
            
            # 3. Alpha ORB scanning
            try:
                # Resolve expired predictions first
                current_prices = {}
                for s in self.symbols:
                    if s in self.market_data and not self.market_data[s].empty:
                        last_close = self.market_data[s].iloc[-1]["close"]
                        if isinstance(last_close, pd.Series):
                            current_prices[s] = float(last_close.iloc[-1])
                        else:
                            current_prices[s] = float(last_close)
                self.prediction_registry.resolve_expired(current_prices)
                
                # Check strategy demotions
                demoted_strategies = self.prediction_registry.check_demotions()
                
                # Scan symbols for new signals
                raw_signals = scan_symbols(self.market_data, self.current_time)
                
                # Filter out demoted strategies
                signals = []
                for sig in raw_signals:
                    strat = sig.get("strategy", "orb")
                    if strat in demoted_strategies:
                        logger.warning(f"Discarding signal from demoted strategy '{strat}' due to low IC.")
                        continue
                    signals.append(sig)

                # ── Regime-conditional signal weighting ──────────────────
                regime_multiplier = 1.0
                regime_str = str(regime).lower()
                if regime_str in ('high_vol', '2'):
                    regime_multiplier = 0.3
                elif regime_str in ('sideways', '1'):
                    regime_multiplier = 0.5
                # else trend_up / 0 → 1.0

                for sig in signals:
                    sig['confidence'] = sig.get('confidence', 0.5) * regime_multiplier
                    sig['regime'] = regime_str
                
                # Log new signals as predictions
                for sig in signals:
                    direction = sig.get("direction", 0)
                    if direction != 0:
                        symbol = sig["symbol"]
                        if symbol in self.market_data and not self.market_data[symbol].empty:
                            close_price = self.market_data[symbol].iloc[-1]["close"]
                            if isinstance(close_price, pd.Series):
                                close_price = float(close_price.iloc[-1])
                            else:
                                close_price = float(close_price)
                            
                            pred_record = PredictionRecord(
                                symbol=symbol,
                                strategy=sig.get("strategy", "orb"),
                                direction="long" if direction > 0 else "short",
                                predicted_return=float(sig.get("expected_return", 0.01)),
                                confidence=float(sig.get("confidence", 0.5)),
                                entry_price=close_price,
                                timestamp=self.current_time,
                                horizon_minutes=390
                            )
                            self.prediction_registry.record_prediction(pred_record)
            except Exception as e:
                logger.error(f"Alpha scanning/prediction registry handling failed: {e}")
                signals = []
            
            # 4. Allocations
            try:
                allocs = self.portfolio_allocator.allocate(signals)
            except Exception as e:
                logger.error(f"Portfolio allocation failed: {e}")
                allocs = []

            # 5. SEBI compliance pre-trade checks
            compliant_allocs = []
            for alloc in allocs:
                try:
                    sym = getattr(alloc, 'symbol', 'UNKNOWN')
                    weight = getattr(alloc, 'weight', 0.0)
                    price = current_prices.get(sym, 0)
                    qty = int(abs(weight * self.portfolio_allocator.get_current_capital() / max(price, 1)))
                    sebi_order = SEBIOrder(
                        order_id=f"ORD_{datetime.now().timestamp():.0f}",
                        symbol=sym,
                        side="BUY" if weight > 0 else "SELL",
                        quantity=max(qty, 1),
                        price=price,
                        order_type="LIMIT",
                        client_id="SYSTEM",
                        strategy_id=getattr(alloc, 'strategy', 'orb'),
                        timestamp=self.current_time,
                    )
                    is_compliant, checks = self.sebi_compliance.pre_trade_check(sebi_order)
                    if is_compliant:
                        compliant_allocs.append(alloc)
                    else:
                        failed = [c.message for c in checks if c.status.value != 'COMPLIANT']
                        logger.warning(f"SEBI rejected {sym}: {failed}")
                except Exception as e:
                    logger.error(f"SEBI check failed for allocation: {e}")
                    compliant_allocs.append(alloc)  # fail-open for now
            allocs = compliant_allocs
            
            # 6. Compute REALIZED PnL from actual price changes
            realized_pnl = 0.0
            for alloc in allocs:
                sym = getattr(alloc, 'symbol', None)
                weight = getattr(alloc, 'weight', 0.0)
                if sym and sym in self.market_data and len(self.market_data[sym]) >= 2:
                    try:
                        prev_close = float(self.market_data[sym]['close'].iloc[-2])
                        curr_close = float(self.market_data[sym]['close'].iloc[-1])
                        if prev_close > 0:
                            position_return = (curr_close - prev_close) / prev_close
                            realized_pnl += weight * self.portfolio_allocator.get_current_capital() * position_return
                    except Exception:
                        pass

            self.daily_pnl = realized_pnl
            new_nav = self.portfolio_allocator.get_current_capital() + self.daily_pnl
            self.portfolio_allocator.update_nav(new_nav)
            self.risk_engine.capital = new_nav
            
            # 7. Circuit breaker on REALIZED PnL
            self.risk_engine.update_daily_pnl(self.daily_pnl)
            cb_triggered, cb_reason = self.risk_engine.check_circuit_breaker(self.daily_pnl)
            if cb_triggered:
                logger.critical(f"Circuit breaker active: {cb_reason} - trading halted")
                allocs = []
            
            state = {
                "nav": new_nav,
                "daily_pnl": self.daily_pnl,
                "regime": regime,
                "regime_confidence": conf,
                "risk": {"var": var, "cvar": cvar, "tail_risk": tail},
                "signals": signals,
                "allocations": allocs,
                "updated_at": datetime.now().isoformat(),
            }

            logger.info(f"State Update: Regime {regime} (Conf {conf:.2f}), VaR ₹{var:,.2f}, NAV ₹{new_nav:,.2f}, Realized PnL ₹{self.daily_pnl:,.2f}, Allocations {len(allocs)}")
            return state
            
        except Exception as e:
            logger.error(f"Critical error in _evaluate_state: {e}")
            return {
                "nav": self.risk_engine.capital,
                "daily_pnl": 0.0,
                "regime": "sideways",
                "regime_confidence": 0.5,
                "risk": {"var": 0.0, "cvar": 0.0, "tail_risk": 0.0},
                "signals": [],
                "allocations": [],
                "updated_at": datetime.now().isoformat(),
            }

    async def run_live(self) -> None:
        await self.initialize()
        logger.info("Entering live loop. Press Ctrl+C to exit.")
        
        stream = NSEWebSocketStream(self.symbols, self.on_bar)
        uri = os.getenv("NSE_WEBSOCKET_URL") or os.getenv("KITE_WEBSOCKET_URL")
        
        if uri:
            logger.info("Connecting to real market WebSocket stream with automatic reconnection...")
            while True:
                try:
                    await stream.connect()
                except Exception as e:
                    logger.error(f"WebSocket stream disconnected or failed: {e}")
                logger.info("Attempting reconnection in 5 seconds...")
                await asyncio.sleep(5)
        else:
            logger.warning("No WebSocket URL set. Running in SIMULATION mode (historical replay).")
            # Historical replay: walk through loaded data bar-by-bar
            # instead of generating random prices
            max_bars = max(
                (len(self.market_data.get(s, pd.DataFrame())) for s in self.symbols),
                default=0,
            )
            if max_bars == 0:
                logger.error("No historical data loaded for replay. Exiting.")
                return

            logger.info(f"Replaying {max_bars} historical bars...")
            for bar_idx in range(max_bars):
                await asyncio.sleep(0.05)  # Throttle replay
                for sym in self.symbols:
                    df = self.market_data.get(sym, pd.DataFrame())
                    if df.empty or bar_idx >= len(df):
                        continue
                    row = df.iloc[bar_idx]
                    sim_bar = {
                        "symbol": sym,
                        "timestamp": row.name if isinstance(row.name, datetime) else datetime.now(),
                        "open": float(row.get('open', 0)),
                        "high": float(row.get('high', 0)),
                        "low": float(row.get('low', 0)),
                        "close": float(row.get('close', 0)),
                        "volume": float(row.get('volume', 0)),
                    }
                    self.on_bar(sim_bar)
            logger.info("Historical replay complete.")

    async def run_backtest(self) -> dict[str, Any]:
        """Run walk-forward backtest over historical data."""
        await self.initialize()
        if not self._has_required_backtest_data():
            missing = [
                s for s in self.symbols
                if s not in self.market_data or self.market_data[s].empty
            ]
            return {
                "status": "failed", "mode": "backtest",
                "reason": "data_unavailable",
                "message": "Backtest aborted — required OHLCV data missing.",
                "missing_symbols": missing,
            }

        nifty = self.market_data.get("NIFTY", pd.DataFrame())
        if nifty.empty:
            return {"status": "failed", "mode": "backtest", "reason": "no_nifty_data"}

        # Snapshot original data for point-in-time slicing
        full_data = {sym: df.copy() for sym, df in self.market_data.items()}
        nav_history = [self.portfolio_allocator.get_current_capital()]
        results = []
        warmup = min(100, len(nifty) // 2)

        for i in range(warmup, len(nifty)):
            bar_date = nifty.index[i]
            if hasattr(bar_date, 'date') and not self.nse_calendar.is_trading_day(bar_date.date()):
                continue

            # Point-in-time slice: only data up to current bar
            for sym in self.symbols:
                if sym in full_data and not full_data[sym].empty:
                    self.market_data[sym] = full_data[sym].iloc[:i + 1]

            self.current_time = bar_date.to_pydatetime() if hasattr(bar_date, 'to_pydatetime') else datetime.now()
            state = self._evaluate_state()
            if state:
                results.append(state)
                nav_history.append(state.get("nav", nav_history[-1]))

        # Restore original data
        self.market_data = full_data

        # Compute honest backtest metrics
        nav_series = pd.Series(nav_history)
        bt_returns = nav_series.pct_change().dropna()
        sharpe = float(bt_returns.mean() / bt_returns.std() * np.sqrt(252)) if bt_returns.std() > 0 else 0.0
        max_dd = float(((nav_series / nav_series.cummax()) - 1).min())

        return {
            "status": "success", "mode": "backtest",
            "total_bars": len(results),
            "final_nav": float(nav_history[-1]),
            "total_return": float(nav_history[-1] / nav_history[0] - 1),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

def _convert_to_native(obj: Any) -> Any:
    """Recursively convert numpy types and dataclasses to Python native types for YAML/JSON serialization."""
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        obj = obj.to_dict()
    elif hasattr(obj, "__dict__"):
        obj = obj.__dict__

    if isinstance(obj, dict):
        return {_convert_to_native(k): _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_convert_to_native(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj

def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Research OS")
    parser.add_argument("--config", default="core/config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["backtest", "live"], default="live", help="Override system mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    system = QuantResearchOS(args.config)
    mode = args.mode or system.config.get("system", {}).get("mode", "live")

    if mode == "live":
        asyncio.run(system.run_live())
    else:
        summary = asyncio.run(system.run_backtest())
        native_summary = _convert_to_native(summary)
        print(yaml.safe_dump(native_summary, sort_keys=False))

if __name__ == "__main__":
    main()
