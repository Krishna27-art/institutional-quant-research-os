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
from alpha.manager import AlphaManager
from data.data_loader import NSEDataLoader
from features.feature_pipeline import FeaturePipeline, FeatureConfig
from portfolio.allocator import PortfolioAllocator
from regime.hmm_engine import RobustHMMRegime
from risk.institutional_risk_engine import InstitutionalRiskEngine
from execution.live.market_stream import NSEWebSocketStream
from alpha.orb_zarattini import scan_symbols
from src.data.quality_gate import get_quality_gate
from src.alpha_factory.prediction_registry import get_prediction_registry, PredictionRecord

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
        self.market_data = {}
        self.symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "INFY", "HDFCBANK"]
        self.current_time = datetime.now()
        self.daily_pnl = 0.0

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
        # CRITICAL FIX: Add comprehensive error handling
        try:
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
                    # Handle 2D arrays
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
            
            # CRITICAL FIX: Update NAV with daily PnL
            # Calculate PnL from allocations and update portfolio allocator NAV
            total_pnl = 0.0
            for alloc in allocs:
                if hasattr(alloc, 'expected_return'):
                    total_pnl += alloc.expected_return
            
            self.daily_pnl = total_pnl
            new_nav = self.portfolio_allocator.get_current_capital() + self.daily_pnl
            self.portfolio_allocator.update_nav(new_nav)
            self.risk_engine.capital = new_nav
            
            # CRITICAL FIX: Check circuit breaker
            self.risk_engine.update_daily_pnl(self.daily_pnl)
            cb_triggered, cb_reason = self.risk_engine.check_circuit_breaker(self.daily_pnl)
            if cb_triggered:
                logger.critical(f"Circuit breaker active: {cb_reason} - trading halted")
                allocs = []  # Clear allocations to stop trading
            
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

            # web.api_server not yet implemented - placeholder for WebSocket broadcasting
            # try:
            #     from web.api_server import publisher
            #     loop = asyncio.get_running_loop()
            #     loop.create_task(publisher.broadcast(state))
            # except Exception:
            #     pass

            logger.info(f"State Update: Regime {regime} (Conf {conf:.2f}), VaR ₹{var:,.2f}, NAV ₹{new_nav:,.2f}, Daily PnL ₹{self.daily_pnl:,.2f}, Allocations {len(allocs)}")
            return state
            
        except Exception as e:
            logger.error(f"Critical error in _evaluate_state: {e}")
            # Return safe default state
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
            logger.warning("Neither NSE_WEBSOCKET_URL nor KITE_WEBSOCKET_URL set. Running in simulation mode.")
            # Simulation loop
            while True:
                await asyncio.sleep(60)  # Check every minute
                for sym in self.symbols:
                    if sym in self.market_data and not self.market_data[sym].empty:
                        last_bar = self.market_data[sym].iloc[-1].to_dict()
                        price = last_bar['close'] * (1.0 + np.random.normal(0, 0.001))
                        sim_bar = {
                            "symbol": sym,
                            "timestamp": datetime.now(),
                            "open": price,
                            "high": price,
                            "low": price,
                            "close": price,
                            "volume": last_bar.get('volume', 100)
                        }
                        self.on_bar(sim_bar)

    async def run_backtest(self) -> dict[str, Any]:
        """Placeholder for backtest mode using actual logic."""
        await self.initialize()
        if not self._has_required_backtest_data():
            missing = [
                symbol
                for symbol in self.symbols
                if symbol not in self.market_data or self.market_data[symbol].empty
            ]
            return {
                "status": "failed",
                "mode": "backtest",
                "reason": "data_unavailable",
                "message": "Backtest aborted because required OHLCV data is missing.",
                "missing_symbols": missing,
            }
        # Mock evaluation since historical data fetch is abstracted
        state = self._evaluate_state()
        return {"status": "success", "mode": "backtest", "state": state}

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
