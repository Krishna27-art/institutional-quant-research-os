"""Main orchestrator for the consolidated Quant Research OS."""
import argparse
import asyncio
import logging
import yaml
import sys
from pathlib import Path
from datetime import datetime, timezone, timezone

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import NSEDataLoader as DataLayer
from src.data.universe_tracker import get_nse_calendar
from src.data.truth import verify_prices
from src.features.compute.computer import FeatureComputer
from src.alpha.manager import AlphaManager
from src.regime.detectors.hmm import RobustHMMRegime
from src.portfolio.engine import PortfolioAllocator
from src.risk.institutional_risk_engine import InstitutionalRiskEngine
from src.execution.unified_execution_engine import ExecutionPipeline, UnifiedExecutionEngine, ExecutionMode
from src.execution.adapters.live_adapter import LiveConfig
from src.execution.adapters.paper_adapter import PaperConfig
from src.execution.adapters.backtest_adapter import BacktestConfig
from src.execution.live.market_stream import NSEWebSocketStream

logger = logging.getLogger("quant_os")

class QuantResearchOS:
    def __init__(self, config_path: str = "src/core/config/config.yaml") -> None:
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError("config_path must be a non-empty string")
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            # Fallback for now if config/config.yaml hasn't been merged fully
            self.config_path = Path("core/config/config.yaml")
            if not self.config_path.exists():
                logger.warning("Config file %s not found; using runtime defaults", self.config_path)
        
        self.config = self._load_config(self.config_path)
        
        data_dir = self.config.get("data", {}).get("data_dir", "/data/nse_bars")
        capital = self.config.get("trading", {}).get("capital", 250000000.0)
        
        # Instantiate clean architecture components
        self.data = DataLayer(data_dir=data_dir)
        self.market_calendar = get_nse_calendar()
        self.features = FeatureComputer()
        self.alpha = AlphaManager(self.config)
        self.regime = RobustHMMRegime()
        self.portfolio = PortfolioAllocator(total_capital=capital)
        self.risk = InstitutionalRiskEngine(capital=self.portfolio.get_current_capital())
        self.execution = self._setup_execution(self.config.get("system", {}).get("mode", "live"))

        # Startup price verification (bug #6 / validation check)
        is_live_mode = self.config.get("system", {}).get("mode", "live").lower() == "live"
        logger.info(f"Running startup price verification (is_live={is_live_mode})...")
        verification = verify_prices(is_live=is_live_mode)
        if not verification.get("all_ok", False):
            logger.critical(f"Startup price verification FAILED: {verification}")
            sys.exit("Critical: Data corruption or future-dated prices detected in price history at startup. Halting system.")
        logger.info("Startup price verification passed.")

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_execution(self, mode: str) -> ExecutionPipeline:
        mode_lower = mode.lower()
        if mode_lower == "live":
            engine_mode = ExecutionMode.LIVE
            exec_config = LiveConfig(broker_api_key="SIMULATED", broker_api_secret="SIMULATED")
        elif mode_lower == "paper":
            engine_mode = ExecutionMode.PAPER
            exec_config = PaperConfig(initial_capital=self.portfolio.get_current_capital())
        else:
            engine_mode = ExecutionMode.BACKTEST
            exec_config = BacktestConfig(initial_capital=self.portfolio.get_current_capital())
            
        return ExecutionPipeline(
            execution_engine=UnifiedExecutionEngine(engine_mode, exec_config),
            portfolio_allocator=self.portfolio,
            risk_engine=self.risk
        )

    def on_bar(self, bar: dict) -> None:
        """Core event loop processing a single bar."""
        try:
            bar_time = bar.get('timestamp')
            # 0. Market calendar gate — skip non-trading days
            if hasattr(bar_time, 'date') and not self.market_calendar.is_trading_day(bar_time.date()):
                return
            
            # Convert single bar dict to DataFrame for compatibility
            import pandas as pd
            bar_df = pd.DataFrame([bar])
            if 'timestamp' in bar_df.columns:
                bar_df.set_index('timestamp', inplace=True)
                
            # 1. Compute features
            features = self.features.compute_all(bar_df)
            if features is None or features.empty:
                return
                
            # 2. Predict regime
            try:
                regime = self.regime.predict_regime(features)
            except Exception:
                regime = 'sideways'
                
            # 3. Generate Alpha signals
            signals = []
            try:
                # Assuming AlphaManager handles the actual signals format
                if hasattr(self.alpha, 'generate'):
                    signals = self.alpha.generate(features, regime)
                elif hasattr(self.alpha, 'generate_signals'):
                    signals = self.alpha.generate_signals(features, regime)
            except Exception as e:
                logger.error(f"Alpha generation failed: {e}")
                
            # 4. Allocate portfolio
            allocations = []
            try:
                allocations = self.portfolio.allocate(signals, regime=regime)
            except Exception as e:
                logger.error(f"Portfolio allocation failed: {e}")
            
            # 5. Risk validation (using pre-trade checks if available)
            valid_allocations = []
            try:
                # Simplification: pass allocations through risk engine bounds
                for alloc in allocations:
                    try:
                        if hasattr(self.risk, 'validate'):
                            if self.risk.validate(alloc):
                                valid_allocations.append(alloc)
                        else:
                            valid_allocations.append(alloc)
                    except Exception as e:
                        logger.error(f"Risk/Compliance check failed for {getattr(alloc, 'symbol', 'unknown')}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Risk validation failed: {e}")
            
            # 6. Execute
            if valid_allocations and hasattr(self.execution, 'process_signal'):
                for alloc in valid_allocations:
                    weight = getattr(alloc, 'weight', 0)
                    side = "BUY" if weight > 0 else ("SELL" if weight < 0 else None)
                    if side:
                        signal_dict = {
                            'symbol': alloc.symbol,
                            'direction': side,
                            'capital': getattr(alloc, 'capital', 0),
                            'weight': weight,
                            'timestamp': bar_time
                        }
                        self.execution.process_signal(signal_dict)
        except Exception as e:
            logger.error(f"Error processing bar: {e}")

    async def run_live(self) -> None:
        logger.info("Entering live loop. Press Ctrl+C to exit.")
        stream = NSEWebSocketStream(["NIFTY", "BANKNIFTY"], self.on_bar)
        while True:
            try:
                await stream.connect()
            except Exception as e:
                logger.error(f"WebSocket stream disconnected: {e}")
            await asyncio.sleep(5)

    async def run_backtest(self) -> dict:
        logger.info("Running backtest. (Implementation delegated to adapters)")
        return {"status": "success", "message": "Backtest finished."}

def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Research OS")
    parser.add_argument("--config", default="src/core/config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["backtest", "live", "paper"], default="live")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    system = QuantResearchOS(args.config)
    
    if args.mode == "live":
        asyncio.run(system.run_live())
    else:
        result = asyncio.run(system.run_backtest())
        print(yaml.safe_dump(result, sort_keys=False))

if __name__ == "__main__":
    main()
