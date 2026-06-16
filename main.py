"""
Main orchestrator for the Institutional Quant Research OS.
Replaces the old predictive architecture with an event-driven, 
inventory-aware decision factory.
"""
import argparse
import asyncio
import logging
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "production"))

from src.data.universe_tracker import get_nse_calendar
from src.features.compute.computer import FeatureComputer
from src.regime.detectors.hmm import RobustHMMRegime

# Institutional Architecture Imports
from src.research.knowledge_graph import KnowledgeGraph, GraphNode, NodeType
from src.data.event_store import EventStore
from src.alpha.marketplace.registry import AlphaMarketplace
from src.portfolio.institutional_allocator import CapitalAllocationEngine
from src.alpha.marketplace.assets.orb_alpha import ORBAlphaAsset

# Execution layer
from src.execution.unified_execution_engine import ExecutionPipeline, UnifiedExecutionEngine, ExecutionMode
from src.execution.adapters.live_adapter import LiveConfig
from src.execution.adapters.paper_adapter import PaperConfig
from src.execution.adapters.backtest_adapter import BacktestConfig
from src.execution.live.market_stream import NSEWebSocketStream

logger = logging.getLogger("quant_os")

class InstitutionalQuantOS:
    def __init__(self, config_path: str = "src/core/config/config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        
        capital = self.config.get("trading", {}).get("capital", 250000000.0)
        
        # 1. Base State
        self.market_calendar = get_nse_calendar()
        self.features = FeatureComputer()
        self.regime = RobustHMMRegime()
        self.execution = self._setup_execution(self.config.get("system", {}).get("mode", "live"), capital)
        
        # 2. Institutional Systems
        self.knowledge_graph = KnowledgeGraph()
        self.event_store = EventStore()
        self.alpha_marketplace = AlphaMarketplace()
        self.capital_allocator = CapitalAllocationEngine(total_capital=capital)
        
        # Current Inventory tracker
        self.inventory: dict[str, float] = {}

        self._initialize_institutional_systems()

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
        
    def _initialize_institutional_systems(self):
        """Register assets and knowledge."""
        # Setup Knowledge Graph Node for ORB
        orb_node = GraphNode(
            id="orb_zarattini",
            type=NodeType.ALPHA,
            hypothesis=ORBAlphaAsset.get_causal_hypothesis()
        )
        self.knowledge_graph.add_node(orb_node)
        self.knowledge_graph.validate_graph()
        
        # Register Alpha in Marketplace
        self.alpha_marketplace.register_alpha(
            alpha_id="orb_zarattini",
            metadata=ORBAlphaAsset.get_metadata()
        )
        
        # Instantiate the asset instances
        self.active_alpha_assets = {
            "orb_zarattini": ORBAlphaAsset(self.config)
        }
        logger.info("Institutional systems initialized.")

    def _setup_execution(self, mode: str, capital: float) -> ExecutionPipeline:
        mode_lower = mode.lower()
        if mode_lower == "live":
            engine_mode = ExecutionMode.LIVE
            exec_config = LiveConfig(broker_api_key="SIMULATED", broker_api_secret="SIMULATED")
        elif mode_lower == "paper":
            engine_mode = ExecutionMode.PAPER
            exec_config = PaperConfig(initial_capital=capital)
        else:
            engine_mode = ExecutionMode.BACKTEST
            exec_config = BacktestConfig(initial_capital=capital)
            
        # Mocking portfolio/risk interfaces for the pipeline initialization if needed
        class DummyPortRisk:
            def get_current_capital(self): return capital
        
        return ExecutionPipeline(
            execution_engine=UnifiedExecutionEngine(engine_mode, exec_config),
            portfolio_allocator=DummyPortRisk(),
            risk_engine=DummyPortRisk()
        )

    def on_bar(self, bar: dict) -> None:
        """Core decision factory loop."""
        try:
            bar_time = bar.get('timestamp')
            if hasattr(bar_time, 'date') and not self.market_calendar.is_trading_day(bar_time.date()):
                return
            
            import pandas as pd
            bar_df = pd.DataFrame([bar])
            if 'timestamp' in bar_df.columns:
                bar_df.set_index('timestamp', inplace=True)
                
            features = self.features.compute_all(bar_df)
            if features is None or features.empty:
                return
                
            try:
                regime = self.regime.predict_regime(features)
            except Exception:
                regime = 'sideways'
                
            # Marketplace Alpha Evaluation
            all_signals = []
            for alpha_id in self.alpha_marketplace.get_available_alphas():
                if alpha_id in self.active_alpha_assets:
                    asset = self.active_alpha_assets[alpha_id]
                    signals = asset.generate_signals(features, regime)
                    all_signals.extend(signals)
            
            # Continuous Self Evaluation
            self.alpha_marketplace.evaluate_alphas()
            
            # Institutional Capital Allocation (cvxpy)
            allocations = self.capital_allocator.allocate(
                marketplace=self.alpha_marketplace,
                current_inventory=self.inventory
            )
            
            # Execute allocations
            if hasattr(self.execution, 'process_signal') and allocations:
                for alloc in allocations:
                    # Look up the signal direction
                    matching_signal = next((s for s in all_signals if s['alpha_id'] == alloc.alpha_id), None)
                    if not matching_signal:
                        continue
                        
                    direction = "BUY" if matching_signal.get('signal_strength', 0) > 0 else "SELL"
                    
                    signal_dict = {
                        'symbol': matching_signal['symbol'],
                        'direction': direction,
                        'capital': alloc.target_capital,
                        'weight': alloc.target_weight,
                        'timestamp': bar_time
                    }
                    self.execution.process_signal(signal_dict)
                    
        except Exception as e:
            logger.error(f"Error processing bar in OS: {e}")

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
        logger.info("Running backtest via Cluster Engine.")
        from src.backtest.distributed.cluster_engine import ClusterEngine, ExperimentTask
        engine = ClusterEngine()
        tasks = [
            ExperimentTask(
                experiment_id="test_run_1",
                alpha_class=None,
                parameters={},
                start_date="2022-01-01",
                end_date="2022-12-31"
            )
        ]
        results = engine.execute_batch(tasks)
        engine.shutdown()
        return {"status": "success", "processed_tasks": len(results)}

def main() -> None:
    parser = argparse.ArgumentParser(description="Institutional Quant OS")
    parser.add_argument("--config", default="src/core/config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["backtest", "live", "paper"], default="live")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    system = InstitutionalQuantOS(args.config)
    
    if args.mode == "live":
        asyncio.run(system.run_live())
    else:
        result = asyncio.run(system.run_backtest())
        print(yaml.safe_dump(result, sort_keys=False))

if __name__ == "__main__":
    main()
