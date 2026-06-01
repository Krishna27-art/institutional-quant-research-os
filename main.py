"""
NIFTY Quant Trading System - Main Orchestrator
Coordinates data, regime detection, alpha generation, risk, and execution.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Core
from core.data_layer import DataManager
from core.feature_layer import FeatureEngine
from core.regime_engine import HybridRegimeEngine, StrategyType

# Alpha
from alpha.orb_strategy import ORBStrategy, ORBSignal
from alpha.vwap_strategy import VWAPStrategy, VWAPSignal
from alpha.chaotic_gcn import ChaoticGCNAlpha
from alpha.game_theoretic import GameTheoreticAlpha

# Risk
from risk.risk_engine import RiskEngine, RiskAction

# Execution (try C++ first, fallback to Python)
try:
    from niftyquant_cpp import ExecutionEngine as CppExecutionEngine, Side, VWAPParams
    USE_CPP_ENGINE = True
    logging.info("C++ Execution Engine loaded successfully")
except ImportError:
    USE_CPP_ENGINE = False
    logging.warning("C++ Engine not found, using Python fallback")

# Backtest
from backtest.backtester import VectorizedBacktester, BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("niftyquant.log")
    ]
)
logger = logging.getLogger("NiftyQuant")


class NiftyQuantSystem:
    """
    Production-ready quantitative trading system for Indian markets.
    """
    
    # NIFTY 50 universe (subset for illustration)
    TRADING_UNIVERSE = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "LT",
        "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "SUNPHARMA",
        "TITAN", "WIPRO", "ULTRACEMCO", "ADANIENT", "NTPC",
    ]
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize components
        self.data_manager = DataManager(self.config)
        self.feature_engine = FeatureEngine()
        self.regime_engine = HybridRegimeEngine(self.config)
        self.risk_engine = RiskEngine(self.config)
        
        # Alpha strategies
        self.orb_strategy = ORBStrategy(self.config)
        self.vwap_strategy = VWAPStrategy(self.config)
        self.chaotic_gcn = ChaoticGCNAlpha(self.config)
        self.game_alpha = GameTheoreticAlpha(self.config)
        
        # Execution
        self.execution_engine = None
        self._init_execution_engine()
        
        # State
        self._intraday_data = {}
        self._daily_data = {}
        self._active_positions = {}
        self._regime_state = None
        
        logger.info(f"NiftyQuant System initialized | Mode: {self.config['system']['mode']}")
    
    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    
    def _init_execution_engine(self):
        if USE_CPP_ENGINE:
            self.execution_engine = CppExecutionEngine(num_threads=2)
            self.execution_engine.start()
            
            # Set fill callback
            def on_fill(fill):
                logger.info(
                    f"FILL: {fill.symbol} {fill.side} "
                    f"{fill.fill_quantity}@{fill.fill_price/100:.2f} "
                    f"Slip: {fill.slippage_bps:.1f}bps"
                )
            self.execution_engine.set_fill_callback(on_fill)
        else:
            self.execution_engine = None
    
    async def initialize(self):
        """Initialize data feeds and pre-load historical data."""
        logger.info("Initializing system...")
        
        await self.data_manager.initialize_feeds()
        
        # Load historical daily data for regime detection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)  # ~252 trading days
        
        for symbol in ["NIFTY"] + self.TRADING_UNIVERSE[:5]:
            df = await self.data_manager.get_data(
                symbol, start_date, end_date, interval="day"
            )
            if not df.empty:
                self._daily_data[symbol] = df
                logger.info(f"Loaded {len(df)} daily bars for {symbol}")
        
        # Fit regime engine
        if "NIFTY" in self._daily_data:
            self.regime_engine.fit(self._daily_data["NIFTY"])
        
        # Train Chaotic GCN (if enough data)
        if len(self._daily_data) > 5:
            try:
                labels = {}
                for sym, df in self._daily_data.items():
                    if len(df) > 5:
                        ret = df["Close"].pct_change(5).iloc[-1]
                        labels[sym] = 2 if ret > 0.01 else (0 if ret < -0.01 else 1)
                
                self.chaotic_gcn.train_model(
                    self._daily_data,
                    self.data_manager.SECTOR_MAP,
                    labels,
                    epochs=50
                )
            except Exception as e:
                logger.warning(f"GCN training skipped: {e}")
        
        logger.info("System initialization complete")
    
    async def run_backtest(self):
        """Run historical backtest on all strategies."""
        logger.info("=" * 60)
        logger.info("RUNNING BACKTEST MODE")
        logger.info("=" * 60)
        
        backtester = VectorizedBacktester(BacktestConfig())
        
        # Backtest ORB
        if self.config["alpha"]["orb"]["enabled"]:
            logger.info("Backtesting ORB Strategy...")
            orb_result = backtester.run_orb_backtest(
                self._daily_data,
                self.config["alpha"]["orb"]
            )
            self._print_backtest_result(orb_result)
        
        logger.info("Backtest complete")
    
    async def run_live(self):
        """Run live/paper trading loop."""
        logger.info("=" * 60)
        logger.info(f"STARTING LIVE TRADING | Mode: {self.config['system']['mode']}")
        logger.info("=" * 60)
        
        self.running = True
        
        # Main trading loop
        while self.running:
            now = datetime.now()
            market_open = now.replace(hour=9, minute=15, second=0)
            market_close = now.replace(hour=15, minute=30, second=0)
            
            # Check market hours
            if now < market_open or now > market_close:
                # Outside market hours - sleep until open
                if now > market_close:
                    # EOD cleanup
                    await self._eod_cleanup()
                    # Sleep until next market open
                    next_open = (now + timedelta(days=1)).replace(
                        hour=9, minute=15, second=0
                    )
                    sleep_secs = (next_open - now).total_seconds()
                    logger.info(f"Market closed. Sleeping {sleep_secs/3600:.1f} hours")
                    await asyncio.sleep(min(sleep_secs, 60))  # Check every min
                    continue
                else:
                    await asyncio.sleep(10)
                    continue
            
            # PRE-MARKET (9:00 - 9:15): Update regime
            if now.hour == 9 and now.minute < 15:
                await self._pre_market_routine()
                await asyncio.sleep(30)
                continue
            
            # OPENING RANGE (9:15 - 9:20): Collect data
            if now.hour == 9 and 15 <= now.minute < 20:
                await self._opening_range_routine()
                await asyncio.sleep(5)
                continue
            
            # TRADING HOURS (9:20 - 15:15): Generate signals and execute
            await self._trading_routine()
            
            # Throttle loop
            await asyncio.sleep(1)
        
        logger.info("Trading loop stopped")
    
    async def _pre_market_routine(self):
        """Pre-market preparation."""
        logger.info("PRE-MARKET: Updating regime and data")
        
        # Update daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400)
        
        for symbol in ["NIFTY"] + self.TRADING_UNIVERSE[:5]:
            df = await self.data_manager.get_data(
                symbol, start_date, end_date, interval="day", use_cache=False
            )
            if not df.empty:
                self._daily_data[symbol] = df
        
        # Detect regime
        if "NIFTY" in self._daily_data:
            self._regime_state = self.regime_engine.detect(
                self._daily_data["NIFTY"]
            )
            logger.info(
                f"Regime: {self._regime_state.regime.value} | "
                f"VVG: {self._regime_state.vvg_label} | "
                f"Pos Mult: {self._regime_state.position_multiplier:.2f}"
            )
    
    async def _opening_range_routine(self):
        """Collect opening range data."""
        # Fetch first 5 minutes of data
        for symbol in self.TRADING_UNIVERSE:
            df = await self.data_manager.get_data(
                symbol,
                datetime.now() - timedelta(minutes=10),
                datetime.now(),
                interval="1m"
            )
            if not df.empty:
                self._intraday_data[symbol] = df
    
    async def _trading_routine(self):
        """Main intraday trading logic."""
        now = datetime.now()
        
        # Time-based strategy selection
        is_morning = now.hour < 12
        is_afternoon = now.hour >= 14
        is_closing = now.hour == 15 and now.minute >= 15
        
        if is_closing:
            # Close all positions before market close
            await self._close_all_positions()
            return
        
        # Update intraday data
        for symbol in self.TRADING_UNIVERSE[:10]:  # Limit API calls
            df = await self.data_manager.get_data(
                symbol,
                datetime.now() - timedelta(hours=2),
                datetime.now(),
                interval="1m"
            )
            if not df.empty:
                self._intraday_data[symbol] = df
        
        # Detect intraday regime
        if "NIFTY" in self._intraday_data and self._regime_state:
            _, vvg_features = self.regime_engine.vvg.classify(
                self._intraday_data["NIFTY"]
            )
            vvg_label = vvg_features.get("label", "trending_moderate")
            self._regime_state.vvg_label = vvg_label
        
        # Get recommended strategies
        if self._regime_state:
            active_strategies = self._regime_state.recommended_strategies
        else:
            active_strategies = [StrategyType.VWAP_REVERSION]
        
        # Execute strategies
        for strategy_type in active_strategies:
            if strategy_type == StrategyType.ORB and is_morning:
                await self._execute_orb()
            elif strategy_type == StrategyType.VWAP_TREND:
                await self._execute_vwap(trend_mode=True)
            elif strategy_type == StrategyType.VWAP_REVERSION:
                await self._execute_vwap(trend_mode=False)
            elif strategy_type == StrategyType.GCN_MOMENTUM:
                await self._execute_gcn()
            elif strategy_type == StrategyType.RISK_OFF:
                await self._close_all_positions()
    
    async def _execute_orb(self):
        """Execute ORB signals."""
        if not self._intraday_data:
            return
        
        candidates = self.orb_strategy.scan_opening_range(self._intraday_data)
        
        for symbol in candidates[:5]:
            if symbol in self._intraday_data and len(self._intraday_data[symbol]) > 0:
                current_bar = self._intraday_data[symbol].iloc[-1]
                signal, position = self.orb_strategy.generate_signal(
                    symbol, current_bar,
                    self._regime_state.position_multiplier if self._regime_state else 1.0
                )
                
                if signal in [ORBSignal.LONG_BREAKOUT, ORBSignal.SHORT_BREAKOUT]:
                    await self._submit_order(
                        symbol=symbol,
                        direction=position.direction,
                        quantity=self._compute_position_size(symbol, position.entry_price),
                        price=position.entry_price,
                        stop_loss=position.stop_loss,
                        target=position.target_price,
                        strategy="ORB"
                    )
    
    async def _execute_vwap(self, trend_mode: bool = True):
        """Execute VWAP signals."""
        for symbol in list(self._intraday_data.keys())[:5]:
            if symbol in self._active_positions:
                continue
            
            df = self._intraday_data[symbol]
            if len(df) < 60:
                continue
            
            vvg_label = "trending_strong" if trend_mode else "choppy_volatile"
            signal, position = self.vwap_strategy.generate_signal(
                symbol, df, vvg_label
            )
            
            if signal in [VWAPSignal.TREND_LONG, VWAPSignal.TREND_SHORT,
                          VWAPSignal.REVERSION_LONG, VWAPSignal.REVERSION_SHORT]:
                await self._submit_order(
                    symbol=symbol,
                    direction=position.direction,
                    quantity=self._compute_position_size(symbol, position.entry_price),
                    price=position.entry_price,
                    stop_loss=position.stop_loss,
                    target=position.target_price,
                    strategy="VWAP"
                )
    
    async def _execute_gcn(self):
        """Execute Chaotic GCN signals."""
        if not self.chaotic_gcn._is_trained:
            return
        
        for symbol in list(self._intraday_data.keys())[:5]:
            if symbol in self._active_positions:
                continue
            
            signal = self.chaotic_gcn.generate_signal(
                self._daily_data,
                self.data_manager.SECTOR_MAP,
                symbol
            )
            
            if signal["confidence"] > 0.6:
                direction = "long" if signal["direction"] == 2 else "short"
                price = self._intraday_data[symbol].iloc[-1]["Close"]
                
                await self._submit_order(
                    symbol=symbol,
                    direction=direction,
                    quantity=self._compute_position_size(
                        symbol, price, signal["position_scale"]
                    ),
                    price=price,
                    strategy="GCN"
                )
    
    async def _submit_order(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        stop_loss: float = 0,
        target: float = 0,
        strategy: str = ""
    ):
        """Submit order through risk engine and execution engine."""
        # Get sector
        sector = self.data_manager.SECTOR_MAP.get(symbol, "Unknown")
        
        # Pre-trade risk check
        action, risk_info = self.risk_engine.pre_trade_check(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            sector=sector,
            strategy=strategy,
            regime_multiplier=self._regime_state.position_multiplier if self._regime_state else 1.0
        )
        
        if action == RiskAction.REJECT:
            logger.warning(f"Order REJECTED: {risk_info.get('reason', 'Risk limit')}")
            return
        elif action == RiskAction.REDUCE_SIZE:
            quantity = risk_info.get("adjusted_quantity", quantity)
            logger.info(f"Order REDUCED: {risk_info.get('reasons')}")
        elif action == RiskAction.FORCE_LIQUIDATE:
            await self._close_all_positions()
            return
        
        # Execute
        if USE_CPP_ENGINE and self.execution_engine:
            side = Side.BUY if direction == "long" else Side.SELL
            price_paise = int(price * 100)
            
            # Determine order type based on strategy
            if strategy == "ORB":
                # ORB needs fast fills
                order_id = self.execution_engine.submit_market_order(
                    symbol, side, int(quantity)
                )
            else:
                # Use VWAP execution for larger orders
                params = VWAPParams()
                params.urgency = 0.3 if strategy == "VWAP" else 0.6
                params.participation_rate = 0.08
                
                order_id = self.execution_engine.submit_vwap_order(
                    symbol, side, int(quantity), price_paise, params
                )
            
            logger.info(
                f"ORDER SUBMITTED: {strategy} {direction} {symbol} "
                f"qty={quantity:.0f} @ {price:.2f} id={order_id}"
            )
        
        # Track position
        self._active_positions[symbol] = {
            "direction": direction,
            "quantity": quantity,
            "entry_price": price,
            "stop_loss": stop_loss,
            "target": target,
            "strategy": strategy,
            "entry_time": datetime.now()
        }
    
    def _compute_position_size(
        self,
        symbol: str,
        price: float,
        confidence_scale: float = 1.0
    ) -> float:
        """
        Compute position size using volatility targeting
        and regime-adjusted sizing.
        """
        portfolio_value = self.risk_engine.portfolio_value
        max_position_pct = self.config["risk"]["max_position_size_pct"]
        vol_target = self.config["risk"]["volatility_target"]
        
        # Base size
        base_size = portfolio_value * max_position_pct
        
        # Volatility adjustment
        if symbol in self._daily_data and len(self._daily_data[symbol]) > 21:
            returns = self._daily_data[symbol]["Close"].pct_change().tail(21)
            asset_vol = returns.std() * np.sqrt(252)
            adj_size = self.risk_engine.volatility_target_sizing(
                base_size / price, asset_vol,
                self._regime_state.position_multiplier if self._regime_state else 1.0
            )
            adj_size *= confidence_scale
        else:
            adj_size = base_size / price * 0.5  # Conservative default
        
        # Round to lot size (1 for equities, 25/50 for F&O)
        return max(int(adj_size), 1)
    
    async def _close_all_positions(self):
        """Close all open positions."""
        for symbol, pos in list(self._active_positions.items()):
            logger.info(f"CLOSING position: {symbol} {pos['direction']}")
            
            if USE_CPP_ENGINE and self.execution_engine:
                side = Side.SELL if pos["direction"] == "long" else Side.BUY
                self.execution_engine.submit_market_order(
                    symbol, side, int(pos["quantity"])
                )
            
            del self._active_positions[symbol]
    
    async def _eod_cleanup(self):
        """End-of-day cleanup and reporting."""
        logger.info("=" * 40)
        logger.info("EOD CLEANUP")
        logger.info(f"Active positions: {len(self._active_positions)}")
        logger.info(f"Portfolio value: {self.risk_engine.portfolio_value:,.2f}")
        logger.info(f"Drawdown: {self.risk_engine.current_drawdown:.2%}")
        logger.info("=" * 40)
        
        await self._close_all_positions()
        self._intraday_data.clear()
        self.orb_strategy.force_close_all({})
    
    def _print_backtest_result(self, result):
        """Print formatted backtest results."""
        logger.info(f"\n{'='*60}")
        logger.info(f"BACKTEST RESULTS: {result.strategy_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Total Return:      {result.total_return:>10.2%}")
        logger.info(f"CAGR:              {result.cagr:>10.2%}")
        logger.info(f"Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
        logger.info(f"Sortino Ratio:     {result.sortino_ratio:>10.2f}")
        logger.info(f"Max Drawdown:      {result.max_drawdown:>10.2%}")
        logger.info(f"Calmar Ratio:      {result.calmar_ratio:>10.2f}")
        logger.info(f"Win Rate:          {result.win_rate:>10.2%}")
        logger.info(f"Profit Factor:     {result.profit_factor:>10.2f}")
        logger.info(f"Avg Trade Return:  {result.avg_trade_return:>10.4f}")
        logger.info(f"Total Trades:      {result.total_trades:>10d}")
        logger.info(f"{'='*60}\n")
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False
        
        if self.execution_engine:
            self.execution_engine.stop()


async def main():
    # Determine mode
    mode = "backtest"  # Change to "live" for paper/live trading
    
    system = NiftyQuantSystem()
    
    # Graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, system.shutdown)
    
    await system.initialize()
    
    if mode == "backtest":
        await system.run_backtest()
    else:
        await system.run_live()


if __name__ == "__main__":
    asyncio.run(main())
