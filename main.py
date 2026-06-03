"""Main orchestrator for the consolidated Quant Research OS."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from alpha.manager import AlphaManager
from data.manager import DataManager
from features.pipeline import FeaturePipeline
from portfolio.allocator import PortfolioAllocator
from regime.manager import RegimeManager
from risk.institutional_risk_engine import InstitutionalRiskEngine, Position

logger = logging.getLogger("quant_os")


class QuantResearchOS:
    """Compact orchestrator that wires data, features, alpha, regime, and risk."""

    DEFAULT_UNIVERSE = [
        "RELIANCE",
        "TCS",
        "HDFCBANK",
        "INFY",
        "ICICIBANK",
        "HINDUNILVR",
        "ITC",
        "SBIN",
        "BHARTIARTL",
        "LT",
    ]

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.data_manager = DataManager(self.config)
        self.feature_pipeline = FeaturePipeline()
        self.alpha_manager = AlphaManager(self.config)
        self.regime_manager = RegimeManager(self.config)
        self.portfolio_allocator = PortfolioAllocator(
            max_position_pct=float(self.config.get("risk", {}).get("max_position_size_pct", 0.05)),
            max_sector_pct=float(self.config.get("risk", {}).get("sector_concentration", 0.30)),
        )
        self.risk_engine = InstitutionalRiskEngine(capital=250_000_000)
        self.market_data: pd.DataFrame = pd.DataFrame()

    def _load_config(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    async def initialize(self) -> None:
        """Load market context and fit the regime engine."""
        logger.info("Initializing Quant Research OS")
        await self.data_manager.initialize_feeds()

        end = datetime.now()
        start = end - timedelta(days=400)
        try:
            frame = await self.data_manager.get_data("NIFTY", start, end, interval="day")
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Data fetch failed, falling back to synthetic sample: %s", exc)
            frame = self._synthetic_market_data()

        if frame.empty:
            frame = self._synthetic_market_data()

        self.market_data = frame
        self.regime_manager.fit(self.market_data)
        logger.info("Initialization complete with %d rows of market data", len(self.market_data))

    async def run_backtest(self) -> dict[str, Any]:
        """Run a lightweight backtest-style walkthrough using the current market sample."""
        if self.market_data.empty:
            await self.initialize()

        latest_features = self.feature_pipeline.compute_features(self.market_data)
        regime = self.regime_manager.detect(self.market_data)
        symbol = "NIFTY"

        options_context = {
            "spot": float(self.market_data["close"].iloc[-1]),
            "strike": float(self.market_data["close"].iloc[-1]),
            "call_price": max(float(self.market_data["close"].iloc[-1]) * 0.01, 1.0),
            "put_price": max(float(self.market_data["close"].iloc[-1]) * 0.01, 1.0),
            "days_to_expiry": 7,
            "risk_free_rate": 0.05,
            "atm_iv": float(latest_features["implied_volatility"] or 0.18),
            "otm_put_iv": float(latest_features["implied_volatility"] or 0.18),
            "realized_vol": float(latest_features["realized_volatility_20d"] or 0.15),
            "pcr": float(latest_features["pcr"] or 1.0),
        }

        signals = self.alpha_manager.generate_signals(
            symbol,
            self.market_data,
            regime_label=regime.regime,
            options_context=options_context,
            timestamp=self.market_data.index[-1].to_pydatetime() if isinstance(self.market_data.index, pd.DatetimeIndex) else None,
        )
        combined = self.alpha_manager.combine_signals(
            signals,
            regime_label=regime.regime,
            market_data=self.market_data,
        )

        allocations = self.portfolio_allocator.allocate_from_alpha_signals(
            capital=self.risk_engine.capital,
            signals=signals,
            regime_label=regime.regime,
            current_prices={symbol: float(self.market_data["close"].iloc[-1])},
            symbol_volatilities={symbol: float(max(latest_features["realized_volatility_20d"], 0.05))},
            symbol_sectors={symbol: "INDEX"},
        )

        positions = [
            Position(
                symbol=allocation.symbol,
                sector="INDEX",
                quantity=max(1, int(allocation.capital / max(float(self.market_data["close"].iloc[-1]), 1.0))),
                entry_price=float(self.market_data["close"].iloc[-1]),
                current_price=float(self.market_data["close"].iloc[-1]),
                side="LONG",
            )
            for allocation in allocations
        ]
        risk = self.risk_engine.calculate_risk_metrics(positions, self._risk_market_frame(), daily_pnl=0.0, weekly_pnl=0.0)

        result = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime.to_dict(),
            "features": latest_features,
            "signals": [signal.to_dict() for signal in signals],
            "combined_signal": combined,
            "allocations": [allocation.to_dict() for allocation in allocations],
            "risk": asdict(risk),
        }
        logger.info("Backtest summary: %s", result["combined_signal"])
        return result

    async def run_live(self) -> None:
        """Run a lightweight live loop that refreshes regime and signal state."""
        if self.market_data.empty:
            await self.initialize()

        logger.info("Entering live loop. Press Ctrl+C to exit.")
        while True:
            regime = self.regime_manager.detect(self.market_data)
            features = self.feature_pipeline.compute_features(self.market_data)
            signals = self.alpha_manager.generate_signals(
                "NIFTY",
                self.market_data,
                regime_label=regime.regime,
                options_context={"spot": float(self.market_data["close"].iloc[-1]), "strike": float(self.market_data["close"].iloc[-1]), "call_price": 1.0, "put_price": 1.0, "days_to_expiry": 7},
            )
            logger.info(
                "Live snapshot | regime=%s confidence=%.2f combined=%.4f rv=%.2f",
                regime.regime,
                regime.probability,
                self.alpha_manager.combine_signals(signals, regime_label=regime.regime)["combined_signal"],
                features["relative_volume"],
            )
            await asyncio.sleep(60)

    def _risk_market_frame(self) -> pd.DataFrame:
        frame = self.market_data[["close"]].copy()
        frame["NIFTY"] = frame["close"]
        return frame[["NIFTY"]]

    def _synthetic_market_data(self, rows: int = 260) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        index = pd.date_range(end=datetime.now(), periods=rows, freq="D")
        rets = rng.normal(0.0006, 0.015, rows)
        close = 20_000 * np.cumprod(1 + rets)
        open_ = close * (1 + rng.normal(0.0, 0.003, rows))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.002, 0.002, rows)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.002, 0.002, rows)))
        volume = rng.integers(1_000_000, 5_000_000, rows)
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant Research OS")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", choices=["backtest", "live"], default=None, help="Override system mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    system = QuantResearchOS(args.config)
    mode = args.mode or system.config.get("system", {}).get("mode", "backtest")

    if mode == "live":
        asyncio.run(system.run_live())
    else:
        summary = asyncio.run(system.run_backtest())
        print(yaml.safe_dump(summary, sort_keys=False, default_flow_style=False))


if __name__ == "__main__":
    main()
