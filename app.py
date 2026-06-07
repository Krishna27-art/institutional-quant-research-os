#!/usr/bin/env python3
"""Entry point for the new market research slice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd

from config import DEFAULT_CONFIG
from core.event_engine import EventDrivenEngine
from core.events import EventType
from data.audit import CorporateActionAudit, SurvivorshipAudit
from data.corporate_actions import CorporateAction
from data.nse_adapter import NSELibAdapter
from data.universe import UniverseRegistry
from market.state import MarketStateEngine
from market.smart_money import SmartMoneyStructure
from research.experiment import ExperimentRecord, ExperimentStore
from research.replay import ReplayJournal
from signals.gap_fade import GapFadeSignalGenerator
from signals.validator import SignalValidator
from stats.leakage import FeatureValidator, LeakageGuard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Market OS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("state-demo", help="Build a sample market state")
    subparsers.add_parser("event-demo", help="Run a small event-driven replay")
    subparsers.add_parser("smc-demo", help="Run smart-money structure checks")
    subparsers.add_parser("data-audit-demo", help="Run corporate-action and survivorship audits")
    subparsers.add_parser("leakage-demo", help="Run feature leakage and quality checks")
    subparsers.add_parser("nse-normalize-demo", help="Normalize sample NSE context datasets")
    subparsers.add_parser("replay-demo", help="Write and verify deterministic replay events")

    gap_parser = subparsers.add_parser("gap-demo", help="Run the gap fade gate on sample inputs")
    gap_parser.add_argument("--gap-pct", type=float, default=0.55)
    gap_parser.add_argument("--vix", type=float, default=15.0)
    gap_parser.add_argument("--fii-flow", type=float, default=-1.0)
    gap_parser.add_argument("--expiry-week", action="store_true")
    gap_parser.add_argument("--mechanism-score", type=float, default=0.65)
    gap_parser.add_argument("--symbol", type=str, default="RELIANCE")

    exp_parser = subparsers.add_parser("log-experiment", help="Store an immutable experiment record")
    exp_parser.add_argument("--hypothesis-id", required=True)
    exp_parser.add_argument("--experiment-id", required=True)
    exp_parser.add_argument("--created-at", default=datetime.now(timezone.utc).isoformat())
    exp_parser.add_argument("--fingerprint", required=True)
    exp_parser.add_argument("--params", default="{}")
    exp_parser.add_argument("--metrics", default="{}")

    return parser


def main() -> None:
    DEFAULT_CONFIG.ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "state-demo":
        engine = MarketStateEngine()
        state = engine.build(
            {
                "trend_strength": 0.72,
                "daily_volatility": 0.011,
                "breadth_score": 0.34,
                "liquidity_score": 0.81,
                "participation_score": 0.55,
                "correlation_score": 0.62,
            }
        )
        print(state.to_dict())
        return

    if args.command == "event-demo":
        journal_path = DEFAULT_CONFIG.experiment_dir / "event_demo_v2.sqlite3"
        engine = EventDrivenEngine.with_journal(journal_path)
        engine.run_bars(
            [
                {"symbol": "RELIANCE", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
                {"symbol": "RELIANCE", "open": 100.5, "high": 102.0, "low": 100.2, "close": 101.8},
            ]
        )
        engine.emit(EventType.MARKET_STATE, {"regime": "medium_mixed_range_bound"})
        engine.emit(EventType.SIGNAL, {"symbol": "RELIANCE", "direction": -1, "mechanism_score": 0.71})
        reloaded = EventDrivenEngine().replay(ReplayJournal(journal_path))
        print({"journal_verified": ReplayJournal(journal_path).verify(), "events": [event.to_dict() for event in reloaded]})
        return

    if args.command == "smc-demo":
        frame = pd.DataFrame(
            {
                "open": [100, 101, 102, 101, 103, 104],
                "high": [101, 103, 104, 102, 105, 106],
                "low": [99, 100, 101, 98, 102, 103],
                "close": [100.5, 102.5, 101.5, 101.0, 104.5, 103.8],
            }
        )
        smc = SmartMoneyStructure()
        signals = [item for item in [smc.latest_break_of_structure(frame), smc.liquidity_sweep(frame), smc.fair_value_gap(frame)] if item]
        print([signal.to_dict() for signal in signals])
        return

    if args.command == "data-audit-demo":
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "open": [100.0, 50.0, 51.0],
                "high": [101.0, 51.0, 52.0],
                "low": [99.0, 49.0, 50.0],
                "close": [100.0, 50.5, 51.5],
                "adjusted_close": [50.0, 50.5, 51.5],
                "volume": [1000, 2000, 2100],
            }
        )
        corp = CorporateActionAudit().verify_adjustment(prices, [CorporateAction("ABC", pd.Timestamp("2024-01-02"), "split", 2.0)])
        universe = UniverseRegistry()
        universe._table = pd.DataFrame({"symbol": ["ABC"], "start_date": [pd.Timestamp("2024-01-01")], "end_date": [pd.NaT]})
        trades = pd.DataFrame({"date": [pd.Timestamp("2024-01-03")], "symbol": ["ABC"]})
        survivorship = SurvivorshipAudit().verify_trades(trades, universe)
        print({"corporate_action": corp.to_dict(), "survivorship": survivorship.to_dict()})
        return

    if args.command == "leakage-demo":
        frame = pd.DataFrame(
            {
                "feature_timestamp": pd.to_datetime(["2024-01-01 09:15", "2024-01-01 09:30"]),
                "decision_timestamp": pd.to_datetime(["2024-01-01 09:20", "2024-01-01 09:25"]),
                "feature_a": [0.1, 0.2],
                "constant_feature": [1.0, 1.0],
                "target": [0.01, 0.02],
            }
        )
        leakage = LeakageGuard().validate_feature_dates(frame)
        target = LeakageGuard().validate_target_shift(frame, ["feature_a", "constant_feature"], "target")
        feature = FeatureValidator().validate(frame, ["feature_a", "constant_feature"])
        print({"date_leakage": leakage.to_dict(), "target_leakage": target.to_dict(), "feature_quality": feature.to_dict()})
        return

    if args.command == "nse-normalize-demo":
        adapter = NSELibAdapter()
        fii_dii = adapter.normalize_fii_dii(
            pd.DataFrame({"date": ["2024-01-01"], "fii_net": [-1200.0], "dii_net": [900.0]})
        )
        vix = adapter.normalize_vix(pd.DataFrame({"date": ["2024-01-01"], "close": [14.8]}))
        delivery = adapter.normalize_delivery(pd.DataFrame({"date": ["2024-01-01"], "symbol": ["RELIANCE"], "deliverable_qty": [100], "traded_qty": [400]}))
        print({"fii_dii": fii_dii.to_dict("records"), "vix": vix.to_dict("records"), "delivery": delivery.to_dict("records")})
        return

    if args.command == "replay-demo":
        journal = ReplayJournal(DEFAULT_CONFIG.experiment_dir / "replay_demo_v2.sqlite3")
        journal.append("market_state", {"regime": "medium_mixed_range_bound"}, timestamp="2024-01-01T09:20:00+05:30")
        journal.append("signal", {"symbol": "RELIANCE", "direction": -1, "mechanism_score": 0.72}, timestamp="2024-01-01T09:21:00+05:30")
        print({"verified": journal.verify(), "events": [event.to_dict() for event in journal.load()]})
        return

    if args.command == "gap-demo":
        engine = MarketStateEngine()
        market_state = engine.build({"trend_strength": 0.18, "daily_volatility": 0.009, "breadth_score": 0.05})
        generator = GapFadeSignalGenerator()
        signal = generator.generate(
            market_state,
            {
                "gap_pct": args.gap_pct,
                "vix": args.vix,
                "fii_flow": args.fii_flow,
                "expiry_week": args.expiry_week,
                "mechanism_score": args.mechanism_score,
                "symbol": args.symbol,
            },
        )
        if signal is None:
            print("no_signal")
            return
        validator = SignalValidator()
        allowed = validator.validate(signal, market_state.regime)
        print({"signal": signal.to_dict(), "allowed": allowed})
        return

    if args.command == "log-experiment":
        store = ExperimentStore(DEFAULT_CONFIG.experiment_dir / "experiments.sqlite3")
        record = ExperimentRecord(
            experiment_id=args.experiment_id,
            hypothesis_id=args.hypothesis_id,
            data_fingerprint=args.fingerprint,
            created_at=args.created_at,
            params=json.loads(args.params),
            metrics=json.loads(args.metrics),
        )
        store.save(record)
        print(record.to_dict())
        return


if __name__ == "__main__":
    main()
