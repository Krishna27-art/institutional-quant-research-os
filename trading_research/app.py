"""Command-line entrypoint for the trading research OS."""

from __future__ import annotations

import argparse
from pathlib import Path

from .extract_gap_events_phase1 import main as extract_gap_events
from .paper_trading_helper import main as run_paper_trading


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Research OS")
    parser.add_argument(
        "command",
        choices=["extract-gaps", "analyze-participants", "paper-trade", "validate"],
        help="Command to run",
    )
    parser.add_argument("--symbol", type=str, help="Specific symbol to analyze")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.command == "extract-gaps":
        extract_gap_events()
    elif args.command == "analyze-participants":
        from .hypothesis.gap_participant_models import ParticipantRegimeClassifier

        print("Participant analysis is available via trading_research.hypothesis.")
        print(ParticipantRegimeClassifier().__class__.__name__)
    elif args.command == "paper-trade":
        run_paper_trading()
    elif args.command == "validate":
        print("Validation entrypoint is ready; run the test suite for full checks.")


if __name__ == "__main__":
    main()
