"""Compatibility wrapper for the regime alpha connector."""

from research.experiments.regime.regime_alpha_connector import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.experiments.regime.regime_alpha_connector", run_name="__main__")
