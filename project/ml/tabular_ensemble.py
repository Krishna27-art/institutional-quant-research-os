"""Compatibility wrapper for Tabular Ensemble."""

from research.experiments.ml.tabular_ensemble import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.experiments.ml.tabular_ensemble", run_name="__main__")
