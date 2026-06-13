"""Compatibility wrapper for Unified Model Hub."""

from research.experiments.ml.model_hub import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.experiments.ml.model_hub", run_name="__main__")
