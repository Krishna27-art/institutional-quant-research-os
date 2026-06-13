"""Compatibility wrapper for FinGPT Research Summarizer."""

from research.experiments.ml.fingpt_research import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.experiments.ml.fingpt_research", run_name="__main__")
