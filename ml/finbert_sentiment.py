"""Compatibility wrapper for FinBERT Sentiment Analyzer."""

from research.experiments.ml.finbert_sentiment import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("research.experiments.ml.finbert_sentiment", run_name="__main__")
