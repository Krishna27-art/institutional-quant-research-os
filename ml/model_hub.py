"""
Model Hub — Unified Lazy-Loading Model Registry
=================================================
Single entry point for all models in the Quant Research OS.

Provides:
  - Lazy loading (models downloaded/loaded on first use)
  - Automatic memory management (LRU eviction after idle timeout)
  - Health tracking per model (last use, error count)
  - Clean API: ModelHub.get_forecaster("chronos"), .get_sentiment("finbert"), etc.

Usage
-----
    hub = ModelHub()

    # Time-series forecasting
    chronos = hub.get_forecaster("chronos")
    result  = chronos.forecast(series, horizon=5)

    # Sentiment analysis
    finbert = hub.get_sentiment("finbert")
    scores  = finbert.analyze_batch(headlines)

    # Tabular alpha
    ensemble = hub.get_tabular("ensemble")
    ensemble.fit_with_cv(X, y)
    preds   = ensemble.predict(X_test)

    # Regime connector
    connector = hub.get_regime_connector(hmm_engine)
    alloc     = connector.get_alpha_allocation(features, timestamp)
"""

import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    FORECASTER   = "forecaster"
    SENTIMENT    = "sentiment"
    TABULAR      = "tabular"
    REGIME       = "regime"


class ModelStatus(str, Enum):
    NOT_LOADED  = "not_loaded"
    LOADED      = "loaded"
    ERROR       = "error"
    UNLOADED    = "unloaded"


class ModelEntry:
    """Tracks a single model's lifecycle in the hub."""

    def __init__(self, name: str, model_type: ModelType, is_research_only: bool = True):
        self.name             = name
        self.model_type       = model_type
        self.is_research_only = is_research_only
        self.status           = ModelStatus.NOT_LOADED
        self.instance         = None
        self.load_time_s      = None
        self.last_used        = None
        self.use_count        = 0
        self.error_count      = 0
        self.last_error       = None

    def mark_used(self) -> None:
        self.last_used = datetime.now()
        self.use_count += 1

    def is_idle(self, timeout_minutes: int = 30) -> bool:
        if self.last_used is None:
            return False
        return datetime.now() - self.last_used > timedelta(minutes=timeout_minutes)

    def __repr__(self) -> str:
        return (
            f"ModelEntry({self.name} | {self.model_type.value} | "
            f"status={self.status.value} | uses={self.use_count})"
        )


class ModelHub:
    """
    Unified model registry with lazy loading and memory management.

    Design philosophy:
    ──────────────────
    1. No model is loaded until first requested.
    2. Models are reused across requests (cached in memory).
    3. Idle models (unused for >idle_timeout_min) are unloaded to free RAM/VRAM.
    4. All models self-report their research_only status.

    Memory Management:
    ──────────────────
    Call `hub.cleanup_idle_models()` periodically (e.g., every 30 min)
    to unload models that haven't been used recently.
    """

    # ── Registry of all available models ─────────────────────────────────────
    REGISTRY = {
        # Forecasting models
        "chronos":      (ModelType.FORECASTER,  True,  "ml.chronos_forecaster",  "ChronosForecaster"),
        "timesfm":      (ModelType.FORECASTER,  True,  "ml.timesfm_forecaster",  "TimesFMForecaster"),
        "patchtst":     (ModelType.FORECASTER,  True,  "ml.patchtst_forecaster", "PatchTSTForecaster"),

        # Sentiment models
        "finbert":      (ModelType.SENTIMENT,   True,  "ml.finbert_sentiment",   "FinBERTSentiment"),
        "fingpt":       (ModelType.SENTIMENT,   True,  "ml.fingpt_research",     "FinGPTResearch"),

        # Tabular models
        "xgboost":      (ModelType.TABULAR,     False, "ml.tabular_ensemble",    "XGBoostAlphaModel"),
        "catboost":     (ModelType.TABULAR,     False, "ml.tabular_ensemble",    "CatBoostAlphaModel"),
        "lightgbm":     (ModelType.TABULAR,     False, "ml.tabular_ensemble",    "LightGBMAlphaModel"),
        "ensemble":     (ModelType.TABULAR,     False, "ml.tabular_ensemble",    "TabularEnsemble"),

        # Regime connector
        "regime":       (ModelType.REGIME,      False, "regime.regime_alpha_connector", "RegimeAlphaConnector"),
    }

    def __init__(self, idle_timeout_min: int = 30):
        self.idle_timeout_min = idle_timeout_min
        self._entries: Dict[str, ModelEntry] = {}

        # Pre-create entries (without loading)
        for name, (mtype, research_only, _, _) in self.REGISTRY.items():
            self._entries[name] = ModelEntry(name, mtype, research_only)

        logger.info(
            f"ModelHub initialized | "
            f"{len(self._entries)} models registered | "
            f"idle_timeout={idle_timeout_min}min"
        )

    # ── Generic model getter ──────────────────────────────────────────────────
    def get(self, name: str, **kwargs) -> Optional[Any]:
        """
        Get a model by name. Loads it on first access.

        Parameters
        ----------
        name : str
            Model name from REGISTRY.
        **kwargs
            Passed to model constructor on first load.

        Returns
        -------
        Model instance or None if loading fails.
        """
        name = name.lower()
        if name not in self._entries:
            logger.error(f"Unknown model: '{name}'. Available: {list(self.REGISTRY.keys())}")
            return None

        entry = self._entries[name]

        if entry.status == ModelStatus.LOADED and entry.instance is not None:
            entry.mark_used()
            return entry.instance

        # Load the model
        success = self._load_model(name, entry, **kwargs)
        if success:
            entry.mark_used()
            return entry.instance
        return None

    # ── Typed getters (for IDE autocompletion) ────────────────────────────────
    def get_forecaster(self, name: str = "chronos", **kwargs):
        """Get a time-series forecasting model."""
        return self.get(name, **kwargs)

    def get_sentiment(self, name: str = "finbert", **kwargs):
        """Get a financial NLP sentiment model."""
        return self.get(name, **kwargs)

    def get_tabular(self, name: str = "ensemble", **kwargs):
        """Get a tabular alpha model or ensemble."""
        return self.get(name, **kwargs)

    def get_regime_connector(self, hmm_engine=None, **kwargs):
        """Get the regime alpha connector (wired to an HMM engine)."""
        if hmm_engine is None:
            logger.error("hmm_engine is required for regime connector.")
            return None
        return self.get("regime", hmm_engine=hmm_engine, **kwargs)

    # ── Internal loader ───────────────────────────────────────────────────────
    def _load_model(self, name: str, entry: ModelEntry, **kwargs) -> bool:
        """Dynamically import and instantiate a model."""
        if name not in self.REGISTRY:
            return False

        model_type, research_only, module_path, class_name = self.REGISTRY[name]

        logger.info(f"Loading model '{name}' from {module_path}.{class_name}...")
        t0 = time.time()

        try:
            import importlib
            module = importlib.import_module(module_path)
            cls    = getattr(module, class_name)
            instance = cls(**kwargs)

            entry.instance    = instance
            entry.status      = ModelStatus.LOADED
            entry.load_time_s = time.time() - t0
            logger.info(f"'{name}' loaded in {entry.load_time_s:.2f}s")
            return True

        except Exception as e:
            entry.status     = ModelStatus.ERROR
            entry.error_count += 1
            entry.last_error  = str(e)
            logger.error(f"Failed to load '{name}': {e}")
            return False

    # ── Memory management ─────────────────────────────────────────────────────
    def unload(self, name: str) -> bool:
        """Manually unload a model to free memory."""
        entry = self._entries.get(name)
        if entry is None or entry.instance is None:
            return False
        try:
            if hasattr(entry.instance, "unload"):
                entry.instance.unload()
            del entry.instance
            entry.instance = None
            entry.status   = ModelStatus.UNLOADED
            logger.info(f"'{name}' unloaded.")
            return True
        except Exception as e:
            logger.error(f"Error unloading '{name}': {e}")
            return False

    def cleanup_idle_models(self) -> List[str]:
        """
        Unload all models idle longer than idle_timeout_min.

        Returns list of model names that were unloaded.
        """
        unloaded = []
        for name, entry in self._entries.items():
            if entry.status == ModelStatus.LOADED and entry.is_idle(self.idle_timeout_min):
                logger.info(f"'{name}' idle for >{self.idle_timeout_min}min — unloading.")
                self.unload(name)
                unloaded.append(name)
        return unloaded

    # ── Status & reporting ────────────────────────────────────────────────────
    def status_report(self) -> str:
        """Return a formatted status report for all models."""
        lines = [
            "=" * 65,
            f"{'MODEL HUB STATUS':^65}",
            "=" * 65,
            f"{'Name':<15} {'Type':<12} {'Status':<12} {'Uses':>5} {'Research?':>10}",
            "-" * 65,
        ]
        for name, entry in self._entries.items():
            research = "YES" if entry.is_research_only else "no"
            lines.append(
                f"{entry.name:<15} {entry.model_type.value:<12} "
                f"{entry.status.value:<12} {entry.use_count:>5} {research:>10}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def get_loaded_models(self) -> Dict[str, ModelEntry]:
        """Return all currently loaded models."""
        return {name: e for name, e in self._entries.items()
                if e.status == ModelStatus.LOADED}

    def get_available_models(self) -> Dict[str, Dict]:
        """Return metadata for all registered models."""
        result = {}
        for name, (mtype, research_only, module, cls) in self.REGISTRY.items():
            entry = self._entries.get(name)
            result[name] = {
                "type":           mtype.value,
                "is_research_only": research_only,
                "status":         entry.status.value if entry else "unknown",
                "module":         module,
                "class":          cls,
            }
        return result


# ── Module-level singleton (optional) ─────────────────────────────────────────
_default_hub: Optional[ModelHub] = None


def get_default_hub() -> ModelHub:
    """Get or create the module-level ModelHub singleton."""
    global _default_hub
    if _default_hub is None:
        _default_hub = ModelHub()
    return _default_hub


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from typing import List
    logging.basicConfig(level=logging.INFO)
    print("=== Model Hub Smoke Test ===\n")

    hub = ModelHub(idle_timeout_min=5)

    print("Available models:")
    for name, meta in hub.get_available_models().items():
        ro = "📚 research" if meta["is_research_only"] else "⚙️  production"
        print(f"  {name:<15} {meta['type']:<12} [{ro}]")

    print("\n" + hub.status_report())

    # Test lazy loading of tabular ensemble (no GPU needed)
    print("\nLoading tabular ensemble...")
    ensemble = hub.get_tabular("ensemble")
    if ensemble is not None:
        print(f"  Available models in ensemble: {ensemble.available_models}")
        print("  ✓ Tabular ensemble loaded via hub.")

    # Test FinBERT loading
    print("\nLoading FinBERT...")
    finbert = hub.get_sentiment("finbert")
    if finbert is not None:
        result = finbert.analyze_text("HDFC Bank posts record quarterly profit.")
        print(f"  Sentiment: {result['label']} (net={result['net_score']:+.3f})")
        print("  ✓ FinBERT loaded and inference succeeded.")

    print("\nFinal status:")
    print(hub.status_report())
    print("\n✓ Model Hub smoke test complete.")
