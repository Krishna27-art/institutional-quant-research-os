"""
Chronos Forecaster — Amazon Chronos-T5 Wrapper
===============================================
Wraps Amazon's Chronos foundation model for time-series forecasting.

⚠  USAGE GUARD: This model produces RESEARCH FORECASTS only.
   Do NOT use raw Chronos output as a direct trading signal.
   Always validate out-of-sample before any live use.

Suitable for:
  - Returns forecasting (5d, 20d horizon)
  - Volume forecasting
  - Realized volatility forecasting
  - Liquidity (bid-ask spread) forecasting

Model: amazon/chronos-t5-large (default, ~700MB)
       amazon/chronos-t5-tiny  (fallback, ~50MB, CPU-friendly)

Install:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

Reference: https://huggingface.co/amazon/chronos-t5-large
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Guard Chronos import ──────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — ChronosForecaster will be disabled.")

try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
    logger.info("chronos-forecasting package loaded.")
except ImportError:
    ChronosPipeline = None
    CHRONOS_AVAILABLE = False
    logger.warning(
        "chronos-forecasting not installed. "
        "Run: pip install git+https://github.com/amazon-science/chronos-forecasting.git"
    )


# ── Constants ─────────────────────────────────────────────────────────────────
RESEARCH_ONLY = True        # Hard flag — never remove
MODEL_VARIANTS = {
    "large":  "amazon/chronos-t5-large",    # ~700MB, best quality
    "small":  "amazon/chronos-t5-small",    # ~250MB, balanced
    "tiny":   "amazon/chronos-t5-tiny",     # ~50MB, CPU-friendly
    "base":   "amazon/chronos-t5-base",     # ~400MB
}
DEFAULT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


class ChronosForecaster:
    """
    Amazon Chronos-T5 wrapper for quantitative research forecasting.

    Features
    --------
    - Lazy model loading (downloads on first forecast call)
    - Auto device detection (CUDA → MPS → CPU)
    - Quantile forecasts (10th, 25th, 50th, 75th, 90th percentile)
    - Batch forecasting for multiple series
    - In-memory caching of loaded model

    Example
    -------
    >>> forecaster = ChronosForecaster(model_size="tiny")
    >>> series = pd.Series(nifty_returns[-252:])
    >>> result = forecaster.forecast(series, horizon=5)
    >>> print(result['median'])   # 5-day return forecast
    """

    is_research_only: bool = RESEARCH_ONLY

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        cache_predictions: bool = True,
    ):
        """
        Parameters
        ----------
        model_size : str
            One of 'tiny', 'small', 'base', 'large'.
        device : str, optional
            'cuda', 'mps', 'cpu', or None (auto-detect).
        dtype : str
            Torch dtype string ('bfloat16', 'float32').
        cache_predictions : bool
            Cache last prediction result to avoid re-inference on same data.
        """
        if model_size not in MODEL_VARIANTS:
            raise ValueError(f"model_size must be one of {list(MODEL_VARIANTS.keys())}")

        self.model_id = MODEL_VARIANTS[model_size]
        self.model_size = model_size
        self._requested_device = device
        self._dtype_str = dtype
        self._cache = cache_predictions

        self._pipeline = None          # Lazy-loaded
        self._last_cache_key: Optional[str] = None
        self._last_result: Optional[Dict] = None

        self._device = self._detect_device(device)
        logger.info(
            f"ChronosForecaster configured | model={self.model_id} | "
            f"device={self._device} | dtype={dtype}"
        )

    # ── Device detection ──────────────────────────────────────────────────────
    @staticmethod
    def _detect_device(requested: Optional[str]) -> str:
        if requested is not None:
            return requested
        if not TORCH_AVAILABLE:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_torch_dtype(self):
        if not TORCH_AVAILABLE:
            return None
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
            "float16":  torch.float16,
        }
        return dtype_map.get(self._dtype_str, torch.bfloat16)

    # ── Lazy model loading ────────────────────────────────────────────────────
    def _load_model(self) -> bool:
        """Download and load model on first use. Returns True on success."""
        if self._pipeline is not None:
            return True
        if not CHRONOS_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("Cannot load Chronos — package not installed.")
            return False

        logger.info(f"Loading {self.model_id} (this may take a moment on first run)...")
        t0 = time.time()
        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self._device,
                torch_dtype=self._get_torch_dtype(),
            )
            logger.info(f"Chronos model loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load Chronos model: {e}")
            self._pipeline = None
            return False

    # ── Core forecast API ─────────────────────────────────────────────────────
    def forecast(
        self,
        series: pd.Series,
        horizon: int = 5,
        num_samples: int = 100,
        quantiles: Optional[List[float]] = None,
    ) -> Dict:
        """
        Generate probabilistic forecasts for a univariate time series.

        Parameters
        ----------
        series : pd.Series
            Historical time series (e.g., daily log returns, vol, volume).
            Minimum recommended length: 30 observations.
        horizon : int
            Number of future steps to forecast.
        num_samples : int
            Number of Monte Carlo samples for quantile estimation.
        quantiles : list of float, optional
            Quantile levels to return. Default: [0.1, 0.25, 0.5, 0.75, 0.9].

        Returns
        -------
        dict with keys:
            'median' : np.ndarray (horizon,)
            'quantiles' : dict {q_level: np.ndarray (horizon,)}
            'mean' : np.ndarray (horizon,)
            'std' : np.ndarray (horizon,)
            'model' : str
            'is_research_only' : bool

        Notes
        -----
        Returns a fallback dict of zeros if model cannot be loaded.
        """
        quantiles = quantiles or DEFAULT_QUANTILES

        if len(series) < 10:
            logger.warning("Series too short for Chronos (<10 points). Returning zeros.")
            return self._zero_result(horizon, quantiles)

        # Check cache
        cache_key = f"{hash(tuple(series.values[-20:]))}_{horizon}"
        if self._cache and cache_key == self._last_cache_key and self._last_result:
            logger.debug("Returning cached Chronos forecast.")
            return self._last_result

        if not self._load_model():
            logger.warning("Chronos model unavailable — returning statistical baseline.")
            return self._statistical_fallback(series, horizon, quantiles)

        try:
            context = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0)
            forecast_tensor = self._pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=num_samples,
            )
            # forecast_tensor shape: (1, num_samples, horizon)
            samples = forecast_tensor[0].numpy()  # (num_samples, horizon)

            q_dict = {}
            for q in quantiles:
                q_dict[q] = np.percentile(samples, q * 100, axis=0)

            result = {
                "median":          np.median(samples, axis=0),
                "mean":            np.mean(samples, axis=0),
                "std":             np.std(samples, axis=0),
                "quantiles":       q_dict,
                "samples":         samples,
                "model":           self.model_id,
                "is_research_only": RESEARCH_ONLY,
            }

            if self._cache:
                self._last_cache_key = cache_key
                self._last_result = result

            return result

        except Exception as e:
            logger.error(f"Chronos inference error: {e}")
            return self._statistical_fallback(series, horizon, quantiles)

    def forecast_batch(
        self,
        series_dict: Dict[str, pd.Series],
        horizon: int = 5,
        num_samples: int = 50,
    ) -> Dict[str, Dict]:
        """
        Forecast multiple series in one call.

        Parameters
        ----------
        series_dict : dict
            {'series_name': pd.Series, ...}
        horizon : int
        num_samples : int

        Returns
        -------
        dict of {series_name: forecast_result}
        """
        results = {}
        for name, series in series_dict.items():
            logger.info(f"Forecasting series: {name}")
            results[name] = self.forecast(series, horizon=horizon, num_samples=num_samples)
        return results

    # ── Fallbacks ─────────────────────────────────────────────────────────────
    def _statistical_fallback(
        self,
        series: pd.Series,
        horizon: int,
        quantiles: List[float],
    ) -> Dict:
        """Simple statistical baseline when model is unavailable."""
        arr = series.dropna().values
        mu  = float(np.mean(arr[-20:]))   # Recent mean
        sig = float(np.std(arr[-20:]))    # Recent std

        q_dict = {}
        from scipy import stats
        for q in quantiles:
            q_dict[q] = np.full(horizon, stats.norm.ppf(q, loc=mu, scale=sig))

        return {
            "median":          np.full(horizon, mu),
            "mean":            np.full(horizon, mu),
            "std":             np.full(horizon, sig),
            "quantiles":       q_dict,
            "samples":         None,
            "model":           "statistical_fallback",
            "is_research_only": RESEARCH_ONLY,
        }

    def _zero_result(self, horizon: int, quantiles: List[float]) -> Dict:
        return {
            "median":          np.zeros(horizon),
            "mean":            np.zeros(horizon),
            "std":             np.zeros(horizon),
            "quantiles":       {q: np.zeros(horizon) for q in quantiles},
            "samples":         None,
            "model":           "zero_fallback",
            "is_research_only": RESEARCH_ONLY,
        }

    def unload(self) -> None:
        """Free GPU/CPU memory by unloading the model."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Chronos model unloaded from memory.")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Chronos Forecaster Smoke Test ===")
    print(f"Chronos available: {CHRONOS_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")

    # Generate synthetic NIFTY-like log return series
    rng = np.random.default_rng(42)
    synthetic_returns = pd.Series(rng.normal(0.0005, 0.012, 300))

    forecaster = ChronosForecaster(model_size="tiny")
    result = forecaster.forecast(synthetic_returns, horizon=5)

    print(f"\nModel: {result['model']}")
    print(f"Research Only: {result['is_research_only']}")
    print(f"5-day Median Return Forecast: {result['median']}")
    print(f"5-day Std:                    {result['std']}")
    print(f"\nQuantile Forecast (day 1):")
    for q, vals in result['quantiles'].items():
        print(f"  P{int(q*100):2d}: {vals[0]:.5f}")
    print("\n✓ Chronos smoke test complete.")
