"""
TimesFM Forecaster — Google TimesFM 2.0 Wrapper
================================================
Wraps Google's TimesFM 2.0 (500M parameter) foundation model for
multi-horizon time-series forecasting.

⚠  USAGE GUARD: Research forecasts only. NOT for direct trading signals.

Suitable for:
  - Multi-horizon return prediction (1d, 5d, 10d, 20d)
  - Regime transition probability forecasting
  - Volatility surface forecasting
  - Volume forecasting

Model: google/timesfm-2.0-500m-pytorch (~4GB on GPU, ~2GB float16)
Fallback: google/timesfm-1.0-200m-pytorch (~1.5GB)

Install:
    pip install timesfm>=1.2.6

Reference: https://huggingface.co/google/timesfm-2.0-500m-pytorch
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Guard timesfm import ──────────────────────────────────────────────────────
try:
    import timesfm
    TIMESFM_AVAILABLE = True
    logger.info("timesfm package loaded.")
except ImportError:
    timesfm = None
    TIMESFM_AVAILABLE = False
    logger.warning("timesfm not installed. Run: pip install timesfm>=1.2.6")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

RESEARCH_ONLY = True


class TimesFMForecaster:
    """
    Google TimesFM 2.0 wrapper for quantitative research.

    TimesFM achieves strong zero-shot performance across diverse time-series
    datasets. Key advantage over Chronos: multi-patch forecasting with
    variable context and frequency support.

    Example
    -------
    >>> tfm = TimesFMForecaster(model_version="2.0", use_gpu=False)
    >>> series = pd.Series(nifty_vix_history)
    >>> result = tfm.forecast(series, horizons=[1, 5, 20])
    >>> print(result['horizon_5']['median'])
    """

    is_research_only: bool = RESEARCH_ONLY

    SUPPORTED_FREQUENCIES = ["D", "W", "M"]  # Daily, Weekly, Monthly

    def __init__(
        self,
        model_version: str = "2.0",
        use_gpu: bool = True,
        backend: str = "pytorch",
    ):
        """
        Parameters
        ----------
        model_version : str
            '2.0' (500M, recommended) or '1.0' (200M, lighter).
        use_gpu : bool
            If False, forces CPU inference (slower but no GPU required).
        backend : str
            'pytorch' (default) or 'jax'.
        """
        self.model_version = model_version
        self.use_gpu = use_gpu and TORCH_AVAILABLE and (
            torch.cuda.is_available() if TORCH_AVAILABLE else False
        )
        self.backend = backend

        self._model = None
        self._context_len = 512
        self._patch_len = 32

        # Select model repo based on version
        if model_version == "2.0":
            self._repo_id = "google/timesfm-2.0-500m-pytorch"
        else:
            self._repo_id = "google/timesfm-1.0-200m-pytorch"

        logger.info(
            f"TimesFMForecaster configured | version={model_version} | "
            f"repo={self._repo_id} | gpu={self.use_gpu}"
        )

    def _load_model(self) -> bool:
        """Lazy-load TimesFM model. Returns True on success."""
        if self._model is not None:
            return True
        if not TIMESFM_AVAILABLE:
            logger.error("timesfm package not installed.")
            return False

        logger.info(f"Loading TimesFM {self.model_version} from {self._repo_id}...")
        t0 = time.time()
        try:
            hparams = timesfm.TimesFmHparams(
                backend=self.backend,
                per_core_batch_size=32,
                horizon_len=128,
                input_patch_len=self._patch_len,
                output_patch_len=self._patch_len,
                num_layers=50 if self.model_version == "2.0" else 20,
                model_dims=1280 if self.model_version == "2.0" else 512,
                use_positional_embedding=False,
            )
            checkpoint = timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self._repo_id
            )
            self._model = timesfm.TimesFm(
                hparams=hparams,
                checkpoint=checkpoint,
            )
            logger.info(f"TimesFM loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load TimesFM: {e}")
            self._model = None
            return False

    def forecast(
        self,
        series: pd.Series,
        horizons: Union[int, List[int]] = 5,
        frequency: str = "D",
    ) -> Dict:
        """
        Generate point forecasts for one or multiple horizons.

        Parameters
        ----------
        series : pd.Series
            Historical time series. Minimum 30 observations.
        horizons : int or list of int
            Forecast horizons. E.g., 5 for 5-day-ahead, or [1,5,20].
        frequency : str
            'D' (daily), 'W' (weekly), 'M' (monthly).

        Returns
        -------
        dict
            If horizons is int:
                {'median': float, 'model': str, 'is_research_only': bool}
            If horizons is list:
                {'horizon_N': {'median': float, ...}, ...}
        """
        if isinstance(horizons, int):
            horizons = [horizons]

        if len(series) < 20:
            logger.warning("Series too short (<20 points). Returning statistical baseline.")
            return self._statistical_fallback(series, horizons)

        max_horizon = max(horizons)

        if not self._load_model():
            return self._statistical_fallback(series, horizons)

        try:
            freq_int = self._frequency_to_int(frequency)
            forecast_input = [series.values.tolist()]
            freq_input = [freq_int]

            # TimesFM returns point forecasts up to max_horizon
            point_forecast, _ = self._model.forecast(
                inputs=forecast_input,
                freq=freq_input,
                prediction_length=max_horizon,
            )

            forecasts_arr = np.array(point_forecast[0])  # (max_horizon,)

            results = {}
            for h in horizons:
                val = float(forecasts_arr[h - 1]) if h <= len(forecasts_arr) else 0.0
                key = f"horizon_{h}"
                results[key] = {
                    "median":           val,
                    "point_forecast":   val,
                    "model":            self._repo_id,
                    "is_research_only": RESEARCH_ONLY,
                }

            return results

        except Exception as e:
            logger.error(f"TimesFM inference error: {e}")
            return self._statistical_fallback(series, horizons)

    def forecast_vol_regimes(
        self,
        returns: pd.Series,
        vol_series: pd.Series,
        horizons: List[int] = [1, 5, 20],
    ) -> Dict:
        """
        Forecast volatility at multiple horizons — useful for regime transitions.

        Returns separate forecasts for returns and volatility, each labeled by horizon.
        """
        return_forecasts = self.forecast(returns, horizons=horizons)
        vol_forecasts    = self.forecast(vol_series, horizons=horizons)

        combined = {}
        for h in horizons:
            key = f"horizon_{h}"
            combined[key] = {
                "return_forecast": return_forecasts.get(key, {}).get("median", 0.0),
                "vol_forecast":    vol_forecasts.get(key, {}).get("median", 0.0),
                "model":           self._repo_id,
                "is_research_only": RESEARCH_ONLY,
            }
        return combined

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _frequency_to_int(freq: str) -> int:
        """Convert frequency string to TimesFM integer code."""
        freq_map = {"D": 0, "W": 1, "M": 2, "Q": 3, "Y": 4}
        return freq_map.get(freq.upper(), 0)

    def _statistical_fallback(
        self,
        series: pd.Series,
        horizons: List[int],
    ) -> Dict:
        """Return exponentially-weighted mean as fallback."""
        arr = series.dropna().values
        ewm_mean = float(pd.Series(arr).ewm(span=20).mean().iloc[-1])
        ewm_std  = float(pd.Series(arr).ewm(span=20).std().iloc[-1])

        results = {}
        for h in horizons:
            results[f"horizon_{h}"] = {
                "median":           ewm_mean,
                "point_forecast":   ewm_mean,
                "std":              ewm_std,
                "model":            "ewm_fallback",
                "is_research_only": RESEARCH_ONLY,
            }
        return results

    def unload(self) -> None:
        """Free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("TimesFM model unloaded.")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TimesFM Forecaster Smoke Test ===")
    print(f"TimesFM available: {TIMESFM_AVAILABLE}")

    rng = np.random.default_rng(42)
    synthetic_vol = pd.Series(rng.uniform(10, 40, 252))   # Simulate India VIX

    tfm = TimesFMForecaster(model_version="1.0", use_gpu=False)
    result = tfm.forecast(synthetic_vol, horizons=[1, 5, 20])

    print("\nForecast results:")
    for key, val in result.items():
        print(f"  {key}: median={val['median']:.4f}  [model={val['model']}]")
    print("\n✓ TimesFM smoke test complete.")
