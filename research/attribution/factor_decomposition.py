"""Factor Decomposition.

Factor decomposition breaks down portfolio returns into constituent factors
to understand:

1. What factors are driving returns (momentum, value, size, volatility, etc.)
2. How much alpha is genuine vs factor exposure
3. Factor timing (when to be long/short specific factors)
4. Factor crowding (when too many strategies are exposed to same factor)
5. Factor decay (when factor effectiveness changes)
6. Pure alpha extraction (returns orthogonal to known factors)

This transforms portfolio analysis from "black box returns" to transparent
factor-based understanding, enabling better risk management and strategy
development.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Types of factors."""
    MOMENTUM = "momentum"
    VALUE = "value"
    SIZE = "size"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    DIVIDEND_YIELD = "dividend_yield"
    LIQUIDITY = "liquidity"
    BETA = "beta"
    SECTOR = "sector"


@dataclass
class FactorExposure:
    """Exposure of a strategy or portfolio to a factor."""
    factor_name: str
    factor_type: FactorType
    exposure: float  # Beta to the factor
    t_stat: float  # Statistical significance
    p_value: float  # Statistical significance
    r_squared: float  # How much variance explained

    # Time-varying exposure
    exposure_history: List[float] = field(default_factory=list)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if exposure is statistically significant."""
        return self.p_value < alpha


@dataclass
class FactorDecompositionResult:
    """Result of factor decomposition."""
    strategy_name: str
    total_return: float
    alpha: float  # Return not explained by factors
    alpha_t_stat: float
    alpha_p_value: float

    # Factor exposures
    factor_exposures: Dict[str, FactorExposure] = field(default_factory=dict)

    # Model fit
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0

    # Factor timing
    factor_timing_scores: Dict[str, float] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.now)

    def get_pure_alpha(self) -> float:
        """Get the pure alpha (return orthogonal to factors)."""
        return self.alpha

    def get_factor_exposure_summary(self) -> Dict[str, float]:
        """Get summary of significant factor exposures."""
        return {
            name: exp.exposure
            for name, exp in self.factor_exposures.items()
            if exp.is_significant()
        }


class FactorModel:
    """Factor model for decomposition."""

    def __init__(self):
        # Factor definitions
        self.factor_definitions: Dict[str, FactorType] = {
            "momentum": FactorType.MOMENTUM,
            "value": FactorType.VALUE,
            "size": FactorType.SIZE,
            "volatility": FactorType.VOLATILITY,
            "quality": FactorType.QUALITY,
            "low_volatility": FactorType.LOW_VOLATILITY,
            "dividend_yield": FactorType.DIVIDEND_YIELD,
            "liquidity": FactorType.LIQUIDITY,
            "beta": FactorType.BETA,
        }

        # Historical factor data
        self.factor_returns: pd.DataFrame = pd.DataFrame()

        # Decomposition history
        self.decomposition_history: List[FactorDecompositionResult] = []

        logger.info("FactorModel initialized")

    def load_factor_data(self, factor_returns: pd.DataFrame) -> None:
        """Load historical factor returns.

        Args:
            factor_returns: DataFrame with factor returns (columns = factors)
        """
        self.factor_returns = factor_returns
        logger.info(f"Loaded factor data: {len(factor_returns)} observations, {len(factor_returns.columns)} factors")

    def calculate_factor_exposures(
        self,
        strategy_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> FactorDecompositionResult:
        """Calculate factor exposures for a strategy.

        Args:
            strategy_returns: Strategy returns (time series)
            factor_returns: Factor returns (if None, use loaded data)

        Returns:
            FactorDecompositionResult with exposures
        """
        factor_returns = factor_returns or self.factor_returns

        if factor_returns is None or len(factor_returns) == 0:
            logger.warning("No factor data available")
            return FactorDecompositionResult(
                strategy_name="unknown",
                total_return=strategy_returns.sum(),
                alpha=strategy_returns.sum(),
                alpha_t_stat=0.0,
                alpha_p_value=1.0,
            )

        # Align data
        common_index = strategy_returns.index.intersection(factor_returns.index)
        if len(common_index) < 30:
            logger.warning(f"Insufficient aligned data: {len(common_index)} observations")
            return FactorDecompositionResult(
                strategy_name="unknown",
                total_return=strategy_returns.sum(),
                alpha=strategy_returns.sum(),
                alpha_t_stat=0.0,
                alpha_p_value=1.0,
            )

        y = strategy_returns.loc[common_index]
        X = factor_returns.loc[common_index]

        # Run regression
        model = LinearRegression()
        model.fit(X, y)

        # Calculate alpha (intercept)
        alpha = model.intercept_

        # Calculate R-squared
        r_squared = model.score(X, y)
        n = len(y)
        k = X.shape[1]
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Calculate factor exposures and t-stats
        factor_exposures = {}
        for i, factor_name in enumerate(X.columns):
            exposure = model.coef_[i]

            # Calculate t-stat (simplified)
            residuals = y - model.predict(X)
            std_error = np.std(residuals) / np.sqrt(len(residuals))
            t_stat = exposure / (std_error + 1e-6)

            # Calculate p-value (two-tailed)
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(residuals) - k - 1))

            factor_exposures[factor_name] = FactorExposure(
                factor_name=factor_name,
                factor_type=self.factor_definitions.get(factor_name, FactorType.MOMENTUM),
                exposure=exposure,
                t_stat=t_stat,
                p_value=p_value,
                r_squared=r_squared,
            )

        result = FactorDecompositionResult(
            strategy_name="strategy",
            total_return=y.sum(),
            alpha=alpha,
            alpha_t_stat=alpha / (std_error + 1e-6),
            alpha_p_value=2 * (1 - stats.t.cdf(abs(alpha / (std_error + 1e-6)), df=len(residuals) - k - 1)),
            factor_exposures=factor_exposures,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
        )

        return result

    def decompose_portfolio(
        self,
        portfolio_returns: pd.Series,
        strategy_weights: Dict[str, float],
        strategy_returns: Dict[str, pd.Series]
    ) -> FactorDecompositionResult:
        """Decompose portfolio returns into factors.

        Args:
            portfolio_returns: Portfolio returns
            strategy_weights: Weights of each strategy in portfolio
            strategy_returns: Returns of each strategy

        Returns:
            FactorDecompositionResult for the portfolio
        """
        # Calculate weighted factor exposures
        weighted_exposures = {}

        for strategy_name, weight in strategy_weights.items():
            if strategy_name not in strategy_returns:
                continue

            strategy_decomp = self.calculate_factor_exposures(strategy_returns[strategy_name])

            for factor_name, exposure in strategy_decomp.factor_exposures.items():
                if factor_name not in weighted_exposures:
                    weighted_exposures[factor_name] = FactorExposure(
                        factor_name=factor_name,
                        factor_type=exposure.factor_type,
                        exposure=0.0,
                        t_stat=0.0,
                        p_value=1.0,
                        r_squared=0.0,
                    )

                weighted_exposures[factor_name].exposure += exposure.exposure * weight

        # Calculate portfolio alpha
        portfolio_decomp = self.calculate_factor_exposures(portfolio_returns)

        # Update with weighted exposures
        portfolio_decomp.factor_exposures = weighted_exposures

        return portfolio_decomp

    def detect_factor_crowding(
        self,
        strategy_decompositions: Dict[str, FactorDecompositionResult]
    ) -> Dict[str, List[str]]:
        """Detect when multiple strategies are crowded in same factors.

        Args:
            strategy_decompositions: Decomposition results by strategy

        Returns:
            Dictionary mapping factor to list of crowded strategies
        """
        factor_crowding: Dict[str, List[str]] = {}

        # Group strategies by significant factor exposures
        for strategy_name, decomp in strategy_decompositions.items():
            for factor_name, exposure in decomp.factor_exposures.items():
                if exposure.is_significant() and abs(exposure.exposure) > 0.5:
                    if factor_name not in factor_crowding:
                        factor_crowding[factor_name] = []
                    factor_crowding[factor_name].append(strategy_name)

        # Filter for factors with 3+ strategies
        factor_crowding = {
            factor: strategies
            for factor, strategies in factor_crowding.items()
            if len(strategies) >= 3
        }

        return factor_crowding

    def calculate_factor_timing_scores(
        self,
        factor_returns: pd.DataFrame,
        lookback_days: int = 60
    ) -> Dict[str, float]:
        """Calculate factor timing scores (which factors to be long/short).

        Args:
            factor_returns: Factor returns
            lookback_days: Lookback period for timing

        Returns:
            Dictionary mapping factor to timing score (-1 to 1)
        """
        timing_scores = {}

        for factor in factor_returns.columns:
            recent_returns = factor_returns[factor].tail(lookback_days)

            # Calculate momentum of the factor itself
            factor_momentum = recent_returns.sum()

            # Normalize to -1 to 1
            timing_score = np.tanh(factor_momentum * 10)
            timing_scores[factor] = timing_score

        return timing_scores

    def extract_pure_alpha(
        self,
        strategy_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """Extract pure alpha (returns orthogonal to factors).

        Args:
            strategy_returns: Strategy returns
            factor_returns: Factor returns

        Returns:
            Pure alpha returns
        """
        decomp = self.calculate_factor_exposures(strategy_returns, factor_returns)

        # Calculate factor contribution
        factor_returns = factor_returns or self.factor_returns
        if factor_returns is None or len(factor_returns) == 0:
            return strategy_returns

        common_index = strategy_returns.index.intersection(factor_returns.index)
        if len(common_index) == 0:
            return strategy_returns

        aligned_factor_returns = factor_returns.loc[common_index]

        # Calculate factor contribution
        factor_contribution = pd.Series(0.0, index=common_index)
        for factor_name, exposure in decomp.factor_exposures.items():
            if factor_name in aligned_factor_returns.columns:
                factor_contribution += aligned_factor_returns[factor_name] * exposure.exposure

        # Pure alpha = total returns - factor contribution
        aligned_strategy_returns = strategy_returns.loc[common_index]
        pure_alpha = aligned_strategy_returns - factor_contribution

        return pure_alpha

    def get_factor_report(self, decomp: FactorDecompositionResult) -> str:
        """Generate a factor decomposition report."""
        lines = []
        lines.append("=" * 70)
        lines.append("FACTOR DECOMPOSITION REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Strategy: {decomp.strategy_name}")
        lines.append(f"Total Return: {decomp.total_return:.2%}")
        lines.append(f"Pure Alpha: {decomp.alpha:.2%} (t-stat: {decomp.alpha_t_stat:.2f})")
        lines.append(f"R-Squared: {decomp.r_squared:.3f}")
        lines.append(f"Adjusted R-Squared: {decomp.adjusted_r_squared:.3f}")
        lines.append("")

        lines.append("FACTOR EXPOSURES")
        lines.append("-" * 70)
        for factor_name, exposure in sorted(
            decomp.factor_exposures.items(),
            key=lambda x: abs(x[1].exposure),
            reverse=True
        ):
            significance = "*" if exposure.is_significant() else ""
            lines.append(
                f"{factor_name}: {exposure.exposure:.3f} "
                f"(t-stat: {exposure.t_stat:.2f}, p-value: {exposure.p_value:.3f}){significance}"
            )
        lines.append("")

        lines.append("* = Significant at 5% level")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)
