"""Capital allocation across validated signals.

CRITICAL FIX: Added MA200 regime filter to reduce long exposure when price < MA200.
Simple but effective filter that would have saved many trend followers in 2022.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional
import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    symbol: str
    weight: float
    capital: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioAllocator:
    """Allocate capital to the strongest validated signals."""

    def __init__(self, max_position_pct: float = 0.05, enable_ma200_filter: bool = True, enable_risk_parity: bool = True, cash_buffer_pct: float = 0.10, max_single_stock_pct: float = 0.05, max_sector_pct: float = 0.30) -> None:
        self.max_position_pct = max_position_pct
        self.enable_ma200_filter = enable_ma200_filter
        self.enable_risk_parity = enable_risk_parity
        self.cash_buffer_pct = cash_buffer_pct
        self.max_single_stock_pct = max_single_stock_pct  # CRITICAL FIX: Limit single stock to 5%
        self.max_sector_pct = max_sector_pct  # CRITICAL FIX: Limit sector to 30%
    
    def get_ma200_multiplier(self, current_price: float, ma200: float) -> float:
        """
        Get position size multiplier based on MA200 regime filter.
        
        If price < MA200, reduce long exposure by 50%.
        This simple filter would have saved many trend followers in 2022.
        
        Args:
            current_price: Current market price
            ma200: 200-day moving average
            
        Returns:
            Position size multiplier (0.5 to 1.0)
        """
        if not self.enable_ma200_filter:
            return 1.0
        
        if current_price < ma200:
            # Price below MA200 - reduce long exposure
            return 0.5
        
        return 1.0

    def allocate_from_alpha_signals(
        self,
        capital: float,
        signals: Iterable[Any],
        regime_label: str | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        current_prices: Optional[Mapping[str, float]] = None,
        ma200_values: Optional[Mapping[str, float]] = None,
        current_vol: Optional[float] = None,
        target_vol: float = 0.15,
        symbol_volatilities: Optional[Mapping[str, float]] = None,
        symbol_sectors: Optional[Mapping[str, str]] = None,
    ) -> list[PortfolioAllocation]:
        """Allocate capital from standardized alpha signals."""
        investable_capital = capital * (1 - self.cash_buffer_pct)
        signal_list = list(signals)
        if not signal_list:
            return []

        regime_multiplier = self._regime_multiplier(regime_label)
        scored: list[tuple[str, float, float, Any]] = []
        for signal in signal_list:
            symbol = str(getattr(signal, "symbol", "UNKNOWN"))
            direction = float(getattr(signal, "direction", 0.0))
            strength = float(getattr(signal, "strength", 0.0))
            confidence = float(getattr(signal, "confidence", 0.0))
            score = abs(direction) * max(strength, 0.0) * max(confidence, 0.0)
            scored.append((symbol, score, direction, signal))

        if not scored:
            return []

        total_score = sum(score for _, score, _, _ in scored) or 1.0
        allocations: list[PortfolioAllocation] = []
        max_capital = investable_capital * self.max_position_pct

        vol_multiplier = 1.0
        if current_vol is not None and current_vol > 0:
            vol_multiplier = target_vol / max(current_vol, 0.05)
            vol_multiplier = min(2.0, max(0.5, vol_multiplier))

        inv_vol_weights: dict[str, float] = {}
        if self.enable_risk_parity and symbol_volatilities:
            total_inv = 0.0
            for symbol, _, _, _ in scored:
                inv_vol = 1.0 / max(float(symbol_volatilities.get(symbol, 0.15)), 0.01)
                inv_vol_weights[symbol] = inv_vol
                total_inv += inv_vol
            if total_inv > 0:
                for symbol in inv_vol_weights:
                    inv_vol_weights[symbol] /= total_inv

        weighted_scores = self._apply_correlation_penalty(
            {symbol: score for symbol, score, _, _ in scored},
            correlation_matrix,
        )

        sector_exposure: dict[str, float] = {}
        for symbol, score, direction, signal in sorted(scored, key=lambda item: item[1], reverse=True):
            weight = inv_vol_weights.get(symbol, score / total_score)
            weight = weighted_scores.get(symbol, weight)
            weight = max(weight, 0.0)

            # Kelly sizing proxy: use confidence and signal strength to cap exposure.
            kelly_fraction = min(0.25, max(0.01, float(getattr(signal, "confidence", 0.0)) * max(float(getattr(signal, "strength", 0.0)), 0.0) * 0.15))
            allocated = min(investable_capital * weight * kelly_fraction * regime_multiplier, max_capital)
            allocated *= vol_multiplier
            allocated = min(allocated, capital * self.max_single_stock_pct)

            if symbol_sectors and symbol in symbol_sectors:
                sector = symbol_sectors[symbol]
                sector_current = sector_exposure.get(sector, 0.0)
                sector_limit = capital * self.max_sector_pct
                if sector_current >= sector_limit:
                    continue
                allocated = min(allocated, sector_limit - sector_current)
                sector_exposure[sector] = sector_current + allocated

            if current_prices and ma200_values and symbol in current_prices and symbol in ma200_values:
                allocated *= self.get_ma200_multiplier(current_prices[symbol], ma200_values[symbol])

            allocations.append(PortfolioAllocation(symbol=symbol, weight=weight, capital=allocated, score=score * direction))

        return allocations

    def allocate(
        self,
        capital: float,
        signals: Iterable[Any],
        evidence_scores: Mapping[str, float] | None = None,
        current_prices: Optional[Mapping[str, float]] = None,
        ma200_values: Optional[Mapping[str, float]] = None,
        current_vol: Optional[float] = None,
        target_vol: float = 0.15,
        symbol_volatilities: Optional[Mapping[str, float]] = None,
        symbol_sectors: Optional[Mapping[str, str]] = None
    ) -> list[PortfolioAllocation]:
        """
        Allocate capital to the strongest validated signals.
        
        CRITICAL FIX: Added MA200 filter, volatility scaling, risk parity, cash buffer, and concentration limits.
        - MA200 filter: Reduce long exposure when price < MA200
        - Volatility scaling: Position size = base * target_vol / current_vol
        - Risk parity: Weight by inverse volatility
        - Cash buffer: Keep 10-20% cash to survive drawdowns
        - Concentration limits: Single stock max 5%, sector max 30%
        """
        # Apply cash buffer (CRITICAL FIX)
        investable_capital = capital * (1 - self.cash_buffer_pct)
        
        scored: list[tuple[str, float, Any]] = []
        for signal in signals:
            symbol = str(getattr(signal, "symbol", "UNKNOWN"))
            strength = float(getattr(signal, "strength", 0.0))
            mechanism = float(getattr(signal, "mechanism_score", 0.0))
            evidence = float((evidence_scores or {}).get(symbol, 0.5))
            score = max(0.0, strength * 0.4 + mechanism * 0.35 + evidence * 0.25)
            scored.append((symbol, score, signal))

        if not scored:
            return []

        total_score = sum(score for _, score, _ in scored) or 1.0
        allocations: list[PortfolioAllocation] = []
        max_capital = investable_capital * self.max_position_pct
        
        # Volatility scaling multiplier
        vol_multiplier = 1.0
        if current_vol is not None and current_vol > 0:
            vol_multiplier = target_vol / max(current_vol, 0.05)
            vol_multiplier = min(2.0, max(0.5, vol_multiplier))
        
        # Risk parity weights (CRITICAL FIX)
        if self.enable_risk_parity and symbol_volatilities:
            # Calculate inverse volatility weights
            inv_vol_weights = {}
            total_inv_vol = 0.0
            for symbol, _, _ in scored:
                vol = symbol_volatilities.get(symbol, 0.15)
                inv_vol = 1.0 / max(vol, 0.01)
                inv_vol_weights[symbol] = inv_vol
                total_inv_vol += inv_vol
            
            # Normalize to sum to 1
            for symbol in inv_vol_weights:
                inv_vol_weights[symbol] /= total_inv_vol
        
        # Track sector exposure (CRITICAL FIX)
        sector_exposure = {}
        
        for symbol, score, _ in sorted(scored, key=lambda item: item[1], reverse=True):
            # Use risk parity weights if enabled, otherwise use score-based weights
            if self.enable_risk_parity and symbol_volatilities and symbol in inv_vol_weights:
                weight = inv_vol_weights[symbol]
            else:
                weight = score / total_score
            
            allocated = min(investable_capital * weight, max_capital)
            
            # Apply single stock concentration limit (CRITICAL FIX)
            max_single_stock_capital = capital * self.max_single_stock_pct
            allocated = min(allocated, max_single_stock_capital)
            
            # Apply sector concentration limit (CRITICAL FIX)
            if symbol_sectors and symbol in symbol_sectors:
                sector = symbol_sectors[symbol]
                sector_current = sector_exposure.get(sector, 0.0)
                max_sector_capital = capital * self.max_sector_pct
                available_sector_capital = max_sector_capital - sector_current
                
                if available_sector_capital <= 0:
                    continue  # Skip this symbol, sector limit reached
                
                allocated = min(allocated, available_sector_capital)
                sector_exposure[sector] = sector_current + allocated
            
            # Apply MA200 filter
            if current_prices and ma200_values and symbol in current_prices and symbol in ma200_values:
                ma200_mult = self.get_ma200_multiplier(current_prices[symbol], ma200_values[symbol])
                allocated *= ma200_mult
            
            # Apply volatility scaling
            allocated *= vol_multiplier
            
            allocations.append(PortfolioAllocation(symbol=symbol, weight=weight, capital=allocated, score=score))
        return allocations

    def _apply_correlation_penalty(
        self,
        weights: dict[str, float],
        correlation_matrix: pd.DataFrame | None,
    ) -> dict[str, float]:
        if correlation_matrix is None or correlation_matrix.empty:
            return weights

        penalized = weights.copy()
        for symbol, weight in weights.items():
            penalty = 0.0
            for other, other_weight in weights.items():
                if other == symbol:
                    continue
                corr = 0.0
                if symbol in correlation_matrix.index and other in correlation_matrix.columns:
                    corr = float(correlation_matrix.loc[symbol, other])
                if corr > 0.5:
                    penalty += other_weight * (corr - 0.5)
            penalized[symbol] = max(0.0, weight - penalty)

        total = sum(penalized.values())
        return {k: (v / total if total > 0 else 0.0) for k, v in penalized.items()}

    def _regime_multiplier(self, regime_label: str | None) -> float:
        regime = (regime_label or "").lower()
        if regime in {"high_vol", "crisis"}:
            return 0.5
        if regime in {"bear_trend"}:
            return 0.75
        if regime in {"bull_trend", "sideways"}:
            return 1.0
        return 0.9
