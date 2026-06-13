"""
Portfolio Construction Engine
Converts signals into position sizes with risk-based allocation.
Consolidated with HRP and Black-Litterman implementations.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Iterable, Mapping, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSpec:
    """Output of position sizing for a single symbol."""
    symbol: str
    position_size: float          # in shares (positive = long, negative = short)
    capital_allocated: float      # INR
    expected_risk: float          # INR (1-day VaR)
    expected_return: float        # INR (expected daily PnL)
    weight: float                 # portfolio weight (decimal)


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    """Compatibility class for backwards compatibility with risk.py."""
    symbol: str
    weight: float
    capital: float
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlackLittermanResult:
    """Result of Black-Litterman computation"""
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    prior_returns: np.ndarray
    weights: np.ndarray
    view_contributions: Optional[np.ndarray] = None


@dataclass
class View:
    """Investor view on asset returns"""
    assets: List[str]  # Assets involved in the view
    weights: np.ndarray  # Weights of assets in the view (sum to 1)
    expected_return: float  # Expected return of the view
    confidence: float  # Confidence level (0-1)


class BlackLitterman:
    """
    Black-Litterman Portfolio Model
    """
    def __init__(self, tau: float = 0.025, risk_free_rate: float = 0.05):
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        
    def implied_returns(self, market_weights: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        market_variance = market_weights @ covariance_matrix @ market_weights
        market_excess_return = 0.05
        lambda_risk = market_excess_return / market_variance if market_variance > 0 else 2.5
        implied = lambda_risk * (covariance_matrix @ market_weights)
        return implied + self.risk_free_rate
    
    def black_litterman(
        self,
        prior_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        views: Optional[List[View]] = None,
        asset_names: Optional[List[str]] = None
    ) -> BlackLittermanResult:
        n_assets = len(prior_returns)
        if not views:
            weights = self._inverse_variance_weights(prior_returns, covariance_matrix)
            return BlackLittermanResult(
                posterior_returns=prior_returns,
                posterior_covariance=covariance_matrix,
                prior_returns=prior_returns,
                weights=weights
            )
            
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))
        
        # Build P, Q
        for i, view in enumerate(views):
            for asset_name, weight in zip(view.assets, view.weights):
                if asset_names is not None and asset_name in asset_names:
                    asset_idx = asset_names.index(asset_name)
                else:
                    try:
                        import re
                        digits = re.findall(r'\d+', str(asset_name))
                        asset_idx = int(digits[0]) if digits else 0
                    except Exception:
                        asset_idx = 0
                if asset_idx < n_assets:
                    P[i, asset_idx] = weight
            Q[i] = view.expected_return
            
        # Build omega matrix (view uncertainty)
        view_variances = np.diag(P @ covariance_matrix @ P.T).copy()
        for i, view in enumerate(views):
            confidence_factor = 1.0 / (view.confidence + 0.01)
            view_variances[i] *= confidence_factor
        omega = np.diag(view_variances)
        
        tau_sigma = self.tau * covariance_matrix
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)
        
        M1 = inv_tau_sigma + P.T @ inv_omega @ P
        M2 = inv_tau_sigma @ prior_returns + P.T @ inv_omega @ Q
        
        posterior_returns = np.linalg.inv(M1) @ M2
        posterior_covariance = np.linalg.inv(M1)
        weights = self._inverse_variance_weights(posterior_returns, posterior_covariance)
        
        return BlackLittermanResult(
            posterior_returns=posterior_returns,
            posterior_covariance=posterior_covariance,
            prior_returns=prior_returns,
            weights=weights
        )
        
    def _inverse_variance_weights(self, returns: np.ndarray, covariance_matrix: np.ndarray) -> np.ndarray:
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            weights = inv_cov @ returns
            if weights.sum() != 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(returns)) / len(returns)
        except np.linalg.LinAlgError:
            weights = np.ones(len(returns)) / len(returns)
        return weights


class PortfolioAllocator:
    """
    Converts signals to position sizes using multiple strategies.
    Consolidated implementation supporting HRP, Black-Litterman, Equal Weight.
    """
    
    def __init__(
        self,
        total_capital: float = 250_000_000.0,
        max_leverage: float = 1.0,
        max_position_pct: float = 0.05,
        enable_ma200_filter: bool = True,
        enable_risk_parity: bool = True,
        cash_buffer_pct: float = 0.10,
        max_single_stock_pct: float = 0.05,
        max_sector_pct: float = 0.30,
        enable_liquidity_decay: bool = True,
        capacity_limit: float = 200_000_000.0,
        participation_rate_cap: float = 0.10
    ) -> None:
        self.total_capital = total_capital
        self.current_nav = total_capital  # Current Net Asset Value (changes with PnL)
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.enable_ma200_filter = enable_ma200_filter
        self.enable_risk_parity = enable_risk_parity
        self.cash_buffer_pct = cash_buffer_pct
        self.max_single_stock_pct = max_single_stock_pct  # Limit single stock to 5%
        self.max_sector_pct = max_sector_pct  # Limit sector to 30%
        self.enable_liquidity_decay = enable_liquidity_decay
        self.capacity_limit = capacity_limit
        self.participation_rate_cap = participation_rate_cap
        
        self.current_positions: Dict[str, PositionSpec] = {}
        self.risk_per_trade: float = 0.01  # 1% risk per trade default
        self.black_litterman_model = BlackLitterman()

    def update_nav(self, new_nav: float) -> None:
        """Update current NAV after PnL changes."""
        self.current_nav = new_nav

    def get_current_capital(self) -> float:
        """Get current capital (NAV)."""
        return self.current_nav

    def equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate equal weights for a list of symbols (default baseline)."""
        if not symbols:
            return {}
        w = 1.0 / len(symbols)
        return {sym: w for sym in symbols}

    def hierarchical_risk_parity(self, covariance: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Hierarchical Risk Parity (HRP) weights (used for 10+ assets).
        """
        if covariance.empty:
            return {}
        symbols = list(covariance.index)
        if len(symbols) == 1:
            return {symbols[0]: 1.0}
            
        cov = covariance.values
        std = np.sqrt(np.diag(cov))
        std = np.maximum(std, 1e-8)
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        
        from scipy.cluster.hierarchy import linkage
        import scipy.spatial.distance as ssd
        try:
            condensed_dist = ssd.squareform(dist, checks=False)
            link = linkage(condensed_dist, method='single')
        except Exception as e:
            logger.warning(f"Linkage failed in HRP: {e}, falling back to equal weights")
            return self.equal_weight(symbols)
            
        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            while sort_ix.max() >= len(dist):
                sort_ix = sort_ix.to_frame()
                df0 = sort_ix[sort_ix[0] >= len(dist)]
                df0 = pd.DataFrame(link[df0[0].values - len(dist), :2], index=df0.index)
                sort_ix = sort_ix.merge(df0, how='left', left_index=True, right_index=True)
                sort_ix = sort_ix.stack().reset_index(level=1, drop=True).to_frame()
            return sort_ix.iloc[:, 0].tolist()
            
        try:
            sort_ix = get_quasi_diag(link)
            sorted_symbols = [symbols[i] for i in sort_ix]
        except Exception as e:
            logger.warning(f"Quasi-diagonalization failed in HRP: {e}")
            sorted_symbols = symbols
            
        def get_cluster_var(cov_df, cluster_items):
            cov_slice = cov_df.loc[cluster_items, cluster_items]
            diag = np.diag(cov_slice.values)
            diag = np.maximum(diag, 1e-8)
            inv_diag = 1.0 / diag
            inv_diag /= inv_diag.sum()
            cluster_var = inv_diag @ cov_slice.values @ inv_diag
            return cluster_var

        def recurse_bisection(w_series, cov_df, items):
            if len(items) <= 1:
                return
            mid = len(items) // 2
            c1 = items[:mid]
            c2 = items[mid:]
            v1 = get_cluster_var(cov_df, c1)
            v2 = get_cluster_var(cov_df, c2)
            alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            w_series[c1] *= alpha
            w_series[c2] *= (1.0 - alpha)
            recurse_bisection(w_series, cov_df, c1)
            recurse_bisection(w_series, cov_df, c2)
            
        w_series = pd.Series(1.0, index=sorted_symbols)
        recurse_bisection(w_series, covariance, sorted_symbols)
        
        # Normalize
        total_w = w_series.sum()
        if total_w > 0:
            w_series /= total_w
        return w_series.to_dict()

    def black_litterman(
        self,
        market_weights: np.ndarray,
        covariance: np.ndarray,
        views: Optional[List[View]] = None,
        asset_names: Optional[List[str]] = None
    ) -> BlackLittermanResult:
        """
        Calculate Black-Litterman posterior returns and weights (used with analyst views).
        """
        prior_returns = self.black_litterman_model.implied_returns(market_weights, covariance)
        return self.black_litterman_model.black_litterman(prior_returns, covariance, views, asset_names)

    def fixed_fractional(
        self,
        symbol: str,
        signal: float,
        entry_price: float,
        stop_loss_price: float
    ) -> PositionSpec:
        capital = self.get_current_capital()
        risk = self.risk_per_trade * capital
        
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return PositionSpec(symbol, 0, 0, 0, 0, 0)
            
        shares = int(risk / risk_per_share)
        if signal < 0:
            shares = -shares
            
        capital_allocated = abs(shares) * entry_price
        weight = capital_allocated / capital
        
        # Enforce max position size limit (5% default)
        max_position_value = capital * self.max_single_stock_pct
        if capital_allocated > max_position_value:
            capital_allocated = max_position_value
            shares = int(capital_allocated / entry_price)
            if signal < 0:
                shares = -shares
            weight = capital_allocated / capital
        
        return PositionSpec(
            symbol=symbol,
            position_size=shares,
            capital_allocated=capital_allocated,
            expected_risk=risk,
            expected_return=capital_allocated * 0.05,
            weight=weight
        )

    def kelly_fractional(
        self,
        symbol: str,
        win_prob: float,
        win_loss_ratio: float,
        entry_price: float,
        signal_strength: float
    ) -> PositionSpec:
        if win_loss_ratio == 0:
            return PositionSpec(symbol, 0, 0, 0, 0, 0)
            
        f_star = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Quarter Kelly
        weight = float(f_star * 0.25)
        weight = float(np.clip(weight, 0, self.max_leverage))
        weight = round(weight, 8)
        
        capital_allocated = float(self.get_current_capital() * weight)
        capital_allocated = round(capital_allocated, 4)
        
        # Enforce max position size limit
        max_position_value = self.get_current_capital() * self.max_single_stock_pct
        if capital_allocated > max_position_value:
            capital_allocated = max_position_value
            weight = capital_allocated / self.get_current_capital()
        
        shares = int(capital_allocated / entry_price)
        if signal_strength < 0:
            shares = -shares
            
        return PositionSpec(
            symbol=symbol,
            position_size=shares,
            capital_allocated=capital_allocated,
            expected_risk=round(float(capital_allocated * 0.02), 4),
            expected_return=round(float(capital_allocated * win_prob * win_loss_ratio), 4),
            weight=weight
        )

    def volatility_targeting(
        self,
        symbol: str,
        signal: float,
        entry_price: float,
        volatility: float,
        target_vol: float = 0.15
    ) -> PositionSpec:
        if volatility <= 0:
            return PositionSpec(symbol, 0, 0, 0, 0, 0)
            
        daily_target = target_vol / np.sqrt(252)
        daily_vol = volatility / 100.0
        if daily_vol <= 0:
            daily_vol = 0.01
            
        weight = (daily_target / daily_vol) * np.sign(signal)
        weight = np.clip(weight, -self.max_leverage, self.max_leverage)
        
        capital_allocated = self.get_current_capital() * abs(weight)
        shares = int(capital_allocated / entry_price)
        if signal < 0 and shares > 0:
            shares = -shares
            
        return PositionSpec(
            symbol=symbol,
            position_size=shares,
            capital_allocated=capital_allocated,
            expected_risk=capital_allocated * 0.02,
            expected_return=capital_allocated * abs(weight),
            weight=weight
        )

    def risk_parity(
        self,
        symbols: List[str],
        signals: List[float],
        volatilities: List[float],
        correlations: np.ndarray
    ) -> List[PositionSpec]:
        if not symbols:
            return []
            
        raw_weights = 1.0 / np.maximum(volatilities, 0.01)
        raw_weights /= np.sum(raw_weights)
        
        for i, sig in enumerate(signals):
            if sig < 0:
                raw_weights[i] = -raw_weights[i]
                
        raw_weights = raw_weights / np.sum(np.abs(raw_weights))
        
        positions = []
        for i, sym in enumerate(symbols):
            capital = self.get_current_capital() * abs(raw_weights[i])
            shares = int(capital / 100)
            positions.append(PositionSpec(
                symbol=sym,
                position_size=shares if raw_weights[i] > 0 else -shares,
                capital_allocated=capital,
                expected_risk=capital * volatilities[i] / 100.0,
                expected_return=capital * signals[i] / 100.0,
                weight=raw_weights[i]
            ))
        return positions

    def apply_sector_limits(
        self,
        positions: List[PositionSpec],
        sector_map: Dict[str, str],
        sector_limits: Dict[str, float]
    ) -> List[PositionSpec]:
        sector_exposure = {}
        for pos in positions:
            sector = sector_map.get(pos.symbol, 'Other')
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + abs(pos.weight)
            
        adjusted = []
        for pos in positions:
            sector = sector_map.get(pos.symbol, 'Other')
            limit = sector_limits.get(sector, 0.5)
            current = sector_exposure.get(sector, 0.0)
            if current > limit:
                scale = limit / current
                pos.weight *= scale
                pos.capital_allocated *= scale
                pos.position_size = int(pos.position_size * scale)
            adjusted.append(pos)
        return adjusted

    def apply_position_limits(
        self,
        positions: List[PositionSpec],
        max_position_weight: float = 0.10
    ) -> List[PositionSpec]:
        for pos in positions:
            if abs(pos.weight) > max_position_weight:
                scale = max_position_weight / abs(pos.weight)
                pos.weight *= scale
                pos.capital_allocated *= scale
                pos.position_size = int(pos.position_size * scale)
        return positions

    def portfolio_heat(
        self,
        positions: List[PositionSpec],
        cov_matrix: np.ndarray,
        max_portfolio_var: float = 0.02
    ) -> float:
        if not positions or cov_matrix.size == 0:
            return 1.0
            
        weights = np.array([pos.weight for pos in positions])
        if len(weights) != cov_matrix.shape[0]:
            return 1.0
            
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_std = np.sqrt(max(portfolio_var, 0.0))
        portfolio_var_95 = portfolio_std * 1.645
        
        if portfolio_var_95 > max_portfolio_var:
            return max_portfolio_var / portfolio_var_95
        return 1.0

    def apply_portfolio_heat(
        self,
        positions: List[PositionSpec],
        cov_matrix: np.ndarray,
        max_portfolio_var: float = 0.02
    ) -> List[PositionSpec]:
        scale = self.portfolio_heat(positions, cov_matrix, max_portfolio_var)
        if scale < 1.0:
            for pos in positions:
                pos.weight *= scale
                pos.capital_allocated *= scale
                pos.position_size = int(pos.position_size * scale)
                pos.expected_risk *= scale
        return positions

    def calculate_liquidity_decay(self, position_value: float, symbol_volume: float) -> float:
        if not self.enable_liquidity_decay or symbol_volume <= 0:
            return 1.0
        participation_rate = position_value / symbol_volume
        return float(np.exp(-2.0 * participation_rate))

    def check_participation_rate(self, position_value: float, symbol_volume: float) -> bool:
        if symbol_volume <= 0:
            return True
        participation_rate = position_value / symbol_volume
        return participation_rate <= self.participation_rate_cap

    def get_ma200_multiplier(self, current_price: float, ma200: float) -> float:
        if not self.enable_ma200_filter:
            return 1.0
        if current_price < ma200:
            return 0.5
        return 1.0

    def _signal_value(self, signal: Any, name: str, default: Any = 0.0) -> Any:
        if isinstance(signal, Mapping):
            return signal.get(name, default)
        return getattr(signal, name, default)

    def _signal_strength(self, signal: Any) -> float:
        raw_strength = self._signal_value(signal, "strength", None)
        if raw_strength is not None:
            return max(float(raw_strength), 0.0)
        rv = float(self._signal_value(signal, "rv", 0.0) or 0.0)
        if rv > 0:
            return float(np.clip(rv / 3.0, 0.0, 1.0))
        expected_return = abs(float(self._signal_value(signal, "expected_return", 0.0) or 0.0))
        return float(np.clip(expected_return / 0.03, 0.0, 1.0))

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
        investable_capital = capital * (1 - self.cash_buffer_pct)
        signal_list = list(signals)
        if not signal_list:
            return []

        regime_multiplier = self._regime_multiplier(regime_label)
        scored: list[tuple[str, float, float, Any]] = []
        for signal in signal_list:
            symbol = str(self._signal_value(signal, "symbol", "UNKNOWN"))
            direction = float(self._signal_value(signal, "direction", 0.0) or 0.0)
            strength = self._signal_strength(signal)
            confidence = float(self._signal_value(signal, "confidence", 0.0) or 0.0)
            score = abs(direction) * max(strength, 0.0) * max(confidence, 0.0)
            if symbol != "UNKNOWN" and score > 0:
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
            rp_weight = inv_vol_weights.get(symbol, score / total_score)
            score_weight = weighted_scores.get(symbol, rp_weight)
            weight = 0.5 * rp_weight + 0.5 * score_weight
            weight = max(weight, 0.0)

            confidence = float(self._signal_value(signal, "confidence", 0.0) or 0.0)
            strength = self._signal_strength(signal)
            kelly_fraction = min(0.25, max(0.01, confidence * strength * 0.15))
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
        *args,
        **kwargs
    ) -> list[PortfolioAllocation]:
        capital = None
        signals = None
        
        if args and isinstance(args[0], (int, float)):
            capital = args[0]
            if len(args) > 1:
                signals = args[1]
        elif args:
            signals = args[0]
            if len(args) > 1:
                capital = args[1]
                
        if capital is None:
            capital = kwargs.get("capital", self.current_nav)
        if signals is None:
            signals = kwargs.get("signals", [])
            
        evidence_scores = kwargs.get("evidence_scores", None)
        current_prices = kwargs.get("current_prices", None)
        ma200_values = kwargs.get("ma200_values", None)
        current_vol = kwargs.get("current_vol", None)
        target_vol = kwargs.get("target_vol", 0.15)
        symbol_volatilities = kwargs.get("symbol_volatilities", None)
        symbol_sectors = kwargs.get("symbol_sectors", None)
        price_history = kwargs.get("price_history", None)
        method = kwargs.get("method", "default")
        
        investable_capital = capital * (1 - self.cash_buffer_pct)
        scored = []
        for signal in signals:
            symbol = str(self._signal_value(signal, "symbol", "UNKNOWN"))
            direction = float(self._signal_value(signal, "direction", 0.0) or 0.0)
            strength = self._signal_strength(signal)
            confidence = float(self._signal_value(signal, "confidence", 0.0) or 0.0)
            mechanism = float(self._signal_value(signal, "mechanism_score", confidence) or 0.0)
            evidence = float((evidence_scores or {}).get(symbol, 0.5))
            score = max(0.0, abs(direction or 1.0) * (strength * 0.4 + mechanism * 0.35 + evidence * 0.25))
            if symbol != "UNKNOWN" and score > 0:
                scored.append((symbol, score, signal))

        if not scored:
            return []

        total_score = sum(score for _, score, _ in scored) or 1.0
        max_capital = investable_capital * self.max_position_pct
        
        # HRP Path
        if method == "hrp":
            if price_history is not None:
                returns = price_history.pct_change().dropna()
                covariance = returns.cov()
            else:
                covariance = kwargs.get("covariance", pd.DataFrame())
            
            weights_dict = self.hierarchical_risk_parity(covariance)
            
            allocations = []
            for symbol, weight in weights_dict.items():
                allocated = min(investable_capital * weight, max_capital)
                allocated = min(allocated, capital * self.max_single_stock_pct)
                direction = 1.0
                for sig in signals:
                    if self._signal_value(sig, "symbol") == symbol:
                        direction = float(self._signal_value(sig, "direction", 1.0) or 1.0)
                        break
                allocations.append(PortfolioAllocation(
                    symbol=symbol,
                    weight=weight,
                    capital=allocated,
                    score=weight * direction
                ))
            return allocations
            
        # Black-Litterman Path
        elif method == "black_litterman":
            if price_history is not None:
                returns = price_history.pct_change().dropna()
                covariance = returns.cov()
            else:
                covariance = kwargs.get("covariance", pd.DataFrame())
                
            if isinstance(covariance, pd.DataFrame):
                asset_names = list(covariance.columns)
                cov_values = covariance.values
            else:
                asset_names = [s[0] for s in scored]
                cov_values = covariance
                
            views = kwargs.get("views", None)
            bl_views = []
            if isinstance(views, dict):
                for asset, target_ret in views.items():
                    bl_views.append(View(
                        assets=[asset],
                        weights=np.array([1.0]),
                        expected_return=target_ret,
                        confidence=0.5
                    ))
            elif isinstance(views, list):
                bl_views = views
                
            market_weights = kwargs.get("market_weights", None)
            if market_weights is None:
                market_weights = np.ones(len(asset_names)) / len(asset_names)
                
            bl_res = self.black_litterman(market_weights, cov_values, bl_views, asset_names)
            
            allocations = []
            for i, symbol in enumerate(asset_names):
                weight = bl_res.weights[i]
                allocated = min(investable_capital * abs(weight), max_capital)
                allocated = min(allocated, capital * self.max_single_stock_pct)
                direction = 1.0
                for sig in signals:
                    if self._signal_value(sig, "symbol") == symbol:
                        direction = float(self._signal_value(sig, "direction", 1.0) or 1.0)
                        break
                allocations.append(PortfolioAllocation(
                    symbol=symbol,
                    weight=weight,
                    capital=allocated,
                    score=weight * direction
                ))
            return allocations

        # Default Allocations Path
        allocations = []
        vol_multiplier = 1.0
        if current_vol is not None and current_vol > 0:
            vol_multiplier = target_vol / max(current_vol, 0.05)
            vol_multiplier = min(2.0, max(0.5, vol_multiplier))
            
        inv_vol_weights = {}
        if self.enable_risk_parity and symbol_volatilities:
            total_inv_vol = 0.0
            for symbol, _, _ in scored:
                vol = symbol_volatilities.get(symbol, 0.15)
                inv_vol = 1.0 / max(vol, 0.01)
                inv_vol_weights[symbol] = inv_vol
                total_inv_vol += inv_vol
            for symbol in inv_vol_weights:
                inv_vol_weights[symbol] /= total_inv_vol
                
        sector_exposure = {}
        for symbol, score, _ in sorted(scored, key=lambda item: item[1], reverse=True):
            if self.enable_risk_parity and symbol_volatilities and symbol in inv_vol_weights:
                weight = inv_vol_weights[symbol]
            else:
                weight = score / total_score
                
            allocated = min(investable_capital * weight, max_capital)
            allocated = min(allocated, capital * self.max_single_stock_pct)
            
            if symbol_sectors and symbol in symbol_sectors:
                sector = symbol_sectors[symbol]
                sector_current = sector_exposure.get(sector, 0.0)
                max_sector_capital = capital * self.max_sector_pct
                available_sector_capital = max_sector_capital - sector_current
                if available_sector_capital <= 0:
                    continue
                allocated = min(allocated, available_sector_capital)
                sector_exposure[sector] = sector_current + allocated
                
            if current_prices and ma200_values and symbol in current_prices and symbol in ma200_values:
                ma200_mult = self.get_ma200_multiplier(current_prices[symbol], ma200_values[symbol])
                allocated *= ma200_mult
                
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


KellyVolatilityAllocator = PortfolioAllocator
