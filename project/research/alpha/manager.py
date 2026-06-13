"""Unified alpha manager with standardized signal formatting and combination."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .orb_zarattini import ORBSignal, ORBStrategy
from .putcall_parity_carry_alpha import PutCallParityCarryAlpha
from .vwap_strategy import VWAPSignal, VWAPStrategy
from features.pipeline import FeaturePipeline

try:  # pragma: no cover - optional dependency
    from .lightgbm_ensemble import LightGBMEnsemble
except Exception:  # pragma: no cover
    LightGBMEnsemble = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from .capped_vol_selling_alpha import CappedVolSellingAlpha
except Exception:  # pragma: no cover
    CappedVolSellingAlpha = None  # type: ignore[assignment]

# Import new alpha strategies
try:
    from alpha.momentum_strategies import get_momentum_signals
except Exception:
    get_momentum_signals = None

try:
    from alpha.mean_reversion_strategies import get_mean_reversion_signals
except Exception:
    get_mean_reversion_signals = None

try:
    from alpha.volatility_strategies import get_volatility_signals
except Exception:
    get_volatility_signals = None

try:
    from alpha.xgboost_predictor import get_xgboost_predictor
except Exception:
    get_xgboost_predictor = None

# Import theoretical foundation modules
try:
    from foundation.market_efficiency import MarketEfficiencyTests
    from foundation.agency_theory import AgencyTheoryMonitor, EventType
    FOUNDATION_AVAILABLE = True
except Exception:
    FOUNDATION_AVAILABLE = False
    MarketEfficiencyTests = None
    AgencyTheoryMonitor = None
    EventType = None


@dataclass(slots=True)
class AlphaSignal:
    """Standardized alpha signal emitted by the alpha manager."""

    strategy: str
    symbol: str
    direction: float
    strength: float
    confidence: float
    timestamp: datetime
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlphaManager:
    """Manage active strategies and combine them into a single actionable view."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}
        self.orb = ORBStrategy(self.config)
        self.vwap = VWAPStrategy(self.config)
        self.pcp = PutCallParityCarryAlpha()
        self.vol_carry = CappedVolSellingAlpha() if CappedVolSellingAlpha is not None else None
        self.feature_pipeline = FeaturePipeline()
        
        # Initialize new alpha strategy modules
        self.momentum_signals_fn = get_momentum_signals
        self.mean_reversion_signals_fn = get_mean_reversion_signals
        self.volatility_signals_fn = get_volatility_signals
        self.xgboost_predictor = get_xgboost_predictor() if get_xgboost_predictor else None
        
        # Initialize theoretical foundation modules
        if FOUNDATION_AVAILABLE and MarketEfficiencyTests is not None:
            self.efficiency_tests = MarketEfficiencyTests()
            self.efficiency_gate_enabled = self.config.get('efficiency_gate_enabled', True)
        else:
            self.efficiency_tests = None
            self.efficiency_gate_enabled = False
        
        if FOUNDATION_AVAILABLE and AgencyTheoryMonitor is not None:
            self.agency_monitor = AgencyTheoryMonitor()
            self.agency_signals_enabled = self.config.get('agency_signals_enabled', True)
        else:
            self.agency_monitor = None
            self.agency_signals_enabled = False
        
        # Updated strategy weights with new strategies
        self.strategy_weights = {
            "orb": 0.15,
            "vwap": 0.15,
            "pcp": 0.10,
            "vol_carry": 0.10,
            "momentum": 0.20,
            "mean_reversion": 0.15,
            "volatility": 0.15,
        }
        self.ensemble = (
            LightGBMEnsemble(list(self.strategy_weights)) if LightGBMEnsemble is not None else None
        )

    def generate_signals(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        *,
        regime_label: str | None = None,
        options_context: Mapping[str, Any] | None = None,
        flow_context: Mapping[str, Any] | None = None,
        order_book: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> list[AlphaSignal]:
        """Generate a standardized signal set for the requested symbol."""
        # Market hours guard
        try:
            from core.market_hours import is_market_open, market_status
            status = market_status()
            if not status["is_open"]:
                logger.warning(
                    "Market is CLOSED (%s). No signals generated. Next open: %s",
                    status["current_time"],
                    status["next_open"],
                )
                return []
        except ImportError:
            logger.warning("Market hours module not available, proceeding without market check")
        
        if market_data.empty:
            return []

        ts = timestamp or self._latest_timestamp(market_data)
        latest_bar = market_data.iloc[-1]
        signals: list[AlphaSignal] = []

        orb_signal, orb_position = self.orb.generate_signal(symbol, latest_bar)
        signals.append(self._standardize_signal("orb", symbol, orb_signal, orb_position, ts))

        vwap_signal, vwap_position = self.vwap.generate_signal(symbol, market_data, regime_label or "neutral")
        signals.append(self._standardize_signal("vwap", symbol, vwap_signal, vwap_position, ts))

        if options_context:
            pcp_signal = self._pcp_signal(symbol, options_context, ts)
            if pcp_signal is not None:
                signals.append(pcp_signal)

            vol_signal = self._vol_carry_signal(symbol, latest_bar, options_context, ts)
            if vol_signal is not None:
                signals.append(vol_signal)
        
        # Add momentum signals
        if self.momentum_signals_fn:
            try:
                momentum_signals = self._get_momentum_signals(symbol, market_data, ts)
                signals.extend(momentum_signals)
            except Exception as e:
                logger.warning(f"Failed to generate momentum signals: {e}")
        
        # Add mean reversion signals
        if self.mean_reversion_signals_fn:
            try:
                mr_signals = self._get_mean_reversion_signals(symbol, market_data, ts)
                signals.extend(mr_signals)
            except Exception as e:
                logger.warning(f"Failed to generate mean reversion signals: {e}")
        
        # Add volatility signals
        if self.volatility_signals_fn:
            try:
                vol_signals = self._get_volatility_signals(symbol, market_data, ts)
                signals.extend(vol_signals)
            except Exception as e:
                logger.warning(f"Failed to generate volatility signals: {e}")
        
        # Add XGBoost prediction
        if self.xgboost_predictor:
            try:
                xgb_signal = self._get_xgboost_signal(symbol, market_data, ts)
                if xgb_signal:
                    signals.append(xgb_signal)
            except Exception as e:
                logger.warning(f"Failed to generate XGBoost signal: {e}")
        
        # Add agency theory signals (event-driven alpha)
        if self.agency_signals_enabled and self.agency_monitor is not None:
            try:
                agency_signals = self._get_agency_signals(symbol, ts)
                signals.extend(agency_signals)
            except Exception as e:
                logger.warning(f"Failed to generate agency signals: {e}")

        return [signal for signal in signals if signal.strength != 0.0 or signal.confidence > 0.0]
    
    def check_market_efficiency_gate(
        self,
        symbol: str,
        market_data: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Check if market efficiency gate allows signal generation.
        
        Uses theoretical foundation market efficiency tests to determine
        if the market is efficient enough for alpha strategies to work.
        
        Args:
            symbol: Stock symbol
            market_data: Market data DataFrame
            
        Returns:
            Dictionary with gate decision and efficiency metrics
        """
        if not self.efficiency_gate_enabled or self.efficiency_tests is None:
            return {'gate_passed': True, 'reason': 'efficiency_gate_disabled'}
        
        if len(market_data) < 50:
            return {'gate_passed': True, 'reason': 'insufficient_data_for_test'}
        
        try:
            # Calculate returns
            prices = market_data['close'].values if 'close' in market_data.columns else market_data.iloc[:, 0].values
            returns = np.diff(np.log(prices))
            
            # Run variance ratio test
            vr_result = self.efficiency_tests.variance_ratio_test(prices, lags=[2, 4, 8])
            
            # Run runs test
            runs_result = self.efficiency_tests.runs_test(returns)
            
            # Calculate overall efficiency score
            test_results = {
                'variance_ratio': vr_result,
                'runs': runs_result,
            }
            efficiency_score = self.efficiency_tests.efficiency_score(test_results)
            
            # Gate decision: allow signals if market is not perfectly efficient
            # (some inefficiency is needed for alpha to work)
            gate_passed = efficiency_score['efficiency_score'] < 0.9
            
            return {
                'gate_passed': gate_passed,
                'efficiency_score': efficiency_score['efficiency_score'],
                'interpretation': efficiency_score['interpretation'],
                'variance_ratio': vr_result,
                'runs': runs_result,
                'reason': 'market_too_efficient' if not gate_passed else 'market_sufficiently_inefficient',
            }
            
        except Exception as e:
            logger.warning(f"Market efficiency gate check failed for {symbol}: {e}")
            return {'gate_passed': True, 'reason': 'efficiency_test_failed'}

    def combine_signals(
        self,
        signals: Iterable[AlphaSignal],
        *,
        regime_label: str | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        market_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Combine signals using regime weights and a lightweight correlation penalty."""
        signal_list = list(signals)
        if not signal_list:
            return {"combined_signal": 0.0, "weights": {}, "signals": []}

        regime_weights = self._regime_weights(regime_label)
        raw_weights: dict[str, float] = {}
        for signal in signal_list:
            base_weight = self.strategy_weights.get(signal.strategy, 0.0)
            regime_weight = regime_weights.get(signal.strategy, 1.0)
            raw_weights[signal.strategy] = base_weight * regime_weight * max(signal.confidence, 0.05)

        raw_weights = self._apply_correlation_penalty(raw_weights, correlation_matrix)
        total_weight = sum(raw_weights.values()) or 1.0
        weights = {k: float(v / total_weight) for k, v in raw_weights.items()}

        combined = 0.0
        for signal in signal_list:
            combined += signal.direction * signal.strength * signal.confidence * weights.get(signal.strategy, 0.0)

        if self.ensemble is not None and market_data is not None and not market_data.empty:
            try:
                model_signal = self.ensemble.predict(
                    {sig.strategy: sig.direction * sig.strength for sig in signal_list},
                    market_data,
                    regime_weights=regime_weights,
                )
                combined = 0.5 * combined + 0.5 * model_signal.combined_signal
            except Exception:
                pass

        return {
            "combined_signal": float(combined),
            "weights": weights,
            "signals": [signal.to_dict() for signal in signal_list],
            "regime_label": regime_label,
        }

    def fit_ensemble(
        self,
        alpha_signals: Mapping[str, list[float]],
        returns: list[float],
        market_data: pd.DataFrame,
    ) -> None:
        """Fit the optional LightGBM ensemble if the dependency is available."""
        if self.ensemble is None:
            return
        self.ensemble.fit(dict(alpha_signals), list(returns), market_data)

    def _standardize_signal(
        self,
        strategy: str,
        symbol: str,
        signal: Any,
        payload: Any,
        timestamp: datetime,
    ) -> AlphaSignal:
        direction_map = {
            ORBSignal.LONG_BREAKOUT: 1.0,
            ORBSignal.SHORT_BREAKOUT: -1.0,
            VWAPSignal.TREND_LONG: 1.0,
            VWAPSignal.REVERSION_LONG: 0.5,
            VWAPSignal.TREND_SHORT: -1.0,
            VWAPSignal.REVERSION_SHORT: -0.5,
            "LONG": 1.0,
            "SHORT": -1.0,
            "BUY": 1.0,
            "SELL": -1.0,
            None: 0.0,
        }
        direction = direction_map.get(signal, 0.0)
        strength = 0.0 if payload is None else min(1.0, max(0.0, abs(direction)))
        
        # CRITICAL FIX: Replace hardcoded confidence with signal-strength-based confidence
        # Instead of constant 0.7, vary confidence based on signal strength
        if signal is None or signal == ORBSignal.NO_SIGNAL or signal == VWAPSignal.NO_SIGNAL:
            confidence = 0.0
        else:
            # Confidence based on signal strength (clipped between 40% and 95%)
            raw_confidence = min(strength * 1.5, 0.95)  # Scale strength to confidence
            confidence = max(raw_confidence, 0.40)  # Floor at 40%
        
        metadata = {"raw_signal": str(signal), "payload": self._safe_payload(payload)}
        return AlphaSignal(strategy=strategy, symbol=symbol, direction=direction, strength=strength, confidence=confidence, timestamp=timestamp, metadata=metadata)

    def _pcp_signal(self, symbol: str, options_context: Mapping[str, Any], timestamp: datetime) -> AlphaSignal | None:
        call_price = float(options_context.get("call_price", 0.0) or 0.0)
        put_price = float(options_context.get("put_price", 0.0) or 0.0)
        strike = float(options_context.get("strike", 0.0) or 0.0)
        spot = float(options_context.get("spot", 0.0) or 0.0)
        dte = float(options_context.get("days_to_expiry", options_context.get("dte", 0.0)) or 0.0)
        risk_free_rate = float(options_context.get("risk_free_rate", 0.05) or 0.05)
        if min(call_price, put_price, strike, spot) <= 0 or dte <= 0:
            return None

        signal = self.pcp.generate_signal(
            symbol,
            spot=spot,
            strike=strike,
            call_price=call_price,
            put_price=put_price,
            time_to_expiry=dte / 365.0,
            risk_free_rate=risk_free_rate,
            timestamp=timestamp,
        )
        if signal is None:
            return None
        return AlphaSignal(
            strategy="pcp",
            symbol=symbol,
            direction=float(signal.signal),
            strength=min(1.0, abs(float(signal.signal))),
            confidence=float(signal.confidence),
            timestamp=timestamp,
            metadata={"regime": signal.regime.value, "expected_arbitrage": signal.expected_arbitrage},
        )

    def _vol_carry_signal(self, symbol: str, bar: pd.Series, options_context: Mapping[str, Any], timestamp: datetime) -> AlphaSignal | None:
        if self.vol_carry is None:
            return None
        atm_vol = float(options_context.get("atm_iv", options_context.get("iv", 0.0)) or 0.0)
        otm_put_vol = float(options_context.get("otm_put_iv", atm_vol) or atm_vol)
        realized_vol = float(options_context.get("realized_vol", options_context.get("rv", 0.0)) or 0.0)
        spot = float(options_context.get("spot", bar.get("close", bar.get("Close", 0.0))) or 0.0)
        if min(atm_vol, otm_put_vol, realized_vol, spot) <= 0:
            return None

        signal = self.vol_carry.generate_signal(
            symbol=symbol,
            spot=spot,
            atm_vol=atm_vol,
            otm_put_vol=otm_put_vol,
            realized_vol=realized_vol,
            timestamp=timestamp,
        )
        if signal is None:
            return None
        return AlphaSignal(
            strategy="vol_carry",
            symbol=symbol,
            direction=float(signal.signal),
            strength=min(1.0, abs(float(signal.signal))),
            confidence=float(signal.confidence),
            timestamp=timestamp,
            metadata={"regime": signal.regime.value, "max_loss": signal.max_loss},
        )

    def _regime_weights(self, regime_label: str | None) -> dict[str, float]:
        mapping = {
            "bull_trend": {"orb": 0.40, "vwap": 0.30, "pcp": 0.15, "vol_carry": 0.10},
            "bear_trend": {"orb": 0.20, "vwap": 0.40, "pcp": 0.20, "vol_carry": 0.15},
            "sideways": {"orb": 0.10, "vwap": 0.10, "pcp": 0.30, "vol_carry": 0.40},
            "high_vol": {"orb": 0.15, "vwap": 0.15, "pcp": 0.20, "vol_carry": 0.40},
        }
        weights = mapping.get((regime_label or "").lower(), self.strategy_weights)
        return {k: float(v) for k, v in weights.items()}

    def _apply_correlation_penalty(
        self,
        weights: dict[str, float],
        correlation_matrix: pd.DataFrame | None,
    ) -> dict[str, float]:
        if correlation_matrix is None or correlation_matrix.empty:
            return weights

        penalized = weights.copy()
        for strategy, base_weight in weights.items():
            penalty = 0.0
            for other, other_weight in weights.items():
                if other == strategy:
                    continue
                corr = 0.0
                if strategy in correlation_matrix.index and other in correlation_matrix.columns:
                    corr = float(correlation_matrix.loc[strategy, other])
                if corr > 0.5:
                    penalty += other_weight * (corr - 0.5)
            penalized[strategy] = float(max(0.01, base_weight - penalty))
        return penalized

    def _latest_timestamp(self, market_data: pd.DataFrame) -> datetime:
        if isinstance(market_data.index, pd.DatetimeIndex) and len(market_data.index):
            ts = market_data.index[-1].to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        return datetime.now(tz=timezone.utc)

    def _safe_payload(self, payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if hasattr(payload, "__dataclass_fields__"):
            return asdict(payload)
        if hasattr(payload, "__dict__"):
            return dict(payload.__dict__)
        if isinstance(payload, Mapping):
            return dict(payload)
        return {"value": str(payload)}
    
    def _get_agency_signals(self, symbol: str, timestamp: datetime) -> list[AlphaSignal]:
        """Generate agency theory signals (event-driven alpha)."""
        if not self.agency_monitor:
            return []
        
        try:
            # Get recent events for the symbol
            recent_events = self.agency_monitor.get_recent_events(symbol, days=30)
            
            if not recent_events:
                return []
            
            # Generate event-driven signals
            signals_dict = self.agency_monitor.event_driven_signals(recent_events, min_confidence=0.5)
            
            signals = []
            for event_symbol, signal_info in signals_dict.items():
                if event_symbol == symbol:
                    alpha_signal = AlphaSignal(
                        strategy=f"agency_{signal_info['signal']}",
                        symbol=symbol,
                        direction=1.0 if signal_info['signal'] == 'LONG' else -1.0,
                        strength=signal_info['confidence'],
                        confidence=signal_info['confidence'],
                        timestamp=timestamp,
                        metadata={
                            'event_count': signal_info['event_count'],
                            'latest_event_date': signal_info['latest_event_date'],
                            'expected_return': signal_info['expected_return'],
                        }
                    )
                    signals.append(alpha_signal)
            
            return signals
            
        except Exception as e:
            logger.warning(f"Failed to generate agency signals for {symbol}: {e}")
            return []
    
    def _get_momentum_signals(self, symbol: str, market_data: pd.DataFrame, timestamp: datetime) -> list[AlphaSignal]:
        """Generate momentum signals using new momentum strategies."""
        if not self.momentum_signals_fn:
            return []
        
        try:
            data_dict = {symbol: market_data}
            momentum_results = self.momentum_signals_fn(data_dict, strategies=["TSMOM", "VolatilityManagedMomentum"])
            
            signals = []
            for strategy_name, signal_list in momentum_results.items():
                for signal in signal_list:
                    if signal.symbol == symbol:
                        alpha_signal = AlphaSignal(
                            strategy=f"momentum_{strategy_name}",
                            symbol=symbol,
                            direction=signal.signal,
                            strength=abs(signal.signal),
                            confidence=signal.confidence,
                            timestamp=timestamp,
                            metadata=signal.metadata
                        )
                        signals.append(alpha_signal)
            return signals
        except Exception as e:
            logger.warning(f"Momentum signals failed: {e}")
            return []
    
    def _get_mean_reversion_signals(self, symbol: str, market_data: pd.DataFrame, timestamp: datetime) -> list[AlphaSignal]:
        """Generate mean reversion signals using new strategies."""
        if not self.mean_reversion_signals_fn:
            return []
        
        try:
            data_dict = {symbol: market_data}
            mr_results = self.mean_reversion_signals_fn(data_dict, strategies=["VWAPReversion", "BollingerReversion"])
            
            signals = []
            for strategy_name, signal_list in mr_results.items():
                for signal in signal_list:
                    if signal.symbol == symbol:
                        alpha_signal = AlphaSignal(
                            strategy=f"mean_reversion_{strategy_name}",
                            symbol=symbol,
                            direction=signal.signal,
                            strength=abs(signal.signal),
                            confidence=signal.confidence,
                            timestamp=timestamp,
                            metadata=signal.metadata
                        )
                        signals.append(alpha_signal)
            return signals
        except Exception as e:
            logger.warning(f"Mean reversion signals failed: {e}")
            return []
    
    def _get_volatility_signals(self, symbol: str, market_data: pd.DataFrame, timestamp: datetime) -> list[AlphaSignal]:
        """Generate volatility signals using new strategies."""
        if not self.volatility_signals_fn:
            return []
        
        try:
            data_dict = {symbol: market_data}
            vol_results = self.volatility_signals_fn(data_dict, strategies=["VRP", "VolatilityTargeting"])
            
            signals = []
            for strategy_name, signal_list in vol_results.items():
                for signal in signal_list:
                    if signal.symbol == symbol:
                        alpha_signal = AlphaSignal(
                            strategy=f"volatility_{strategy_name}",
                            symbol=symbol,
                            direction=signal.signal,
                            strength=abs(signal.signal),
                            confidence=signal.confidence,
                            timestamp=timestamp,
                            metadata=signal.metadata
                        )
                        signals.append(alpha_signal)
            return signals
        except Exception as e:
            logger.warning(f"Volatility signals failed: {e}")
            return []
    
    def _get_xgboost_signal(self, symbol: str, market_data: pd.DataFrame, timestamp: datetime) -> AlphaSignal | None:
        """Generate XGBoost prediction signal."""
        if not self.xgboost_predictor:
            return None
        
        try:
            prediction = self.xgboost_predictor.predict(market_data, symbol)
            if prediction:
                return AlphaSignal(
                    strategy="xgboost",
                    symbol=symbol,
                    direction=prediction.prediction_value,
                    strength=abs(prediction.prediction_value),
                    confidence=prediction.confidence,
                    timestamp=timestamp,
                    metadata={
                        "model_version": prediction.model_version,
                        "features_used": prediction.features_used
                    }
                )
        except Exception as e:
            logger.warning(f"XGBoost signal failed: {e}")
        return None
