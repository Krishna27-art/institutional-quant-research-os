"""
Risk Checks - Pre-trade and post-trade risk validation
Enhanced with theoretical foundation limits to arbitrage constraints
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import theoretical foundation modules
try:
    from foundation.limits_to_arbitrage import LimitsToArbitrage, PositionConstraints, VolatilityRegime
    FOUNDATION_AVAILABLE = True
except Exception:
    FOUNDATION_AVAILABLE = False
    LimitsToArbitrage = None
    PositionConstraints = None
    VolatilityRegime = None


class RiskLimitType(Enum):
    """Types of risk limits"""
    POSITION_SIZE = "position_size"
    SECTOR_EXPOSURE = "sector_exposure"
    LEVERAGE = "leverage"
    CONCENTRATION = "concentration"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    TURNOVER = "turnover"


class PreTradeRiskChecker:
    """
    Pre-trade risk checks before order execution
    Enhanced with theoretical foundation limits to arbitrage constraints
    
    Checks:
    - Position size limits
    - Sector exposure limits
    - Capital availability
    - Leverage limits
    - Concentration limits
    - Liquidity constraints (from theoretical foundation)
    - Volatility constraints (from theoretical foundation)
    - Correlation constraints (from theoretical foundation)
    """
    
    def __init__(self):
        self.limits = {
            RiskLimitType.POSITION_SIZE: 0.05,  # 5% of portfolio
            RiskLimitType.SECTOR_EXPOSURE: 0.30,  # 30% of portfolio
            RiskLimitType.LEVERAGE: 2.0,  # 2x leverage
            RiskLimitType.CONCENTRATION: 0.10,  # 10% in single position
        }
        self.positions: Dict[str, float] = {}
        self.sector_exposure: Dict[str, float] = {}
        self.current_capital: float = 1_000_000
        self.current_leverage: float = 1.0
        
        # Initialize theoretical foundation modules
        if FOUNDATION_AVAILABLE and LimitsToArbitrage is not None:
            self.limits_to_arbitrage = LimitsToArbitrage()
            self.constraints = PositionConstraints()
            logger.info("PreTradeRiskChecker initialized with limits to arbitrage constraints")
        else:
            self.limits_to_arbitrage = None
            self.constraints = None
            logger.info("PreTradeRiskChecker initialized (basic mode)")
    
    def set_limit(self, limit_type: RiskLimitType, value: float) -> None:
        """Set a risk limit"""
        self.limits[limit_type] = value
    
    def update_position(self, symbol: str, quantity: float, sector: Optional[str] = None) -> None:
        """Update current position"""
        self.positions[symbol] = quantity
        if sector:
            self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + quantity
    
    def check(self, signal: Dict) -> Dict:
        """
        Perform all pre-trade risk checks
        
        Args:
            signal: Trading signal
            
        Returns:
            Dict with approval status and reason if rejected
        """
        checks = [
            self._check_position_size(signal),
            self._check_sector_exposure(signal),
            self._check_capital(signal),
            self._check_leverage(signal),
            self._check_concentration(signal)
        ]
        
        # Add limits to arbitrage checks if available
        if self.limits_to_arbitrage is not None:
            checks.append(self._check_limits_to_arbitrage(signal))
        
        failed_checks = [c for c in checks if not c['passed']]
        
        if failed_checks:
            return {
                'approved': False,
                'reason': failed_checks[0]['reason'],
                'failed_checks': failed_checks
            }
        
        return {'approved': True, 'reason': 'All checks passed'}
    
    def _check_limits_to_arbitrage(self, signal: Dict) -> Dict:
        """
        Check limits to arbitrage constraints using theoretical foundation.
        
        Args:
            signal: Trading signal
            
        Returns:
            Dict with check result
        """
        if self.limits_to_arbitrage is None:
            return {'passed': True, 'reason': 'limits_to_arbitrage_not_available'}
        
        try:
            symbol = signal['symbol']
            quantity = signal['quantity']
            price = signal.get('price', 100.0)
            signal_strength = signal.get('strength', 0.5)
            
            # Get market data from signal if available
            daily_volume = signal.get('daily_volume', 1_000_000)
            volatility = signal.get('volatility', 0.2)
            correlation = signal.get('correlation', 0.0)
            
            # Calculate capacity-limited position size
            base_size = abs(quantity) * price
            adjusted_size, constraint_info = self.limits_to_arbitrage.capacity_limited_size(
                signal=signal_strength,
                liquidity=daily_volume,
                vol_regime=self._classify_volatility_regime(volatility),
                correlation=correlation,
                portfolio_value=self.current_capital,
                base_kelly=0.02
            )
            
            # Check if requested position exceeds capacity-limited size
            if base_size > adjusted_size:
                # Calculate adjusted quantity
                adjusted_quantity = (adjusted_size / price) * (1 if quantity > 0 else -1)
                return {
                    'passed': False,
                    'reason': f'Position size exceeds capacity limit: {base_size:.2f} > {adjusted_size:.2f}',
                    'constraint_info': constraint_info,
                    'suggested_quantity': adjusted_quantity
                }
            
            return {'passed': True, 'reason': 'limits_to_arbitrage_passed', 'constraint_info': constraint_info}
            
        except Exception as e:
            logger.warning(f"Limits to arbitrage check failed: {e}")
            return {'passed': True, 'reason': 'limits_to_arbitrage_check_failed'}
    
    def _classify_volatility_regime(self, volatility: float) -> VolatilityRegime:
        """Classify volatility into regime."""
        if not FOUNDATION_AVAILABLE or VolatilityRegime is None:
            # Simple fallback classification
            if volatility < 0.15:
                return VolatilityRegime.LOW if VolatilityRegime else None
            elif volatility < 0.25:
                return VolatilityRegime.NORMAL if VolatilityRegime else None
            elif volatility < 0.40:
                return VolatilityRegime.HIGH if VolatilityRegime else None
            else:
                return VolatilityRegime.EXTREME if VolatilityRegime else None
        else:
            # Use foundation module classification
            if volatility < 0.15:
                return VolatilityRegime.LOW
            elif volatility < 0.25:
                return VolatilityRegime.NORMAL
            elif volatility < 0.40:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME
    
    def _check_position_size(self, signal: Dict) -> Dict:
        """Check position size limit"""
        symbol = signal['symbol']
        quantity = signal['quantity']
        price = signal.get('price', 100.0)
        
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        position_value = abs(new_position) * price
        portfolio_value = self.current_capital  # Simplified
        
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
        limit = self.limits[RiskLimitType.POSITION_SIZE]
        
        if position_pct > limit:
            return {
                'passed': False,
                'reason': f'Position size {position_pct:.2%} exceeds limit {limit:.2%}',
                'check': 'position_size'
            }
        
        return {'passed': True, 'check': 'position_size'}
    
    def _check_sector_exposure(self, signal: Dict) -> Dict:
        """Check sector exposure limit"""
        sector = signal.get('sector')
        if not sector:
            return {'passed': True, 'check': 'sector_exposure'}
        
        quantity = signal['quantity']
        price = signal.get('price', 100.0)
        trade_value = abs(quantity) * price
        
        current_exposure = self.sector_exposure.get(sector, 0)
        new_exposure = current_exposure + trade_value
        portfolio_value = self.current_capital
        
        exposure_pct = new_exposure / portfolio_value if portfolio_value > 0 else 0
        limit = self.limits[RiskLimitType.SECTOR_EXPOSURE]
        
        if exposure_pct > limit:
            return {
                'passed': False,
                'reason': f'Sector exposure {exposure_pct:.2%} exceeds limit {limit:.2%}',
                'check': 'sector_exposure'
            }
        
        return {'passed': True, 'check': 'sector_exposure'}
    
    def _check_capital(self, signal: Dict) -> Dict:
        """Check capital availability"""
        quantity = signal['quantity']
        price = signal.get('price', 100.0)
        direction = signal['direction']
        
        if direction == 'BUY':
            cost = quantity * price
            if cost > self.current_capital:
                return {
                    'passed': False,
                    'reason': f'Insufficient capital: need {cost:.2f}, have {self.current_capital:.2f}',
                    'check': 'capital'
                }
        
        return {'passed': True, 'check': 'capital'}
    
    def _check_leverage(self, signal: Dict) -> Dict:
        """Check leverage limit"""
        quantity = signal['quantity']
        price = signal.get('price', 100.0)
        direction = signal['direction']
        
        if direction == 'BUY':
            new_exposure = quantity * price
            total_exposure = sum(abs(p * price) for p in self.positions.values()) + new_exposure
            portfolio_value = self.current_capital
            
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            limit = self.limits[RiskLimitType.LEVERAGE]
            
            if leverage > limit:
                return {
                    'passed': False,
                    'reason': f'Leverage {leverage:.2f}x exceeds limit {limit:.2f}x',
                    'check': 'leverage'
                }
        
        return {'passed': True, 'check': 'leverage'}
    
    def _check_concentration(self, signal: Dict) -> Dict:
        """Check concentration limit"""
        symbol = signal['symbol']
        quantity = signal['quantity']
        price = signal.get('price', 100.0)
        
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        position_value = abs(new_position) * price
        
        total_gross_exposure = sum(abs(p * price) for p in self.positions.values()) + position_value
        
        if total_gross_exposure > 0:
            concentration = position_value / total_gross_exposure
            limit = self.limits[RiskLimitType.CONCENTRATION]
            
            if concentration > limit:
                return {
                    'passed': False,
                    'reason': f'Concentration {concentration:.2%} exceeds limit {limit:.2%}',
                    'check': 'concentration'
                }
        
        return {'passed': True, 'check': 'concentration'}


class PostTradeRiskChecker:
    """
    Post-trade risk checks after order execution
    
    Checks:
    - Daily loss limits
    - Drawdown monitoring
    - Exposure drift
    - Turnover limits
    """
    
    def __init__(self):
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.start_of_day_capital: float = 1_000_000
        self.peak_capital: float = 1_000_000
        self.current_capital: float = 1_000_000
        self.daily_turnover: float = 0.0
        
        self.limits = {
            RiskLimitType.DAILY_LOSS: 0.03,  # 3% daily loss limit
            RiskLimitType.DRAWDOWN: 0.15,  # 15% max drawdown
            RiskLimitType.TURNOVER: 0.50,  # 50% daily turnover limit
        }
    
    def set_limit(self, limit_type: RiskLimitType, value: float) -> None:
        """Set a risk limit"""
        self.limits[limit_type] = value
    
    def check(self, execution_result: Dict) -> Dict:
        """
        Perform post-trade risk checks
        
        Args:
            execution_result: Execution result from order
            
        Returns:
            Dict with check results
        """
        checks = [
            self._check_daily_loss(),
            self._check_drawdown(),
            self._check_turnover()
        ]
        
        warnings = [c for c in checks if not c['passed']]
        
        return {
            'passed': len(warnings) == 0,
            'warnings': warnings,
            'daily_pnl': self.daily_pnl,
            'current_drawdown': self._calculate_drawdown()
        }
    
    def update_pnl(self, pnl: float) -> None:
        """Update daily PnL"""
        self.daily_pnl += pnl
        self.current_capital += pnl
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
    
    def update_turnover(self, turnover: float) -> None:
        """Update daily turnover"""
        self.daily_turnover += turnover
    
    def _check_daily_loss(self) -> Dict:
        """Check daily loss limit"""
        loss_pct = -self.daily_pnl / self.start_of_day_capital if self.daily_pnl < 0 else 0
        limit = self.limits[RiskLimitType.DAILY_LOSS]
        
        if loss_pct > limit:
            return {
                'passed': False,
                'reason': f'Daily loss {loss_pct:.2%} exceeds limit {limit:.2%}',
                'check': 'daily_loss',
                'severity': 'critical'
            }
        
        return {'passed': True, 'check': 'daily_loss'}
    
    def _check_drawdown(self) -> Dict:
        """Check drawdown limit"""
        drawdown = self._calculate_drawdown()
        limit = self.limits[RiskLimitType.DRAWDOWN]
        
        if drawdown > limit:
            return {
                'passed': False,
                'reason': f'Drawdown {drawdown:.2%} exceeds limit {limit:.2%}',
                'check': 'drawdown',
                'severity': 'warning'
            }
        
        return {'passed': True, 'check': 'drawdown'}
    
    def _check_turnover(self) -> Dict:
        """Check turnover limit"""
        turnover_pct = self.daily_turnover / self.start_of_day_capital
        limit = self.limits[RiskLimitType.TURNOVER]
        
        if turnover_pct > limit:
            return {
                'passed': False,
                'reason': f'Turnover {turnover_pct:.2%} exceeds limit {limit:.2%}',
                'check': 'turnover',
                'severity': 'warning'
            }
        
        return {'passed': True, 'check': 'turnover'}
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if self.peak_capital > 0:
            return (self.peak_capital - self.current_capital) / self.peak_capital
        return 0.0
    
    def reset_day(self) -> None:
        """Reset daily metrics at start of new day"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.start_of_day_capital = self.current_capital
        self.daily_turnover = 0.0


class CircuitBreaker:
    """
    Circuit breaker to halt trading on extreme conditions
    
    Triggers:
    - Daily loss limit breach
    - Extreme volatility
    - System failures
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None
        self.cooldown_minutes: int = 30
    
    def trigger(self, reason: str) -> None:
        """Trigger circuit breaker"""
        if self.enabled:
            self.triggered = True
            self.trigger_time = datetime.now()
            self.trigger_reason = reason
    
    def reset(self) -> None:
        """Reset circuit breaker after cooldown"""
        if self.triggered and self.trigger_time:
            elapsed = (datetime.now() - self.trigger_time).total_seconds() / 60
            if elapsed > self.cooldown_minutes:
                self.triggered = False
                self.trigger_time = None
                self.trigger_reason = None
    
    def is_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        self.reset()
        return self.triggered
