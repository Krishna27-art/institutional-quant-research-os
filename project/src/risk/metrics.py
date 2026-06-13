"""
Risk Metrics Module
Implements statistical risk metrics: VaR (Parametric, Historical, EVT), CVaR, tail risk,
portfolio heat, volatility targeting, and Kelly sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_portfolio_returns(positions: List[Any], market_data: pd.DataFrame, capital: float) -> np.ndarray:
    """Calculate portfolio returns from positions and historical market data."""
    if not positions or market_data.empty:
        return np.array([])
    
    # Calculate position weights
    total_value = sum(pos.quantity * pos.current_price for pos in positions)
    if total_value <= 0:
        return np.array([])
        
    weights = {pos.symbol: (pos.quantity * pos.current_price) / total_value for pos in positions}
    
    returns_list = []
    for pos in positions:
        if pos.symbol in market_data.columns:
            symbol_returns = market_data[pos.symbol].pct_change().dropna()
            direction = -1.0 if pos.side.upper() == "SHORT" else 1.0
            weighted_returns = symbol_returns * weights[pos.symbol] * direction
            returns_list.append(weighted_returns)
            
    if not returns_list:
        return np.array([])
        
    portfolio_returns = pd.concat(returns_list, axis=1).sum(axis=1).values
    return portfolio_returns


def calculate_moments(returns: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate moments of returns (mean, std, skew, kurtosis)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    
    if len(returns) > 3:
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
    else:
        skew = 0.0
        kurt = 0.0
    
    return mu, sigma, skew, kurt


def calculate_var(returns: np.ndarray, capital: float, confidence: float = 0.99, use_cornish_fisher: bool = True) -> float:
    """Calculate Value at Risk using Cornish-Fisher expansion or parametric method."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0

    mu, sigma, skew, kurt = calculate_moments(returns)
    if sigma <= 0:
        return 0.0
    
    if use_cornish_fisher:
        z = stats.norm.ppf(1 - confidence)
        z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
        quantile = mu + z_cf * sigma
    else:
        z = stats.norm.ppf(1 - confidence)
        quantile = mu + z * sigma
    
    return capital * abs(quantile) if quantile < 0 else 0.0


def calculate_var_historical(returns: np.ndarray, capital: float, confidence: float = 0.99) -> float:
    """Calculate Historical Simulation VaR."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0
    
    sorted_returns = np.sort(returns)
    idx = int(len(sorted_returns) * (1 - confidence))
    quantile = sorted_returns[idx] if idx < len(sorted_returns) else sorted_returns[-1]
    
    return capital * abs(quantile) if quantile < 0 else 0.0


def calculate_var_evt(returns: np.ndarray, capital: float, confidence: float = 0.99, threshold_percentile: float = 0.90) -> float:
    """Calculate EVT VaR using Peaks-Over-Threshold and Generalized Pareto Distribution."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 50:
        return calculate_var_historical(returns, capital, confidence)
    
    negative_returns = -returns[returns < 0]
    if len(negative_returns) < 20:
        return calculate_var_historical(returns, capital, confidence)
    
    threshold = np.percentile(negative_returns, threshold_percentile * 100)
    excesses = negative_returns[negative_returns > threshold] - threshold
    
    if len(excesses) < 10:
        return calculate_var_historical(returns, capital, confidence)
    
    try:
        mean_excess = np.mean(excesses)
        var_excess = np.var(excesses, ddof=1)
        
        if var_excess > 0:
            xi = -0.5 * ((mean_excess**2 / var_excess) - 1)
            xi = max(min(xi, 0.5), -0.5)
        else:
            xi = 0
        
        beta = 0.5 * mean_excess * (1 + xi**2)
        beta = max(beta, 0.001)
        
        n = len(negative_returns)
        n_excess = len(excesses)
        u = threshold
        alpha = 1 - confidence
        
        if xi != 0:
            evt_var = u + (beta / xi) * (((n / n_excess) * alpha) ** (-xi) - 1)
        else:
            evt_var = u - beta * np.log((n / n_excess) * alpha)
        
        return capital * abs(evt_var) if evt_var > 0 else 0.0
    except Exception:
        return calculate_var_historical(returns, capital, confidence)


def calculate_cvar(returns: np.ndarray, capital: float, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0

    sorted_returns = np.sort(returns)
    idx = int(len(sorted_returns) * (1 - confidence))
    
    if idx == 0:
        cvar_return = sorted_returns[0]
    else:
        cvar_return = np.mean(sorted_returns[:idx])
        
    return capital * abs(cvar_return) if cvar_return < 0 else 0.0


def calculate_liquidity_adjusted_var(positions: List[Any], returns: np.ndarray, capital: float, confidence: float = 0.99, adv_data: Optional[Dict] = None) -> float:
    """Calculate Liquidity-adjusted VaR (L-VaR)."""
    base_var = calculate_var(returns, capital, confidence)
    if adv_data is None:
        adv_data = {}
        
    liquidity_adjustment = 0.0
    for pos in positions:
        position_value = pos.quantity * pos.current_price
        adv = adv_data.get(pos.symbol, 1e9)
        adjustment = position_value / adv
        liquidity_adjustment += adjustment
        
    l_var = base_var * (1 + liquidity_adjustment)
    return l_var


def calculate_volatility_target_multiplier(returns: np.ndarray, target_vol: float = 0.15) -> float:
    """Calculate volatility targeting multiplier."""
    if len(returns) < 2:
        return 1.0
    current_vol = np.std(returns) * np.sqrt(252)
    if current_vol < 0.01:
        current_vol = 0.01
    vol_mult = target_vol / current_vol
    vol_mult = min(2.0, vol_mult)
    vol_mult = max(0.5, vol_mult)
    return vol_mult


def calculate_kelly_fraction(
    expected_return: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    win_rate_std: float = 0.05,
    volatility: float = 0.15,
    confidence_level: float = 0.95
) -> float:
    """Calculate Kelly fraction for position sizing with safety adjustments."""
    if avg_loss == 0 or volatility == 0:
        return 0.0
    
    b = avg_win / abs(avg_loss)
    p = win_rate
    
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p_conservative = p - z_score * win_rate_std
    p_conservative = max(0.01, min(0.99, p_conservative))
    
    kelly = (p_conservative * b - (1 - p_conservative)) / b
    
    target_vol = 0.15
    vol_scaling = target_vol / max(volatility, 0.05)
    vol_scaling = min(1.5, max(0.5, vol_scaling))
    
    kelly = kelly * 0.25 * vol_scaling
    
    max_drawdown_allowed = 0.25
    worst_case_loss = avg_loss
    max_kelly_by_dd = max_drawdown_allowed / worst_case_loss if worst_case_loss > 0 else 0.25
    
    kelly = min(0.25, max_kelly_by_dd, max(0, kelly))
    return kelly


def calculate_portfolio_heat(positions: List[Any], market_data: pd.DataFrame) -> float:
    """Calculate portfolio heat (correlation-based concentration)."""
    if len(positions) < 2 or market_data.empty:
        return 0.0
    
    returns_df = pd.DataFrame()
    for pos in positions:
        if pos.symbol in market_data.columns:
            returns_df[pos.symbol] = market_data[pos.symbol].pct_change().dropna()
            
    if returns_df.empty:
        return 0.0
        
    corr_matrix = returns_df.corr()
    portfolio_heat = np.mean(np.abs(corr_matrix.values))
    return portfolio_heat


def calculate_tail_risk(returns: np.ndarray, capital: float, percentile: float = 0.15) -> float:
    """Calculate tail risk (worst X% of returns)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return 0.0

    tail_fraction = percentile if 0 < percentile <= 1 else percentile / 100
    n_tail = max(1, int(np.ceil(len(returns) * tail_fraction)))
    tail_return = float(np.mean(np.sort(returns)[:n_tail]))
    
    return capital * abs(tail_return) if tail_return < 0 else 0.0
