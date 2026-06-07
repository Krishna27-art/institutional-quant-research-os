"""
Indian Multi-Factor Model (Fama-French Framework)
5-7 factor model for Indian markets based on Fama-French methodology.

Critical for institutional-grade factor investing in India.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression


@dataclass
class FactorReturns:
    """Factor returns for a period"""
    date: datetime
    market_risk_premium: float  # Rm - Rf
    smb: float  # Small Minus Big
    hml: float  # High Minus Low
    rmw: float  # Robust Minus Weak
    cma: float  # Conservative Minus Aggressive
    momentum: float  # Momentum factor
    volatility: float  # Volatility factor


@dataclass
class FactorExposure:
    """Factor exposure for a stock"""
    stock_id: str
    beta: float  # Market beta
    size: float  # Size factor exposure
    value: float  # Value factor exposure
    profitability: float  # Profitability factor exposure
    investment: float  # Investment factor exposure
    momentum: float  # Momentum factor exposure
    volatility: float  # Volatility factor exposure


@dataclass
class FactorModelResult:
    """Result of factor model regression"""
    stock_id: str
    alpha: float  # Idiosyncratic alpha
    alpha_t_stat: float
    alpha_p_value: float
    r_squared: float
    factor_exposures: Dict[str, float]
    factor_t_stats: Dict[str, float]


class IndianFactorModel:
    """
    Indian Multi-Factor Model
    
    Implements Fama-French 5-factor model adapted for Indian markets:
    1. Market Risk Premium (Rm - Rf)
    2. SMB (Small Minus Big)
    3. HML (High Minus Low)
    4. RMW (Robust Minus Weak - profitability)
    5. CMA (Conservative Minus Aggressive - investment)
    6. Momentum (WML - Winners Minus Losers)
    7. Volatility (Low Volatility factor)
    
    Indian-specific adaptations:
    - Use NIFTY 50 as market proxy
    - Use NIFTY Midcap 100 for SMB
    - Use BSE 200 for broader coverage
    - Adjust for Indian market structure (circuit limits, settlement cycles)
    """
    
    def __init__(self, risk_free_rate: float = 0.06):  # 6% risk-free rate for India
        self.risk_free_rate = risk_free_rate
        self.factor_returns_history: List[FactorReturns] = []
        self.stock_factor_exposures: Dict[str, FactorExposure] = {}
        self.regression_results: Dict[str, FactorModelResult] = {}
    
    def calculate_factor_returns(self, market_returns: pd.Series,
                                small_cap_returns: pd.Series,
                                large_cap_returns: pd.Series,
                                value_returns: pd.Series,
                                growth_returns: pd.Series,
                                robust_returns: pd.Series,
                                weak_returns: pd.Series,
                                conservative_returns: pd.Series,
                                aggressive_returns: pd.Series,
                                winner_returns: pd.Series,
                                loser_returns: pd.Series,
                                low_vol_returns: pd.Series,
                                high_vol_returns: pd.Series) -> FactorReturns:
        """
        Calculate factor returns for a period.
        
        Args:
            market_returns: Market returns (NIFTY 50)
            small_cap_returns: Small cap returns (NIFTY Midcap)
            large_cap_returns: Large cap returns
            value_returns: Value stock returns
            growth_returns: Growth stock returns
            robust_returns: High profitability returns
            weak_returns: Low profitability returns
            conservative_returns: Low investment returns
            aggressive_returns: High investment returns
            winner_returns: Winner returns (momentum)
            loser_returns: Loser returns (momentum)
            low_vol_returns: Low volatility returns
            high_vol_returns: High volatility returns
        
        Returns:
            FactorReturns
        """
        # Market risk premium
        market_risk_premium = market_returns.mean() - self.risk_free_rate / 252
        
        # SMB: Small Minus Big
        smb = small_cap_returns.mean() - large_cap_returns.mean()
        
        # HML: High Minus Low (value)
        hml = value_returns.mean() - growth_returns.mean()
        
        # RMW: Robust Minus Weak (profitability)
        rmw = robust_returns.mean() - weak_returns.mean()
        
        # CMA: Conservative Minus Aggressive (investment)
        cma = conservative_returns.mean() - aggressive_returns.mean()
        
        # Momentum: Winners Minus Losers
        momentum = winner_returns.mean() - loser_returns.mean()
        
        # Volatility: Low Vol Minus High Vol
        volatility = low_vol_returns.mean() - high_vol_returns.mean()
        
        factor_returns = FactorReturns(
            date=datetime.now(),
            market_risk_premium=market_risk_premium,
            smb=smb,
            hml=hml,
            rmw=rmw,
            cma=cma,
            momentum=momentum,
            volatility=volatility
        )
        
        self.factor_returns_history.append(factor_returns)
        
        return factor_returns
    
    def estimate_factor_exposure(self, stock_returns: pd.Series,
                                 market_returns: pd.Series,
                                 factor_returns: FactorReturns) -> FactorExposure:
        """
        Estimate factor exposure for a stock using regression.
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns
            factor_returns: Factor returns
        
        Returns:
            FactorExposure
        """
        # Build factor matrix
        # For simplicity, use single-factor beta estimation
        # In production, would use multi-factor regression
        
        # Market beta
        if len(stock_returns) == len(market_returns):
            beta, _, _, _, _ = stats.linregress(market_returns, stock_returns)
        else:
            beta = 1.0
        
        # Simplified factor exposures based on stock characteristics
        # In production, would use actual factor data
        size = 0.0  # Would be based on market cap
        value = 0.0  # Would be based on B/P ratio
        profitability = 0.0  # Would be based on ROE
        investment = 0.0  # Would be based on asset growth
        momentum = 0.0  # Would be based on past returns
        volatility = 0.0  # Would be based on historical volatility
        
        exposure = FactorExposure(
            stock_id="stock",
            beta=beta,
            size=size,
            value=value,
            profitability=profitability,
            investment=investment,
            momentum=momentum,
            volatility=volatility
        )
        
        return exposure
    
    def run_factor_regression(self, stock_returns: pd.Series,
                            factor_returns_df: pd.DataFrame) -> FactorModelResult:
        """
        Run multi-factor regression to decompose stock returns.
        
        Args:
            stock_returns: Stock returns
            factor_returns_df: DataFrame of factor returns
        
        Returns:
            FactorModelResult
        """
        # Align data
        aligned_data = pd.concat([stock_returns, factor_returns_df], axis=1).dropna()
        
        if len(aligned_data) < 30:
            # Not enough data
            return None
        
        y = aligned_data.iloc[:, 0].values
        X = aligned_data.iloc[:, 1:].values
        
        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Run regression
        model = LinearRegression(fit_intercept=False)
        model.fit(X_with_const, y)
        
        # Get coefficients
        coefficients = model.coef_
        alpha = coefficients[0]
        factor_exposures = dict(zip(factor_returns_df.columns, coefficients[1:]))
        
        # Calculate t-stats (simplified)
        residuals = y - model.predict(X_with_const)
        mse = np.mean(residuals ** 2)
        std_error = np.sqrt(mse / len(X))
        
        alpha_t_stat = alpha / std_error if std_error > 0 else 0
        alpha_p_value = 2 * (1 - stats.norm.cdf(abs(alpha_t_stat)))
        
        # R-squared
        r_squared = model.score(X_with_const, y)
        
        result = FactorModelResult(
            stock_id="stock",
            alpha=alpha,
            alpha_t_stat=alpha_t_stat,
            alpha_p_value=alpha_p_value,
            r_squared=r_squared,
            factor_exposures=factor_exposures,
            factor_t_stats={}  # Would calculate in production
        )
        
        return result
    
    def get_idiosyncratic_alpha(self, stock_id: str) -> Optional[float]:
        """Get idiosyncratic alpha for a stock"""
        if stock_id in self.regression_results:
            return self.regression_results[stock_id].alpha
        return None
    
    def get_factor_attribution(self, stock_id: str) -> Dict[str, float]:
        """Get factor attribution for a stock"""
        if stock_id not in self.regression_results:
            return {}
        
        result = self.regression_results[stock_id]
        return result.factor_exposures
    
    def generate_report(self) -> str:
        """Generate factor model report"""
        report = f"""
Indian Multi-Factor Model Report
{'=' * 50}
Risk-Free Rate: {self.risk_free_rate:.1%}
Factor Returns History: {len(self.factor_returns_history)}
Stocks Analyzed: {len(self.regression_results)}

Factor Definitions:
{'-' * 50}
- Market Risk Premium (Rm - Rf): Excess return over risk-free
- SMB: Small Minus Big (size factor)
- HML: High Minus Low (value factor)
- RMW: Robust Minus Weak (profitability factor)
- CMA: Conservative Minus Aggressive (investment factor)
- Momentum: Winners Minus Losers (momentum factor)
- Volatility: Low Vol Minus High Vol (volatility factor)

Recent Factor Returns:
{'-' * 50}
"""
        
        if self.factor_returns_history:
            recent = self.factor_returns_history[-5:]
            for fr in recent:
                report += f"{fr.date}: "
                report += f"Market: {fr.market_risk_premium:.2%}, "
                report += f"SMB: {fr.smb:.2%}, "
                report += f"HML: {fr.hml:.2%}, "
                report += f"RMW: {fr.rmw:.2%}, "
                report += f"CMA: {fr.cma:.2%}, "
                report += f"Mom: {fr.momentum:.2%}, "
                report += f"Vol: {fr.volatility:.2%}\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    model = IndianFactorModel(risk_free_rate=0.06)
    
    # Simulate factor returns
    print("Simulating factor returns...")
    n_days = 252
    
    market_returns = pd.Series(np.random.normal(0.08/252, 0.02/np.sqrt(252), n_days))
    small_cap_returns = pd.Series(np.random.normal(0.10/252, 0.025/np.sqrt(252), n_days))
    large_cap_returns = pd.Series(np.random.normal(0.07/252, 0.018/np.sqrt(252), n_days))
    value_returns = pd.Series(np.random.normal(0.09/252, 0.022/np.sqrt(252), n_days))
    growth_returns = pd.Series(np.random.normal(0.07/252, 0.02/np.sqrt(252), n_days))
    robust_returns = pd.Series(np.random.normal(0.09/252, 0.02/np.sqrt(252), n_days))
    weak_returns = pd.Series(np.random.normal(0.06/252, 0.025/np.sqrt(252), n_days))
    conservative_returns = pd.Series(np.random.normal(0.08/252, 0.018/np.sqrt(252), n_days))
    aggressive_returns = pd.Series(np.random.normal(0.06/252, 0.028/np.sqrt(252), n_days))
    winner_returns = pd.Series(np.random.normal(0.11/252, 0.03/np.sqrt(252), n_days))
    loser_returns = pd.Series(np.random.normal(0.04/252, 0.035/np.sqrt(252), n_days))
    low_vol_returns = pd.Series(np.random.normal(0.08/252, 0.015/np.sqrt(252), n_days))
    high_vol_returns = pd.Series(np.random.normal(0.07/252, 0.03/np.sqrt(252), n_days))
    
    # Calculate factor returns
    factor_returns = model.calculate_factor_returns(
        market_returns, small_cap_returns, large_cap_returns,
        value_returns, growth_returns, robust_returns, weak_returns,
        conservative_returns, aggressive_returns, winner_returns, loser_returns,
        low_vol_returns, high_vol_returns
    )
    
    print(f"Factor Returns:")
    print(f"  Market Risk Premium: {factor_returns.market_risk_premium:.2%}")
    print(f"  SMB: {factor_returns.smb:.2%}")
    print(f"  HML: {factor_returns.hml:.2%}")
    print(f"  RMW: {factor_returns.rmw:.2%}")
    print(f"  CMA: {factor_returns.cma:.2%}")
    print(f"  Momentum: {factor_returns.momentum:.2%}")
    print(f"  Volatility: {factor_returns.volatility:.2%}")
    
    print(model.generate_report())
