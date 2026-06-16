"""
Factor Models - Level 2 Foundation

This module provides factor models for asset pricing:
- CAPM (Capital Asset Pricing Model)
- APT (Arbitrage Pricing Theory)
- Fama-French 3-factor model
- Fama-French 5-factor model
- Barra industry factors
- Factor attribution and risk premium estimation

Based on Audit Report Priority 2: Asset Pricing Theories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class FactorModel(Enum):
    """Types of factor models."""
    CAPM = "capm"
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    APT = "apt"
    BARRA = "barra"


@dataclass
class FactorExposure:
    """Factor exposure for a stock or portfolio."""
    symbol: str
    market_beta: float
    size_factor: float = 0.0
    value_factor: float = 0.0
    profitability_factor: float = 0.0
    investment_factor: float = 0.0
    momentum_factor: float = 0.0
    volatility_factor: float = 0.0


@dataclass
class FactorReturn:
    """Factor return data."""
    date: str
    market_return: float
    smb: float = 0.0  # Small minus Big
    hml: float = 0.0  # High minus Low
    rmw: float = 0.0  # Robust minus Weak
    cma: float = 0.0  # Conservative minus Aggressive
    mom: float = 0.0  # Momentum
    vol: float = 0.0  # Low volatility


class FactorModelEngine:
    """
    Factor model engine for asset pricing.
    
    This class implements various factor models (CAPM, APT, Fama-French)
    for the Indian market to decompose returns into factor components.
    """
    
    def __init__(self):
        """Initialize factor model engine."""
        self.factor_returns: List[FactorReturn] = []
        self.factor_exposures: Dict[str, FactorExposure] = {}
        
    def run_factor_model(
        self,
        returns: pd.DataFrame,
        model: FactorModel = FactorModel.CAPM,
        indian_market: bool = False
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Run a factor model by type.
        """
        asset_col = returns.columns[0]
        asset_ret = returns[asset_col]
        
        if model == FactorModel.CAPM:
            market_col = returns.columns[1] if len(returns.columns) > 1 else 'market'
            market_ret = returns[market_col]
            res = self.capm_single_factor(asset_ret, market_ret)
            
            betas = {'market': res['beta_market']}
            expected = res['alpha'] + res['beta_market'] * market_ret.mean()
            return {
                'betas': betas,
                'expected_returns': expected,
                'alpha': res['alpha'],
                'beta_market': res['beta_market'],
                'r_squared': res['r_squared'],
            }
        elif model == FactorModel.FAMA_FRENCH_3:
            res = self.fama_french_3factor(
                asset_ret,
                returns['market'],
                returns['smb'],
                returns['hml']
            )
            betas = {
                'market': res['beta_market'],
                'smb': res['beta_smb'],
                'hml': res['beta_hml']
            }
            expected = res['alpha'] + res['beta_market'] * returns['market'].mean() + res['beta_smb'] * returns['smb'].mean() + res['beta_hml'] * returns['hml'].mean()
            return {
                'betas': betas,
                'expected_returns': expected,
                'alpha': res['alpha'],
                'r_squared': res['r_squared']
            }
        else:
            res = self.apt_multi_factor(asset_ret, returns.drop(columns=[asset_col]))
            betas = res['factor_betas']
            expected = res['alpha'] + sum(betas[f] * returns[f].mean() for f in betas)
            return {
                'betas': betas,
                'expected_returns': expected,
                'alpha': res['alpha'],
                'r_squared': res['r_squared']
            }
    
    def fama_french_3factor(
        self,
        returns: pd.Series,
        market_return: pd.Series,
        smb: pd.Series,
        hml: pd.Series
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Fama-French 3-factor model regression.
        
        R_i - R_f = α + β_MKT * (R_M - R_f) + β_SMB * SMB + β_HML * HML + ε
        
        Args:
            returns: Stock excess returns
            market_return: Market excess returns
            smb: Small minus Big factor returns
            hml: High minus Low factor returns
            
        Returns:
            Dictionary with regression coefficients and statistics
        """
        # Align data
        data = pd.DataFrame({
            'returns': returns,
            'market': market_return,
            'smb': smb,
            'hml': hml,
        }).dropna()
        
        X = data[['market', 'smb', 'hml']]
        y = data['returns']
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # OLS regression
        model = sm.OLS(y, X).fit()
        
        return {
            'alpha': model.params['const'],
            'beta_market': model.params['market'],
            'beta_smb': model.params['smb'],
            'beta_hml': model.params['hml'],
            'r_squared': model.rsquared,
            'p_values': model.pvalues.to_dict(),
            'residuals': model.resid,
        }
    
    def fama_french_5factor(
        self,
        returns: pd.Series,
        market_return: pd.Series,
        smb: pd.Series,
        hml: pd.Series,
        rmw: pd.Series,
        cma: pd.Series
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Fama-French 5-factor model regression.
        
        R_i - R_f = α + β_MKT * (R_M - R_f) + β_SMB * SMB + β_HML * HML 
                 + β_RMW * RMW + β_CMA * CMA + ε
        
        Args:
            returns: Stock excess returns
            market_return: Market excess returns
            smb: Small minus Big factor returns
            hml: High minus Low factor returns
            rmw: Robust minus Weak factor returns
            cma: Conservative minus Aggressive factor returns
            
        Returns:
            Dictionary with regression coefficients and statistics
        """
        # Align data
        data = pd.DataFrame({
            'returns': returns,
            'market': market_return,
            'smb': smb,
            'hml': hml,
            'rmw': rmw,
            'cma': cma,
        }).dropna()
        
        X = data[['market', 'smb', 'hml', 'rmw', 'cma']]
        y = data['returns']
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # OLS regression
        model = sm.OLS(y, X).fit()
        
        return {
            'alpha': model.params['const'],
            'beta_market': model.params['market'],
            'beta_smb': model.params['smb'],
            'beta_hml': model.params['hml'],
            'beta_rmw': model.params['rmw'],
            'beta_cma': model.params['cma'],
            'r_squared': model.rsquared,
            'p_values': model.pvalues.to_dict(),
            'residuals': model.resid,
        }
    
    def barra_industry_factors(
        self,
        returns: pd.Series,
        industry_returns: pd.DataFrame
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Barra industry factor model regression.
        
        Args:
            returns: Stock returns
            industry_returns: DataFrame of industry factor returns
            
        Returns:
            Dictionary with regression coefficients and statistics
        """
        # Align data
        data = industry_returns.copy()
        data['stock'] = returns
        data = data.dropna()
        
        X = data.drop(columns=['stock'])
        y = data['stock']
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # OLS regression
        model = sm.OLS(y, X).fit()
        
        return {
            'alpha': model.params['const'],
            'industry_betas': model.params.drop('const').to_dict(),
            'r_squared': model.rsquared,
            'p_values': model.pvalues.to_dict(),
            'residuals': model.resid,
        }
    
    def capm_single_factor(
        self,
        returns: pd.Series,
        market_return: pd.Series
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        CAPM single-factor model regression.
        
        R_i - R_f = α + β_MKT * (R_M - R_f) + ε
        
        Args:
            returns: Stock excess returns
            market_return: Market excess returns
            
        Returns:
            Dictionary with regression coefficients and statistics
        """
        # Align data
        data = pd.DataFrame({
            'returns': returns,
            'market': market_return,
        }).dropna()
        
        X = data[['market']]
        y = data['returns']
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # OLS regression
        model = sm.OLS(y, X).fit()
        
        return {
            'alpha': model.params['const'],
            'beta_market': model.params['market'],
            'r_squared': model.rsquared,
            'p_value_alpha': model.pvalues['const'],
            'p_value_beta': model.pvalues['market'],
            'residuals': model.resid,
        }
    
    def apt_multi_factor(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        APT multi-factor model regression.
        
        R_i = α + Σ β_j * F_j + ε
        
        Args:
            returns: Stock returns
            factor_returns: DataFrame of factor returns
            
        Returns:
            Dictionary with regression coefficients and statistics
        """
        # Align data
        data = factor_returns.copy()
        data['stock'] = returns
        data = data.dropna()
        
        X = data.drop(columns=['stock'])
        y = data['stock']
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # OLS regression
        model = sm.OLS(y, X).fit()
        
        return {
            'alpha': model.params['const'],
            'factor_betas': model.params.drop('const').to_dict(),
            'r_squared': model.rsquared,
            'p_values': model.pvalues.to_dict(),
            'residuals': model.resid,
        }
    
    def calculate_beta(
        self,
        returns: pd.Series,
        factor_returns: pd.Series
    ) -> float:
        """
        Calculate beta (systematic risk) for a single factor.
        
        Args:
            returns: Asset returns
            factor_returns: Factor returns
            
        Returns:
            Beta coefficient
        """
        # Align data
        data = pd.DataFrame({
            'returns': returns,
            'factor': factor_returns,
        }).dropna()
        
        # Calculate covariance and variance
        covariance = data['returns'].cov(data['factor'])
        variance = data['factor'].var()
        
        if variance > 0:
            beta = covariance / variance
        else:
            beta = 0.0
        
        return beta
    
    def calculate_residual_returns(
        self,
        returns: pd.Series,
        factor_model: Dict[str, Union[float, Dict[str, float]]],
        factor_returns: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Calculate residual returns from factor model.
        
        Args:
            returns: Actual returns
            factor_model: Factor model regression results
            factor_returns: Factor returns (for prediction)
            
        Returns:
            Residual returns (alpha)
        """
        if 'residuals' in factor_model:
            return pd.Series(factor_model['residuals'])
        
        # If residuals not available, calculate from model
        alpha = factor_model.get('alpha', 0)
        
        if factor_returns is not None and 'factor_betas' in factor_model:
            # Calculate predicted returns from factors
            predicted = alpha
            for factor, beta in factor_model['factor_betas'].items():
                if factor in factor_returns.columns:
                    predicted += beta * factor_returns[factor]
            
            residuals = returns - predicted
            return residuals
        
        # Fallback: return alpha as constant residual
        return pd.Series([alpha] * len(returns), index=returns.index)
    
    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_exposures: Dict[str, float],
        factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Attribute portfolio returns to factors.
        
        Args:
            portfolio_returns: Portfolio returns
            factor_exposures: Factor exposures (betas)
            factor_returns: Factor returns
            
        Returns:
            Dictionary with return attribution by factor
        """
        attribution = {}
        
        # Calculate contribution from each factor
        for factor, exposure in factor_exposures.items():
            if factor in factor_returns.columns:
                factor_return = factor_returns[factor].mean()
                contribution = exposure * factor_return
                attribution[factor] = contribution
        
        # Calculate alpha (unexplained return)
        total_return = portfolio_returns.mean()
        explained_return = sum(attribution.values())
        attribution['alpha'] = total_return - explained_return
        
        return attribution
    
    def factor_risk_premium_estimation(
        self,
        factor_returns: pd.DataFrame,
        risk_free_rate: float = 0.06
    ) -> Dict[str, float]:
        """
        Estimate factor risk premiums.
        
        Args:
            factor_returns: Historical factor returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with estimated risk premiums
        """
        premiums = {}
        
        for factor in factor_returns.columns:
            # Calculate average excess return
            avg_return = factor_returns[factor].mean()
            premium = avg_return - risk_free_rate / 252  # Daily risk-free rate
            premiums[factor] = premium
        
        return premiums
    
    def indian_market_factors(
        self,
        nifty_returns: pd.Series,
        small_cap_returns: pd.Series,
        large_cap_returns: pd.Series,
        high_bm_returns: pd.Series,
        low_bm_returns: pd.Series,
        robust_profit_returns: pd.Series,
        weak_profit_returns: pd.Series,
        conservative_inv_returns: pd.Series,
        aggressive_inv_returns: pd.Series,
        winner_returns: pd.Series,
        loser_returns: pd.Series,
        low_vol_returns: pd.Series,
        high_vol_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate Indian market factor returns.
        
        Args:
            nifty_returns: NIFTY50 returns (market)
            small_cap_returns: Small cap index returns
            large_cap_returns: Large cap index returns
            high_bm_returns: High book-to-market returns
            low_bm_returns: Low book-to-market returns
            robust_profit_returns: Robust profitability returns
            weak_profit_returns: Weak profitability returns
            conservative_inv_returns: Conservative investment returns
            aggressive_inv_returns: Aggressive investment returns
            winner_returns: Winner returns (momentum)
            loser_returns: Loser returns (momentum)
            low_vol_returns: Low volatility returns
            high_vol_returns: High volatility returns
            
        Returns:
            DataFrame with factor returns
        """
        factors = pd.DataFrame()
        
        # Market factor (MKT)
        factors['MKT'] = nifty_returns
        
        # Small minus Big (SMB)
        factors['SMB'] = small_cap_returns - large_cap_returns
        
        # High minus Low (HML)
        factors['HML'] = high_bm_returns - low_bm_returns
        
        # Robust minus Weak (RMW)
        factors['RMW'] = robust_profit_returns - weak_profit_returns
        
        # Conservative minus Aggressive (CMA)
        factors['CMA'] = conservative_inv_returns - aggressive_inv_returns
        
        # Momentum (MOM)
        factors['MOM'] = winner_returns - loser_returns
        
        # Low volatility (VOL)
        factors['VOL'] = low_vol_returns - high_vol_returns
        
        return factors
    
    def calculate_factor_exposures(
        self,
        symbol: str,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        model_type: FactorModel = FactorModel.FAMA_FRENCH_5
    ) -> FactorExposure:
        """
        Calculate factor exposures for a stock.
        
        Args:
            symbol: Stock symbol
            returns: Stock returns
            factor_returns: Factor returns
            model_type: Type of factor model
            
        Returns:
            FactorExposure object
        """
        if model_type == FactorModel.CAPM:
            result = self.capm_single_factor(returns, factor_returns['MKT'])
            return FactorExposure(
                symbol=symbol,
                market_beta=result['beta_market'],
            )
        
        elif model_type == FactorModel.FAMA_FRENCH_3:
            result = self.fama_french_3factor(
                returns,
                factor_returns['MKT'],
                factor_returns['SMB'],
                factor_returns['HML']
            )
            return FactorExposure(
                symbol=symbol,
                market_beta=result['beta_market'],
                size_factor=result['beta_smb'],
                value_factor=result['beta_hml'],
            )
        
        elif model_type == FactorModel.FAMA_FRENCH_5:
            result = self.fama_french_5factor(
                returns,
                factor_returns['MKT'],
                factor_returns['SMB'],
                factor_returns['HML'],
                factor_returns['RMW'],
                factor_returns['CMA']
            )
            return FactorExposure(
                symbol=symbol,
                market_beta=result['beta_market'],
                size_factor=result['beta_smb'],
                value_factor=result['beta_hml'],
                profitability_factor=result['beta_rmw'],
                investment_factor=result['beta_cma'],
            )
        
        else:
            # Default to APT with all available factors
            result = self.apt_multi_factor(returns, factor_returns)
            return FactorExposure(
                symbol=symbol,
                market_beta=result['factor_betas'].get('MKT', 0),
                size_factor=result['factor_betas'].get('SMB', 0),
                value_factor=result['factor_betas'].get('HML', 0),
                profitability_factor=result['factor_betas'].get('RMW', 0),
                investment_factor=result['factor_betas'].get('CMA', 0),
                momentum_factor=result['factor_betas'].get('MOM', 0),
                volatility_factor=result['factor_betas'].get('VOL', 0),
            )
