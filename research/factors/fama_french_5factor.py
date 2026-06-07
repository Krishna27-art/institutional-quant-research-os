"""
Fama-French 5-Factor Model Implementation

Implements the Fama-French 5-factor model for asset pricing:
- Market risk (MKT)
- Size factor (SMB - Small Minus Big)
- Value factor (HML - High Minus Low)
- Profitability factor (RMW - Robust Minus Weak)
- Investment factor (CMA - Conservative Minus Aggressive)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FactorReturns:
    """Returns for each factor."""
    mkt: float  # Market excess return
    smb: float  # Small minus big
    hml: float  # High minus low
    rmw: float  # Robust minus weak
    cma: float  # Conservative minus aggressive
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MKT': self.mkt,
            'SMB': self.smb,
            'HML': self.hml,
            'RMW': self.rmw,
            'CMA': self.cma
        }


@dataclass
class FactorExposures:
    """Factor exposures (betas) for a stock."""
    mkt_beta: float
    smb_beta: float
    hml_beta: float
    rmw_beta: float
    cma_beta: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'MKT': self.mkt_beta,
            'SMB': self.smb_beta,
            'HML': self.hml_beta,
            'RMW': self.rmw_beta,
            'CMA': self.cma_beta
        }


class FamaFrench5Factor:
    """
    Fama-French 5-factor model implementation.
    
    For Indian markets, we need to adapt the factors:
    - Use NIFTY 50 as market proxy
    - Use market cap for size sorting
    - Use book-to-market for value sorting
    - Use operating profitability for profitability
    - Use asset growth for investment
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        """
        Initialize Fama-French 5-factor model.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 6% for India)
        """
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
    def calculate_size_factor(
        self,
        market_caps: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate SMB (Small Minus Big) factor.
        
        Args:
            market_caps: Dictionary of symbol to market cap
            
        Returns:
            Dictionary of symbol to size factor exposure
        """
        if not market_caps:
            return {}
        
        # Calculate median market cap
        caps = list(market_caps.values())
        median_cap = np.median(caps)
        
        # Assign size exposure: +1 for small caps, -1 for large caps
        size_exposures = {}
        for symbol, cap in market_caps.items():
            exposure = 1.0 if cap < median_cap else -1.0
            size_exposures[symbol] = exposure
        
        return size_exposures
    
    def calculate_value_factor(
        self,
        book_to_market: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate HML (High Minus Low) factor.
        
        Args:
            book_to_market: Dictionary of symbol to book-to-market ratio
            
        Returns:
            Dictionary of symbol to value factor exposure
        """
        if not book_to_market:
            return {}
        
        # Calculate median book-to-market
        btm = list(book_to_market.values())
        median_btm = np.median(btm)
        
        # Assign value exposure: +1 for high B/M (value), -1 for low B/M (growth)
        value_exposures = {}
        for symbol, btm_ratio in book_to_market.items():
            exposure = 1.0 if btm_ratio > median_btm else -1.0
            value_exposures[symbol] = exposure
        
        return value_exposures
    
    def calculate_profitability_factor(
        self,
        operating_profitability: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate RMW (Robust Minus Weak) factor.
        
        Args:
            operating_profitability: Dictionary of symbol to operating profitability
            
        Returns:
            Dictionary of symbol to profitability factor exposure
        """
        if not operating_profitability:
            return {}
        
        # Calculate median profitability
        op = list(operating_profitability.values())
        median_op = np.median(op)
        
        # Assign profitability exposure: +1 for robust, -1 for weak
        profitability_exposures = {}
        for symbol, op_ratio in operating_profitability.items():
            exposure = 1.0 if op_ratio > median_op else -1.0
            profitability_exposures[symbol] = exposure
        
        return profitability_exposures
    
    def calculate_investment_factor(
        self,
        asset_growth: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate CMA (Conservative Minus Aggressive) factor.
        
        Args:
            asset_growth: Dictionary of symbol to asset growth rate
            
        Returns:
            Dictionary of symbol to investment factor exposure
        """
        if not asset_growth:
            return {}
        
        # Calculate median asset growth
        ag = list(asset_growth.values())
        median_ag = np.median(ag)
        
        # Assign investment exposure: +1 for conservative (low growth), -1 for aggressive (high growth)
        investment_exposures = {}
        for symbol, growth_rate in asset_growth.items():
            exposure = 1.0 if growth_rate < median_ag else -1.0
            investment_exposures[symbol] = exposure
        
        return investment_exposures
    
    def calculate_factor_returns(
        self,
        stock_returns: Dict[str, float],
        market_return: float,
        size_exposures: Dict[str, float],
        value_exposures: Dict[str, float],
        profitability_exposures: Dict[str, float],
        investment_exposures: Dict[str, float]
    ) -> FactorReturns:
        """
        Calculate factor returns for a given period.
        
        Args:
            stock_returns: Dictionary of symbol to stock return
            market_return: Market return (excess of risk-free)
            size_exposures: Size factor exposures
            value_exposures: Value factor exposures
            profitability_exposures: Profitability factor exposures
            investment_exposures: Investment factor exposures
            
        Returns:
            FactorReturns object
        """
        # Market factor
        mkt = market_return
        
        # Size factor (SMB)
        if size_exposures:
            weighted_returns = [stock_returns[s] * size_exposures[s] 
                             for s in size_exposures if s in stock_returns]
            smb = np.mean(weighted_returns) if weighted_returns else 0.0
        else:
            smb = 0.0
        
        # Value factor (HML)
        if value_exposures:
            weighted_returns = [stock_returns[s] * value_exposures[s] 
                             for s in value_exposures if s in stock_returns]
            hml = np.mean(weighted_returns) if weighted_returns else 0.0
        else:
            hml = 0.0
        
        # Profitability factor (RMW)
        if profitability_exposures:
            weighted_returns = [stock_returns[s] * profitability_exposures[s] 
                             for s in profitability_exposures if s in stock_returns]
            rmw = np.mean(weighted_returns) if weighted_returns else 0.0
        else:
            rmw = 0.0
        
        # Investment factor (CMA)
        if investment_exposures:
            weighted_returns = [stock_returns[s] * investment_exposures[s] 
                             for s in investment_exposures if s in stock_returns]
            cma = np.mean(weighted_returns) if weighted_returns else 0.0
        else:
            cma = 0.0
        
        return FactorReturns(mkt=mkt, smb=smb, hml=hml, rmw=rmw, cma=cma)
    
    def estimate_factor_exposures(
        self,
        stock_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> FactorExposures:
        """
        Estimate factor exposures (betas) for a stock using regression.
        
        Args:
            stock_returns: Stock excess returns
            factor_returns: DataFrame with factor returns (MKT, SMB, HML, RMW, CMA)
            
        Returns:
            FactorExposures object
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare data
            X = factor_returns.values
            y = stock_returns.values
            
            # Fit regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Get betas
            betas = model.coef_
            
            return FactorExposures(
                mkt_beta=betas[0] if len(betas) > 0 else 0.0,
                smb_beta=betas[1] if len(betas) > 1 else 0.0,
                hml_beta=betas[2] if len(betas) > 2 else 0.0,
                rmw_beta=betas[3] if len(betas) > 3 else 0.0,
                cma_beta=betas[4] if len(betas) > 4 else 0.0
            )
        except Exception as e:
            logger.error(f"Failed to estimate factor exposures: {e}")
            return FactorExposures(0.0, 0.0, 0.0, 0.0, 0.0)


# Singleton instance
_ff5f_model = None

def get_ff5f_model() -> FamaFrench5Factor:
    """Get the singleton Fama-French 5-factor model instance."""
    global _ff5f_model
    if _ff5f_model is None:
        _ff5f_model = FamaFrench5Factor()
    return _ff5f_model


if __name__ == "__main__":
    # Test Fama-French 5-factor model
    print("Testing Fama-French 5-Factor Model...")
    
    model = FamaFrench5Factor()
    
    # Create sample data
    market_caps = {'STOCK_A': 1000, 'STOCK_B': 500, 'STOCK_C': 2000}
    book_to_market = {'STOCK_A': 1.5, 'STOCK_B': 0.8, 'STOCK_C': 1.2}
    operating_profitability = {'STOCK_A': 0.15, 'STOCK_B': 0.10, 'STOCK_C': 0.20}
    asset_growth = {'STOCK_A': 0.05, 'STOCK_B': 0.15, 'STOCK_C': 0.08}
    
    size_exposures = model.calculate_size_factor(market_caps)
    value_exposures = model.calculate_value_factor(book_to_market)
    profitability_exposures = model.calculate_profitability_factor(operating_profitability)
    investment_exposures = model.calculate_investment_factor(asset_growth)
    
    print("Size exposures:", size_exposures)
    print("Value exposures:", value_exposures)
    print("Profitability exposures:", profitability_exposures)
    print("Investment exposures:", investment_exposures)
