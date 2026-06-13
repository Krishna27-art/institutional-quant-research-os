"""
Arbitrage Pricing Theory (APT) Factor Engine
Based on the critique: Build Factor Engine, Factor Exposure, Factor Neutralizer

Instead of:
    Price features only

Build factor model with multiple factors:
- Market
- Size
- Value
- Quality
- Momentum
- Volatility
- Liquidity
- Options
- Flows

Then train LightGBM on factor residuals for huge improvement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.linear_model import LinearRegression


class FactorType(Enum):
    """Types of factors."""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    QUALITY = "quality"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    OPTIONS = "options"
    FLOWS = "flows"


@dataclass
class Factor:
    """Factor definition."""
    name: str
    factor_type: FactorType
    description: str
    data_source: str


@dataclass
class FactorExposure:
    """Factor exposure for a stock."""
    symbol: str
    timestamp: datetime
    factor_name: str
    exposure: float
    z_score: float
    percentile: float


@dataclass
class FactorModel:
    """Factor model for a stock."""
    symbol: str
    timestamp: datetime
    factor_exposures: Dict[str, float]
    r_squared: float
    residual_return: float


class APTFactorEngine:
    """
    Arbitrage Pricing Theory Factor Engine.
    
    Features:
    - Factor definitions
    - Factor exposure calculation
    - Factor neutralization
    - Factor model fitting
    - Residual return calculation
    """
    
    def __init__(self):
        self.factors: Dict[str, Factor] = {}
        self.factor_exposures: Dict[str, List[FactorExposure]] = {}
        self.factor_models: Dict[str, List[FactorModel]] = {}
        
        # Initialize standard factors
        self._initialize_standard_factors()
    
    def _initialize_standard_factors(self):
        """Initialize standard APT factors."""
        factors = [
            Factor("Market", FactorType.MARKET, "Market return", "market_data"),
            Factor("Size", FactorType.SIZE, "Market capitalization", "fundamental_data"),
            Factor("Value", FactorType.VALUE, "Book-to-market ratio", "fundamental_data"),
            Factor("Quality", FactorType.QUALITY, "Profitability metrics", "fundamental_data"),
            Factor("Momentum", FactorType.MOMENTUM, "12-month price momentum", "price_data"),
            Factor("Volatility", FactorType.VOLATILITY, "Return volatility", "price_data"),
            Factor("Liquidity", FactorType.LIQUIDITY, "Trading volume / market cap", "price_data"),
            Factor("Options", FactorType.OPTIONS, "Options flow / IV", "options_data"),
            Factor("Flows", FactorType.FLOWS, "FII/DII flows", "flow_data"),
        ]
        
        for factor in factors:
            self.factors[factor.name] = factor
    
    def calculate_factor_exposure(
        self,
        symbol: str,
        timestamp: datetime,
        price_data: pd.DataFrame,
        fundamental_data: Optional[Dict] = None,
        options_data: Optional[Dict] = None,
        flow_data: Optional[Dict] = None
    ) -> Dict[str, FactorExposure]:
        """
        Calculate factor exposures for a stock.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp
            price_data: Price data (OHLCV)
            fundamental_data: Fundamental data (market cap, book value, etc.)
            options_data: Options data (IV, options flow)
            flow_data: Flow data (FII/DII flows)
            
        Returns:
            Dictionary of factor exposures
        """
        exposures = {}
        
        # Market factor (beta)
        if len(price_data) > 60:
            market_return = price_data['close'].pct_change().iloc[-60:].mean()
            stock_return = price_data['close'].pct_change().iloc[-60:].mean()
            market_exposure = stock_return / market_return if market_return != 0 else 1.0
        else:
            market_exposure = 1.0
        
        exposures['Market'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Market',
            exposure=market_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Size factor (log market cap)
        if fundamental_data and 'market_cap' in fundamental_data:
            size_exposure = np.log(fundamental_data['market_cap'])
        else:
            size_exposure = np.log(price_data['close'].iloc[-1] * 1000000)  # Proxy
        
        exposures['Size'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Size',
            exposure=size_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Value factor (book-to-market)
        if fundamental_data and 'book_value' in fundamental_data and 'market_cap' in fundamental_data:
            value_exposure = fundamental_data['book_value'] / fundamental_data['market_cap']
        else:
            value_exposure = 1.0 / price_data['close'].iloc[-1]  # Proxy
        
        exposures['Value'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Value',
            exposure=value_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Quality factor (profitability)
        if fundamental_data and 'roe' in fundamental_data:
            quality_exposure = fundamental_data['roe']
        else:
            quality_exposure = price_data['close'].pct_change().iloc[-252:].mean() * 252  # Proxy
        
        exposures['Quality'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Quality',
            exposure=quality_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Momentum factor (12-month momentum)
        if len(price_data) > 252:
            momentum_exposure = (price_data['close'].iloc[-1] / price_data['close'].iloc[-252]) - 1
        else:
            momentum_exposure = price_data['close'].pct_change().iloc[-60:].sum()
        
        exposures['Momentum'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Momentum',
            exposure=momentum_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Volatility factor
        if len(price_data) > 20:
            volatility_exposure = price_data['close'].pct_change().iloc[-20:].std()
        else:
            volatility_exposure = 0.02
        
        exposures['Volatility'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Volatility',
            exposure=volatility_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Liquidity factor
        if len(price_data) > 20:
            avg_volume = price_data['volume'].iloc[-20:].mean()
            market_cap = price_data['close'].iloc[-1] * 1000000
            liquidity_exposure = avg_volume / market_cap
        else:
            liquidity_exposure = 0.01
        
        exposures['Liquidity'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Liquidity',
            exposure=liquidity_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Options factor
        if options_data and 'iv' in options_data:
            options_exposure = options_data['iv']
        else:
            options_exposure = price_data['close'].pct_change().iloc[-20:].std()  # Proxy
        
        exposures['Options'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Options',
            exposure=options_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Flows factor
        if flow_data and 'fii_dii_flow' in flow_data:
            flows_exposure = flow_data['fii_dii_flow']
        else:
            flows_exposure = 0.0
        
        exposures['Flows'] = FactorExposure(
            symbol=symbol,
            timestamp=timestamp,
            factor_name='Flows',
            exposure=flows_exposure,
            z_score=0.0,
            percentile=0.5
        )
        
        # Store in history
        for factor_name, exposure in exposures.items():
            if symbol not in self.factor_exposures:
                self.factor_exposures[symbol] = []
            self.factor_exposures[symbol].append(exposure)
        
        return exposures
    
    def normalize_factor_exposures(
        self,
        symbol: str,
        exposures: Dict[str, FactorExposure],
        cross_sectional_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, FactorExposure]:
        """
        Normalize factor exposures using z-scores.
        
        Args:
            symbol: Trading symbol
            exposures: Factor exposures
            cross_sectional_data: Cross-sectional data for normalization
            
        Returns:
            Normalized factor exposures
        """
        normalized = {}
        
        for factor_name, exposure in exposures.items():
            if cross_sectional_data and factor_name in cross_sectional_data:
                # Calculate cross-sectional z-score
                all_values = [data.get(symbol, exposure.exposure) for symbol, data in cross_sectional_data.items()]
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                
                if std_val > 0:
                    z_score = (exposure.exposure - mean_val) / std_val
                else:
                    z_score = 0.0
                
                # Calculate percentile
                percentile = len([v for v in all_values if v < exposure.exposure]) / len(all_values)
            else:
                z_score = 0.0
                percentile = 0.5
            
            normalized[factor_name] = FactorExposure(
                symbol=symbol,
                timestamp=exposure.timestamp,
                factor_name=factor_name,
                exposure=exposure.exposure,
                z_score=z_score,
                percentile=percentile
            )
        
        return normalized
    
    def fit_factor_model(
        self,
        symbol: str,
        returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> FactorModel:
        """
        Fit factor model to stock returns.
        
        R_i = alpha + sum(beta_j * F_j) + epsilon
        
        Args:
            symbol: Trading symbol
            returns: Stock returns
            factor_returns: DataFrame of factor returns
            
        Returns:
            FactorModel
        """
        # Align data
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        
        if len(aligned) < 30:
            return FactorModel(
                symbol=symbol,
                timestamp=datetime.now(),
                factor_exposures={},
                r_squared=0.0,
                residual_return=0.0
            )
        
        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        # Calculate residual return
        predicted = model.predict(X)
        residual_return = y.iloc[-1] - predicted[-1]
        
        # Get factor exposures (betas)
        factor_exposures = {}
        for i, factor_name in enumerate(X.columns):
            factor_exposures[factor_name] = model.coef_[i]
        
        factor_model = FactorModel(
            symbol=symbol,
            timestamp=datetime.now(),
            factor_exposures=factor_exposures,
            r_squared=r_squared,
            residual_return=residual_return
        )
        
        # Store in history
        if symbol not in self.factor_models:
            self.factor_models[symbol] = []
        self.factor_models[symbol].append(factor_model)
        
        return factor_model
    
    def neutralize_factor_exposure(
        self,
        signal: float,
        factor_exposure: float,
        target_exposure: float = 0.0
    ) -> float:
        """
        Neutralize factor exposure from signal.
        
        Neutralized signal = signal - factor_exposure * hedge_ratio
        
        Args:
            signal: Original signal
            factor_exposure: Factor exposure
            target_exposure: Target exposure (default 0 for neutral)
            
        Returns:
            Factor-neutralized signal
        """
        hedge_ratio = factor_exposure - target_exposure
        neutralized_signal = signal - hedge_ratio * 0.5  # Adjust hedge ratio
        
        return neutralized_signal
    
    def get_factor_exposure_summary(self, symbol: str) -> pd.DataFrame:
        """Get summary of factor exposures for a symbol."""
        if symbol not in self.factor_exposures or not self.factor_exposures[symbol]:
            return pd.DataFrame()
        
        data = []
        for exposure in self.factor_exposures[symbol][-10:]:  # Last 10
            data.append({
                'Timestamp': exposure.timestamp,
                'Factor': exposure.factor_name,
                'Exposure': exposure.exposure,
                'Z-Score': exposure.z_score,
                'Percentile': exposure.percentile
            })
        
        return pd.DataFrame(data)
    
    def get_factor_model_summary(self, symbol: str) -> pd.DataFrame:
        """Get summary of factor models for a symbol."""
        if symbol not in self.factor_models or not self.factor_models[symbol]:
            return pd.DataFrame()
        
        data = []
        for model in self.factor_models[symbol][-10:]:  # Last 10
            data.append({
                'Timestamp': model.timestamp,
                'R-Squared': model.r_squared,
                'Residual Return': model.residual_return,
                **model.factor_exposures
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the APT Factor Engine
    print("Testing Arbitrage Pricing Theory Factor Engine...")
    
    engine = APTFactorEngine()
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    price_data = pd.DataFrame({
        'open': np.random.normal(100, 10, n).cumsum(),
        'high': np.random.normal(100, 10, n).cumsum(),
        'low': np.random.normal(100, 10, n).cumsum(),
        'close': np.random.normal(100, 10, n).cumsum(),
        'volume': np.random.normal(1000000, 200000, n)
    }, index=dates)
    
    fundamental_data = {
        'market_cap': 1000000000000,
        'book_value': 500000000000,
        'roe': 0.15
    }
    
    options_data = {
        'iv': 0.25
    }
    
    flow_data = {
        'fii_dii_flow': 100000000
    }
    
    # Calculate factor exposures
    print("\nCalculating Factor Exposures...")
    exposures = engine.calculate_factor_exposure(
        symbol="RELIANCE",
        timestamp=datetime.now(),
        price_data=price_data,
        fundamental_data=fundamental_data,
        options_data=options_data,
        flow_data=flow_data
    )
    
    print("\nFactor Exposures:")
    for factor_name, exposure in exposures.items():
        print(f"  {factor_name}: {exposure.exposure:.4f}")
    
    # Normalize exposures
    print("\nNormalizing Factor Exposures...")
    cross_sectional_data = {
        'RELIANCE': {'Market': 1.2, 'Size': 20.0, 'Value': 0.5},
        'TCS': {'Market': 0.9, 'Size': 18.0, 'Value': 0.4},
        'HDFCBANK': {'Market': 1.1, 'Size': 19.0, 'Value': 0.6}
    }
    
    normalized = engine.normalize_factor_exposures("RELIANCE", exposures, cross_sectional_data)
    
    print("\nNormalized Factor Exposures:")
    for factor_name, exposure in normalized.items():
        print(f"  {factor_name}: Z-Score={exposure.z_score:.2f}, Percentile={exposure.percentile:.2%}")
    
    # Fit factor model
    print("\nFitting Factor Model...")
    returns = price_data['close'].pct_change().dropna()
    
    factor_returns = pd.DataFrame({
        'Market': np.random.normal(0.0005, 0.015, len(returns)),
        'Size': np.random.normal(0.0001, 0.01, len(returns)),
        'Value': np.random.normal(0.0002, 0.008, len(returns)),
        'Momentum': np.random.normal(0.0003, 0.012, len(returns))
    }, index=returns.index)
    
    factor_model = engine.fit_factor_model("RELIANCE", returns, factor_returns)
    
    print(f"R-Squared: {factor_model.r_squared:.2%}")
    print(f"Residual Return: {factor_model.residual_return:.4%}")
    print(f"Factor Exposures:")
    for factor_name, exposure in factor_model.factor_exposures.items():
        print(f"  {factor_name}: {exposure:.4f}")
    
    # Neutralize factor exposure
    print("\nNeutralizing Factor Exposure...")
    signal = 0.8
    factor_exposure = 0.5
    neutralized = engine.neutralize_factor_exposure(signal, factor_exposure)
    print(f"Original Signal: {signal:.2f}")
    print(f"Factor Exposure: {factor_exposure:.2f}")
    print(f"Neutralized Signal: {neutralized:.2f}")
    
    # Get summaries
    print("\nFactor Exposure Summary:")
    exposure_summary = engine.get_factor_exposure_summary("RELIANCE")
    print(exposure_summary.to_string(index=False))
    
    print("\nFactor Model Summary:")
    model_summary = engine.get_factor_model_summary("RELIANCE")
    print(model_summary.to_string(index=False))
