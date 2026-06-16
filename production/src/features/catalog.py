"""
Feature Catalog Definitions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import pandas as pd
import numpy as np


class FeatureCategory(Enum):
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    BREADTH = "breadth"
    SECTOR = "sector"
    OPTIONS = "options"
    CROSS_ASSET = "cross_asset"
    SENTIMENT = "sentiment"


@dataclass
class FeatureDefinition:
    """Definition of a single feature"""
    name: str
    category: FeatureCategory
    formula: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    update_frequency: str = "1min"  # 1min, 5min, 1hour, 1day
    complexity: str = "O(1)"  # O(1), O(N), O(N log N), etc.
    version: str = "v1"
    deprecated: bool = False
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute the feature from raw data"""
        raise NotImplementedError("Subclasses must implement compute method")


class FeatureRegistry:
    """Registry of all feature definitions"""
    
    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._initialize_default_features()
    
    def register(self, feature: FeatureDefinition) -> None:
        """Register a new feature"""
        self._features[feature.name] = feature
    
    def get(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature by name"""
        return self._features.get(name)
    
    def list_by_category(self, category: FeatureCategory) -> List[FeatureDefinition]:
        """List all features in a category"""
        return [f for f in self._features.values() if f.category == category]
    
    def list_all(self) -> List[FeatureDefinition]:
        """List all features"""
        return list(self._features.values())
    
    def _initialize_default_features(self) -> None:
        """Initialize default feature definitions"""
        
        # Price Features
        self.register(FeatureDefinition(
            name="returns_1d",
            category=FeatureCategory.PRICE,
            formula="log(P(t)/P(t-1))",
            description="1-day log returns",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="returns_5d",
            category=FeatureCategory.PRICE,
            formula="log(P(t)/P(t-5))",
            description="5-day log returns",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="distance_to_sma20",
            category=FeatureCategory.PRICE,
            formula="(P - SMA20) / SMA20",
            description="Distance to 20-day SMA",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="distance_to_sma50",
            category=FeatureCategory.PRICE,
            formula="(P - SMA50) / SMA50",
            description="Distance to 50-day SMA",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="distance_to_sma200",
            category=FeatureCategory.PRICE,
            formula="(P - SMA200) / SMA200",
            description="Distance to 200-day SMA",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="pct_from_52w_high",
            category=FeatureCategory.PRICE,
            formula="(P - 52w_high) / 52w_high",
            description="Percentage from 52-week high",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="pct_from_52w_low",
            category=FeatureCategory.PRICE,
            formula="(P - 52w_low) / 52w_low",
            description="Percentage from 52-week low",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="gap_pct",
            category=FeatureCategory.PRICE,
            formula="(open - prev_close) / prev_close",
            description="Gap percentage from previous close",
            update_frequency="1min",
            complexity="O(1)"
        ))

        self.register(FeatureDefinition(
            name="fracdiff_close_d04",
            category=FeatureCategory.PRICE,
            formula="fixed-width fractional difference of close with d=0.4",
            description="Stationary long-memory close feature using fractional differencing",
            parameters={"d": 0.4, "threshold": 1e-3},
            dependencies=["close"],
            update_frequency="1day",
            complexity="O(NW)",
            version="v2"
        ))

        self.register(FeatureDefinition(
            name="chaos_logistic_return",
            category=FeatureCategory.PRICE,
            formula="3.8 * x * (1 - x), x = trailing min-max normalized return",
            description="Logistic chaotic-map transform of normalized log returns",
            parameters={"r": 3.8, "normalization_window": 63},
            dependencies=["close"],
            update_frequency="1day",
            complexity="O(N)",
            version="v2"
        ))

        self.register(FeatureDefinition(
            name="chaos_tent_return",
            category=FeatureCategory.PRICE,
            formula="1.8*x if x<0.5 else 1.8*(1-x)",
            description="Tent chaotic-map transform of normalized log returns",
            parameters={"mu": 1.8, "normalization_window": 63},
            dependencies=["close"],
            update_frequency="1day",
            complexity="O(N)",
            version="v2"
        ))
        
        # Volume Features
        self.register(FeatureDefinition(
            name="volume_ratio_20d",
            category=FeatureCategory.VOLUME,
            formula="V / SMA(V, 20)",
            description="Volume ratio to 20-day average",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="volume_zscore",
            category=FeatureCategory.VOLUME,
            formula="(V - mean) / std",
            description="Volume Z-score (20-day window)",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="obv",
            category=FeatureCategory.VOLUME,
            formula="cumulative(sign(ret) * V)",
            description="On-Balance Volume",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        # Volatility Features
        self.register(FeatureDefinition(
            name="atr_14",
            category=FeatureCategory.VOLATILITY,
            formula="mean(TR, 14)",
            description="Average True Range (14-day)",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="realized_vol_10d",
            category=FeatureCategory.VOLATILITY,
            formula="std(ret) * sqrt(252)",
            description="10-day realized volatility (annualized)",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="realized_vol_21d",
            category=FeatureCategory.VOLATILITY,
            formula="std(ret) * sqrt(252)",
            description="21-day realized volatility (annualized)",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="realized_vol_63d",
            category=FeatureCategory.VOLATILITY,
            formula="std(ret) * sqrt(252)",
            description="63-day realized volatility (annualized)",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="parkinson_vol",
            category=FeatureCategory.VOLATILITY,
            formula="sqrt((1/(4*ln2)) * mean(ln(H/L)^2))",
            description="Parkinson volatility estimator",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="vol_of_vol",
            category=FeatureCategory.VOLATILITY,
            formula="std(vol, 21)",
            description="Volatility of volatility (21-day)",
            update_frequency="1min",
            complexity="O(N)"
        ))

        self.register(FeatureDefinition(
            name="hurst_60d",
            category=FeatureCategory.VOLATILITY,
            formula="slope(log(lag), log(std(r_t - r_{t-lag})))",
            description="Rolling 60-day Hurst exponent for long-memory and rough-volatility regimes",
            parameters={"window": 60, "max_lag": 20},
            dependencies=["close"],
            update_frequency="1day",
            complexity="O(NL)",
            version="v2"
        ))

        self.register(FeatureDefinition(
            name="rough_vol_regime_60d",
            category=FeatureCategory.VOLATILITY,
            formula="1 if hurst_60d < 0.45 else 0",
            description="Rough-volatility regime indicator derived from rolling Hurst exponent",
            parameters={"hurst_threshold": 0.45},
            dependencies=["hurst_60d"],
            update_frequency="1day",
            complexity="O(NL)",
            version="v2"
        ))
        
        # Breadth Features
        self.register(FeatureDefinition(
            name="advance_decline_ratio",
            category=FeatureCategory.BREADTH,
            formula="advances / declines",
            description="Advance-Decline ratio",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="pct_above_ma50",
            category=FeatureCategory.BREADTH,
            formula="count(P > MA50) / total",
            description="Percentage of stocks above 50-day MA",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="pct_above_ma200",
            category=FeatureCategory.BREADTH,
            formula="count(P > MA200) / total",
            description="Percentage of stocks above 200-day MA",
            update_frequency="1min",
            complexity="O(N)"
        ))
        
        self.register(FeatureDefinition(
            name="new_highs_minus_lows",
            category=FeatureCategory.BREADTH,
            formula="NH - NL",
            description="New highs minus new lows",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        # Options Features
        self.register(FeatureDefinition(
            name="put_call_ratio_volume",
            category=FeatureCategory.OPTIONS,
            formula="Put Volume / Call Volume",
            description="Put-Call Ratio (volume)",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="put_call_ratio_oi",
            category=FeatureCategory.OPTIONS,
            formula="Put OI / Call OI",
            description="Put-Call Ratio (open interest)",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="iv_skew_25delta",
            category=FeatureCategory.OPTIONS,
            formula="IV_put - IV_call",
            description="IV Skew (25-delta)",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="iv_term_structure",
            category=FeatureCategory.OPTIONS,
            formula="IV_1w - IV_1m",
            description="IV Term Structure",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        # Cross-Asset Features
        self.register(FeatureDefinition(
            name="usdinr_returns",
            category=FeatureCategory.CROSS_ASSET,
            formula="log(USDINR(t)/USDINR(t-1))",
            description="USD/INR returns",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="crude_returns",
            category=FeatureCategory.CROSS_ASSET,
            formula="log(CRUDE(t)/CRUDE(t-1))",
            description="Crude oil returns",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="gold_returns",
            category=FeatureCategory.CROSS_ASSET,
            formula="log(GOLD(t)/GOLD(t-1))",
            description="Gold returns",
            update_frequency="1min",
            complexity="O(1)"
        ))
        
        # Sentiment Features
        self.register(FeatureDefinition(
            name="fii_net_buy_sell",
            category=FeatureCategory.SENTIMENT,
            formula="FII buy - FII sell",
            description="FII net buy/sell",
            update_frequency="1day",
            complexity="O(1)"
        ))
        
        self.register(FeatureDefinition(
            name="dii_net_buy_sell",
            category=FeatureCategory.SENTIMENT,
            formula="DII buy - DII sell",
            description="DII net buy/sell",
            update_frequency="1day",
            complexity="O(1)"
        ))


feature_registry = FeatureRegistry()
