import pandas as pd
import numpy as np

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range.
    df must contain columns: high, low, close.
    """
    required = {'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_atr requires columns {sorted(required)}; missing {sorted(missing)}")

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    prev_close = df['close'].astype(float).shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    return atr


def parkinson_volatility(df: pd.DataFrame, period: int = 14, annualize: bool = False) -> pd.Series:
    """Compute Parkinson range volatility from high/low bars."""
    required = {'high', 'low'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parkinson_volatility requires columns {sorted(required)}; missing {sorted(missing)}")

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    high_low_ratio = np.log(high / low)
    variance = (high_low_ratio ** 2).rolling(period).mean() / (4 * np.log(2))
    vol = np.sqrt(variance)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
