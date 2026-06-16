"""
Time Utilities - Time alignment and resampling
"""

import pandas as pd
import numpy as np
from typing import Optional


def align_timeframes(data_dict: dict, timeframe: str = '1min') -> dict:
    """
    Align multiple time series to common timeframe
    
    Args:
        data_dict: Dict mapping symbol to DataFrame
        timeframe: Target timeframe (e.g., '1min', '5min', '1H')
        
    Returns:
        Dict with aligned DataFrames
    """
    aligned = {}
    
    # Get common dates
    if not data_dict:
        return aligned
    
    common_index = data_dict[list(data_dict.keys())[0]].index
    for data in data_dict.values():
        common_index = common_index.intersection(data.index)
    
    # Resample and align
    for symbol, data in data_dict.items():
        aligned_data = data.loc[common_index]
        if timeframe != '1min':
            aligned_data = resample_ohlcv(aligned_data, timeframe)
        aligned[symbol] = aligned_data
    
    return aligned


def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe
    
    Args:
        data: DataFrame with OHLCV columns
        timeframe: Target timeframe (e.g., '5min', '1H', '1D')
        
    Returns:
        Resampled DataFrame
    """
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        raise ValueError("Data must contain OHLC columns")
    
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    
    if 'volume' in data.columns:
        agg_dict['volume'] = 'sum'
    
    resampled = data.resample(timeframe).agg(agg_dict).dropna()
    
    return resampled
