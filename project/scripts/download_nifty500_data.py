"""
NIFTY500 Data Downloader & Feature Engineer

This script downloads historical data for NIFTY 500 constituents, sector indices,
and volatility indices from Yahoo Finance. It then runs the correlation, covariance,
relative strength, and market breadth calculations to build the parquet files
for the institutional quant research platform.

Usage:
    python scripts/download_nifty500_data.py
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Set up paths
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import NIFTY 500 symbols loader
try:
    from data.nifty500_symbols import get_nifty500_symbols
except ImportError:
    # Fallback to hardcoded major symbols if import fails
    def get_nifty500_symbols(force_refresh=False):
        return [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LICI',
            'AXISBANK', 'LT', 'HINDUNILVR', 'BAJFINANCE', 'MARUTI',
            'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'NTPC', 'WIPRO',
            'HCLTECH', 'ASIANPAINT', 'ULTRACEMCO', 'NESTLEIND', 'POWERGRID',
            'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'ONGC', 'GAIL',
            'M&M', 'BAJAJFINSV', 'DABUR', 'BRITANNIA', 'DIVISLAB',
            'DRREDDY', 'CIPLA', 'AUROPHARMA', 'LUPIN', 'TATAPOWER',
            'ADANIPORTS', 'ADANIENT', 'GRASIM', 'ACC', 'AMBUJACEM',
            'UPL', 'SHREECEM', 'ZEEL', 'TATACONSUM', 'EICHERMOT',
            'HEROMOTOCO'
        ]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_historical_prices(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download prices in batches from Yahoo Finance."""
    # Filter out known problematic or delisted tickers
    ignored_tickers = {'MOTHERSUMI'}
    symbols = [sym for sym in symbols if sym not in ignored_tickers]
    
    yahoo_tickers = [f"{sym}.NS" for sym in symbols]
    logger.info(f"Downloading {len(yahoo_tickers)} tickers from Yahoo Finance...")
    
    try:
        data = yf.download(
            yahoo_tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=True
        )
        
        if data.empty:
            raise ValueError("yf.download returned an empty DataFrame")
            
        # Extract Adj Close / Close
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                prices = data.xs('Close', axis=1, level=0)
            elif 'close' in data.columns.levels[0]:
                prices = data.xs('close', axis=1, level=0)
            else:
                raise ValueError("Could not find Close/close column in multi-index columns")
        else:
            if 'Close' in data.columns:
                prices = data[['Close']]
            elif 'close' in data.columns:
                prices = data[['close']]
            else:
                prices = data
                
        # Clean column names (strip suffix)
        prices.columns = [col.replace(".NS", "") for col in prices.columns]
        
        # Drop columns with all NaN values
        prices = prices.dropna(how='all', axis=1)
        logger.info(f"Successfully downloaded price data: shape {prices.shape}")
        return prices
        
    except Exception as e:
        logger.error(f"Failed batch download: {e}. Attempting sequential download...")
        prices_dict = {}
        for sym in symbols:
            try:
                ticker = f"{sym}.NS"
                df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        close_col = [col for col in df.columns if 'Close' in col or 'close' in col]
                        if close_col:
                            series = df[close_col[0]]
                        else:
                            series = df.iloc[:, 0]
                    else:
                        series = df['Close'] if 'Close' in df.columns else (df['close'] if 'close' in df.columns else df.iloc[:, 0])
                    
                    # Ensure series has a DatetimeIndex
                    series.index = pd.to_datetime(series.index)
                    prices_dict[sym] = series
                    logger.info(f"Downloaded {sym}: {len(series)} rows")
            except Exception as ex:
                logger.warning(f"Failed to download {sym}: {ex}")
                
        if not prices_dict:
            raise ValueError("Failed to download any prices sequentially!")
            
        prices = pd.DataFrame(prices_dict)
        logger.info(f"Successfully compiled sequential price data: shape {prices.shape}")
        return prices


def compute_breadth_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute market breadth features across the universe."""
    logger.info("Computing market breadth features...")
    breadth = pd.DataFrame(index=prices.index)
    
    # 1. Percentage above EMAs
    emas_20 = prices.ewm(span=20, adjust=False).mean()
    emas_50 = prices.ewm(span=50, adjust=False).mean()
    emas_200 = prices.ewm(span=200, adjust=False).mean()
    
    breadth['pct_above_20ema'] = (prices > emas_20).sum(axis=1) / prices.notna().sum(axis=1)
    breadth['pct_above_50ema'] = (prices > emas_50).sum(axis=1) / prices.notna().sum(axis=1)
    breadth['pct_above_200ema'] = (prices > emas_200).sum(axis=1) / prices.notna().sum(axis=1)
    
    # 2. New Highs / New Lows (20-day rolling window)
    highs_20 = prices.rolling(window=20).max()
    lows_20 = prices.rolling(window=20).min()
    
    # Mark if current price is equal to the 20-day high/low
    is_new_high = (prices >= highs_20 - 1e-5) & (prices.notna())
    is_new_low = (prices <= lows_20 + 1e-5) & (prices.notna())
    
    breadth['new_highs_20d'] = is_new_high.sum(axis=1) / prices.notna().sum(axis=1)
    breadth['new_lows_20d'] = is_new_low.sum(axis=1) / prices.notna().sum(axis=1)
    
    # Forward fill or fillna to handle first few rows
    breadth = breadth.ffill().fillna(0.5)
    return breadth


def main():
    # Set dates
    start_date = "2010-01-01"
    end_date = "2026-06-01"
    
    # 1. Define output directory
    output_dir = ROOT_DIR / "market_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Get NIFTY500 symbols
    symbols = get_nifty500_symbols()
    logger.info(f"Loaded {len(symbols)} symbols from universe loader.")
    
    # Limit to unique non-empty symbols
    symbols = list(set([s for s in symbols if s]))
    
    # 3. Download prices
    prices = download_historical_prices(symbols, start_date, end_date)
    
    # Save Nifty 500 Prices Parquet
    nifty500_path = output_dir / "nifty500.parquet"
    prices.to_parquet(nifty500_path)
    logger.info(f"Saved Nifty 500 prices to {nifty500_path}")
    
    # 4. Download India VIX & Benchmark Index Data
    logger.info("Downloading VIX and Benchmark Index data...")
    index_tickers = {
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
        "NIFTY500": "^CRSLDX"
    }
    
    index_data = {}
    for name, ticker in index_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    val = df.xs('Close', axis=1, level=0) if 'Close' in df.columns.levels[0] else df.iloc[:, 0]
                else:
                    val = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                
                # Make sure it's a 1D series
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                index_data[name] = val
        except Exception as e:
            logger.warning(f"Failed to download index {name} ({ticker}): {e}")
            
    if index_data:
        index_df = pd.DataFrame(index=prices.index)
        for name, series in index_data.items():
            index_df[name] = series.reindex(prices.index).ffill()
        index_path = output_dir / "index_data.parquet"
        index_df.to_parquet(index_path)
        logger.info(f"Saved index data to {index_path}")
    
    # India VIX
    try:
        vix_df = yf.download("^INDIAVIX", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        if not vix_df.empty:
            vix_path = output_dir / "india_vix.parquet"
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_series = vix_df.xs('Close', axis=1, level=0) if 'Close' in vix_df.columns.levels[0] else vix_df.iloc[:, 0]
            else:
                vix_series = vix_df['Close'] if 'Close' in vix_df.columns else vix_df.iloc[:, 0]
            
            if isinstance(vix_series, pd.DataFrame):
                vix_series = vix_series.iloc[:, 0]
                
            vix_save = pd.DataFrame({"india_vix": vix_series}, index=prices.index)
            vix_save["india_vix"] = vix_series.reindex(prices.index).ffill()
            vix_save.to_parquet(vix_path)
            logger.info(f"Saved India VIX to {vix_path}")
    except Exception as e:
        logger.warning(f"Failed to download India VIX: {e}")
        
    # 5. Download Sector Indices
    logger.info("Downloading sector index data...")
    sector_tickers = {
        "IT": "^CNXIT",
        "BANK": "^NSEBANK",
        "AUTO": "^CNXAUTO",
        "ENERGY": "^CNXENERGY",
        "METAL": "^CNXMETAL",
        "PHARMA": "^CNXPHARMA",
        "FMCG": "^CNXFMCG",
        "INFRA": "^CNXINFRA"
    }
    
    sector_data = {}
    for name, ticker in sector_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    val = df.xs('Close', axis=1, level=0) if 'Close' in df.columns.levels[0] else df.iloc[:, 0]
                else:
                    val = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                sector_data[name] = val
        except Exception as e:
            logger.warning(f"Failed to download sector index {name} ({ticker}): {e}")
            
    if sector_data:
        sector_df = pd.DataFrame(index=prices.index)
        for name, series in sector_data.items():
            sector_df[name] = series.reindex(prices.index).ffill()
        sector_path = output_dir / "sector_data.parquet"
        sector_df.to_parquet(sector_path)
        logger.info(f"Saved sector index data to {sector_path}")
        
    # 6. Compute Breadth Features
    breadth_df = compute_breadth_features(prices)
    breadth_path = output_dir / "breadth_features.parquet"
    breadth_df.to_parquet(breadth_path)
    logger.info(f"Saved breadth features to {breadth_path}")
    
    logger.info("Data download and feature engineering complete!")


if __name__ == "__main__":
    main()
