import pandas as pd
import os
from typing import Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CorporateActionAdjuster:
    """Apply splits and dividends adjustments to historical price series."""

    @staticmethod
    def adjust(df: pd.DataFrame, actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust price columns (open, high, low, close) and volume in df
        using corporate actions data in reverse chronological order.
        """
        if df.empty or actions_df.empty:
            return df

        df = df.copy()

        # Sort actions chronologically
        actions = actions_df.sort_values("date")

        # Ensure tz-naive DatetimeIndex
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)

        for _, action in actions.iterrows():
            action_date = pd.Timestamp(action["date"]).tz_localize(None)
            action_type = str(action["action_type"]).lower()
            val = float(action["value"])

            # Mask for all records strictly before the corporate action
            mask = df.index < action_date

            if not mask.any():
                continue

            if action_type == "split" and val > 0:
                for col in ["open", "high", "low", "close"]:
                    if col in df.columns:
                        df.loc[mask, col] = df.loc[mask, col] / val
                if "volume" in df.columns:
                    df.loc[mask, "volume"] = df.loc[mask, "volume"] * val
            elif action_type == "dividend" and val > 0:
                # Proportional adjustment: use close of day prior to ex-date
                prior_df = df[mask]
                if not prior_df.empty:
                    base_price = prior_df["close"].iloc[-1]
                    if base_price > 0:
                        adj_factor = (base_price - val) / base_price
                        for col in ["open", "high", "low", "close"]:
                            if col in df.columns:
                                df.loc[mask, col] = df.loc[mask, col] * adj_factor
        return df


class NSEDataLoader:
    """
    Loads real NSE/BSE 1-minute bar data from historical files and live WebSocket.
    
    CRITICAL FIX: Falls back to Yahoo Finance if parquet files don't exist.
    This ensures the system can run with real data even without local data files.
    """
    
    def __init__(self, data_dir: str = "/data/nse_bars", symbols: List[str] = None):
        self.data_dir = data_dir
        self.symbols = symbols or self._get_active_symbols()
        self.cache = {}
        self.live_stream = None
        self._using_yahoo_fallback = False
        
    def _get_active_symbols(self) -> List[str]:
        """Load current NIFTY500 + BANKNIFTY constituents."""
        csv_path = f"{self.data_dir}/nifty500_constituents.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df['symbol'].tolist()
        return ["NIFTY", "BANKNIFTY"]
    
    def get_historical_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load parquet file for given symbol and date range, and apply corporate action adjustments.
        
        CRITICAL FIX: Falls back to Yahoo Finance if parquet file doesn't exist.
        """
        file_path = f"{self.data_dir}/{symbol}/1min_bars.parquet"
        if not os.path.exists(file_path):
            logger.warning(f"No local data for {symbol}, attempting Yahoo Finance fallback")
            return self._get_yahoo_finance_data(symbol, start_date, end_date)
        
        df = pd.read_parquet(file_path)
        df = self._standardize_ohlcv(df, symbol)

        # Apply corporate action adjustments to raw 1min bars
        try:
            from src.data.truth import get_corporate_actions
            actions_df = get_corporate_actions(symbol)
            if not actions_df.empty:
                df = CorporateActionAdjuster.adjust(df, actions_df)
                logger.info(f"Dynamically adjusted {symbol} 1min bars for {len(actions_df)} corporate actions")
        except Exception as e:
            logger.error(f"Failed to adjust historical bars for corporate actions: {e}")

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        return df[(df.index >= start) & (df.index <= end)]
    
    def _get_yahoo_finance_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        P1-1 Fix: Fetch 5m intraday data from yfinance for recent dates,
        fallback to trusted daily prices from market_truth.db.
        """
        import yfinance as yf
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Attempt 5m fetch if within 60 days
        if (pd.Timestamp.now() - start).days < 59:
            try:
                yf_symbol = symbol + ".NS" if not symbol.startswith("^") else symbol
                if yf_symbol == "NIFTY.NS": yf_symbol = "^NSEI"
                if yf_symbol == "BANKNIFTY.NS": yf_symbol = "^NSEBANK"
                
                df_5m = yf.download(yf_symbol, start=start, end=end, interval="5m", progress=False)
                if not df_5m.empty:
                    # Handle yfinance multi-index columns
                    if isinstance(df_5m.columns, pd.MultiIndex):
                        df_5m.columns = [c[0].lower() for c in df_5m.columns]
                    else:
                        df_5m.columns = [c.lower() for c in df_5m.columns]
                    
                    df_5m["symbol"] = symbol
                    self._using_yahoo_fallback = True
                    logger.info(f"Loaded {len(df_5m)} rows of 5m data for {symbol} from yfinance")
                    return self._standardize_ohlcv(df_5m, symbol)
            except Exception as e:
                logger.warning(f"Failed to fetch 5m data for {symbol}: {e}")

        from src.data.truth import get_price_history, refresh_prices
        
        # Load from truth DB
        df_history = get_price_history(symbol, days=2000)
        if df_history.empty:
            logger.info(f"No trusted data for {symbol} found. Refreshing truth prices...")
            try:
                refresh_prices(period="5y")
            except Exception as e:
                logger.error(f"Failed to refresh truth prices: {e}")
            df_history = get_price_history(symbol, days=2000)
            
        if df_history.empty:
            logger.error(f"Could not load trusted data for {symbol}")
            return pd.DataFrame()
            
        # Filter by date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df_filtered = df_history[(df_history["date"] >= start) & (df_history["date"] <= end)].copy()
        
        # Format df to match standardize_ohlcv inputs
        df_filtered = df_filtered.set_index("date")
        df_filtered["symbol"] = symbol
        
        # Standardize and validate
        df = self._standardize_ohlcv(df_filtered, symbol)
        self._using_yahoo_fallback = True
        logger.info(f"Loaded {len(df)} rows of trusted data for {symbol} from truth DB")
        return df

    def _standardize_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Return a clean, single-symbol OHLCV frame with a DatetimeIndex."""
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "symbol"])

        normalized = df.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            if symbol in normalized.columns.get_level_values(-1):
                normalized = normalized.xs(symbol, axis=1, level=-1)
            elif len(normalized.columns.levels[-1]) == 1:
                normalized.columns = normalized.columns.droplevel(-1)
            else:
                normalized = normalized.droplevel(-1, axis=1)

        rename_map = {
            str(col): str(col).strip().lower().replace(" ", "_")
            for col in normalized.columns
        }
        normalized = normalized.rename(columns=rename_map)

        aliases: dict[str, str] = {
            "adj_close": "close",
            "last": "close",
            "ltp": "close",
            "qty": "volume",
            "vol": "volume",
        }
        normalized = normalized.rename(columns={k: v for k, v in aliases.items() if k in normalized.columns})

        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in normalized.columns]
        if missing:
            raise ValueError(f"OHLCV data for {symbol} missing columns: {missing}")

        if not isinstance(normalized.index, pd.DatetimeIndex):
            timestamp_col = next((col for col in ("timestamp", "date", "datetime") if col in normalized.columns), None)
            if timestamp_col is None:
                raise ValueError(f"OHLCV data for {symbol} has no DatetimeIndex or timestamp column")
            normalized[timestamp_col] = pd.to_datetime(normalized[timestamp_col], errors="coerce")
            normalized = normalized.set_index(timestamp_col)

        normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
        normalized = normalized.sort_index()
        normalized = normalized[~normalized.index.duplicated(keep="last")]

        for col in required:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        
        # CRITICAL FIX: Add data validation
        normalized = self._validate_ohlcv_data(normalized, symbol)
        
        normalized["symbol"] = symbol
        return normalized[required + ["symbol"]]
    
    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate OHLCV data for quality issues.
        
        CRITICAL FIX: Added comprehensive data validation to prevent garbage data.
        """
        required = ["open", "high", "low", "close", "volume"]
        
        # Remove rows with any NaN values in required columns
        df = df.dropna(subset=required)
        
        # Validate price continuity - prices must be positive
        for col in ["open", "high", "low", "close"]:
            df = df[df[col] > 0]
        
        # Validate volume - must be non-negative
        df = df[df["volume"] >= 0]
        
        # Validate OHLC consistency: high >= low, high >= open/close, low <= open/close
        df = df[df["high"] >= df["low"]]
        df = df[df["high"] >= df["open"]]
        df = df[df["high"] >= df["close"]]
        df = df[df["low"] <= df["open"]]
        df = df[df["low"] <= df["close"]]
        
        # Remove extreme outliers (more than 50% daily change)
        df["daily_change"] = (df["close"] - df["open"]) / df["open"]
        df = df[abs(df["daily_change"]) < 0.50]
        df = df.drop(columns=["daily_change"])
        
        # Remove zero-volume bars (unless it's the first bar)
        if len(df) > 1:
            df = df[df["volume"] > 0]
        
        # Log validation results
        if len(df) == 0:
            logger.warning(f"All data filtered out for {symbol} due to validation")
        else:
            logger.info(f"Validation passed for {symbol}: {len(df)} rows")
        
        return df
    
    def _symbol_to_yahoo_ticker(self, symbol: str) -> Optional[str]:
        """
        Map Indian symbol to Yahoo Finance ticker.
        """
        symbol_upper = symbol.upper()
        
        if symbol_upper == "NIFTY":
            return "^NSEI"
        elif symbol_upper == "BANKNIFTY":
            return "^NSEBANK"
        elif symbol_upper in ["NIFTY50", "NIFTY 50"]:
            return "^NSEI"
        elif symbol_upper in ["SENSEX", "S&P BSE SENSEX"]:
            return "^BSESN"
        
        clean_symbol = symbol_upper.replace(".NS", "").replace(".BO", "")
        
        ticker_map = {
            "RELIANCE": "RELIANCE.NS",
            "INFY": "INFY.NS",
            "HDFCBANK": "HDFCBANK.NS",
            "ICICIBANK": "ICICIBANK.NS",
            "TCS": "TCS.NS",
            "KOTAKBANK": "KOTAKBANK.NS",
            "HINDUNILVR": "HINDUNILVR.NS",
            "AXISBANK": "AXISBANK.NS",
            "BAJFINANCE": "BAJFINANCE.NS",
            "SBIN": "SBIN.NS",
        }
        
        if clean_symbol in ticker_map:
            return ticker_map[clean_symbol]
        
        return f"{clean_symbol}.NS"
    
    async def connect_live_stream(self):
        """Connect to NSE WebSocket for real-time 1-minute bars."""
        try:
            from kiteconnect import KiteTicker
        except ImportError:
            logger.error("kiteconnect not installed. Run: pip install kiteconnect")
            return None
        
        api_key = os.getenv('KITE_API_KEY')
        access_token = os.getenv('KITE_ACCESS_TOKEN')
        
        if not api_key or not access_token:
            logger.error("KITE_API_KEY and KITE_ACCESS_TOKEN must be set")
            return None
            
        kws = KiteTicker(api_key, access_token)
        
        def on_tick(ws, tick):
            self._aggregate_tick(tick)
        
        def on_connect(ws, response):
            ws.subscribe(self.symbols)
            ws.set_mode(ws.MODE_FULL, self.symbols)
        
        kws.on_ticks = on_tick
        kws.on_connect = on_connect
        kws.connect(threaded=True)
        return kws
        
    def _aggregate_tick(self, tick):
        from datetime import datetime
        
        if not hasattr(self, '_minute_bars'):
            self._minute_bars = {}
        
        instrument_token = tick.get('instrument_token')
        if instrument_token not in self._minute_bars:
            self._minute_bars[instrument_token] = {
                'open': tick.get('last_price'),
                'high': tick.get('last_price'),
                'low': tick.get('last_price'),
                'close': tick.get('last_price'),
                'volume': tick.get('last_traded_quantity', 0)
            }
        else:
            bar = self._minute_bars[instrument_token]
            bar['high'] = max(bar['high'], tick.get('last_price', bar['high']))
            bar['low'] = min(bar['low'], tick.get('last_price', bar['low']))
            bar['close'] = tick.get('last_price')
            bar['volume'] += tick.get('last_traded_quantity', 0)
        
        current_minute = datetime.now().replace(second=0, microsecond=0)
        if not hasattr(self, '_last_minute'):
            self._last_minute = current_minute
        
        if current_minute != self._last_minute:
            for token, bar in self._minute_bars.items():
                logger.info(f"Completed 1-minute bar for token {token}: {bar}")
            self._minute_bars = {}
            self._last_minute = current_minute
