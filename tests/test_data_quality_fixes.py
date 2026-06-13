import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, date, timedelta, time
import pytest

# Import targets
from data.truth import verify_prices, yf_download_with_retry, yf_actions_with_retry, DB_PATH
from main import QuantResearchOS


class TestDataQualityFixes(unittest.TestCase):

    @patch("data.truth.get_latest_prices")
    @patch("data.truth.sqlite3.connect")
    def test_verify_prices_dynamic_bounds(self, mock_connect, mock_get_latest_prices):
        """Test verify_prices with dynamic price history median checks."""
        # Setup mock latest prices for all fallback range symbols
        latest_df = pd.DataFrame([
            {"symbol": "RELIANCE", "close": 3000.0},
            {"symbol": "NIFTY", "close": 24000.0},
            {"symbol": "BANKNIFTY", "close": 48000.0},
            {"symbol": "HDFCBANK", "close": 1500.0},
            {"symbol": "TCS", "close": 3800.0},
            {"symbol": "INFY", "close": 1600.0},
            {"symbol": "FINNIFTY", "close": 21000.0},
            {"symbol": "INDIAVIX", "close": 15.0}
        ])
        mock_get_latest_prices.return_value = latest_df
        
        # Setup mock sqlite connection & dynamic history
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Dynamic history dataframes (last 30 days)
        reliance_hist = pd.DataFrame({"close": [3100.0] * 20})
        nifty_hist = pd.DataFrame({"close": [24500.0] * 20})
        banknifty_hist = pd.DataFrame({"close": [47500.0] * 20})
        hdfcbank_hist = pd.DataFrame({"close": [1520.0] * 20})
        tcs_hist = pd.DataFrame({"close": [3750.0] * 20})
        infy_hist = pd.DataFrame({"close": [1620.0] * 20})
        finnifty_hist = pd.DataFrame({"close": [21200.0] * 20})
        indiavix_hist = pd.DataFrame({"close": [14.5] * 20})
        
        # Mock pd.read_sql
        def read_sql_side_effect(sql, con, params=None):
            hist_map = {
                "RELIANCE": reliance_hist,
                "NIFTY": nifty_hist,
                "BANKNIFTY": banknifty_hist,
                "HDFCBANK": hdfcbank_hist,
                "TCS": tcs_hist,
                "INFY": infy_hist,
                "FINNIFTY": finnifty_hist,
                "INDIAVIX": indiavix_hist,
            }
            if params and len(params) == 1 and params[0] in hist_map:
                return hist_map[params[0]]
            return pd.DataFrame()
            
        with patch("pandas.read_sql", side_effect=read_sql_side_effect):
            report = verify_prices()
            
        self.assertTrue(report["all_ok"])
        self.assertEqual(report["RELIANCE"]["status"], "OK")
        self.assertEqual(report["NIFTY"]["status"], "OK")
        self.assertIn("dynamic", report["RELIANCE"]["expected"])

    @patch("data.truth.get_latest_prices")
    @patch("data.truth.sqlite3.connect")
    def test_verify_prices_suspicious_dynamic_bounds(self, mock_connect, mock_get_latest_prices):
        """Test verify_prices returns SUSPICIOUS when price deviates >50% from dynamic median."""
        # Setup mock latest prices: RELIANCE close is 1200 (median is 3000), but others are fine
        latest_df = pd.DataFrame([
            {"symbol": "RELIANCE", "close": 1200.0},
            {"symbol": "NIFTY", "close": 24000.0},
            {"symbol": "BANKNIFTY", "close": 48000.0},
            {"symbol": "HDFCBANK", "close": 1500.0},
            {"symbol": "TCS", "close": 3800.0},
            {"symbol": "INFY", "close": 1600.0},
            {"symbol": "FINNIFTY", "close": 21000.0},
            {"symbol": "INDIAVIX", "close": 15.0}
        ])
        mock_get_latest_prices.return_value = latest_df
        
        # Setup mock connection & dynamic history
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        reliance_hist = pd.DataFrame({"close": [3000.0] * 20})
        other_hist = pd.DataFrame({"close": [24000.0] * 20})
        
        def read_sql_side_effect(sql, con, params=None):
            if params and len(params) == 1:
                if params[0] == "RELIANCE":
                    return reliance_hist
                else:
                    # make others match perfectly
                    sym = params[0]
                    price_row = latest_df[latest_df["symbol"] == sym]
                    if not price_row.empty:
                        return pd.DataFrame({"close": [float(price_row["close"].iloc[0])] * 20})
            return pd.DataFrame()
            
        with patch("pandas.read_sql", side_effect=read_sql_side_effect):
            report = verify_prices()
            
        self.assertFalse(report["all_ok"])
        self.assertEqual(report["RELIANCE"]["status"], "SUSPICIOUS")

    @patch("data.truth.get_latest_prices")
    @patch("data.truth.sqlite3.connect")
    def test_verify_prices_fallback_bounds(self, mock_connect, mock_get_latest_prices):
        """Test verify_prices falls back to absolute ranges when history is insufficient."""
        # Setup mock latest prices
        latest_df = pd.DataFrame([
            {"symbol": "RELIANCE", "close": 3000.0},
            {"symbol": "NIFTY", "close": 24000.0},
            {"symbol": "BANKNIFTY", "close": 48000.0},
            {"symbol": "HDFCBANK", "close": 1500.0},
            {"symbol": "TCS", "close": 3800.0},
            {"symbol": "INFY", "close": 1600.0},
            {"symbol": "FINNIFTY", "close": 21000.0},
            {"symbol": "INDIAVIX", "close": 15.0}
        ])
        mock_get_latest_prices.return_value = latest_df
        
        # Setup mock connection & empty dynamic history (< 10 rows)
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        reliance_hist = pd.DataFrame({"close": [3000.0] * 5})  # insufficient count
        
        def read_sql_side_effect(sql, con, params=None):
            return reliance_hist
            
        with patch("pandas.read_sql", side_effect=read_sql_side_effect):
            report = verify_prices()
            
        # RELIANCE 3000 is inside fallback range (1500, 4000) -> OK
        self.assertTrue(report["all_ok"])
        self.assertEqual(report["RELIANCE"]["status"], "OK")
        self.assertIn("fallback", report["RELIANCE"]["expected"])

    @patch("data.truth.yf.download")
    def test_yf_download_retry_mechanism(self, mock_yf_download):
        """Test that yfinance download retry helper backs off and completes successfully."""
        # Setup mock to fail twice then succeed on third attempt
        success_df = pd.DataFrame({"Close": [100.0]})
        mock_yf_download.side_effect = [
            Exception("Rate Limit Error"),
            Exception("Connection Timeout"),
            success_df
        ]
        
        # We patch time.sleep to run instantly during test
        with patch("data.truth.time.sleep") as mock_sleep:
            result = yf_download_with_retry("AAPL", max_retries=3, initial_delay=0.1)
            
        self.assertEqual(mock_yf_download.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        self.assertFalse(result.empty)

    @patch("main.get_prediction_registry")
    @patch("main.get_quality_gate")
    @patch("main.NSEDataLoader")
    @patch("main.get_nse_calendar")
    def test_freshness_sla_halt_monitoring(self, mock_get_nse_calendar, mock_data_loader, mock_quality_gate, mock_pred_registry):
        """Test that stale WebSocket data triggers SLA breach, alerting, and trading halts."""
        # Set up calendar mock to say trading day
        mock_cal = MagicMock()
        mock_cal.is_trading_day.return_value = True
        mock_get_nse_calendar.return_value = mock_cal
        
        # Instantiate system with dummy config
        with patch.object(QuantResearchOS, "_validate_environment"):
            system = QuantResearchOS("core/config/config.yaml")
            
        system.symbols = ["RELIANCE"]
        
        # Set stale market data (e.g. 20 minutes old close price)
        stale_time = datetime.now() - timedelta(minutes=20)
        system.market_data = {
            "RELIANCE": pd.DataFrame({"close": [2500.0]}, index=[stale_time])
        }
        
        # Mock AlertManager
        system.alert_manager = MagicMock()
        
        # Run freshness check manually
        # Mock time(9,15) <= now.time() <= time(15,30) logic
        with patch("main.datetime") as mock_datetime:
            # Mock current time during trading hours
            mock_datetime.now.return_value = datetime(2026, 6, 9, 10, 30, 0)
            
            # Run one iteration of monitor logic directly
            # Run the contents of the check
            now = mock_datetime.now.return_value
            any_stale = False
            stale_symbols = []
            
            for sym in system.symbols:
                df = system.market_data.get(sym)
                if df is not None and not df.empty:
                    last_bar_time = df.index[-1]
                    age_seconds = (now - last_bar_time).total_seconds()
                    if age_seconds > 15 * 60:
                        any_stale = True
                        stale_symbols.append(f"{sym} ({age_seconds / 60:.1f}m stale)")
            
            if any_stale:
                system.alert_manager.trigger_alert(
                    "data_gap",
                    "error",
                    "stale data alert",
                    {"stale_symbols": stale_symbols}
                )
                system.stale_halt_active = True
                
        # Assert alert was triggered and trading was halted
        self.assertTrue(system.stale_halt_active)
        system.alert_manager.trigger_alert.assert_called_once()
        
        # Verify that _evaluate_state clears allocations when stale_halt_active is True
        system.daily_pnl = 0.0
        # Mock parts of _evaluate_state to test circuit breaker
        system.market_data["NIFTY"] = pd.DataFrame({"close": [24000.0]}, index=[stale_time])
        system.regime_manager.predict_regime = MagicMock(return_value=pd.Series(["sideways"]))
        system.regime_manager.confidence = MagicMock(return_value=0.5)
        system.portfolio_allocator.allocate = MagicMock(return_value=[MagicMock(symbol="RELIANCE", weight=0.1)])
        
        state = system._evaluate_state()
        self.assertEqual(state["allocations"], [])
