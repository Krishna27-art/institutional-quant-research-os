import re
with open('dashboard/api/api_server.py', 'r') as f:
    content = f.read()

# Make the screener use all NIFTY50 and bulk download
new_screener = """
@app.get("/api/screener")
async def get_screener():
    \"\"\"Get screener data with real signals and features\"\"\"
    # CRITICAL FIX: Use real NIFTY50 symbols with AlphaManager signals and feature store
    try:
        symbols = get_nifty50_symbols()
        # Fallback if empty
        if not symbols:
            symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            
        data = []
        
        # Batch download for performance
        tickers = [f"{s}.NS" for s in symbols]
        try:
            # We download last 5 days
            hist_data = yf.download(tickers, period="5d", group_by="ticker", threads=True, progress=False)
        except Exception as e:
            hist_data = None
            
        for symbol in symbols:
            try:
                hist = None
                if hist_data is not None:
                    if len(symbols) == 1:
                        hist = hist_data
                    else:
                        hist = hist_data[f"{symbol}.NS"] if f"{symbol}.NS" in hist_data else pd.DataFrame()
                        
                if hist is None or hist.empty:
                    # fallback single fetch
                    ticker = yf.Ticker(f"{symbol}.NS")
                    hist = ticker.history(period="5d")
                    
                if hist is None or hist.empty:
                    continue
                
                # Get latest price and change
                latest_close = float(hist['Close'].iloc[-1])
                prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else latest_close
                change_pct = ((latest_close - prev_close) / prev_close) * 100 if len(hist) > 1 else 0.0
                
                # Market Hours Check
                market_stat = get_market_status_func()
                
                # Generate signals ONLY if market is open or pre-open, or return NEUTRAL otherwise to avoid false predictions when closed.
                if not market_stat.get("is_open", False) and not market_stat.get("is_pre_open", False):
                     signals = [] # No predictions when closed as requested by user
                else:
                     signals = alpha_manager.generate_signals(symbol, hist)
                
                # Get features from feature store if available
                try:
                    features = feature_store.get_features(f"{symbol}_features")
                    if features is not None and not features.empty:
                        latest_features = features.iloc[-1]
                        rv = float(latest_features.get('relative_volume', 1.0))
                        rsi = float(latest_features.get('rsi', 50))
                        conf = float(latest_features.get('confidence', 50))
                    else:
                        # Calculate basic features if not in store
                        volume = float(hist['Volume'].iloc[-1])
                        avg_volume = float(hist['Volume'].mean())
                        rv = volume / avg_volume if avg_volume > 0 else 1.0
                        rsi = 50.0
                        conf = 50.0
                except Exception:
                    # Fallback to basic calculations
                    volume = float(hist['Volume'].iloc[-1])
                    avg_volume = float(hist['Volume'].mean())
                    rv = volume / avg_volume if avg_volume > 0 else 1.0
                    rsi = 50.0
                    conf = 50.0
                
                # Get signal from AlphaManager
                if signals:
                    signal = signals[0]
                    direction = signal.get("direction", 0)
                    signal_str = "BUY" if direction > 0 else "SHORT" if direction < 0 else "NEUTRAL"
                    strength = float(signal.get("strength", signal.get("rv", 0.5)))
                    confidence = float(signal.get("confidence", 0.5))
                    target = float(signal.get("target", 0))
                    stop_loss = float(signal.get("stop", signal.get("stop_loss", 0)))
                    
                    # Calculate risk-reward ratio
                    if target > 0 and stop_loss > 0 and latest_close != stop_loss:
                        if direction > 0:
                            rr = (target - latest_close) / (latest_close - stop_loss)
                        else:
                            rr = (latest_close - target) / (stop_loss - latest_close)
                    else:
                        rr = 2.5
                else:
                    signal_str = "NEUTRAL"
                    strength = 0.5
                    confidence = 0.5
                    target = 0.0
                    stop_loss = 0.0
                    rr = 2.5
                
                data.append({
                    "symbol": symbol,
                    "signal": signal_str,
                    "price": round(latest_close, 2),
                    "change": round(change_pct, 2),
                    "rv": round(rv, 2),
                    "rsi": round(rsi, 2),
                    "conf": round(confidence * 100, 2) if confidence < 1 else round(confidence, 2),
                    "target": round(target, 2),
                    "sl": round(stop_loss, 2),
                    "rr": round(abs(rr), 2)
                })
                
            except Exception as e:
                # Skip symbol if data fetch fails
                continue
        
        # Format the return structure correctly matching the JS expectation
        return {"stocks": data} if data else {"data": data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "stocks": []}
"""

content = re.sub(r'@app\.get\("/api/screener"\).*?(?=@app\.get\("/api/metrics"\))', new_screener, content, flags=re.DOTALL)

with open('dashboard/api/api_server.py', 'w') as f:
    f.write(content)
print("Patched api_server.py")
