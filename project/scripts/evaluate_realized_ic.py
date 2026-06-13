#!/usr/bin/env python3
"""
Diagnostic script to evaluate realized Information Coefficient (IC) by joining
stored predictions with actual market prices from market_truth.db.
"""

import os
import sys
import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ic_evaluator")

def clean_symbol(symbol: str) -> str:
    """Normalize symbol names by stripping common suffixes and uppercasing."""
    if not symbol:
        return ""
    s = str(symbol).upper().strip()
    for suffix in [".NS", ".BO"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s

def load_market_data(db_path: Path) -> dict:
    """
    Load daily prices from market_truth.db and structure them for quick lookup.
    Returns a dict mapping symbol -> sorted list of (date_str, close_price, open_price).
    """
    if not db_path.exists():
        logger.error(f"Market truth database not found at {db_path}")
        return {}
    
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT symbol, date, open, close FROM daily_prices"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Failed to query daily_prices: {e}")
        return {}
    finally:
        conn.close()
        
    df["symbol"] = df["symbol"].apply(clean_symbol)
    df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
    
    market_data = {}
    for sym, group in df.groupby("symbol"):
        market_data[sym] = {
            "dates": group["date"].tolist(),
            "closes": group["close"].tolist(),
            "opens": group["open"].tolist(),
            "date_map": dict(zip(group["date"], range(len(group))))
        }
    
    logger.info(f"Loaded daily prices for {len(market_data)} symbols from {db_path}")
    return market_data

def get_market_price_at_horizon(
    symbol: str, 
    pred_timestamp_str: str, 
    horizon_days: int, 
    market_data: dict
) -> tuple:
    """
    Finds the market entry date, actual entry close/open, and actual exit price at the horizon.
    Returns: (entry_date, entry_price_actual, exit_date, exit_price_actual) or (None, None, None, None)
    """
    symbol = clean_symbol(symbol)
    if symbol not in market_data:
        return None, None, None, None
    
    sym_data = market_data[symbol]
    dates = sym_data["dates"]
    closes = sym_data["closes"]
    opens = sym_data["opens"]
    
    # Parse prediction date
    try:
        pred_date_str = pred_timestamp_str.split("T")[0]
    except Exception:
        pred_date_str = pred_timestamp_str
        
    # Find entry index: closest trading date on or after prediction date
    entry_idx = None
    for idx, d in enumerate(dates):
        if d >= pred_date_str:
            entry_idx = idx
            break
            
    if entry_idx is None:
        return None, None, None, None
    
    # Target exit index is entry_idx + horizon_days
    exit_idx = entry_idx + horizon_days
    if exit_idx >= len(dates):
        # We don't have enough history to resolve this prediction yet
        return None, None, None, None
        
    entry_date = dates[entry_idx]
    entry_price = closes[entry_idx]
    exit_date = dates[exit_idx]
    exit_price = closes[exit_idx]
    
    return entry_date, entry_price, exit_date, exit_price

def load_predictions(db_path: Path, db_type: str) -> pd.DataFrame:
    """Load and normalize predictions from the specified database."""
    if not db_path.exists():
        logger.warning(f"Prediction database not found at {db_path}")
        return pd.DataFrame()
        
    conn = sqlite3.connect(db_path)
    try:
        if db_type == "registry":
            query = """
                SELECT id, symbol, strategy, direction, predicted_return, 
                       confidence, entry_price, timestamp, horizon_minutes,
                       exit_price, realized_return, exit_timestamp
                FROM predictions
            """
        else: # compatibility / standard db
            query = """
                SELECT id, symbol, strategy, direction, confidence, 
                       entry_price, timestamp, target_price, stop_loss,
                       exit_price, realized_return, exit_timestamp
                FROM predictions
            """
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Failed to query predictions from {db_path}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
        
    if df.empty:
        return df
        
    # Normalize fields
    df["symbol"] = df["symbol"].apply(clean_symbol)
    df["direction_num"] = df["direction"].apply(
        lambda x: 1.0 if str(x).lower() in ("long", "buy", "1", "1.0") else -1.0
    )
    
    if "predicted_return" not in df.columns:
        # If predicted_return is not in db (like standard predictions.db),
        # use confidence * direction as a proxy for the signal strength/direction
        df["predicted_return"] = df["confidence"] * df["direction_num"]
        
    if "horizon_minutes" not in df.columns:
        df["horizon_minutes"] = 390 # default 1 day
        
    df["source_db"] = db_path.name
    return df

def evaluate_database(pred_db_path: Path, market_db_path: Path, db_type: str, override_horizon_days: int = None):
    """Run the evaluation on a specific prediction database."""
    print(f"\n================================================================================")
    print(f"EVALUATING PREDICTION DATABASE: {pred_db_path} ({db_type.upper()})")
    print(f"================================================================================")
    
    preds_df = load_predictions(pred_db_path, db_type)
    if preds_df.empty:
        print("No predictions found in database.")
        return
        
    print(f"Loaded {len(preds_df)} total prediction records.")
    
    market_data = load_market_data(market_db_path)
    if not market_data:
        print("Market data is empty. Cannot evaluate.")
        return
        
    # Join and resolve outcome for each prediction
    results = []
    
    for _, pred in preds_df.iterrows():
        # Determine horizon days
        if override_horizon_days is not None:
            horizon_days = override_horizon_days
        else:
            # 390 minutes represents ~1 trading day
            horizon_minutes = pred.get("horizon_minutes", 390)
            horizon_days = max(1, int(round(horizon_minutes / 390)))
            
        m_entry_date, m_entry_close, m_exit_date, m_exit_close = get_market_price_at_horizon(
            pred["symbol"], pred["timestamp"], horizon_days, market_data
        )
        
        if m_entry_date is None:
            # Could not resolve
            continue
            
        direction = pred["direction_num"]
        
        # Calculate returns:
        # 1. Stored Return: Stored realized return in the DB if available, otherwise from stored exit
        stored_ret = pred.get("realized_return")
        if pd.isna(stored_ret) and not pd.isna(pred.get("exit_price")) and pred["entry_price"] > 0:
            stored_ret = direction * (pred["exit_price"] - pred["entry_price"]) / pred["entry_price"]
            
        # 2. Market Realized Return (Stored Entry): Stored entry price vs actual market exit price
        stored_entry = pred["entry_price"]
        market_ret_stored_entry = None
        if stored_entry > 0:
            market_ret_stored_entry = direction * (m_exit_close - stored_entry) / stored_entry
            
        # 3. Pure Market Return: Actual market close at entry vs actual market close at exit
        pure_market_ret = direction * (m_exit_close - m_entry_close) / m_entry_close
        
        # Predicted value for correlation (either predicted return or confidence * direction)
        pred_val = float(pred["predicted_return"])
        # If predicted_return doesn't carry sign, align it with direction
        if pred_val >= 0 and pred["direction_num"] < 0 and db_type == "registry" and pred["strategy"] == "orb":
            # Just to be safe, make sure signal strength is signed
            pred_val = -pred_val if direction < 0 else pred_val
            
        results.append({
            "id": pred["id"],
            "symbol": pred["symbol"],
            "strategy": pred["strategy"],
            "direction": pred["direction"],
            "timestamp": pred["timestamp"],
            "entry_price_stored": stored_entry,
            "entry_price_market": m_entry_close,
            "exit_price_stored": pred.get("exit_price"),
            "exit_price_market": m_exit_close,
            "stored_return": stored_ret if not pd.isna(stored_ret) else None,
            "market_ret_stored_entry": market_ret_stored_entry,
            "pure_market_return": pure_market_ret,
            "predicted_val": pred_val,
            "confidence": pred["confidence"],
            "horizon_days": horizon_days,
            "entry_date": m_entry_date,
            "exit_date": m_exit_date
        })
        
    eval_df = pd.DataFrame(results)
    if eval_df.empty:
        print("Could not resolve any predictions against the market truth dates. (Check if dates in prediction and market truth overlap).")
        return
        
    print(f"Resolved and joined {len(eval_df)} predictions against market truth data.")
    
    # Verify price discrepancies
    eval_df["entry_price_pct_diff"] = abs(eval_df["entry_price_stored"] - eval_df["entry_price_market"]) / eval_df["entry_price_market"]
    avg_price_diff = eval_df["entry_price_pct_diff"].mean()
    print(f"Average entry price discrepancy (stored vs market close): {avg_price_diff:.2%}")
    if avg_price_diff > 0.05:
        print("WARNING: High price discrepancy detected! Check if stored entry prices are intraday or if symbols/units mismatch.")
        
    # Group by strategy and compute IC
    strategies = eval_df["strategy"].unique()
    summary_data = []
    
    for strat in strategies:
        strat_df = eval_df[eval_df["strategy"] == strat]
        n_resolved = len(strat_df)
        
        if n_resolved < 3:
            summary_data.append({
                "Strategy": strat,
                "Resolved": n_resolved,
                "Win Rate": f"{(strat_df['pure_market_return'] > 0).mean():.1%}" if n_resolved > 0 else "N/A",
                "Avg Pure Return": f"{strat_df['pure_market_return'].mean():.2%}" if n_resolved > 0 else "N/A",
                "Spearman IC": 0.0,
                "p-value": 1.0,
                "Significance": "Insufficient Data"
            })
            continue
            
        # Spearman correlation (Rank IC) between predicted_val and pure_market_return
        predicted = strat_df["predicted_val"].values
        realized = strat_df["pure_market_return"].values
        
        # Check if predictions are constant
        if np.all(predicted == predicted[0]) or np.all(realized == realized[0]):
            spearman_ic, p_val = 0.0, 1.0
        else:
            spearman_ic, p_val = stats.spearmanr(predicted, realized)
            
        win_rate = (realized > 0).mean()
        avg_ret = realized.mean()
        
        # Assess significance
        if p_val < 0.01:
            sig = "99% Significant"
        elif p_val < 0.05:
            sig = "95% Significant"
        elif p_val < 0.10:
            sig = "90% Significant"
        else:
            sig = "Not Significant"
            
        summary_data.append({
            "Strategy": strat,
            "Resolved": n_resolved,
            "Win Rate": f"{win_rate:.1%}",
            "Avg Pure Return": f"{avg_ret:.2%}",
            "Spearman IC": spearman_ic if not np.isnan(spearman_ic) else 0.0,
            "p-value": p_val if not np.isnan(p_val) else 1.0,
            "Significance": sig
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="Spearman IC", ascending=False)
    
    print("\nStrategy Realized IC Report (Ranked by Spearman IC against Market Close at Horizon):")
    print("=" * 100)
    print(summary_df.to_string(index=False, formatters={
        "Spearman IC": lambda x: f"{x:.4f}",
        "p-value": lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001"
    }))
    print("=" * 100)
    
    # Print a few samples for validation
    print("\nSample Resolved Predictions:")
    cols_to_show = ["symbol", "strategy", "direction", "timestamp", "entry_price_stored", "entry_price_market", "exit_price_market", "pure_market_return", "predicted_val"]
    print(eval_df[cols_to_show].head(10).to_string(index=False))
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate strategy realized IC against historical daily close prices.")
    parser.add_argument("--pred-db", type=str, help="Path to predictions database. If omitted, checks standard locations.")
    parser.add_argument("--market-db", type=str, default="data/market_truth.db", help="Path to market truth database.")
    parser.add_argument("--horizon-days", type=int, help="Override forecast horizon in trading days.")
    args = parser.parse_args()
    
    market_db_path = Path(args.market_db)
    
    if args.pred_db:
        pred_db = Path(args.pred_db)
        db_type = "registry" if "registry" in pred_db.name else "standard"
        evaluate_database(pred_db, market_db_path, db_type, args.horizon_days)
    else:
        # Check both databases in the default data folder
        registry_db = Path("data/prediction_registry.db")
        standard_db = Path("data/predictions.db")
        
        evaluated_any = False
        
        if registry_db.exists():
            evaluate_database(registry_db, market_db_path, "registry", args.horizon_days)
            evaluated_any = True
            
        if standard_db.exists():
            evaluate_database(standard_db, market_db_path, "standard", args.horizon_days)
            evaluated_any = True
            
        if not evaluated_any:
            print("No prediction databases found at data/prediction_registry.db or data/predictions.db.")

if __name__ == "__main__":
    main()
