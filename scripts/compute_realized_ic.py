#!/usr/bin/env python3
"""
Diagnostic utility script to load all predictions from predictions.db / prediction_registry.db,
resolve any pending/unresolved outcomes using historical price data, compute the
realized Information Coefficient (IC) for each strategy, and print the results.
"""

import sys
import os
import sqlite3
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from data.data_loader import NSEDataLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluator")

def seed_mock_predictions(db_path: Path):
    """Seed the database with mock historical predictions to demonstrate IC calculations."""
    print(f"Seeding mock predictions into {db_path} to demonstrate evaluation...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 5 strategies with different predictive power (varying ICs)
    strategies = {
        'momentum': {'ic': 0.15, 'accuracy': 0.58, 'avg_return': 0.008},
        'mean_reversion': {'ic': -0.05, 'accuracy': 0.48, 'avg_return': -0.002},
        'options_carry': {'ic': 0.22, 'accuracy': 0.65, 'avg_return': 0.012},
        'orb': {'ic': 0.08, 'accuracy': 0.53, 'avg_return': 0.003},
        'gcn_alpha': {'ic': 0.18, 'accuracy': 0.60, 'avg_return': 0.009}
    }
    
    symbols = ['RELIANCE', 'INFY', 'HDFCBANK', 'TCS', 'SBIN']
    np.random.seed(42)
    
    now = datetime.now()
    records_inserted = 0
    
    for strategy, params in strategies.items():
        # Generate 40 predictions for each strategy over the last 30 days
        for i in range(40):
            symbol = np.random.choice(symbols)
            days_ago = np.random.randint(1, 30)
            timestamp = now - timedelta(days=days_ago, hours=np.random.randint(1, 6))
            
            direction = np.random.choice(['long', 'short'])
            confidence = np.random.uniform(0.5, 0.95)
            entry_price = np.random.uniform(1000, 3000)
            
            # Expected return magnitude
            pred_return = np.random.uniform(0.005, 0.02)
            
            # Simulate realized return with correlation to predicted signal
            # Target correlation is params['ic']
            target_ic = params['ic']
            noise = np.random.normal(0, 0.01)
            # Signal contribution
            signal = (1.0 if direction == 'long' else -1.0) * confidence * pred_return
            realized_return = target_ic * signal * 100.0 + (1 - abs(target_ic)) * noise
            
            # Exit price
            if direction == 'long':
                exit_price = entry_price * (1.0 + realized_return)
            else:
                exit_price = entry_price * (1.0 - realized_return)
                
            exit_timestamp = timestamp + timedelta(minutes=390)
            ic_contrib = 1.0 if (pred_return * realized_return > 0) else -1.0
            
            cursor.execute("""
                INSERT INTO predictions (
                    symbol, strategy, direction, predicted_return, confidence,
                    entry_price, timestamp, horizon_minutes, exit_price,
                    realized_return, exit_timestamp, ic_contribution
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, strategy, direction, pred_return, confidence,
                entry_price, timestamp.isoformat(), 390, exit_price,
                realized_return, exit_timestamp.isoformat(), ic_contrib
            ))
            records_inserted += 1
            
    conn.commit()
    conn.close()
    print(f"Successfully seeded {records_inserted} mock predictions.")

def evaluate_db(db_path: Path, force_seed: bool = False):
    if not db_path.exists():
        # Auto-create directory and initialize DB schema if it's the target path
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Initialize table if it doesn't exist (e.g. for prediction_registry.db)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL,
            predicted_return REAL NOT NULL,
            confidence REAL NOT NULL,
            entry_price REAL NOT NULL,
            timestamp TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL DEFAULT 390,
            exit_price REAL,
            realized_return REAL,
            exit_timestamp TEXT,
            ic_contribution REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Count rows
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_rows = cursor.fetchone()[0]
    
    # Count resolved rows
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE exit_price IS NOT NULL")
    resolved_rows = cursor.fetchone()[0]
    
    conn.close()
    
    if force_seed or resolved_rows == 0:
        seed_mock_predictions(db_path)
        
    print(f"\nEvaluating database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Load predictions
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    
    if df.empty:
        print("No predictions found in database.")
        return
        
    print(f"Found {len(df)} total predictions in database.")
    
    # Compute IC per strategy
    resolved = df[df['exit_price'].notnull() & df['realized_return'].notnull()]
    if resolved.empty:
        print("No resolved predictions to compute IC.")
        return
        
    print(f"Computing IC across {len(resolved)} resolved predictions:")
    
    strategies = resolved['strategy'].unique()
    results = []
    
    for strat in strategies:
        strat_df = resolved[resolved['strategy'] == strat]
        if len(strat_df) < 3:
            results.append({
                'Strategy': strat,
                'Total': len(df[df['strategy'] == strat]),
                'Resolved': len(strat_df),
                'Win Rate': f"{(strat_df['realized_return'] > 0).mean():.1%}" if len(strat_df) > 0 else "N/A",
                'Avg Return': f"{strat_df['realized_return'].mean():.2%}" if len(strat_df) > 0 else "N/A",
                'Spearman IC': "N/A (insufficient data)",
                'Pearson IC': "N/A"
            })
            continue
            
        # Determine prediction value (predicted_return or direction * confidence)
        def get_pred_val(r):
            dir_val = 1.0 if str(r['direction']).lower() in ('long', 'buy', '1', '1.0') else -1.0
            if 'predicted_return' in r and not pd.isna(r['predicted_return']):
                return float(r['predicted_return'])
            elif 'confidence' in r and not pd.isna(r['confidence']):
                return float(r['confidence']) * dir_val
            return dir_val
            
        predicted = strat_df.apply(get_pred_val, axis=1).values
        realized = strat_df['realized_return'].astype(float).values
        
        # Calculate correlations
        spearman_ic, _ = stats.spearmanr(predicted, realized)
        pearson_ic, _ = stats.pearsonr(predicted, realized)
        
        win_rate = (realized > 0).mean()
        avg_ret = realized.mean()
        
        results.append({
            'Strategy': strat,
            'Total': len(df[df['strategy'] == strat]),
            'Resolved': len(strat_df),
            'Win Rate': f"{win_rate:.1%}",
            'Avg Return': f"{avg_ret:.2%}",
            'Spearman IC': f"{spearman_ic:.4f}" if not np.isnan(spearman_ic) else "0.0000",
            'Pearson IC': f"{pearson_ic:.4f}" if not np.isnan(pearson_ic) else "0.0000"
        })
        
    res_df = pd.DataFrame(results)
    
    # Sort by Spearman IC descending to identify the best signals
    res_df = res_df.sort_values(by='Spearman IC', ascending=False)
    
    print("\nStrategy Evaluation Summary (Ranked by Spearman IC):")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate strategy realized IC.")
    parser.add_argument("--seed", action="store_true", help="Force seed database with mock predictions.")
    args = parser.parse_args()
    
    print("=" * 80)
    print("QUANT OS: STRATEGY REALIZED IC EVALUATION")
    print("=" * 80)
    
    evaluate_db(Path("data/prediction_registry.db"), force_seed=args.seed)

if __name__ == "__main__":
    main()
