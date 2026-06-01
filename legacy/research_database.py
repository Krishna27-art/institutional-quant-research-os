"""
Research Database with Regime-Conditional Evidence Matrix
Integrated from quant_research_platform folder

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class StrategyEvidence:
    """Evidence for strategy performance"""
    strategy_name: str
    volatility_regime: str
    participation_regime: str
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    last_updated: str


@dataclass
class RetiredStrategy:
    """Retired strategy with failure signature"""
    strategy_name: str
    profitable_regimes: List[Dict[str, str]]
    failure_regimes: List[Dict[str, str]]
    transition_signature: str
    mechanism_explanation: str
    retired_date: str


class ResearchDatabase:
    """
    SQLite database for research evidence and regime analysis.
    
    Features:
    - 3x3 Regime-Conditional Evidence Matrix (Vol x Participation)
    - Strategy performance tracking by regime
    - Retired strategy registry with failure signatures
    - Sensitivity analysis results storage
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Strategy evidence table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                volatility_regime TEXT NOT NULL,
                participation_regime TEXT NOT NULL,
                sharpe_ratio REAL,
                win_rate REAL,
                total_trades INTEGER,
                last_updated TEXT,
                UNIQUE(strategy_name, volatility_regime, participation_regime)
            )
        """)
        
        # Retired strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retired_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT UNIQUE NOT NULL,
                profitable_regimes TEXT,
                failure_regimes TEXT,
                transition_signature TEXT,
                mechanism_explanation TEXT,
                retired_date TEXT
            )
        """)
        
        # Sensitivity analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensitivity_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                parameter_value REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                analysis_date TEXT
            )
        """)
        
        # Trade registry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                trade_id TEXT,
                entry_time TEXT,
                exit_time TEXT,
                symbol TEXT,
                pnl REAL,
                regime_volatility TEXT,
                regime_participation TEXT
            )
        """)
        
        self.conn.commit()
    
    def save_regime_matrix(
        self,
        strategy_name: str,
        trades: List[Dict]
    ) -> None:
        """
        Save regime-conditional evidence matrix.
        
        Args:
            strategy_name: Name of strategy
            trades: List of trade dictionaries with regime information
        """
        cursor = self.conn.cursor()
        
        # Group trades by regime
        regime_stats: Dict[Tuple[str, str], List[float]] = {}
        
        for trade in trades:
            vol_regime = trade.get('regime_volatility', 'Medium')
            part_regime = trade.get('regime_participation', 'Balanced')
            pnl = trade.get('pnl', 0.0)
            
            key = (vol_regime, part_regime)
            if key not in regime_stats:
                regime_stats[key] = []
            regime_stats[key].append(pnl)
        
        # Calculate statistics for each regime cell
        for (vol_regime, part_regime), pnls in regime_stats.items():
            if len(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO strategy_evidence
                    (strategy_name, volatility_regime, participation_regime, sharpe_ratio, win_rate, total_trades, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_name,
                    vol_regime,
                    part_regime,
                    sharpe,
                    win_rate,
                    len(pnls),
                    datetime.now().isoformat()
                ))
        
        self.conn.commit()
    
    def register_retired_strategy(
        self,
        strategy_name: str,
        profitable_regimes: List[Dict[str, str]],
        failure_regimes: List[Dict[str, str]],
        transition_signature: str,
        mechanism_explanation: str
    ) -> None:
        """
        Register a retired strategy with failure signature.
        
        Args:
            strategy_name: Name of retired strategy
            profitable_regimes: List of regimes where strategy was profitable
            failure_regimes: List of regimes where strategy failed
            transition_signature: Market conditions that caused failure
            mechanism_explanation: Explanation of why strategy failed
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO retired_strategies
            (strategy_name, profitable_regimes, failure_regimes, transition_signature, mechanism_explanation, retired_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            strategy_name,
            json.dumps(profitable_regimes),
            json.dumps(failure_regimes),
            transition_signature,
            mechanism_explanation,
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
    
    def get_strategy_evidence(self, strategy_name: str) -> List[StrategyEvidence]:
        """
        Get evidence for a strategy across regimes.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            List of StrategyEvidence objects
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT strategy_name, volatility_regime, participation_regime, sharpe_ratio, win_rate, total_trades, last_updated
            FROM strategy_evidence
            WHERE strategy_name = ?
            ORDER BY sharpe_ratio DESC
        """, (strategy_name,))
        
        rows = cursor.fetchall()
        
        evidence = [
            StrategyEvidence(
                strategy_name=row[0],
                volatility_regime=row[1],
                participation_regime=row[2],
                sharpe_ratio=row[3],
                win_rate=row[4],
                total_trades=row[5],
                last_updated=row[6]
            )
            for row in rows
        ]
        
        return evidence
    
    def get_retired_strategies(self) -> List[RetiredStrategy]:
        """Get all retired strategies."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT strategy_name, profitable_regimes, failure_regimes, transition_signature, mechanism_explanation, retired_date
            FROM retired_strategies
            ORDER BY retired_date DESC
        """)
        
        rows = cursor.fetchall()
        
        retired = [
            RetiredStrategy(
                strategy_name=row[0],
                profitable_regimes=json.loads(row[1]),
                failure_regimes=json.loads(row[2]),
                transition_signature=row[3],
                mechanism_explanation=row[4],
                retired_date=row[5]
            )
            for row in rows
        ]
        
        return retired
    
    def print_evidence_matrix_dashboard(self, strategy_name: str) -> str:
        """
        Print formatted evidence matrix dashboard.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Formatted string representation
        """
        evidence = self.get_strategy_evidence(strategy_name)
        
        if not evidence:
            return f"No evidence found for strategy: {strategy_name}"
        
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"REGIME-CONDITIONAL EVIDENCE MATRIX: {strategy_name}")
        output.append(f"{'='*80}")
        output.append(f"\n{'Volatility Regime':<20} {'Participation':<20} {'Sharpe':<10} {'Win Rate':<10} {'Trades':<10}")
        output.append(f"{'-'*80}")
        
        for ev in evidence:
            output.append(
                f"{ev.volatility_regime:<20} {ev.participation_regime:<20} "
                f"{ev.sharpe_ratio:>8.2f} {ev.win_rate:>8.2%} {ev.total_trades:>8}"
            )
        
        output.append(f"{'-'*80}")
        
        return "\n".join(output)
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
