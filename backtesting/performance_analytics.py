"""
Backtest Performance Analytics
Comprehensive performance metrics for strategy evaluation

Architecture V2 - Quantitative Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk metrics
    volatility_annual: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    recovery_factor: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    
    # Holding period metrics
    avg_holding_minutes: float
    median_holding_minutes: float
    
    # Risk-adjusted metrics
    information_ratio: float  # vs benchmark
    beta: float
    alpha: float
    
    # Tail risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    
    # Other metrics
    skewness: float
    kurtosis: float
    hit_rate: float


class PerformanceAnalytics:
    """
    Comprehensive performance analytics for backtesting.
    
    Calculates all metrics required for Go/No-Go decision:
    - Sharpe Ratio > 1.0
    - Max Drawdown < 12%
    - Win rate, profit factor, etc.
    """
    
    def __init__(self, initial_capital: float = 10000000):
        self.initial_capital = initial_capital
    
    def calculate_metrics(
        self,
        equity_curve: np.ndarray,
        trades: List[Dict],
        benchmark_returns: Optional[np.ndarray] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: Array of equity values over time
            trades: List of trade dictionaries
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with all metrics
        """
        # Return metrics
        total_return, annualized_return, cagr = self._calculate_return_metrics(equity_curve)
        
        # Risk metrics
        volatility_annual, sharpe_ratio, sortino_ratio, calmar_ratio = self._calculate_risk_metrics(
            equity_curve, benchmark_returns
        )
        
        # Drawdown metrics
        max_dd, max_dd_duration, avg_dd, recovery_factor = self._calculate_drawdown_metrics(equity_curve)
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(trades)
        
        # Holding period metrics
        avg_hold, median_hold = self._calculate_holding_period_metrics(trades)
        
        # Risk-adjusted metrics vs benchmark
        info_ratio, beta, alpha = self._calculate_risk_adjusted_metrics(
            equity_curve, benchmark_returns
        )
        
        # Tail risk metrics
        var_95, var_99, cvar_95 = self._calculate_tail_risk_metrics(equity_curve)
        
        # Distribution metrics
        skewness, kurtosis = self._calculate_distribution_metrics(equity_curve)
        
        # Hit rate (daily positive returns)
        hit_rate = self._calculate_hit_rate(equity_curve)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            volatility_annual=volatility_annual,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd,
            recovery_factor=recovery_factor,
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            avg_trade=trade_metrics['avg_trade'],
            avg_holding_minutes=avg_hold,
            median_holding_minutes=median_hold,
            information_ratio=info_ratio,
            beta=beta,
            alpha=alpha,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            hit_rate=hit_rate
        )
    
    def _calculate_return_metrics(self, equity_curve: np.ndarray) -> Tuple[float, float, float]:
        """Calculate return metrics."""
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0.0
        
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Assume daily data
        days = len(equity_curve)
        years = days / 252
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        cagr = (equity_curve[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
        
        return total_return, annualized_return, cagr
    
    def _calculate_risk_metrics(
        self,
        equity_curve: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """Calculate risk metrics."""
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate daily returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Annual volatility
        volatility_annual = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Calmar ratio (return / max drawdown)
        max_dd = self._calculate_max_drawdown(equity_curve)
        cagr = (equity_curve[-1] / self.initial_capital) ** (252 / len(equity_curve)) - 1
        calmar_ratio = cagr / abs(max_dd) if max_dd != 0 else 0.0
        
        return volatility_annual, sharpe_ratio, sortino_ratio, calmar_ratio
    
    def _calculate_drawdown_metrics(self, equity_curve: np.ndarray) -> Tuple[float, int, float, float]:
        """Calculate drawdown metrics."""
        if len(equity_curve) < 2:
            return 0.0, 0, 0.0, 0.0
        
        # Calculate drawdown curve
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Calculate max drawdown duration
        # Find the start of the drawdown period
        peak_idx = np.argmax(running_max[:max_dd_idx + 1])
        max_dd_duration = max_dd_idx - peak_idx
        
        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        avg_dd = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0.0
        
        # Recovery factor (average recovery speed)
        # Simplified: ratio of final value to max drawdown
        recovery_factor = (equity_curve[-1] - equity_curve[max_dd_idx]) / abs(equity_curve[max_dd_idx] - running_max[max_dd_idx]) if max_dd != 0 else 0.0
        
        return abs(max_dd), max_dd_duration, abs(avg_dd), recovery_factor
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trade-based metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0
            }
        
        # Extract PnL from trades
        pnls = [t.get('pnl', 0) for t in trades]
        
        total_trades = len(trades)
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        avg_trade = np.mean(pnls) if pnls else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade
        }
    
    def _calculate_holding_period_metrics(self, trades: List[Dict]) -> Tuple[float, float]:
        """Calculate holding period metrics."""
        if not trades:
            return 0.0, 0.0
        
        holding_periods = [t.get('holding_minutes', 0) for t in trades]
        
        avg_hold = np.mean(holding_periods) if holding_periods else 0.0
        median_hold = np.median(holding_periods) if holding_periods else 0.0
        
        return avg_hold, median_hold
    
    def _calculate_risk_adjusted_metrics(
        self,
        equity_curve: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """Calculate risk-adjusted metrics vs benchmark."""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0, 0.0, 0.0
        
        # Calculate strategy returns
        strategy_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Align lengths
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0.0
        
        # Beta
        if np.var(benchmark_returns) > 0:
            beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 0.0
        
        # Alpha (Jensen's alpha)
        alpha = np.mean(strategy_returns) - beta * np.mean(benchmark_returns)
        
        return information_ratio, beta, alpha
    
    def _calculate_tail_risk_metrics(self, equity_curve: np.ndarray) -> Tuple[float, float, float]:
        """Calculate tail risk metrics (VaR, CVaR)."""
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # VaR at 95% and 99% confidence levels
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR at 95% (average of worst 5% returns)
        var_95_threshold = np.percentile(returns, 5)
        worst_5_percent = returns[returns <= var_95_threshold]
        cvar_95 = np.mean(worst_5_percent) if len(worst_5_percent) > 0 else var_95
        
        return abs(var_95), abs(var_99), abs(cvar_95)
    
    def _calculate_distribution_metrics(self, equity_curve: np.ndarray) -> Tuple[float, float]:
        """Calculate distribution metrics (skewness, kurtosis)."""
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return skewness, kurtosis
    
    def _calculate_hit_rate(self, equity_curve: np.ndarray) -> float:
        """Calculate hit rate (percentage of positive days)."""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        positive_days = sum(1 for r in returns if r > 0)
        hit_rate = positive_days / len(returns) if len(returns) > 0 else 0.0
        
        return hit_rate
    
    def print_metrics(self, metrics: PerformanceMetrics) -> None:
        """Print performance metrics in a formatted way."""
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE ANALYTICS")
        print("="*70)
        
        print("\nRETURN METRICS")
        print("-" * 70)
        print(f"Total Return:          {metrics.total_return:.2%}")
        print(f"Annualized Return:     {metrics.annualized_return:.2%}")
        print(f"CAGR:                  {metrics.cagr:.2%}")
        
        print("\nRISK METRICS")
        print("-" * 70)
        print(f"Annual Volatility:     {metrics.volatility_annual:.2%}")
        print(f"Sharpe Ratio:           {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:          {metrics.sortino_ratio:.2f}")
        print(f"Calmar Ratio:           {metrics.calmar_ratio:.2f}")
        
        print("\nDRAWDOWN METRICS")
        print("-" * 70)
        print(f"Max Drawdown:          {metrics.max_drawdown:.2%}")
        print(f"Max DD Duration:       {metrics.max_drawdown_duration} days")
        print(f"Avg Drawdown:          {metrics.avg_drawdown:.2%}")
        print(f"Recovery Factor:       {metrics.recovery_factor:.2f}")
        
        print("\nTRADE METRICS")
        print("-" * 70)
        print(f"Total Trades:          {metrics.total_trades}")
        print(f"Winning Trades:        {metrics.winning_trades}")
        print(f"Losing Trades:         {metrics.losing_trades}")
        print(f"Win Rate:              {metrics.win_rate:.2%}")
        print(f"Profit Factor:         {metrics.profit_factor:.2f}")
        print(f"Avg Win:               ₹{metrics.avg_win:,.2f}")
        print(f"Avg Loss:              ₹{metrics.avg_loss:,.2f}")
        print(f"Avg Trade:             ₹{metrics.avg_trade:,.2f}")
        
        print("\nHOLDING PERIOD METRICS")
        print("-" * 70)
        print(f"Avg Holding:           {metrics.avg_holding_minutes:.1f} minutes")
        print(f"Median Holding:        {metrics.median_holding_minutes:.1f} minutes")
        
        print("\nRISK-ADJUSTED METRICS")
        print("-" * 70)
        print(f"Information Ratio:     {metrics.information_ratio:.2f}")
        print(f"Beta:                  {metrics.beta:.2f}")
        print(f"Alpha:                 {metrics.alpha:.2%}")
        
        print("\nTAIL RISK METRICS")
        print("-" * 70)
        print(f"VaR (95%):             {metrics.var_95:.2%}")
        print(f"VaR (99%):             {metrics.var_99:.2%}")
        print(f"CVaR (95%):            {metrics.cvar_95:.2%}")
        
        print("\nDISTRIBUTION METRICS")
        print("-" * 70)
        print(f"Skewness:              {metrics.skewness:.2f}")
        print(f"Kurtosis:              {metrics.kurtosis:.2f}")
        print(f"Hit Rate:              {metrics.hit_rate:.2%}")
        
        print("\n" + "="*70)
        print("GO/NO-GO ASSESSMENT")
        print("="*70)
        
        go_criteria = []
        no_go_criteria = []
        
        if metrics.sharpe_ratio >= 1.0:
            go_criteria.append(f"✓ Sharpe {metrics.sharpe_ratio:.2f} >= 1.0")
        else:
            no_go_criteria.append(f"✗ Sharpe {metrics.sharpe_ratio:.2f} < 1.0")
        
        if metrics.max_drawdown <= 0.12:
            go_criteria.append(f"✓ Max DD {metrics.max_drawdown:.2%} <= 12%")
        else:
            no_go_criteria.append(f"✗ Max DD {metrics.max_drawdown:.2%} > 12%")
        
        if go_criteria:
            print("\nPASS CRITERIA:")
            for criterion in go_criteria:
                print(f"  {criterion}")
        
        if no_go_criteria:
            print("\nFAIL CRITERIA:")
            for criterion in no_go_criteria:
                print(f"  {criterion}")
        
        print("\n" + "="*70)
        
        if len(no_go_criteria) == 0:
            print("DECISION: ✓ GO - Proceed to live trading")
        else:
            print("DECISION: ✗ NO-GO - Continue paper trading")
        
        print("="*70 + "\n")
    
    def export_metrics_to_csv(self, metrics: PerformanceMetrics, filepath: str) -> None:
        """Export metrics to CSV file."""
        import csv
        
        metrics_dict = {
            'metric': [],
            'value': []
        }
        
        for field, value in metrics.__dict__.items():
            metrics_dict['metric'].append(field)
            metrics_dict['value'].append(value)
        
        df = pd.DataFrame(metrics_dict)
        df.to_csv(filepath, index=False)
        print(f"Metrics exported to {filepath}")


def analyze_backtest_results(equity_curve: np.ndarray, trades: List[Dict]) -> PerformanceMetrics:
    """
    Convenience function to analyze backtest results.
    
    Args:
        equity_curve: Array of equity values
        trades: List of trade dictionaries
        
    Returns:
        PerformanceMetrics object
    """
    analytics = PerformanceAnalytics()
    metrics = analytics.calculate_metrics(equity_curve, trades)
    analytics.print_metrics(metrics)
    
    return metrics


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Generate synthetic equity curve
    days = 252  # 1 year
    returns = np.random.normal(0.0005, 0.01, days)
    equity = 10000000 * np.cumprod(1 + returns)
    
    # Generate synthetic trades
    trades = []
    for i in range(50):
        trade = {
            'pnl': np.random.normal(1000, 5000),
            'holding_minutes': np.random.randint(10, 120)
        }
        trades.append(trade)
    
    # Analyze
    analytics = PerformanceAnalytics(initial_capital=10000000)
    metrics = analytics.calculate_metrics(equity, trades)
    analytics.print_metrics(metrics)
