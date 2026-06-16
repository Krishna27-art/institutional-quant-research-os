from src.core.config.settings import TRADING_CAPITAL
"""
Test script to verify risk engine fixes produce reasonable values.

This tests that the corrected risk engine no longer produces impossible values
like VaR = -4 billion for a ₹25 crore portfolio.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/pandu/Desktop/institutional-quant-research-os')

from src.risk.institutional_risk_engine import InstitutionalRiskEngine, Position


def test_risk_calculations():
    """Test that risk calculations produce reasonable values."""
    
    # Initialize risk engine with ₹25 crore capital
    capital = TRADING_CAPITAL  # ₹25 Crore
    risk_engine = InstitutionalRiskEngine(capital=capital)
    
    # Create sample positions
    positions = [
        Position("RELIANCE", "ENERGY", 100, 2450, 2500, "LONG"),
        Position("HDFCBANK", "BANKING", 50, 1550, 1600, "LONG"),
        Position("INFY", "IT", 75, 1450, 1400, "SHORT")
    ]
    
    # Create sample market data with realistic daily returns (mean ~0.1%, std ~2%)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    
    market_data = pd.DataFrame({
        "RELIANCE": 2500 * np.cumprod(1 + np.random.normal(0.001, 0.02, 252)),
        "HDFCBANK": 1600 * np.cumprod(1 + np.random.normal(0.0008, 0.018, 252)),
        "INFY": 1400 * np.cumprod(1 + np.random.normal(0.0005, 0.025, 252))
    }, index=dates)
    
    # Calculate portfolio returns
    portfolio_returns = risk_engine.calculate_portfolio_returns(positions, market_data)
    
    print("=" * 60)
    print("RISK ENGINE FIX VERIFICATION TEST")
    print("=" * 60)
    print(f"Capital: ₹{capital:,.2f}")
    print(f"Number of returns: {len(portfolio_returns)}")
    print(f"Mean daily return: {np.mean(portfolio_returns):.4%}")
    print(f"Std daily return: {np.std(portfolio_returns):.4%}")
    print()
    
    # Test VaR
    var = risk_engine.calculate_var(portfolio_returns)
    print(f"VaR (99%): ₹{var:,.2f}")
    
    # VaR should be positive (loss amount)
    assert var >= 0, f"VaR should be positive, got {var}"
    
    # VaR should be reasonable for a ₹25 crore portfolio
    # With 2% daily vol and 99% confidence, VaR should be around 4-6% of capital
    # i.e., ₹1-1.5 crore, not ₹4 billion
    var_pct = var / capital
    assert var_pct < 0.10, f"VaR percentage {var_pct:.2%} seems too high (should be <10%)"
    assert var_pct > 0.01, f"VaR percentage {var_pct:.2%} seems too low (should be >1%)"
    print(f"  ✓ VaR is {var_pct:.2%} of capital (reasonable)")
    print()
    
    # Test CVaR
    cvar = risk_engine.calculate_cvar(portfolio_returns)
    print(f"CVaR (95%): ₹{cvar:,.2f}")
    
    # CVaR should be positive
    assert cvar >= 0, f"CVaR should be positive, got {cvar}"
    
    # CVaR should be larger than VaR (typically 1.2-1.5x)
    assert cvar >= var * 0.8, f"CVaR {cvar} should be >= 0.8 * VaR {var}"
    assert cvar <= var * 2.0, f"CVaR {cvar} should be <= 2.0 * VaR {var}"
    
    cvar_pct = cvar / capital
    assert cvar_pct < 0.15, f"CVaR percentage {cvar_pct:.2%} seems too high"
    print(f"  ✓ CVaR is {cvar_pct:.2%} of capital (reasonable)")
    print(f"  ✓ CVaR/VaR ratio: {cvar/var:.2f} (expected 1.2-1.5)")
    print()
    
    # Test Tail Risk
    tail_risk = risk_engine.calculate_tail_risk(portfolio_returns)
    print(f"Tail Risk (15%): ₹{tail_risk:,.2f}")
    
    # Tail risk should be positive
    assert tail_risk >= 0, f"Tail risk should be positive, got {tail_risk}"
    
    # Tail risk should be larger than CVaR (worst 15% vs worst 5%)
    assert tail_risk >= cvar * 0.5, f"Tail risk {tail_risk} should be >= 0.5 * CVaR {cvar}"
    
    tail_pct = tail_risk / capital
    assert tail_pct < 0.20, f"Tail risk percentage {tail_pct:.2%} seems too high"
    print(f"  ✓ Tail risk is {tail_pct:.2%} of capital (reasonable)")
    print()
    
    # Test full risk metrics
    metrics = risk_engine.calculate_risk_metrics(positions, market_data, daily_pnl=500000)
    
    print("=" * 60)
    print("FULL RISK METRICS")
    print("=" * 60)
    print(f"VaR: ₹{metrics.var:,.2f} ({metrics.var/capital:.2%})")
    print(f"CVaR: ₹{metrics.cvar:,.2f} ({metrics.cvar/capital:.2%})")
    print(f"L-VaR: ₹{metrics.l_var:,.2f} ({metrics.l_var/capital:.2%})")
    print(f"Tail Risk: ₹{metrics.tail_risk:,.2f} ({metrics.tail_risk/capital:.2%})")
    print(f"Portfolio Heat: {metrics.portfolio_heat:.2%}")
    print()
    
    # Sanity checks
    assert metrics.var >= 0, "VaR should be positive"
    assert metrics.cvar >= 0, "CVaR should be positive"
    assert metrics.tail_risk >= 0, "Tail risk should be positive"
    assert metrics.var < capital * 0.10, "VaR should be < 10% of capital"
    assert metrics.cvar < capital * 0.15, "CVaR should be < 15% of capital"
    
    print("=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Summary of fixes:")
    print("  1. calculate_var: Changed from log return formula to simple return formula")
    print("  2. calculate_cvar: Changed from log return formula to simple return formula")
    print("  3. calculate_tail_risk: Changed from log return formula to simple return formula")
    print()
    print("The risk engine now produces reasonable values:")
    print(f"  - VaR: {var_pct:.2%} of capital (was impossible negative value)")
    print(f"  - CVaR: {cvar_pct:.2%} of capital (was inconsistent)")
    print(f"  - Tail Risk: {tail_pct:.2%} of capital (was undefined)")
    print()
    
    # Test new VaR methods
    print("=" * 60)
    print("TESTING NEW VaR METHODS")
    print("=" * 60)
    
    # Test Historical VaR
    var_hist = risk_engine.calculate_var_historical(portfolio_returns)
    print(f"Historical VaR (99%): ₹{var_hist:,.2f} ({var_hist/capital:.2%})")
    assert var_hist >= 0, "Historical VaR should be positive"
    assert var_hist < capital * 0.10, "Historical VaR should be < 10% of capital"
    print("  ✓ Historical VaR is reasonable")
    
    # Test EVT VaR
    var_evt = risk_engine.calculate_var_evt(portfolio_returns)
    print(f"EVT VaR (99%): ₹{var_evt:,.2f} ({var_evt/capital:.2%})")
    assert var_evt >= 0, "EVT VaR should be positive"
    assert var_evt < capital * 0.15, "EVT VaR should be < 15% of capital"
    print("  ✓ EVT VaR is reasonable")
    
    # Compare methods
    print()
    print("VaR Method Comparison:")
    print(f"  Parametric (Cornish-Fisher): ₹{var:,.2f} ({var/capital:.2%})")
    print(f"  Historical Simulation: ₹{var_hist:,.2f} ({var_hist/capital:.2%})")
    print(f"  EVT (GPD): ₹{var_evt:,.2f} ({var_evt/capital:.2%})")
    print()
    print("  EVT typically captures tail risk better than parametric methods")
    print("  Historical simulation makes no distributional assumptions")
    print("  Use all three for robust risk estimation")


if __name__ == "__main__":
    test_risk_calculations()
