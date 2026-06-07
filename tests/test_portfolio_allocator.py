"""
Unit tests for PortfolioAllocator
Tests position sizing strategies to ensure non-zero allocations.
"""

import pytest
import numpy as np
from portfolio.allocator import PortfolioAllocator, PositionSpec


def test_fixed_fractional():
    """Test fixed fractional position sizing."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.fixed_fractional("NIFTY", signal=1.0, entry_price=19500, stop_loss_price=19300)
    
    # risk = 1% of capital = 10,000
    # risk_per_share = 200 -> shares = 50
    # CRITICAL FIX: Position capped at 5% of capital (50,000) due to max_single_stock_pct
    assert pos.position_size == 2  # 50,000 / 19,500 = 2.56 -> int = 2
    assert pos.capital_allocated == 50_000  # Capped at 5% of capital
    assert pos.weight == 0.05  # 5%
    assert pos.expected_risk == 10_000


def test_fixed_fractional_short():
    """Test fixed fractional with short signal."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.fixed_fractional("NIFTY", signal=-1.0, entry_price=19500, stop_loss_price=19700)
    
    # CRITICAL FIX: Position capped at 5% of capital (50,000) due to max_single_stock_pct
    assert pos.position_size == -2  # Negative for short
    assert pos.capital_allocated == 50_000  # Capped at 5% of capital
    assert pos.weight == 0.05  # 5%


def test_fixed_fractional_zero_risk():
    """Test fixed fractional with zero risk per share."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.fixed_fractional("NIFTY", signal=1.0, entry_price=19500, stop_loss_price=19500)
    
    assert pos.position_size == 0
    assert pos.capital_allocated == 0


def test_kelly_fractional():
    """Test Kelly criterion position sizing."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.kelly_fractional("BANKNIFTY", win_prob=0.55, win_loss_ratio=1.5,
                                     entry_price=45000, signal_strength=1.0)
    
    # f* = (0.55*1.5 - 0.45)/1.5 = (0.825-0.45)/1.5 = 0.25
    # CRITICAL FIX: Quarter Kelly (0.25 * 0.25 = 0.0625) capped at 5% max position
    assert pos.weight == 0.05  # Capped at max_single_stock_pct
    assert pos.capital_allocated == 50_000  # 5% of capital
    assert pos.position_size == 1  # 50000 / 45000 = 1.11 -> int = 1


def test_kelly_fractional_short():
    """Test Kelly criterion with short signal."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.kelly_fractional("BANKNIFTY", win_prob=0.55, win_loss_ratio=1.5,
                                     entry_price=45000, signal_strength=-1.0)
    
    # CRITICAL FIX: Quarter Kelly capped at 5% max position
    assert pos.position_size == -1  # Negative for short
    assert pos.capital_allocated == 50_000  # 5% of capital


def test_kelly_fractional_zero_ratio():
    """Test Kelly criterion with zero win/loss ratio."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.kelly_fractional("BANKNIFTY", win_prob=0.55, win_loss_ratio=0,
                                     entry_price=45000, signal_strength=1.0)
    
    assert pos.position_size == 0
    assert pos.capital_allocated == 0


def test_volatility_targeting():
    """Test volatility targeting position sizing."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.volatility_targeting("NIFTY", signal=1.0, entry_price=19500,
                                          volatility=20.0, target_vol=0.15)
    
    # daily_target = 0.15 / sqrt(252) = 0.00945
    # daily_vol = 0.20
    # weight = 0.00945 / 0.20 = 0.047
    assert pos.weight > 0
    assert pos.capital_allocated > 0
    assert pos.position_size > 0


def test_volatility_targeting_short():
    """Test volatility targeting with short signal."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.volatility_targeting("NIFTY", signal=-1.0, entry_price=19500,
                                          volatility=20.0, target_vol=0.15)
    
    assert pos.weight < 0
    assert pos.capital_allocated > 0
    assert pos.position_size < 0


def test_volatility_targeting_zero_vol():
    """Test volatility targeting with zero volatility."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    pos = allocator.volatility_targeting("NIFTY", signal=1.0, entry_price=19500,
                                          volatility=0.0, target_vol=0.15)
    
    assert pos.position_size == 0
    assert pos.capital_allocated == 0


def test_risk_parity():
    """Test risk parity allocation."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    symbols = ["NIFTY", "BANKNIFTY"]
    signals = [1.0, 1.0]
    volatilities = [20.0, 25.0]
    correlations = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    positions = allocator.risk_parity(symbols, signals, volatilities, correlations)
    
    assert len(positions) == 2
    assert all(pos.position_size > 0 for pos in positions)
    assert all(pos.capital_allocated > 0 for pos in positions)


def test_risk_parity_empty():
    """Test risk parity with empty inputs."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    positions = allocator.risk_parity([], [], [], np.array([]))
    
    assert len(positions) == 0


def test_apply_sector_limits():
    """Test sector limits enforcement."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    
    positions = [
        PositionSpec("HDFC", 100, 500_000, 10_000, 5_000, 0.5),
        PositionSpec("ICICI", 100, 500_000, 10_000, 5_000, 0.5)
    ]
    
    sector_map = {"HDFC": "Financial", "ICICI": "Financial"}
    sector_limits = {"Financial": 0.30}  # Cap at 30%
    
    adjusted = allocator.apply_sector_limits(positions, sector_map, sector_limits)
    
    # Total exposure should be capped at 30%
    total_exposure = sum(abs(pos.weight) for pos in adjusted)
    assert total_exposure <= 0.30


def test_apply_position_limits():
    """Test position limits enforcement."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    
    positions = [
        PositionSpec("HDFC", 100, 500_000, 10_000, 5_000, 0.5),  # 50% - too high
        PositionSpec("ICICI", 100, 500_000, 10_000, 5_000, 0.5)   # 50% - too high
    ]
    
    adjusted = allocator.apply_position_limits(positions, max_position_weight=0.10)
    
    # All positions should be capped at 10%
    for pos in adjusted:
        assert abs(pos.weight) <= 0.10


def test_portfolio_heat():
    """Test portfolio heat calculation."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    
    positions = [
        PositionSpec("HDFC", 100, 500_000, 10_000, 5_000, 0.5),
        PositionSpec("ICICI", 100, 500_000, 10_000, 5_000, 0.5)
    ]
    
    cov_matrix = np.array([[0.04, 0.02], [0.02, 0.04]])
    
    scale = allocator.portfolio_heat(positions, cov_matrix, max_portfolio_var=0.02)
    
    # Should return a scaling factor
    assert isinstance(scale, float)
    assert scale > 0


def test_portfolio_heat_empty():
    """Test portfolio heat with empty positions."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    
    scale = allocator.portfolio_heat([], np.array([[1.0]]), max_portfolio_var=0.02)
    
    assert scale == 1.0


def test_apply_portfolio_heat():
    """Test portfolio heat scaling."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    
    positions = [
        PositionSpec("HDFC", 100, 500_000, 10_000, 5_000, 0.5),
        PositionSpec("ICICI", 100, 500_000, 10_000, 5_000, 0.5)
    ]
    
    cov_matrix = np.array([[0.04, 0.02], [0.02, 0.04]])
    
    adjusted = allocator.apply_portfolio_heat(positions, cov_matrix, max_portfolio_var=0.02)
    
    # Positions should be scaled if portfolio VaR exceeds limit
    assert len(adjusted) == 2


def test_allocator_initialization():
    """Test allocator initialization."""
    allocator = PortfolioAllocator(total_capital=1_000_000, max_leverage=2.0)
    
    assert allocator.total_capital == 1_000_000
    assert allocator.max_leverage == 2.0
    assert allocator.risk_per_trade == 0.01
    assert allocator.current_positions == {}


def test_non_zero_allocations():
    """Test that allocations are non-zero when signals are non-zero."""
    allocator = PortfolioAllocator(total_capital=1_000_000)
    
    # Test all methods with valid inputs
    pos1 = allocator.fixed_fractional("TEST", signal=1.0, entry_price=100, stop_loss_price=95)
    assert pos1.position_size > 0
    assert pos1.capital_allocated > 0
    
    pos2 = allocator.kelly_fractional("TEST", win_prob=0.6, win_loss_ratio=2.0,
                                       entry_price=100, signal_strength=1.0)
    assert pos2.position_size > 0
    assert pos2.capital_allocated > 0
    
    pos3 = allocator.volatility_targeting("TEST", signal=1.0, entry_price=100,
                                          volatility=20.0, target_vol=0.15)
    assert pos3.position_size > 0
    assert pos3.capital_allocated > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
