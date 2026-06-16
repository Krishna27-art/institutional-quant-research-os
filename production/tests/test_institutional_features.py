import pytest
import os
from src.alpha.lifecycle_manager import LifecycleManager, AlphaState
from src.alpha.registry import AlphaRegistry, AlphaDefinition, AlphaType, AlphaStatus
from src.alpha.attribution_engine import AlphaAttributionEngine
from src.portfolio.trade_logger import TradeLogger, Trade, TradeSide, TradeStatus
from src.execution.simulation_engine import SimulationEngine, SimulationConfig
from src.execution.brokers.broker_adapter import Order, OrderType, OrderSide
from src.portfolio.correlation_manager import CorrelationManager

def test_lifecycle_manager_creates_graveyard(tmp_path):
    lm = LifecycleManager(registry_dir=str(tmp_path))
    lm.transition_alpha("test_alpha", AlphaState.DEAD, metadata={"hypothesis": "Too much slippage"})
    
    dead_path = tmp_path / "dead" / "test_alpha.yaml"
    assert dead_path.exists(), "Graveyard YAML should be created"
    
    content = dead_path.read_text()
    assert "Too much slippage" in content
    
def test_alpha_registry_demotion(tmp_path):
    registry = AlphaRegistry()
    
    # Overwrite the default LifecycleManager in registry's demote_alpha locally to test
    # (Since we don't have dependency injection for LifecycleManager in registry)
    # We will just verify it runs without crashing for now.
    import datetime
    
    dummy_alpha = AlphaDefinition(
        alpha_id="alpha_123",
        name="Dummy Alpha",
        version=1,
        alpha_type=AlphaType.MOMENTUM,
        logic="Buy high",
        parameters={},
        expected_sharpe=1.0,
        capacity_cr=10,
        decay_months=6,
        confidence=0.8,
        status=AlphaStatus.PHASE1_RESEARCH,
        priority=1
    )
    registry.register(dummy_alpha)
    assert dummy_alpha.status == AlphaStatus.PHASE1_RESEARCH
    
    # Call demote
    try:
        registry.demote_alpha("alpha_123", "Tested poorly")
        assert registry.get("alpha_123").status == AlphaStatus.REJECTED
    except Exception as e:
        pytest.fail(f"demote_alpha raised an exception: {e}")

def test_attribution_engine(tmp_path):
    from datetime import datetime
    logger = TradeLogger(log_dir=str(tmp_path))
    trade = Trade(
        trade_id="t1",
        symbol="AAPL",
        side=TradeSide.BUY,
        quantity=100,
        entry_price=150.0,
        exit_price=155.0,
        exit_time=datetime.now()
    )
    logger.log_trade(trade, alpha_id="alpha_1", regime_id="bull")
    
    trade2 = Trade(
        trade_id="t2",
        symbol="AAPL",
        side=TradeSide.SELL,
        quantity=100,
        entry_price=160.0,
        exit_price=155.0,
        exit_time=datetime.now()
    )
    logger.log_trade(trade2, alpha_id="alpha_1", regime_id="bear")
    
    engine = AlphaAttributionEngine(trade_logger=logger)
    df = engine.calculate_attribution(group_by="alpha_id")
    
    assert not df.empty
    assert 'alpha_1' in df.index
    assert df.loc['alpha_1']['total_trades'] == 2

def test_simulation_engine():
    sim = SimulationEngine(SimulationConfig(queue_position_penalty=1.0))
    order = Order(
        order_id="test_ord",
        symbol="RELIANCE",
        quantity=100000, # Large order
        price=2500.0,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET
    )
    
    market_state = {
        'price': 2500.0,
        'volume': 10000, # L1 volume is small
        'adv': 5000000
    }
    
    fill = sim.simulate_fill(order, market_state)
    assert fill.quantity < order.quantity # Partial fill due to size and penalty

def test_correlation_manager():
    manager = CorrelationManager(correlation_threshold=0.75)
    
    # Log signals for two highly correlated strategies
    # Strategy A: [1.0, 2.0, 3.0, 4.0, 5.0]
    # Strategy B: [1.1, 2.1, 3.1, 4.1, 5.1]
    for val_a, val_b in zip([1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1]):
        manager.log_strategy_state("strat_a", val_a, daily_return=val_a * 0.01)
        manager.log_strategy_state("strat_b", val_b, daily_return=val_b * 0.01)
        
    # Log signals for an uncorrelated strategy C
    for val_c in [5.0, 1.0, 4.0, 2.0, 3.0]:
        manager.log_strategy_state("strat_c", val_c, daily_return=val_c * 0.01)
        
    sig_corr = manager.calculate_signal_correlation_matrix()
    assert not sig_corr.empty
    assert sig_corr.loc["strat_a", "strat_b"] > 0.95
    assert abs(sig_corr.loc["strat_a", "strat_c"]) < 0.5
    
    ret_corr = manager.calculate_strategy_return_correlation_matrix()
    assert not ret_corr.empty
    assert ret_corr.loc["strat_a", "strat_b"] > 0.95
    
    pairs = manager.get_highly_correlated_pairs()
    assert len(pairs) >= 1
    # Check that strat_a and strat_b are identified
    pair_names = {(p[0], p[1]) for p in pairs}
    assert ("strat_a", "strat_b") in pair_names or ("strat_b", "strat_a") in pair_names
    
    # Check trade redundancy adjustment
    pending = {"strat_a": 1.0, "strat_b": 1.0, "strat_c": 1.0}
    adjusted = manager.check_trade_redundancy(pending)
    # Both strat_a and strat_b should be penalized because they have high signal correlation and are trading in the same direction
    assert adjusted["strat_a"] < 1.0
    assert adjusted["strat_b"] < 1.0
    assert adjusted["strat_c"] == 1.0
