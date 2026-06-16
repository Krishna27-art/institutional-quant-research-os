"""
Unit tests for the new Institutional Core Systems:
- Research Knowledge Graph
- Market Event Store
- Alpha Marketplace Registry
- Capital Allocation Engine (cvxpy)
- Distributed Backtest Engine (ray)
"""

import pytest
import numpy as np
from datetime import datetime, timezone
import sys
from pathlib import Path

# Fix path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.research.knowledge_graph import KnowledgeGraph, GraphNode, NodeType, CausalHypothesis
from src.data.event_store import EventStore, EventType, TradeEvent, QuoteEvent
from src.alpha.marketplace.registry import AlphaMarketplace, AlphaMetadata
from src.portfolio.institutional_allocator import CapitalAllocationEngine
from src.backtest.distributed.cluster_engine import ClusterEngine, ExperimentTask


def test_knowledge_graph_basic():
    """Verify that KnowledgeGraph tracks dependencies and validates cycles."""
    kg = KnowledgeGraph()
    
    # 1. Standard nodes
    node_ds = GraphNode(id="raw_order_book", type=NodeType.DATASET)
    node_feat = GraphNode(id="imbalance_feature", type=NodeType.FEATURE)
    
    # Standard alpha node with hypothesis
    hypothesis = CausalHypothesis(
        mechanism="Market maker inventory imbalance",
        market_participant="Retail buyers buying at market",
        incentive="Spread extraction by HFT",
        expected_decay="Fast"
    )
    node_alpha = GraphNode(id="hft_imbalance_alpha", type=NodeType.ALPHA, hypothesis=hypothesis)
    
    kg.add_node(node_ds)
    kg.add_node(node_feat)
    kg.add_node(node_alpha)
    
    # 2. Add edges
    kg.add_edge("raw_order_book", "imbalance_feature")
    kg.add_edge("imbalance_feature", "hft_imbalance_alpha")
    
    assert kg.validate_graph() is True
    assert "hft_imbalance_alpha" in kg.get_downstream_impact("raw_order_book")
    assert "raw_order_book" in kg.get_upstream_dependencies("hft_imbalance_alpha")
    
    # 3. Test Cycle Detection
    kg.add_edge("hft_imbalance_alpha", "raw_order_book")
    assert kg.validate_graph() is False


def test_knowledge_graph_institutional_rules():
    """Alphas must have causal hypothesis, otherwise fail node insertion."""
    kg = KnowledgeGraph()
    
    # Invalid alpha node without causal hypothesis
    node_invalid = GraphNode(id="curve_fitted_alpha", type=NodeType.ALPHA, hypothesis=None)
    
    with pytest.raises(ValueError, match="Alpha 'curve_fitted_alpha' must have a CausalHypothesis"):
        kg.add_node(node_invalid)


def test_event_store_streaming():
    """Verify EventStore tracks and streams Trade and Quote events sequentially."""
    store = EventStore()
    
    t1 = datetime(2026, 6, 14, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 6, 14, 10, 0, 5, tzinfo=timezone.utc)
    t3 = datetime(2026, 6, 14, 10, 0, 10, tzinfo=timezone.utc)
    
    # Append events
    store.append(TradeEvent(timestamp=t1, symbol="RELIANCE", event_type=EventType.TRADE, price=2500.0, quantity=100, is_buyer_maker=False))
    store.append(QuoteEvent(timestamp=t2, symbol="RELIANCE", event_type=EventType.QUOTE_UPDATE, bid_price=2499.0, bid_size=500, ask_price=2501.0, ask_size=400))
    store.append(TradeEvent(timestamp=t3, symbol="INFY", event_type=EventType.TRADE, price=1400.0, quantity=50, is_buyer_maker=True))
    
    # Stream events filter by time
    events = list(store.stream_events(start=t1, end=t2))
    assert len(events) == 2
    assert events[0].symbol == "RELIANCE"
    
    # Stream events filter by symbol
    events_infy = list(store.stream_events(start=t1, end=t3, symbols=["INFY"]))
    assert len(events_infy) == 1
    assert events_infy[0].symbol == "INFY"


def test_alpha_marketplace_evaluation():
    """Verify that AlphaMarketplace registers and evaluates active alphas."""
    amp = AlphaMarketplace()
    
    metadata = AlphaMetadata(
        expected_sharpe=1.8,
        capacity_limit_usd=5_000_000.0,
        turnover_daily_pct=0.5,
        decay_half_life_days=5,
        author="QuantTeam",
        causal_hypothesis_id="hft_imbalance_alpha"
    )
    
    amp.register_alpha("alpha_v1", metadata)
    assert "alpha_v1" in amp.get_available_alphas()
    
    # Simulate a deterioration in observed performance
    amp.performance_stats["alpha_v1"].observed_sharpe = 0.5 # significantly lower than expected 1.8
    amp.evaluate_alphas()
    
    # Check if deactivated
    assert "alpha_v1" not in amp.get_available_alphas()
    assert amp.performance_stats["alpha_v1"].is_active is False
    assert "Regime shift" in amp.performance_stats["alpha_v1"].failure_reason


def test_capital_allocation_cvxpy():
    """Verify that CapitalAllocationEngine solves portfolio weights using cvxpy."""
    amp = AlphaMarketplace()
    
    # Register 2 active alphas
    amp.register_alpha("alpha_1", AlphaMetadata(
        expected_sharpe=2.0, capacity_limit_usd=10_000_000.0,
        turnover_daily_pct=0.2, decay_half_life_days=10,
        author="HFT", causal_hypothesis_id="h1"
    ))
    amp.register_alpha("alpha_2", AlphaMetadata(
        expected_sharpe=1.5, capacity_limit_usd=20_000_000.0,
        turnover_daily_pct=0.1, decay_half_life_days=20,
        author="Trend", causal_hypothesis_id="h2"
    ))
    
    engine = CapitalAllocationEngine(total_capital=100_000_000.0, max_leverage=1.0)
    allocations = engine.allocate(amp, current_inventory={})
    
    assert len(allocations) > 0
    total_weight = sum(alloc.target_weight for alloc in allocations)
    assert total_weight <= 1.0001 # Max leverage constraint
    
    # Capacity limit check:
    # total capital is 100M. alpha_1 capacity is 10M (max weight 0.1).
    # alpha_2 capacity is 20M (max weight 0.2).
    # So weights should not exceed capacity bounds!
    for alloc in allocations:
        if alloc.alpha_id == "alpha_1":
            assert alloc.target_weight <= 0.1001
        elif alloc.alpha_id == "alpha_2":
            assert alloc.target_weight <= 0.2001


@pytest.mark.slow
def test_ray_distributed_cluster_engine():
    """Verify distributed cluster engine runs tasks via Ray."""
    engine = ClusterEngine()
    
    tasks = [
        ExperimentTask(
            experiment_id="test_ray_task",
            alpha_class=None,
            parameters={},
            start_date="2026-01-01",
            end_date="2026-01-02"
        )
    ]
    
    results = engine.execute_batch(tasks)
    assert len(results) == 1
    assert results[0].experiment_id == "test_ray_task"
    assert results[0].sharpe_ratio == 1.5
    assert results[0].is_valid is True
    
    engine.shutdown()
