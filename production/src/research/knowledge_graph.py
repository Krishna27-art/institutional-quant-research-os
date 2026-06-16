"""
Research Knowledge Graph & Causal Mechanism Tracking.
Models the dependencies: Dataset -> Feature -> Alpha -> Portfolio -> PnL.
Enforces institutional standards by requiring causal explanations for alphas.
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class NodeType(Enum):
    DATASET = "dataset"
    FEATURE = "feature"
    ALPHA = "alpha"
    PORTFOLIO = "portfolio"
    PNL = "pnl"

@dataclass
class CausalHypothesis:
    """The 'Why' behind an alpha. Prevents curve-fitting."""
    mechanism: str         # e.g., "Market maker inventory imbalance"
    market_participant: str # e.g., "Retail option buyers"
    incentive: str         # e.g., "Forced liquidation at 15:15"
    expected_decay: str    # e.g., "Half-life of 2 hours"

@dataclass
class GraphNode:
    id: str
    type: NodeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    hypothesis: Optional[CausalHypothesis] = None

class KnowledgeGraph:
    """Directed Acyclic Graph tracking all quant research knowledge."""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        
    def add_node(self, node: GraphNode) -> None:
        if node.type == NodeType.ALPHA and not node.hypothesis:
            raise ValueError(f"Institutional Standard: Alpha '{node.id}' must have a CausalHypothesis.")
        
        self.nodes[node.id] = node
        self.graph.add_node(node.id, type=node.type.value, **node.metadata)
        logger.info(f"Added {node.type.value} node: {node.id}")
        
    def add_edge(self, source_id: str, target_id: str, relationship: str = "depends_on") -> None:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError("Both nodes must exist before adding an edge.")
        self.graph.add_edge(source_id, target_id, relationship=relationship)
        
    def get_downstream_impact(self, node_id: str) -> List[str]:
        """If a feature breaks, what alphas/portfolios are affected?"""
        if node_id not in self.graph:
            return []
        return list(nx.descendants(self.graph, node_id))
        
    def get_upstream_dependencies(self, node_id: str) -> List[str]:
        """What data does this alpha rely on?"""
        if node_id not in self.graph:
            return []
        return list(nx.ancestors(self.graph, node_id))
        
    def validate_graph(self) -> bool:
        """Ensure no cyclical dependencies exist."""
        is_dag = nx.is_directed_acyclic_graph(self.graph)
        if not is_dag:
            logger.error("Knowledge Graph contains cycles! Invalid architecture.")
        return is_dag
