"""
V4 Knowledge Graph Layer
Graph database for storing experiments, features, strategies, regimes, results, and failures.
Enables reasoning about what has been tried and what has not.
"""

from .graph_schema import (
    NodeType,
    EdgeType,
    GraphNode,
    GraphEdge,
    KnowledgeGraphSchema,
)

from .knowledge_graph import (
    KnowledgeGraph,
    GraphQuery,
    QueryResult,
    ReasoningEngine,
)

__all__ = [
    # Schema
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "KnowledgeGraphSchema",
    # Graph
    "KnowledgeGraph",
    "GraphQuery",
    "QueryResult",
    "ReasoningEngine",
]
