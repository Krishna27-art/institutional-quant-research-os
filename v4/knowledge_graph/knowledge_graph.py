"""Knowledge graph for papers, alphas, and features."""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    PAPER = "paper"
    ALPHA = "alpha"
    FEATURE = "feature"
    HYPOTHESIS = "hypothesis"
    CONCEPT = "concept"


class RelationshipType(Enum):
    """Types of relationships between nodes."""
    CITES = "cites"  # Paper cites another paper
    USES = "uses"  # Alpha uses a feature
    IMPLEMENTS = "implements"  # Alpha implements a hypothesis
    DERIVED_FROM = "derived_from"  # Feature derived from a paper
    INSPIRED_BY = "inspired_by"  # Alpha inspired by a paper
    RELATED_TO = "related_to"  # General relationship
    CONTRADICTS = "contradicts"  # Paper contradicts another


@dataclass
class Node:
    """Node in the knowledge graph."""
    node_id: str
    node_type: NodeType
    name: str
    description: str
    metadata: Dict[str, any]
    created_at: datetime
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id


@dataclass
class Edge:
    """Edge in the knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    metadata: Dict[str, any] = None
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.relationship_type))
    
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.source_id, self.target_id, self.relationship_type) == \
               (other.source_id, other.target_id, other.relationship_type)


@dataclass(frozen=True, slots=True)
class GraphQuery:
    node_type: NodeType | None = None
    name_contains: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    shard_year: int | None = None


@dataclass(frozen=True, slots=True)
class QueryResult:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]


class ReasoningEngine:
    """Lightweight graph reasoning helper for research discovery."""

    def __init__(self, graph: "KnowledgeGraph") -> None:
        self.graph = graph

    def recommend_related_nodes(self, node_id: str, max_hops: int = 2) -> list[Node]:
        components = self.graph.get_connected_components(node_id, max_depth=max_hops)
        related: list[Node] = []
        for depth, nodes in components.items():
            if depth == 0:
                continue
            related.extend(nodes)
        return related


class KnowledgeGraph:
    """
    Knowledge graph for papers, alphas, and features.
    
    This class manages a graph structure connecting research papers,
    alpha strategies, and features with various relationships.
    """
    
    def __init__(self):
        """Initialize knowledge graph."""
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, RelationshipType], Edge] = {}
        self.adjacency_list: Dict[str, Dict[str, List[Edge]]] = defaultdict(lambda: defaultdict(list))
        self.temporal_shards: Dict[int, Set[str]] = defaultdict(set)
        
        logger.info("KnowledgeGraph initialized")
    
    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> Node:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of node
            name: Node name
            description: Node description
            metadata: Additional metadata
            
        Returns:
            Created or updated Node
        """
        node = Node(
            node_id=node_id,
            node_type=node_type,
            name=name,
            description=description,
            metadata=metadata or {},
            created_at=created_at or datetime.now()
        )

        self.nodes[node_id] = node
        self.temporal_shards[node.created_at.year].add(node_id)
        logger.debug(f"Added node: {node_id} ({node_type.value})")
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Edge:
        """
        Add an edge to the graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Type of relationship
            weight: Edge weight
            metadata: Additional metadata
            
        Returns:
            Created or updated Edge
        """
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        edge_key = (source_id, target_id, relationship_type)
        self.edges[edge_key] = edge
        
        # Update adjacency list
        self.adjacency_list[source_id][target_id].append(edge)
        
        logger.debug(f"Added edge: {source_id} -> {target_id} ({relationship_type.value})")
        
        return edge

    def get_temporal_shard(self, year: int) -> "KnowledgeGraph":
        """Return a subgraph containing only nodes from a given year."""
        shard = KnowledgeGraph()
        for node_id in self.temporal_shards.get(year, set()):
            node = self.nodes[node_id]
            shard.add_node(
                node.node_id,
                node.node_type,
                node.name,
                node.description,
                dict(node.metadata),
                created_at=node.created_at,
            )
        for edge in self.edges.values():
            if edge.source_id in shard.nodes and edge.target_id in shard.nodes:
                shard.add_edge(edge.source_id, edge.target_id, edge.relationship_type, edge.weight, dict(edge.metadata or {}))
        return shard

    def prune_inactive_nodes(
        self,
        reference_date: Optional[datetime] = None,
        max_age_days: int = 180,
    ) -> Dict[str, int]:
        """Archive nodes with zero citations/uses and no reuse for over six months."""
        reference_date = reference_date or datetime.now()
        cutoff = reference_date - timedelta(days=max_age_days)
        removed_nodes = 0
        removed_edges = 0
        for node_id, node in list(self.nodes.items()):
            metadata = node.metadata or {}
            citations = int(metadata.get("citations", 0))
            uses = int(metadata.get("uses", 0))
            reuse = int(metadata.get("reuse_count", 0))
            last_used = metadata.get("last_used_at")
            if isinstance(last_used, str):
                try:
                    last_used = pd.to_datetime(last_used).to_pydatetime()
                except Exception:
                    last_used = None
            stale = node.created_at < cutoff and (last_used is None or last_used < cutoff)
            if stale and citations == 0 and uses == 0 and reuse == 0:
                removed_nodes += 1
                self.nodes.pop(node_id, None)
                self.temporal_shards[node.created_at.year].discard(node_id)
                for edge_key, edge in list(self.edges.items()):
                    if edge.source_id == node_id or edge.target_id == node_id:
                        removed_edges += 1
                        self.edges.pop(edge_key, None)
                self.adjacency_list.pop(node_id, None)
                for source_id in list(self.adjacency_list.keys()):
                    self.adjacency_list[source_id].pop(node_id, None)
        return {"removed_nodes": removed_nodes, "removed_edges": removed_edges}

    def query(self, query: GraphQuery) -> QueryResult:
        """Run a structured graph query."""
        nodes = list(self.nodes.values())
        if query.node_type is not None:
            nodes = [node for node in nodes if node.node_type == query.node_type]
        if query.name_contains:
            needle = query.name_contains.lower()
            nodes = [node for node in nodes if needle in node.name.lower()]
        if query.created_after:
            nodes = [node for node in nodes if node.created_at >= query.created_after]
        if query.created_before:
            nodes = [node for node in nodes if node.created_at <= query.created_before]
        if query.shard_year is not None:
            allowed = self.temporal_shards.get(query.shard_year, set())
            nodes = [node for node in nodes if node.node_id in allowed]
        node_ids = {node.node_id for node in nodes}
        edges = tuple(edge for edge in self.edges.values() if edge.source_id in node_ids and edge.target_id in node_ids)
        return QueryResult(nodes=tuple(nodes), edges=edges)

    def get_shard_summary(self) -> Dict[int, int]:
        """Return node counts by year shard."""
        return {year: len(node_ids) for year, node_ids in self.temporal_shards.items()}
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> List[Node]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: Node ID
            relationship_type: Filter by relationship type (optional)
            
        Returns:
            List of neighboring nodes
        """
        neighbors = []
        
        if node_id not in self.adjacency_list:
            return neighbors
        
        for target_id, edges in self.adjacency_list[node_id].items():
            for edge in edges:
                if relationship_type is None or edge.relationship_type == relationship_type:
                    if target_id in self.nodes:
                        neighbors.append(self.nodes[target_id])
        
        return neighbors
    
    def get_connected_components(
        self,
        node_id: str,
        max_depth: int = 3
    ) -> Dict[str, List[Node]]:
        """
        Get connected components up to a certain depth.
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Dict mapping depth to list of nodes at that depth
        """
        components = {0: [self.nodes.get(node_id)]}
        visited = {node_id}
        current_level = [node_id]
        
        for depth in range(1, max_depth + 1):
            next_level = []
            components[depth] = []
            
            for current_id in current_level:
                neighbors = self.get_neighbors(current_id)
                
                for neighbor in neighbors:
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        next_level.append(neighbor.node_id)
                        components[depth].append(neighbor)
            
            current_level = next_level
            
            if not current_level:
                break
        
        return components
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> Optional[List[Node]]:
        """
        Find a path between two nodes using BFS.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_length: Maximum path length
            
        Returns:
            List of nodes in the path, or None if no path found
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        from collections import deque
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                return [self.nodes[node_id] for node_id in path]
            
            if len(path) >= max_length:
                continue
            
            neighbors = self.get_neighbors(current_id)
            
            for neighbor in neighbors:
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    queue.append((neighbor.node_id, path + [neighbor.node_id]))
        
        return None
    
    def get_features_for_alpha(self, alpha_id: str) -> List[Node]:
        """
        Get all features used by an alpha.
        
        Args:
            alpha_id: Alpha node ID
            
        Returns:
            List of feature nodes
        """
        return self.get_neighbors(alpha_id, RelationshipType.USES)
    
    def get_alphas_using_feature(self, feature_id: str) -> List[Node]:
        """
        Get all alphas that use a feature.
        
        Args:
            feature_id: Feature node ID
            
        Returns:
            List of alpha nodes
        """
        # Need to traverse edges in reverse
        alphas = []
        
        for edge_key, edge in self.edges.items():
            if edge.target_id == feature_id and edge.relationship_type == RelationshipType.USES:
                if edge.source_id in self.nodes:
                    alphas.append(self.nodes[edge.source_id])
        
        return alphas
    
    def get_papers_for_alpha(self, alpha_id: str) -> List[Node]:
        """
        Get all papers related to an alpha.
        
        Args:
            alpha_id: Alpha node ID
            
        Returns:
            List of paper nodes
        """
        papers = []
        
        # Direct inspiration
        inspired_by = self.get_neighbors(alpha_id, RelationshipType.INSPIRED_BY)
        papers.extend(inspired_by)
        
        # Papers cited by features used by alpha
        features = self.get_features_for_alpha(alpha_id)
        for feature in features:
            derived_from = self.get_neighbors(feature.node_id, RelationshipType.DERIVED_FROM)
            papers.extend(derived_from)
        
        return list(set(papers))  # Remove duplicates
    
    def query_by_type(self, node_type: NodeType) -> List[Node]:
        """
        Query nodes by type.
        
        Args:
            node_type: Node type to filter by
            
        Returns:
            List of nodes of the specified type
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def search_by_name(self, name_pattern: str) -> List[Node]:
        """
        Search nodes by name pattern.
        
        Args:
            name_pattern: Name pattern to search for
            
        Returns:
            List of matching nodes
        """
        pattern_lower = name_pattern.lower()
        return [
            node for node in self.nodes.values()
            if pattern_lower in node.name.lower()
        ]
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """
        Get graph statistics.
        
        Returns:
            Dict with graph statistics
        """
        node_type_counts = defaultdict(int)
        for node in self.nodes.values():
            node_type_counts[node_type.value] += 1
        
        relationship_type_counts = defaultdict(int)
        for edge in self.edges.values():
            relationship_type_counts[edge.relationship_type.value] += 1
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_type_distribution': dict(node_type_counts),
            'relationship_type_distribution': dict(relationship_type_counts),
            'temporal_shards': self.get_shard_summary(),
        }
    
    def print_graph_report(self) -> None:
        """Print graph report."""
        stats = self.get_graph_statistics()
        
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH REPORT")
        print("="*60)
        
        print(f"\nTotal Nodes: {stats['total_nodes']}")
        print(f"Total Edges: {stats['total_edges']}")
        
        print(f"\nNode Type Distribution:")
        for node_type, count in stats['node_type_distribution'].items():
            print(f"  {node_type}: {count}")
        
        print(f"\nRelationship Type Distribution:")
        for rel_type, count in stats['relationship_type_distribution'].items():
            print(f"  {rel_type}: {count}")
        
        print("\n" + "="*60)
    
    def export_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export graph to DataFrames.
        
        Returns:
            (nodes_df, edges_df)
        """
        nodes_data = []
        for node in self.nodes.values():
            nodes_data.append({
                'node_id': node.node_id,
                'node_type': node.node_type.value,
                'name': node.name,
                'description': node.description,
                'created_at': node.created_at
            })
        
        edges_data = []
        for edge in self.edges.values():
            edges_data.append({
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'relationship_type': edge.relationship_type.value,
                'weight': edge.weight
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        
        return nodes_df, edges_df


def sample_knowledge_graph():
    """Demonstrate knowledge graph."""
    print("=== Knowledge Graph Demo ===\n")
    
    # Initialize graph
    kg = KnowledgeGraph()
    
    # Add papers
    kg.add_node("paper_1", NodeType.PAPER, "Almgren-Chriss 2000", "Optimal execution")
    kg.add_node("paper_2", NodeType.PAPER, "Kyle 1985", "Market microstructure")
    kg.add_node("paper_3", NodeType.PAPER, "Easley et al. 1996", "VPIN")
    
    # Add features
    kg.add_node("feature_1", NodeType.FEATURE, "OFI", "Order flow imbalance")
    kg.add_node("feature_2", NodeType.FEATURE, "VPIN", "Volume-synchronized PIN")
    kg.add_node("feature_3", NodeType.FEATURE, "Spread", "Bid-ask spread")
    
    # Add alphas
    kg.add_node("alpha_1", NodeType.ALPHA, "Momentum", "Momentum strategy")
    kg.add_node("alpha_2", NodeType.ALPHA, "Mean Reversion", "Mean reversion strategy")
    kg.add_node("alpha_3", NodeType.ALPHA, "VPIN Strategy", "VPIN-based strategy")
    
    # Add relationships
    kg.add_edge("paper_1", "paper_2", RelationshipType.CITES)
    kg.add_edge("paper_3", "paper_2", RelationshipType.CITES)
    
    kg.add_edge("feature_1", "paper_2", RelationshipType.DERIVED_FROM)
    kg.add_edge("feature_2", "paper_3", RelationshipType.DERIVED_FROM)
    kg.add_edge("feature_3", "paper_2", RelationshipType.DERIVED_FROM)
    
    kg.add_edge("alpha_1", "feature_1", RelationshipType.USES)
    kg.add_edge("alpha_1", "feature_3", RelationshipType.USES)
    kg.add_edge("alpha_2", "feature_1", RelationshipType.USES)
    kg.add_edge("alpha_3", "feature_2", RelationshipType.USES)
    
    kg.add_edge("alpha_1", "paper_1", RelationshipType.INSPIRED_BY)
    kg.add_edge("alpha_3", "paper_3", RelationshipType.IMPLEMENTS)
    
    # Print report
    kg.print_graph_report()
    
    # Query examples
    print("\nQuery Examples:")
    
    print("\nFeatures used by alpha_1:")
    features = kg.get_features_for_alpha("alpha_1")
    for f in features:
        print(f"  - {f.name}")
    
    print("\nAlphas using feature_1:")
    alphas = kg.get_alphas_using_feature("feature_1")
    for a in alphas:
        print(f"  - {a.name}")
    
    print("\nPapers related to alpha_3:")
    papers = kg.get_papers_for_alpha("alpha_3")
    for p in papers:
        print(f"  - {p.name}")
    
    print("\nPath from alpha_1 to paper_2:")
    path = kg.find_path("alpha_1", "paper_2")
    if path:
        print("  -> ".join([node.name for node in path]))
    
    print("\n=== Knowledge Graph Demo Complete ===")
    print("Key capabilities:")
    print("- Graph structure for papers, alphas, and features")
    print("- Relationship types (cites, uses, implements, derived_from)")
    print("- Graph traversal and querying")
    print("- Feature lineage tracking")
    print("- Alpha-paper connections")
    print("- Knowledge discovery and recommendations")


if __name__ == "__main__":
    sample_knowledge_graph()
