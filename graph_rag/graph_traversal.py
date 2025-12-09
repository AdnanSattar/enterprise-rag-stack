"""
Graph traversal algorithms for multi-hop retrieval.

Finds connected evidence 2-3 hops away from seed entities.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .graph_builder import Entity, Relation, SimpleKnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    """Result of graph traversal."""

    seed_entity_id: str
    entities_by_hop: Dict[int, Set[str]]
    paths: List[
        List[Tuple[str, str, str]]
    ]  # [(entity_id, relation_type, entity_id), ...]
    total_entities: int
    max_depth_reached: int


class GraphTraverser:
    """
    Graph traversal for multi-hop evidence retrieval.

    Usage:
        traverser = GraphTraverser(graph)

        # BFS to find all entities within 2 hops
        neighbors = traverser.bfs("customer_001", max_hops=2)

        # Find paths between two entities
        paths = traverser.find_paths("contract_001", "product_001")
    """

    def __init__(self, graph: SimpleKnowledgeGraph):
        self.graph = graph

    def bfs(
        self,
        start_id: str,
        max_hops: int = 2,
        relation_filter: List[str] = None,
    ) -> Dict[int, Set[str]]:
        """
        Breadth-first search from a starting entity.

        Args:
            start_id: Starting entity ID
            max_hops: Maximum traversal depth
            relation_filter: Only traverse these relation types

        Returns:
            Dict mapping hop distance to set of entity IDs
        """
        visited: Set[str] = set()
        result: Dict[int, Set[str]] = {i: set() for i in range(max_hops + 1)}

        queue = [(start_id, 0)]
        visited.add(start_id)
        result[0].add(start_id)

        while queue:
            current_id, hop = queue.pop(0)

            if hop >= max_hops:
                continue

            for neighbor, relation in self.graph.get_neighbors(current_id):
                if relation_filter and relation.type not in relation_filter:
                    continue

                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    result[hop + 1].add(neighbor.id)
                    queue.append((neighbor.id, hop + 1))

        return result

    def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 4,
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Find all paths between two entities.

        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            max_depth: Maximum path length

        Returns:
            List of paths, where each path is a list of (source, relation_type, target) tuples
        """
        paths = []

        def dfs(current: str, target: str, path: List, visited: Set):
            if len(path) > max_depth:
                return

            if current == target:
                paths.append(path.copy())
                return

            for neighbor, relation in self.graph.get_neighbors(current):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    path.append((current, relation.type, neighbor.id))
                    dfs(neighbor.id, target, path, visited)
                    path.pop()
                    visited.remove(neighbor.id)

        dfs(start_id, end_id, [], {start_id})
        return paths

    def get_subgraph_for_query(
        self,
        entity_ids: List[str],
        max_hops: int = 2,
    ) -> TraversalResult:
        """
        Get subgraph for a set of query entities.

        Combines traversals from multiple seed entities.

        Args:
            entity_ids: Seed entity IDs from query
            max_hops: Maximum traversal depth

        Returns:
            TraversalResult with combined subgraph info
        """
        combined_entities: Dict[int, Set[str]] = {}
        all_paths = []

        for seed_id in entity_ids:
            neighbors = self.bfs(seed_id, max_hops)

            for hop, ids in neighbors.items():
                if hop not in combined_entities:
                    combined_entities[hop] = set()
                combined_entities[hop].update(ids)

        # Find paths between all pairs of seed entities
        for i, id1 in enumerate(entity_ids):
            for id2 in entity_ids[i + 1 :]:
                paths = self.find_paths(id1, id2, max_hops * 2)
                all_paths.extend(paths)

        total = sum(len(ids) for ids in combined_entities.values())
        max_depth = max(combined_entities.keys()) if combined_entities else 0

        return TraversalResult(
            seed_entity_id=entity_ids[0] if entity_ids else "",
            entities_by_hop=combined_entities,
            paths=all_paths,
            total_entities=total,
            max_depth_reached=max_depth,
        )

    def pattern_match(
        self,
        pattern: List[Tuple[str, str]],
    ) -> List[List[str]]:
        """
        Match a traversal pattern across the graph.

        Pattern format: [(entity_type, relation_type), ...]

        Example pattern for "Contracts in APAC region":
        [("Contract", "SIGNED_BY"), ("Customer", "LOCATED_IN"), ("Region", None)]

        Args:
            pattern: List of (entity_type, relation_type) tuples

        Returns:
            List of matching entity ID sequences
        """
        if not pattern:
            return []

        # Start with entities matching first type
        first_type, first_rel = pattern[0]
        candidates = [[e.id] for e in self.graph.get_entities_by_type(first_type)]

        # Extend paths for each pattern step
        for i in range(1, len(pattern)):
            entity_type, _ = pattern[i]
            prev_rel = pattern[i - 1][1]

            new_candidates = []
            for path in candidates:
                last_id = path[-1]

                for neighbor, relation in self.graph.get_neighbors(last_id):
                    if prev_rel and relation.type != prev_rel:
                        continue
                    if neighbor.type != entity_type:
                        continue

                    new_candidates.append(path + [neighbor.id])

            candidates = new_candidates

        return candidates
