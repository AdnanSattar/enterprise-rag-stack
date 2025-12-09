"""
Knowledge graph construction and management.

Example entities: Customers, Contracts, Products, Regions
Simple adjacency: Contract -pertains_to-> Product

In production, use Neo4j, Amazon Neptune, or similar.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    type: str  # Customer, Contract, Product, Region, etc.
    name: str
    properties: Dict = field(default_factory=dict)


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    type: str  # PERTAINS_TO, SIGNED_BY, LOCATED_IN, etc.
    properties: Dict = field(default_factory=dict)


class SimpleKnowledgeGraph:
    """
    In-memory knowledge graph for demonstration.

    In production, use:
    - Neo4j for complex graph queries
    - Amazon Neptune for managed service
    - TigerGraph for high-scale analytics

    Usage:
        graph = SimpleKnowledgeGraph()
        graph.add_entity(Entity(id="c1", type="Customer", name="Acme Corp"))
        graph.add_relation(Relation(source_id="c1", target_id="r1", type="LOCATED_IN"))
    """

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self._adjacency: Dict[str, List[Tuple[str, Relation]]] = {}

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        if entity.id not in self._adjacency:
            self._adjacency[entity.id] = []

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the graph."""
        self.relations.append(relation)

        # Update adjacency list (bidirectional for traversal)
        if relation.source_id not in self._adjacency:
            self._adjacency[relation.source_id] = []
        self._adjacency[relation.source_id].append((relation.target_id, relation))

        if relation.target_id not in self._adjacency:
            self._adjacency[relation.target_id] = []
        self._adjacency[relation.target_id].append((relation.source_id, relation))

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.type == entity_type]

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: str = None,
    ) -> List[Tuple[Entity, Relation]]:
        """
        Get neighboring entities.

        Args:
            entity_id: Source entity ID
            relation_type: Filter by relation type (optional)

        Returns:
            List of (entity, relation) tuples
        """
        neighbors = []
        for neighbor_id, relation in self._adjacency.get(entity_id, []):
            if relation_type and relation.type != relation_type:
                continue
            neighbor = self.entities.get(neighbor_id)
            if neighbor:
                neighbors.append((neighbor, relation))
        return neighbors

    def get_subgraph(
        self,
        entity_ids: List[str],
        max_hops: int = 2,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract a subgraph containing the given entities and their neighbors.

        Args:
            entity_ids: Seed entity IDs
            max_hops: Maximum traversal depth

        Returns:
            (entities, relations) in subgraph
        """
        from .graph_traversal import GraphTraverser

        traverser = GraphTraverser(self)
        all_ids: Set[str] = set()

        for eid in entity_ids:
            neighbors = traverser.bfs(eid, max_hops)
            for hop_set in neighbors.values():
                all_ids.update(hop_set)

        # Extract entities
        entities = [self.entities[eid] for eid in all_ids if eid in self.entities]

        # Extract relations between these entities
        relations = [
            r
            for r in self.relations
            if r.source_id in all_ids and r.target_id in all_ids
        ]

        return entities, relations

    def query_by_property(
        self,
        entity_type: str,
        property_name: str,
        property_value: str,
    ) -> List[Entity]:
        """
        Query entities by property value.

        Args:
            entity_type: Type filter
            property_name: Property key
            property_value: Property value to match

        Returns:
            Matching entities
        """
        return [
            e
            for e in self.entities.values()
            if e.type == entity_type
            and e.properties.get(property_name) == property_value
        ]

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        entity_types = {}
        for e in self.entities.values():
            entity_types[e.type] = entity_types.get(e.type, 0) + 1

        relation_types = {}
        for r in self.relations:
            relation_types[r.type] = relation_types.get(r.type, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def to_cypher_export(self) -> str:
        """
        Export graph as Cypher statements for Neo4j import.

        Returns:
            Cypher CREATE statements
        """
        lines = []

        # Create entities
        for entity in self.entities.values():
            props = ", ".join(f'{k}: "{v}"' for k, v in entity.properties.items())
            if props:
                props = ", " + props
            lines.append(
                f'CREATE (n:{entity.type} {{id: "{entity.id}", name: "{entity.name}"{props}}})'
            )

        # Create relations
        for rel in self.relations:
            props = ", ".join(f'{k}: "{v}"' for k, v in rel.properties.items())
            props_clause = f" {{{props}}}" if props else ""
            lines.append(
                f'MATCH (a {{id: "{rel.source_id}"}}), (b {{id: "{rel.target_id}"}}) '
                f"CREATE (a)-[:{rel.type}{props_clause}]->(b)"
            )

        return "\n".join(lines)
