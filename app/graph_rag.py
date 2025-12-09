"""
Graph-RAG integration module (Advanced).
For multi-hop and entity-centric queries, combine knowledge graph with dense retrieval.

Pattern:
1. Entity extraction and linking populates a graph
2. Graph traversal finds connected evidence 2-3 hops away
3. Convert graph subgraphs to natural-language summaries for context
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A node in the knowledge graph."""

    id: str
    type: str  # Customer, Contract, Product, Region, etc.
    name: str
    properties: Dict = None

    def __post_init__(self):
        self.properties = self.properties or {}


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    type: str  # PERTAINS_TO, SIGNED_BY, LOCATED_IN, etc.
    properties: Dict = None

    def __post_init__(self):
        self.properties = self.properties or {}


class SimpleKnowledgeGraph:
    """
    In-memory knowledge graph for demonstration.

    In production, use Neo4j, Amazon Neptune, or similar.
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

        # Update adjacency list
        if relation.source_id not in self._adjacency:
            self._adjacency[relation.source_id] = []
        self._adjacency[relation.source_id].append((relation.target_id, relation))

        # Add reverse edge for traversal
        if relation.target_id not in self._adjacency:
            self._adjacency[relation.target_id] = []
        self._adjacency[relation.target_id].append((relation.source_id, relation))

    def get_neighbors(self, entity_id: str, max_hops: int = 2) -> Dict[str, Set[str]]:
        """
        Get all entities within max_hops of the given entity.

        Returns dict mapping hop distance to set of entity IDs.
        """
        visited: Set[str] = set()
        result: Dict[str, Set[str]] = {i: set() for i in range(max_hops + 1)}

        # BFS
        queue = [(entity_id, 0)]
        visited.add(entity_id)
        result[0].add(entity_id)

        while queue:
            current_id, hop = queue.pop(0)

            if hop >= max_hops:
                continue

            for neighbor_id, _ in self._adjacency.get(current_id, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    result[hop + 1].add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))

        return result

    def get_subgraph(
        self, entity_ids: List[str], max_hops: int = 2
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract a subgraph containing the given entities and their neighbors.
        """
        # Collect all relevant entity IDs
        all_ids: Set[str] = set()
        for eid in entity_ids:
            neighbors = self.get_neighbors(eid, max_hops)
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

    def subgraph_to_text(
        self, entities: List[Entity], relations: List[Relation]
    ) -> str:
        """
        Convert a subgraph to natural language for LLM context.

        Example output:
        - Customer "Acme Corp" is located in region "APAC"
        - Contract "C-2024-001" pertains to Product "Enterprise Suite"
        - Contract "C-2024-001" was signed by Customer "Acme Corp"
        """
        lines = []

        # Entity descriptions
        entity_map = {e.id: e for e in entities}

        for relation in relations:
            source = entity_map.get(relation.source_id)
            target = entity_map.get(relation.target_id)

            if source and target:
                # Format relation as natural language
                rel_type = relation.type.replace("_", " ").lower()
                lines.append(
                    f'{source.type} "{source.name}" {rel_type} '
                    f'{target.type} "{target.name}"'
                )

        return "\n".join(f"- {line}" for line in lines)


class EntityExtractor:
    """
    Extract entities from text for graph population.

    In production, use spaCy NER, GPT-4, or domain-specific models.
    """

    def __init__(self):
        # Simple keyword-based extraction for demo
        self.entity_patterns = {
            "Customer": ["customer", "client", "buyer"],
            "Contract": ["contract", "agreement", "deal"],
            "Product": ["product", "service", "solution"],
            "Region": ["region", "territory", "area", "apac", "emea", "americas"],
        }

    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text.

        This is a simplified implementation. In production, use:
        - spaCy for NER
        - OpenAI function calling for structured extraction
        - Domain-specific entity recognition
        """
        entities = []
        text_lower = text.lower()

        # Very basic extraction (demo only)
        for entity_type, keywords in self.entity_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # In reality, you'd extract the actual entity name
                    entities.append(
                        Entity(
                            id=f"{entity_type.lower()}_{keyword}",
                            type=entity_type,
                            name=keyword.title(),
                        )
                    )

        return entities


class GraphRAGRetriever:
    """
    Combine graph traversal with vector retrieval.
    """

    def __init__(self, graph: SimpleKnowledgeGraph, vector_retriever=None):
        self.graph = graph
        self.vector_retriever = vector_retriever
        self.entity_extractor = EntityExtractor()

    def retrieve(self, query: str, top_k: int = 5, graph_hops: int = 2) -> Dict:
        """
        Hybrid retrieval combining graph and vector search.

        Steps:
        1. Extract entities from query
        2. Get graph subgraph for entities
        3. Run vector retrieval
        4. Merge graph context with retrieved chunks
        """
        # 1. Extract entities from query
        query_entities = self.entity_extractor.extract(query)
        entity_ids = [e.id for e in query_entities]

        # 2. Get graph context
        graph_context = ""
        if entity_ids:
            entities, relations = self.graph.get_subgraph(entity_ids, graph_hops)
            if relations:
                graph_context = self.graph.subgraph_to_text(entities, relations)
                logger.info(
                    f"Graph context: {len(entities)} entities, {len(relations)} relations"
                )

        # 3. Vector retrieval (if available)
        vector_results = []
        if self.vector_retriever:
            vector_results = self.vector_retriever.retrieve(query, top_k)

        return {
            "query_entities": [
                {"id": e.id, "type": e.type, "name": e.name} for e in query_entities
            ],
            "graph_context": graph_context,
            "vector_results": vector_results,
        }


# Example usage for multi-hop query:
# "Which products are impacted by delayed contracts in APAC?"
#
# This requires:
# 1. Finding contracts with status="Delayed"
# 2. Filtering by region="APAC" (via customer relationship)
# 3. Following PERTAINS_TO relation to get products
#
# Neo4j Cypher equivalent:
# MATCH (c:Contract)-[:PERTAINS_TO]->(p:Product)
# MATCH (c)-[:SIGNED_BY]->(cust:Customer)-[:LOCATED_IN]->(r:Region {name: "APAC"})
# WHERE c.status = "Delayed"
# RETURN c, p, cust, r
