"""
Knowledge graph summarization for LLM context.

Converts graph subgraphs to natural-language summaries
that can be added to the LLM prompt context.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .graph_builder import Entity, Relation

logger = logging.getLogger(__name__)


def subgraph_to_text(
    entities: List[Entity],
    relations: List[Relation],
    format_style: str = "natural",
) -> str:
    """
    Convert a subgraph to natural language for LLM context.

    Format styles:
    - natural: Prose-like sentences
    - bullet: Bullet point list
    - structured: Section-organized

    Example output (natural):
    - Customer "Acme Corp" is located in region "APAC"
    - Contract "C-2024-001" pertains to Product "Enterprise Suite"
    - Contract "C-2024-001" was signed by Customer "Acme Corp"

    Args:
        entities: List of entities in subgraph
        relations: List of relations in subgraph
        format_style: Output format

    Returns:
        Natural language description
    """
    if not entities or not relations:
        return ""

    entity_map = {e.id: e for e in entities}

    if format_style == "bullet":
        return _format_bullet(entity_map, relations)
    elif format_style == "structured":
        return _format_structured(entity_map, relations)
    else:
        return _format_natural(entity_map, relations)


def _format_natural(
    entity_map: Dict[str, Entity],
    relations: List[Relation],
) -> str:
    """Format as natural language sentences."""
    lines = []

    for relation in relations:
        source = entity_map.get(relation.source_id)
        target = entity_map.get(relation.target_id)

        if source and target:
            rel_type = relation.type.replace("_", " ").lower()
            lines.append(
                f'{source.type} "{source.name}" {rel_type} '
                f'{target.type} "{target.name}"'
            )

    return "\n".join(f"- {line}" for line in lines)


def _format_bullet(
    entity_map: Dict[str, Entity],
    relations: List[Relation],
) -> str:
    """Format as bullet points grouped by entity."""
    entity_facts: Dict[str, List[str]] = {}

    for relation in relations:
        source = entity_map.get(relation.source_id)
        target = entity_map.get(relation.target_id)

        if source and target:
            rel_type = relation.type.replace("_", " ").lower()

            if source.id not in entity_facts:
                entity_facts[source.id] = []
            entity_facts[source.id].append(f'{rel_type} {target.type} "{target.name}"')

    lines = []
    for entity_id, facts in entity_facts.items():
        entity = entity_map[entity_id]
        lines.append(f"**{entity.type}: {entity.name}**")
        for fact in facts:
            lines.append(f"  - {fact}")

    return "\n".join(lines)


def _format_structured(
    entity_map: Dict[str, Entity],
    relations: List[Relation],
) -> str:
    """Format with sections by entity type."""
    by_type: Dict[str, List[str]] = {}

    for relation in relations:
        source = entity_map.get(relation.source_id)
        target = entity_map.get(relation.target_id)

        if source and target:
            rel_type = relation.type.replace("_", " ").lower()
            statement = (
                f'{source.type} "{source.name}" {rel_type} '
                f'{target.type} "{target.name}"'
            )

            if source.type not in by_type:
                by_type[source.type] = []
            by_type[source.type].append(statement)

    sections = []
    for entity_type, statements in sorted(by_type.items()):
        sections.append(f"[{entity_type} Relationships]")
        for stmt in statements:
            sections.append(f"- {stmt}")
        sections.append("")

    return "\n".join(sections)


def summarize_path(
    path: List[Tuple[str, str, str]],
    entity_map: Dict[str, Entity],
) -> str:
    """
    Summarize a path through the graph as a sentence.

    Args:
        path: List of (source_id, relation_type, target_id) tuples
        entity_map: Entity ID to Entity mapping

    Returns:
        Natural language path description
    """
    if not path:
        return ""

    parts = []

    for i, (source_id, rel_type, target_id) in enumerate(path):
        source = entity_map.get(source_id)
        target = entity_map.get(target_id)

        if not source or not target:
            continue

        rel_text = rel_type.replace("_", " ").lower()

        if i == 0:
            parts.append(f'{source.type} "{source.name}"')

        parts.append(f'{rel_text} {target.type} "{target.name}"')

    return " → ".join(parts) if parts else ""


def generate_evidence_summary(
    entities: List[Entity],
    relations: List[Relation],
    query_context: str = None,
) -> str:
    """
    Generate an evidence summary for LLM context augmentation.

    Creates a focused summary that:
    1. Highlights query-relevant relationships
    2. Provides entity context
    3. Shows multi-hop connections

    Args:
        entities: Entities in subgraph
        relations: Relations in subgraph
        query_context: Original query for relevance weighting

    Returns:
        Evidence summary text
    """
    if not entities:
        return "[No graph evidence found]"

    entity_map = {e.id: e for e in entities}

    # Build summary
    summary_parts = ["[Graph Evidence]"]

    # Entity overview
    entity_types = {}
    for e in entities:
        entity_types[e.type] = entity_types.get(e.type, 0) + 1

    type_summary = ", ".join(f"{count} {t}(s)" for t, count in entity_types.items())
    summary_parts.append(f"Entities found: {type_summary}")
    summary_parts.append("")

    # Key relationships
    summary_parts.append("Key relationships:")
    summary_parts.append(subgraph_to_text(entities, relations, format_style="natural"))

    return "\n".join(summary_parts)


class GraphRAGRetriever:
    """
    Combine graph traversal with vector retrieval.

    Usage:
        retriever = GraphRAGRetriever(graph, vector_retriever)
        results = retriever.retrieve("Which products are delayed in APAC?")
    """

    def __init__(self, graph, vector_retriever=None):
        self.graph = graph
        self.vector_retriever = vector_retriever

        from .entity_linking import EntityExtractor

        self.entity_extractor = EntityExtractor()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        graph_hops: int = 2,
    ) -> Dict:
        """
        Hybrid retrieval combining graph and vector search.

        Steps:
        1. Extract entities from query
        2. Get graph subgraph for entities
        3. Run vector retrieval
        4. Merge graph context with retrieved chunks

        Args:
            query: User query
            top_k: Number of vector results
            graph_hops: Graph traversal depth

        Returns:
            Combined retrieval results
        """
        # 1. Extract entities from query
        query_entities = self.entity_extractor.extract(query)
        entity_ids = [
            f"{e.type.lower()}_{e.text.lower().replace(' ', '_')}"
            for e in query_entities
        ]

        # 2. Get graph context
        graph_context = ""
        if entity_ids:
            # Find matching entities in graph
            matching_ids = [eid for eid in entity_ids if eid in self.graph.entities]

            if matching_ids:
                entities, relations = self.graph.get_subgraph(matching_ids, graph_hops)
                if relations:
                    graph_context = generate_evidence_summary(
                        entities, relations, query
                    )
                    logger.info(
                        f"Graph context: {len(entities)} entities, {len(relations)} relations"
                    )

        # 3. Vector retrieval (if available)
        vector_results = []
        if self.vector_retriever:
            vector_results = self.vector_retriever.retrieve(query, top_k)

        return {
            "query_entities": [
                {"type": e.type, "text": e.text} for e in query_entities
            ],
            "graph_context": graph_context,
            "vector_results": vector_results,
        }
